import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from typing import List, Dict, Optional, Tuple

import math

import gazefuze.utils as utils
from gazefuze.backbone import DinoV2Backbone

from transformers import AutoModel, AutoTokenizer
from .vl_transformer import build_vl_transformer


# 2D正弦位置编码实现
def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model_half = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model_half, 2) *
                         -(math.log(10000.0) / d_model_half))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model_half:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model_half:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model_half::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model_half + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class GazeFuze(nn.Module):
    def __init__(self, 
                 backbone, 
                 inout=False, 
                 dim=256, 
                 num_layers=3, 
                 in_size=(448, 448), 
                 out_size=(64, 64),
                 text_encoder_name: str = 'TextGaze-IJCAI2026/models/bert-base-uncased',
                 args=None):
        super().__init__()
        self.backbone = backbone
        self.dim = dim
        self.num_layers = num_layers
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.inout = inout
        
        # 配置参数初始化
        self.args = args
        if self.args is None:
            from argparse import Namespace
            self.args = Namespace(
                vl_hidden_dim=dim,
                vl_nheads=8,
                vl_enc_layers=num_layers,
                vl_dim_feedforward=2048,
                vl_dropout=0.1,
                vl_activation='relu',
                text_supervision_weight=0.1  # 文本监督权重
            )
        
        # 视觉特征投影
        self.linear = nn.Conv2d(backbone.get_dimension(), self.dim, 1)
        
        # 特殊Token定义
        self.head_token = nn.Embedding(1, self.dim)
        if self.inout:
            self.inout_token = nn.Embedding(1, self.dim)
        
        # 文本处理相关组件
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, self.dim)
        
        # 文本监督投影层（每个Transformer层对应一个）
        self.text_supervision_proj = nn.ModuleList([
            nn.Linear(dim, self.text_encoder.config.hidden_size)
            for _ in range(self.args.vl_enc_layers)
        ])
        
        # 计算各部分token数量
        self.num_visu_token = self.featmap_h * self.featmap_w  # 视觉token数量
        self.num_text_token = 64  # 文本token数量
        self.num_special_tokens = 1 if inout else 0  # 特殊token数量（inout）
        
        # 可学习位置编码（仅用于文本token，不包含inout token）
        self.learnable_pos_embed = nn.Embedding(
            self.num_text_token,  # 只给文本token位置编码
            self.dim
        )
        
        # 视觉区域2D正弦位置编码（固定，非学习）
        self.register_buffer(
            "visu_pos_embed", 
            positionalencoding2d(self.dim, self.featmap_h, self.featmap_w).flatten(start_dim=1).permute(1, 0)
            # 形状: [H*W, dim] - 每个视觉token对应一个位置编码
        )
        
        self.vl_transformer = build_vl_transformer(self.args)
        
        # 预测头
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        if self.inout:
            self.inout_head = nn.Sequential(
                nn.Linear(self.dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

    def get_input_head_maps(self, bboxes: List[List[Tuple[float, float, float, float]]]) -> List[Tensor]:
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for bbox in bbox_list:
                if bbox is None:
                    img_head_maps.append(torch.zeros(self.featmap_h, self.featmap_w))
                else:
                    xmin, ymin, xmax, ymax = bbox
                    width, height = self.featmap_w, self.featmap_h
                    xmin = max(0, min(round(xmin * width), width - 1))
                    ymin = max(0, min(round(ymin * height), height - 1))
                    xmax = max(xmin + 1, min(round(xmax * width), width))
                    ymax = max(ymin + 1, min(round(ymax * height), height))
                    head_map = torch.zeros((height, width))
                    head_map[ymin:ymax, xmin:xmax] = 1
                    img_head_maps.append(head_map)
            head_maps.append(torch.stack(img_head_maps))
        return head_maps

    def process_text_features(self, texts: List[str], device: torch.device, num_ppl_per_img: List[int]) -> Tuple[Tensor, Tensor, Tensor]:
        """扩展返回文本特征作为监督目标"""
        text_inputs = self.tokenizer(
            texts, 
            return_tensors='pt', 
            padding='max_length', 
            truncation=True,
            max_length=self.num_text_token
        ).to(device)
        attention_mask = text_inputs.attention_mask
        
        text_outputs = self.text_encoder(** text_inputs)
        text_feat = text_outputs.last_hidden_state  # 用于监督的目标特征
        text_proj_feat = self.text_proj(text_feat)  # 用于跨模态融合的特征
        
        text_proj_feat = utils.repeat_tensors(text_proj_feat, num_ppl_per_img)
        text_src = text_proj_feat.permute(1, 0, 2)
        
        text_mask = (attention_mask == 0)
        text_mask = utils.repeat_tensors(text_mask, num_ppl_per_img)
        
        # 重复原始文本特征用于监督
        text_feat = utils.repeat_tensors(text_feat, num_ppl_per_img)
        
        return text_src, text_mask, text_feat

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Optional[Tensor]]:
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        bs_total = sum(num_ppl_per_img)
        device = input["images"].device

        # 1. 图像特征提取
        x = self.backbone.forward(input["images"])
        visu_feat = self.linear(x)
        
        # 2. 生成人脸区域掩码并与视觉特征融合
        head_maps = self.get_input_head_maps(input["bboxes"])
        head_maps = torch.cat(head_maps, dim=0).to(device)
        head_map_embeddings = head_maps.unsqueeze(1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        
        # 3. 重复视觉特征并融入人脸掩码信息
        visu_feat_repeated = utils.repeat_tensors(visu_feat, num_ppl_per_img)
        visu_feat_with_head = visu_feat_repeated + head_map_embeddings
        
        # 保存原始视觉特征用于残差连接
        original_visu_feat = visu_feat_with_head
        
        # 4. 文本特征处理（获取监督目标）
        text_src, text_mask, text_feat_target = self.process_text_features(input["texts"], device, num_ppl_per_img)
        
        # 5. 特征序列组装
        token_list = []
        mask_list = []
        
        if self.inout:
            # inout token放在序列首位，不添加位置编码
            inout_src = self.inout_token.weight.unsqueeze(1).repeat(1, bs_total, 1)
            inout_mask = torch.zeros((bs_total, 1), device=device, dtype=torch.bool)
            token_list.append(inout_src)
            mask_list.append(inout_mask)
        
        token_list.append(text_src)
        mask_list.append(text_mask)
        
        visu_src = visu_feat_with_head.flatten(start_dim=2).permute(2, 0, 1)
        visu_mask = torch.zeros((bs_total, self.num_visu_token), device=device, dtype=torch.bool)
        token_list.append(visu_src)
        mask_list.append(visu_mask)
        
        vl_src = torch.cat(token_list, dim=0)
        vl_mask = torch.cat(mask_list, dim=1)
        
        # 6. 混合位置编码
        # 6.1 生成可学习位置编码（仅文本token）
        learnable_pos_ids = torch.arange(
            self.num_text_token,  # 只生成文本token的位置编码
            device=device
        )
        learnable_pos = self.learnable_pos_embed(learnable_pos_ids).unsqueeze(1).repeat(1, bs_total, 1)
        
        # 6.2 获取视觉区域的正弦位置编码并扩展到批次维度
        visu_pos = self.visu_pos_embed.unsqueeze(1).repeat(1, bs_total, 1).to(device)
        
        # 6.3 拼接所有位置编码（inout token没有位置编码）
        if self.inout:
            # 序列结构: [inout_token(无编码), text_tokens(有编码), visu_tokens(有编码)]
            vl_pos = torch.cat([
                torch.zeros(1, bs_total, self.dim, device=device),  # inout token位置编码为0（不使用）
                learnable_pos, 
                visu_pos
            ], dim=0)
        else:
            # 无inout token的情况
            vl_pos = torch.cat([learnable_pos, visu_pos], dim=0)
        
        # 验证位置编码长度匹配
        assert vl_pos.shape[0] == vl_src.shape[0], \
            f"位置编码长度与特征序列不匹配：pos_len={vl_pos.shape[0]}, src_len={vl_src.shape[0]}"
        
        # 7. 跨模态融合（获取各层输出用于监督）
        layer_outputs, final_output = self.vl_transformer(vl_src, vl_mask, vl_pos)
        
        # 8. 计算文本细粒度监督损失
        text_supervision_loss = 0.0
        # 文本token起始位置：如果有inout token则从1开始，否则从0开始
        token_idx = 1 if self.inout else 0
        
        for i, layer_out in enumerate(layer_outputs):
            # 提取文本token部分的输出
            text_layer_out = layer_out[token_idx:token_idx+self.num_text_token]  # [T, B, D]
            text_layer_out = text_layer_out.permute(1, 0, 2)  # [B, T, D]
            
            # 投影到原始文本特征空间
            pred_text_feat = self.text_supervision_proj[i](text_layer_out)
            
            # 计算MSE损失
            text_supervision_loss += F.mse_loss(
                pred_text_feat, 
                text_feat_target,
                reduction='mean'
            )
        
        # 平均各层损失并加权
        text_supervision_loss = text_supervision_loss / len(layer_outputs) * self.args.text_supervision_weight
        
        # 9. 预测输出
        vg_hs = final_output
        token_idx = 0
        inout_preds = None
        
        if self.inout:
            inout_tokens = vg_hs[token_idx]
            inout_preds = self.inout_head(inout_tokens).squeeze(dim=-1)
            inout_preds = utils.split_tensors(inout_preds, num_ppl_per_img)
            token_idx += 1
        
        # 提取Transformer输出的视觉特征
        visu_features_transformer = vg_hs[token_idx + self.num_text_token:]
        visu_features_transformer = visu_features_transformer.permute(1, 2, 0).view(bs_total, self.dim, self.featmap_h, self.featmap_w)
        
        # 增加视觉特征残差连接：Transformer输出 + 原始视觉特征
        visu_features = visu_features_transformer + original_visu_feat
        
        x = self.heatmap_head(visu_features).squeeze(dim=1)
        x = torchvision.transforms.functional.resize(x, self.out_size, antialias=True)
        heatmap_preds = utils.split_tensors(x, num_ppl_per_img)
        
        return {
            "heatmap": heatmap_preds, 
            "inout": inout_preds if self.inout else None,
            "text_supervision_loss": text_supervision_loss
        }

    def get_gazefuze_state_dict(self, include_backbone: bool = False) -> Dict[str, Tensor]:
        if include_backbone:
            return self.state_dict()
        else:
            return {k: v for k, v in self.state_dict().items() if not k.startswith("backbone")}
        
    def load_gazefuze_state_dict(self, ckpt_state_dict: Dict[str, Tensor], include_backbone: bool = False) -> None:
        current_state_dict = self.state_dict()
        keys1 = set(current_state_dict.keys())
        keys2 = set(ckpt_state_dict.keys())

        if not include_backbone:
            keys1 = {k for k in keys1 if not k.startswith("backbone")}
            keys2 = {k for k in keys2 if not k.startswith("backbone")}

        if len(keys2 - keys1) > 0:
            print("WARNING unused keys in checkpoint: ", keys2 - keys1)
        if len(keys1 - keys2) > 0:
            print("WARNING missing keys in checkpoint: ", keys1 - keys2)

        for k in keys1 & keys2:
            current_state_dict[k] = ckpt_state_dict[k]
        self.load_state_dict(current_state_dict, strict=False)


def get_gazefuze_model(model_name: str, args=None) -> Tuple[nn.Module, callable]:
    factory = {
        "gazefuze_dinov2_vitb14": lambda: gazefuze_dinov2_vitb14(args),
        "gazefuze_dinov2_vitl14": lambda: gazefuze_dinov2_vitl14(args),
        "gazefuze_dinov2_vitb14_inout": lambda: gazefuze_dinov2_vitb14_inout(args),
        "gazefuze_dinov2_vitl14_inout": lambda: gazefuze_dinov2_vitl14_inout(args),
    }
    assert model_name in factory.keys(), f"invalid model name: {model_name}"
    return factory[model_name]()

def gazefuze_dinov2_vitb14(args) -> Tuple[nn.Module, callable]:
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = GazeFuze(backbone, args=args)
    return model, transform

def gazefuze_dinov2_vitl14(args) -> Tuple[nn.Module, callable]:
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = GazeFuze(backbone, args=args)
    return model, transform

def gazefuze_dinov2_vitb14_inout(args) -> Tuple[nn.Module, callable]:
    backbone = DinoV2Backbone('dinov2_vitb14')
    transform = backbone.get_transform((448, 448))
    model = GazeFuze(backbone, inout=True, args=args)
    return model, transform

def gazefuze_dinov2_vitl14_inout(args) -> Tuple[nn.Module, callable]:
    backbone = DinoV2Backbone('dinov2_vitl14')
    transform = backbone.get_transform((448, 448))
    model = GazeFuze(backbone, inout=True, args=args)
    return model, transform