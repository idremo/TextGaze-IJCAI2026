import argparse
import torch
from PIL import Image
import json
import os
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from gazefuze.model import get_gazefuze_model
from gazefuze.utils import vat_auc, vat_l2
import warnings
warnings.filterwarnings("ignore", message="The default value of the antialias parameter")


def str2bool(value):
    if isinstance(value, bool):
        return value
    return value.lower() in ("1", "true", "yes", "y")

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="datasets/childplay_Qwen_En/images")
parser.add_argument("--model_name", type=str, default="gazefuze_dinov2_vitb14_inout")
parser.add_argument("--ckpt_path", type=str, default="TextGaze_VAT_ViT-B_epoch6.pt")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--filter_empty_frames", type=str2bool, default=True, help="过滤无人员标注的帧")
# 新增：控制是否使用文本（便于对比实验）
parser.add_argument("--use_text", type=str2bool, default=True, help="是否使用dataloader返回的texts字段")
args = parser.parse_args()

class VideoAttentionTarget(torch.utils.data.Dataset):
    def __init__(self, path, img_transform, filter_empty_frames=True, use_text=True):
        self.json_path = os.path.join(os.path.dirname(path), "test_preprocessed.json")
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"未找到JSON标注文件：{self.json_path}")
        
        self.sequences = json.load(open(self.json_path, "r", encoding="utf-8"))
        self.frames = []
        self.use_text = use_text  # 控制是否读取文本
        
        # 收集有效帧
        for seq_idx in range(len(self.sequences)):
            seq = self.sequences[seq_idx]
            for frame_idx in range(len(seq['frames'])):
                frame = seq['frames'][frame_idx]
                if filter_empty_frames and len(frame.get('heads', [])) == 0:
                    continue
                self.frames.append((seq_idx, frame_idx))
        
        self.path = path
        self.transform = img_transform
        print(f"数据集加载完成：共{len(self.sequences)}个序列，{len(self.frames)}个有效帧")
        print(f"是否启用文本输入：{self.use_text}")

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.frames[idx]
        seq = self.sequences[seq_idx]
        frame = seq['frames'][frame_idx]
        
        # 构建图像路径
        img_full_path = os.path.join(self.path, frame['path'])
        if not os.path.exists(img_full_path):
            raise FileNotFoundError(f"图像文件不存在：{img_full_path}")
        
        # 加载图像
        image = self.transform(Image.open(img_full_path).convert("RGB"))
        
        # 收集标注（支持多人员）
        bboxes = []
        gazex = []
        gazey = []
        inout = []
        texts = []  # 新增：存储每个人员的text
        
        for head in frame.get('heads', []):
            # 校验边界框
            bbox_norm = head.get('bbox_norm', [])
            if len(bbox_norm) != 4 or not (0 <= bbox_norm[0] < bbox_norm[2] <= 1) or not (0 <= bbox_norm[1] < bbox_norm[3] <= 1):
                print(f"警告：跳过无效边界框（序列{seq['path']}，帧{frame['path']}，person_id={head.get('person_id', '未知')}）")
                continue
            
            # 收集基础标注
            bboxes.append(bbox_norm)
            gazex.append(head.get('gazex_norm', [-1.0]))
            gazey.append(head.get('gazey_norm', [-1.0]))
            inout.append(head.get('inout', 2))
            
            # 新增：读取文本（与人员一一对应）
            if self.use_text:
                text = head.get('text', "").strip()  # 无text时用空字符串
                texts.append(text)
            else:
                texts.append("")  # 不使用文本时传入空字符串
        
        # 返回值新增texts，与其他标注一一对应
        return image, bboxes, gazex, gazey, inout, texts

    def __len__(self):
        return len(self.frames)
    
def collate(batch):
    """ 修复：处理texts字段，保持与其他标注维度一致 """
    # 新增texts接收
    images, bboxes, gazex, gazey, inout, texts = zip(*batch)
    
    # 过滤无有效标注的样本
    valid_indices = [i for i in range(len(bboxes)) if len(bboxes[i]) > 0]
    if len(valid_indices) < len(bboxes):
        print(f"警告：当前batch中有{len(bboxes)-len(valid_indices)}个样本无有效人员标注，已过滤")
    
    # 保留有效样本（同步过滤texts）
    images = torch.stack([images[i] for i in valid_indices])
    bboxes = [bboxes[i] for i in valid_indices]
    gazex = [gazex[i] for i in valid_indices]
    gazey = [gazey[i] for i in valid_indices]
    inout = [inout[i] for i in valid_indices]
    texts = [texts[i] for i in valid_indices]  # 过滤后仍为嵌套列表：[["text1", "text2"], ["text3"], ...]
    
    # 关键：展平texts为一维列表（适配模型tokenizer批量处理）
    flattened_texts = []
    for img_texts in texts:
        flattened_texts.extend(img_texts)
    
    # 记录每个图像的人员数（用于模型内部拆分文本特征）
    num_ppl_per_img = [len(bbox) for bbox in bboxes]
    
    return images, bboxes, gazex, gazey, inout, flattened_texts, num_ppl_per_img


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    # 加载模型
    model, transform = get_gazefuze_model(args.model_name)
    model.load_gazefuze_state_dict(torch.load(args.ckpt_path, weights_only=True))
    model.to(device)
    model.eval()

    # 加载数据集（支持文本输入）
    dataset = VideoAttentionTarget(
        args.data_path, 
        transform, 
        filter_empty_frames=args.filter_empty_frames,
        use_text=args.use_text
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate,
        shuffle=False,
        pin_memory=True if device == "cuda" else False
    )

    # 初始化指标
    aucs = []
    l2s = []
    inout_preds = []
    inout_gts = []

    # 开始评测
    for batch_idx, (images, bboxes, gazex, gazey, inout, flattened_texts, num_ppl_per_img) in tqdm(
        enumerate(dataloader), desc="Evaluating", total=len(dataloader)
    ):
        if images.shape[0] == 0:
            continue
        
        # 模型输入：新增texts和num_ppl_per_img（适配多人员文本）
        model_input = {
            "images": images.to(device), 
            "bboxes": bboxes,
            "num_ppl_per_img": num_ppl_per_img  # 告知模型每个图像的人员数
        }
        if args.use_text:
            model_input["texts"] = flattened_texts  # 仅在启用时传入文本
        
        preds = model.forward(model_input)
        
        # 逐样本、逐人员计算指标（顺序与输入一致）
        img_idx = 0
        for ppl_cnt in num_ppl_per_img:  # 遍历每个图像的人员数
            for head_idx in range(ppl_cnt):  # 遍历每个人员
                # 获取真实标签
                gt_inout = inout[img_idx][head_idx]
                gt_gazex = gazex[img_idx][head_idx][0]
                gt_gazey = gazey[img_idx][head_idx][0]
                
                # 计算AUC和L2
                if gt_inout == 1 and gt_gazex >= 0 and gt_gazey >= 0:
                    try:
                        auc = vat_auc(preds['heatmap'][img_idx][head_idx], gt_gazex, gt_gazey)
                        l2 = vat_l2(preds['heatmap'][img_idx][head_idx], gt_gazex, gt_gazey)
                        aucs.append(auc)
                        l2s.append(l2)
                    except Exception as e:
                        print(f"警告：计算AUC/L2失败（图像索引{img_idx}，人员索引{head_idx}），错误：{e}")
                
                # 收集Inout分类指标
                if gt_inout in [0, 1]:
                    try:
                        inout_preds.append(preds['inout'][img_idx][head_idx].item())
                        inout_gts.append(gt_inout)
                    except Exception as e:
                        print(f"警告：收集inout预测失败（图像索引{img_idx}，人员索引{head_idx}），错误：{e}")
            img_idx += 1  # 切换到下一个图像

    # 输出核心指标
    print("\n===== 核心评估指标（仅inout=1的有效样本） =====")
    print(f"AUC: {np.array(aucs).mean():.4f}" if len(aucs) > 0 else "AUC: 无有效样本")
    print(f"Avg L2: {np.array(l2s).mean():.4f}" if len(l2s) > 0 else "Avg L2: 无有效样本")
    print(f"参与AUC/L2计算的样本数: {len(aucs)}")

    # 输出Inout AP指标
    print("\n===== Inout分类评估指标（仅inout=0/1的样本） =====")
    print(f"参与AP计算的样本数: {len(inout_gts)}")
    if len(inout_gts) > 0:
        inout_ap = average_precision_score(inout_gts, inout_preds)
        print(f"Inout AP: {inout_ap:.4f}")
        print(f"类别分布 - inout=0: {inout_gts.count(0)}个, inout=1: {inout_gts.count(1)}个")
    else:
        print("警告：无有效样本参与Inout AP计算（所有样本inout=2或标注无效）")
    
    # 输出文本配置信息
    print(f"\n===== 文本配置 =====")
    print(f"是否使用文本：{args.use_text}")
    if args.use_text:
        print(f"本次评测传入文本")
    else:
        print("本次评测未使用文本")

if __name__ == "__main__":
    main()