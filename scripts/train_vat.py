import argparse
from datetime import datetime
import numpy as np
import os
import random
from sklearn.metrics import average_precision_score
import torch
import torch.nn as nn
import logging
from tqdm import tqdm

from gazefuze.dataloader import GazeDataset, collate_fn
from gazefuze.model import get_gazefuze_model
from gazefuze.utils import vat_auc, vat_l2

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="gazefuze_dinov2_vitb14_inout")
parser.add_argument('--init_ckpt', type=str, default='TextGaze-IJCAI2026/experiments/gf_base_1/2025-11-14_15-47-26/epoch_20.pt', 
                    help='checkpoint for initialization (trained on GazeFollow)')
parser.add_argument('--data_path', type=str, default='TextGaze-IJCAI2026/datasets/videoattentiontarget_Qwen_En')
parser.add_argument('--frame_sample_every', type=int, default=6)
parser.add_argument('--ckpt_save_dir', type=str, default='./experiments')
parser.add_argument('--exp_name', type=str, default='vat_base_1_epoch20')
parser.add_argument('--log_iter', type=int, default=10, help='how often to log loss during training')
parser.add_argument('--max_epochs', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=60)
parser.add_argument('--inout_loss_lambda', type=float, default=1.0)
parser.add_argument('--lr_non_inout', type=float, default=1e-5)
parser.add_argument('--lr_inout', type=float, default=1e-2)
parser.add_argument('--n_workers', type=int, default=8)
# 文本监督相关参数（与train_gazefollow保持一致）
parser.add_argument('--text_supervision_weight', type=float, default=0.1, help='文本监督损失权重')
parser.add_argument('--vl_hidden_dim', type=int, default=256)
parser.add_argument('--vl_nheads', type=int, default=8)
parser.add_argument('--vl_enc_layers', type=int, default=3)
parser.add_argument('--vl_dim_feedforward', type=int, default=2048)
parser.add_argument('--vl_dropout', type=float, default=0.1)
parser.add_argument('--vl_activation', type=str, default='relu')
args = parser.parse_args()

# 配置日志记录
exp_dir = os.path.join(args.ckpt_save_dir, args.exp_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(exp_dir)
log_file_path = os.path.join(exp_dir, 'training_log.txt')

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

# 同时将日志输出到终端
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


def main():
    # 传递参数给模型（包括文本监督相关参数）
    model, transform = get_gazefuze_model(args.model, args)
    logging.info("Initializing from {}".format(args.init_ckpt))
    model.load_gazefuze_state_dict(torch.load(args.init_ckpt, weights_only=True))
    model.cuda()

    for param in model.backbone.parameters():  # freeze backbone
        param.requires_grad = False
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Learnable parameters: {learnable_params}")

    # 加载数据集时读取text字段
    train_dataset = GazeDataset(
        'videoattentiontarget', 
        args.data_path, 
        'train', 
        transform, 
        in_frame_only=False, 
        sample_rate=args.frame_sample_every
    )
    train_dl = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=args.n_workers
    )
    
    eval_dataset = GazeDataset(
        'videoattentiontarget', 
        args.data_path, 
        'test', 
        transform, 
        in_frame_only=False, 
        sample_rate=args.frame_sample_every
    )
    eval_dl = torch.utils.data.DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=args.n_workers
    )

    heatmap_loss_fn = nn.BCELoss()
    inout_loss_fn = nn.BCELoss()
    param_groups = [
        {'params': [param for name, param in model.named_parameters() if "inout" in name], 'lr': args.lr_inout},
        {'params': [param for name, param in model.named_parameters() if "inout" not in name], 'lr': args.lr_non_inout}
    ]
    optimizer = torch.optim.Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)

    for epoch in range(args.max_epochs):
        # TRAIN EPOCH
        model.train()
        for cur_iter, batch in enumerate(train_dl):
            # 从batch中获取text字段
            imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps, texts = batch

            optimizer.zero_grad()
            # 将texts传入模型
            preds = model({
                "images": imgs.cuda(), 
                "bboxes": [[bbox] for bbox in bboxes],
                "texts": texts
            })
            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            inout_preds = torch.stack(preds['inout']).squeeze(dim=1)

            # 计算热图损失（仅针对in-frame目标）
            heatmap_loss = heatmap_loss_fn(heatmap_preds[inout.bool()], heatmaps[inout.bool()].cuda())
            inout_loss = inout_loss_fn(inout_preds, inout.float().cuda())
            # 总损失 = 热图损失 + inout损失 + 文本监督损失
            total_loss = heatmap_loss + args.inout_loss_lambda * inout_loss + preds['text_supervision_loss']

            total_loss.backward()
            optimizer.step()

            if cur_iter % args.log_iter == 0:
                log_msg = f"TRAIN EPOCH {epoch}, iter {cur_iter}/{len(train_dl)}, " \
                          f"heatmap_loss={round(heatmap_loss.item(), 4)}, " \
                          f"inout_loss={round(inout_loss.item(), 4)}, " \
                          f"text_loss={round(preds['text_supervision_loss'].item(), 4)}, " \
                          f"total_loss={round(total_loss.item(), 4)}"
                logging.info(log_msg)

        scheduler.step()

        # 仅按epoch保存checkpoint，不记录最佳模型
        ckpt_path = os.path.join(exp_dir, 'epoch_{}.pt'.format(epoch))
        torch.save(model.get_gazefuze_state_dict(), ckpt_path)
        logging.info(f"Saved checkpoint to {ckpt_path}")

        # EVAL EPOCH
        logging.info("Running evaluation")
        model.eval()
        l2s = []
        aucs = []
        all_inout_preds = []
        all_inout_gts = []
        
        for cur_iter, batch in tqdm(enumerate(eval_dl), total=len(eval_dl), desc=f"Eval Epoch {epoch}"):
            # 评估时同样读取text字段
            imgs, bboxes, gazex, gazey, inout, heights, widths, texts = batch

            with torch.no_grad():
                preds = model({
                    "images": imgs.cuda(), 
                    "bboxes": [[bbox] for bbox in bboxes],
                    "texts": texts
                })

            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            inout_preds = torch.stack(preds['inout']).squeeze(dim=1)
            
            for i in range(heatmap_preds.shape[0]):
                if inout[i] == 1:  # in-frame
                    auc = vat_auc(heatmap_preds[i], gazex[i][0], gazey[i][0])
                    l2 = vat_l2(heatmap_preds[i], gazex[i][0], gazey[i][0])
                    aucs.append(auc)
                    l2s.append(l2)
                all_inout_preds.append(inout_preds[i].item())
                all_inout_gts.append(inout[i])

        epoch_l2 = np.mean(l2s) if l2s else 0.0
        epoch_auc = np.mean(aucs) if aucs else 0.0
        epoch_inout_ap = average_precision_score(all_inout_gts, all_inout_preds)

        log_msg = f"EVAL EPOCH {epoch}: AUC={round(epoch_auc, 4)}, L2={round(epoch_l2, 4)}, Inout AP={round(epoch_inout_ap, 4)}"
        logging.info(log_msg)

    logging.info(f"Completed training. All checkpoints saved to {exp_dir}")


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()
