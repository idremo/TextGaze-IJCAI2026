import argparse
from datetime import datetime
import numpy as np
import os
import random
import torch
import torch.nn as nn
import logging

from gazefuze.dataloader import GazeDataset, collate_fn
from gazefuze.model import get_gazefuze_model
from gazefuze.utils import gazefollow_auc, gazefollow_l2

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="gazefuze_dinov2_vitb14")
parser.add_argument('--data_path', type=str, default='TextGaze-IJCAI2026/datasets/gazefollow_extended_Qwen_En')
parser.add_argument('--ckpt_save_dir', type=str, default='./experiments')
parser.add_argument('--exp_name', type=str, default='train_gazefollow')
parser.add_argument('--log_iter', type=int, default=10, help='how often to log loss during training')
parser.add_argument('--max_epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_workers', type=int, default=8)
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
    # 传递文本监督权重参数
    model, transform = get_gazefuze_model(args.model, args)
    model.cuda()

    for param in model.backbone.parameters():  # freeze backbone
        param.requires_grad = False
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Learnable parameters: {learnable_params}")

    train_dataset = GazeDataset('gazefollow', args.data_path, 'train', transform)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                                           num_workers=args.n_workers)
    eval_dataset = GazeDataset('gazefollow', args.data_path, 'test', transform)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                                          num_workers=args.n_workers)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)

    best_min_l2 = 1.0
    best_epoch = None

    for epoch in range(args.max_epochs):
        # TRAIN EPOCH
        model.train()
        for cur_iter, batch in enumerate(train_dl):
            imgs, bboxes, gazex, gazey, inout, heights, widths, heatmaps, texts = batch

            optimizer.zero_grad()
            preds = model({"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes], "texts": texts})
            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)

            # 主损失（热图预测）
            heatmap_loss = loss_fn(heatmap_preds, heatmaps.cuda())
            # 总损失 = 主损失 + 文本监督损失
            total_loss = heatmap_loss + preds['text_supervision_loss']

            total_loss.backward()
            optimizer.step()

            if cur_iter % args.log_iter == 0:
                log_msg = f"TRAIN EPOCH {epoch}, iter {cur_iter}/{len(train_dl)}, " \
                          f"heatmap_loss={round(heatmap_loss.item(), 4)}, " \
                          f"text_loss={round(preds['text_supervision_loss'].item(), 4)}, " \
                          f"total_loss={round(total_loss.item(), 4)}"
                logging.info(log_msg)

        scheduler.step()

        ckpt_path = os.path.join(exp_dir, 'epoch_{}.pt'.format(epoch))
        torch.save(model.get_gazefuze_state_dict(), ckpt_path)
        logging.info(f"Saved checkpoint to {ckpt_path}")

        # EVAL EPOCH（评估过程不变）
        logging.info("Running evaluation")
        model.eval()
        avg_l2s = []
        min_l2s = []
        aucs = []
        for cur_iter, batch in enumerate(eval_dl):
            imgs, bboxes, gazex, gazey, inout, heights, widths, texts = batch

            with torch.no_grad():
                preds = model({"images": imgs.cuda(), "bboxes": [[bbox] for bbox in bboxes], "texts":texts})

            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            for i in range(heatmap_preds.shape[0]):
                auc = gazefollow_auc(heatmap_preds[i], gazex[i], gazey[i], heights[i], widths[i])
                avg_l2, min_l2 = gazefollow_l2(heatmap_preds[i], gazex[i], gazey[i])
                aucs.append(auc)
                avg_l2s.append(avg_l2)
                min_l2s.append(min_l2)

        epoch_avg_l2 = np.mean(avg_l2s)
        epoch_min_l2 = np.mean(min_l2s)
        epoch_auc = np.mean(aucs)

        log_msg = f"EVAL EPOCH {epoch}: AUC={round(epoch_auc, 4)}, Min L2={round(epoch_min_l2, 4)}, Avg L2={round(epoch_avg_l2, 4)}"
        logging.info(log_msg)

        if epoch_min_l2 < best_min_l2:
            best_min_l2 = epoch_min_l2
            best_epoch = epoch

    logging.info(f"Completed training. Best Min L2 of {round(best_min_l2, 4)} obtained at epoch {best_epoch}")


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    main()