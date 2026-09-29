import argparse
import numpy as np
import os
import torch
import logging
from datetime import datetime

from gazefuze.dataloader import GazeDataset, collate_fn
from gazefuze.model import get_gazelle_model
from gazefuze.utils import gazefollow_auc, gazefollow_l2

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="gazelle_dinov2_vitb14", 
                    help='与训练时一致的模型名称')
parser.add_argument('--data_path', type=str, 
                    default='/data/sjh/EyeProject/DataSets/gooreal_Qwen_En', 
                    help='新数据所在目录（包含test_preprocessed.json）')
parser.add_argument('--ckpt_path', type=str, required=True, 
                    help='训练好的模型权重路径')
parser.add_argument('--batch_size', type=int, default=60)
parser.add_argument('--n_workers', type=int, default=8)
parser.add_argument('--log_dir', type=str, default='./test_logs', 
                    help='测试日志保存目录')
args = parser.parse_args()


def main():
    # 配置日志
    os.makedirs(args.log_dir, exist_ok=True)
    log_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(args.log_dir, f'test_gooreal_{log_time}.txt')
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    logging.info(f"测试配置: {args}")
    logging.info(f"使用模型权重: {args.ckpt_path}")
    logging.info(f"测试数据路径: {args.data_path}")

    # 加载模型
    model, transform = get_gazelle_model(args.model)
    model.cuda()
    
    # 加载训练好的权重
    if os.path.exists(args.ckpt_path):
        state_dict = torch.load(args.ckpt_path)
        model.load_state_dict(state_dict, strict=False)
        logging.info(f"成功加载模型权重: {args.ckpt_path}（已忽略主干网络参数）")
    else:
        logging.error(f"模型权重文件不存在: {args.ckpt_path}")
        return

    # 冻结模型
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # 修正：GazeDataset参数不使用dataset_type关键字，按位置传递
    test_dataset = GazeDataset(
        'gazefollow',  # 第一个参数：dataset_name（对应训练时的'dataset_type'）
        args.data_path,
        'test',  # 数据集划分（test对应test_preprocessed.json）
        transform  # 第四个参数：transform（无需关键字）
    )
    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.n_workers
    )
    logging.info(f"测试集加载完成，共 {len(test_dataset)} 样本")

    # 测试指标计算
    avg_l2s = []
    aucs = []

    with torch.no_grad():
        for cur_iter, batch in enumerate(test_dl):
            imgs, bboxes, gazex, gazey, inout, heights, widths, texts = batch

            # 模型预测
            preds = model({
                "images": imgs.cuda(),
                "bboxes": [[bbox] for bbox in bboxes],
                "texts": texts
            })
            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)

            # 计算指标
            for i in range(heatmap_preds.shape[0]):
                auc = gazefollow_auc(heatmap_preds[i], gazex[i], gazey[i], heights[i], widths[i])
                avg_l2, _ = gazefollow_l2(heatmap_preds[i], gazex[i], gazey[i])
                aucs.append(auc)
                avg_l2s.append(avg_l2)

            if (cur_iter + 1) % 10 == 0:
                logging.info(f"已处理 {cur_iter + 1}/{len(test_dl)} 批次")

    # 输出结果
    total_avg_l2 = np.mean(avg_l2s)
    total_auc = np.mean(aucs)
    logging.info("\n===== 测试结果 =====")
    logging.info(f"测试样本数: {len(aucs)}")
    logging.info(f"AUC: {round(total_auc, 4)}")
    logging.info(f"Avg L2距离: {round(total_avg_l2, 4)}")
    logging.info("====================")


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    main()