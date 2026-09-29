import argparse
import torch
from PIL import Image
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter  # 新增：高斯平滑
from gazefuze.model import get_gazelle_model
from gazefuze.utils import visualize_heatmap
from tqdm import tqdm  # 用于进度条显示

# ===================== 配置参数（无需修改，直接使用）=====================
parser = argparse.ArgumentParser()
parser.add_argument("--json_path", type=str, default="/data/sjh/EyeProject/GazeFuze/comparison_best3_samples/test3.json",
                    help="测试集JSON标注文件路径（新模型对应的带text标注）")
parser.add_argument("--img_root", type=str, default="/data/sjh/EyeProject/GazeFuze/comparison_best3_samples",
                    help="图片根目录（用于拼接绝对路径）")
parser.add_argument("--old_model_ckpt", type=str, default="/data/sjh/EyeProject/GAZELLE/experiments/train_gazelle_vitb_gazefollow/2025-07-08_09-31-08/epoch_11.pt",
                    help="旧模型权重路径")
parser.add_argument("--new_model_ckpt", type=str, default="/data/sjh/EyeProject/GazeFuze-A/experiments/gf_base_1/2025-11-14_15-47-26/epoch_20.pt",
                    help="新模型权重路径")
parser.add_argument("--model_name", type=str, default="gazelle_dinov2_vitb14",
                    help="模型名称（两个模型一致）")
parser.add_argument("--save_dir", type=str, default="./comparison_best3_samples",
                    help="最优样本可视化结果保存目录")
parser.add_argument("--top_k", type=int, default=3,
                    help="筛选最优样本数量（默认3个）")
parser.add_argument("--min_old_error", type=float, default=50.0,
                    help="旧模型最小误差阈值（确保偏差足够大，单位：像素）")
args = parser.parse_args()

# 创建保存目录（按样本序号分文件夹，更清晰）
os.makedirs(args.save_dir, exist_ok=True)
for i in range(1, args.top_k + 1):
    os.makedirs(os.path.join(args.save_dir, f"sample_{i}"), exist_ok=True)

# 颜色配置（统一为limegreen）
MAIN_COLOR = 'limegreen'       # 主颜色：青柠绿（所有线条、点）
BBBOX_COLOR = 'limegreen'      # 边界框：青柠绿
GT_POINT_COLOR = 'limegreen'   # 真实点：青柠绿
OLD_PRED_COLOR = 'limegreen'   # 旧模型预测点：青柠绿
NEW_PRED_COLOR = 'limegreen'   # 新模型预测点：青柠绿
OLD_LINE_COLOR = 'limegreen'   # 旧模型视线：青柠绿
NEW_LINE_COLOR = 'limegreen'   # 新模型视线：青柠绿

# 热图配置（统一参数，确保新旧模型效果一致）
GAUSSIAN_SIGMA = 1.5  # 高斯平滑系数（越大越平滑，1.5兼顾平滑和细节）
HEATMAP_ALPHA = 0.6   # 热图透明度（和新模型保持一致）

# 尺寸配置（大幅加粗线条、放大点大小，满足视觉需求）
LINE_WIDTH_BBOX = 8.0          # 边界框线条宽度（原2.5→8.0）
LINE_WIDTH_GAZE_OLD = 10.0     # 旧模型凝视线宽度（原3.5→10.0）
LINE_WIDTH_GAZE_NEW = 10.0     # 新模型凝视线宽度（原3.5→10.0）
LINE_WIDTH_COMPARE_OLD = 9.0    # 对比图旧模型凝视线宽度（原3→9.0）
LINE_WIDTH_COMPARE_NEW = 9.0    # 对比图新模型凝视线宽度（原3→9.0）

POINT_SIZE_GT = 300             # GT图真实点大小（原80→300）
POINT_SIZE_GT_REF = 220         # 预测图中参考真实点大小（原60→220）
POINT_SIZE_OLD_PRED = 300       # 旧模型预测点大小（原80→300）
POINT_SIZE_NEW_PRED = 300       # 新模型预测点大小（原80→300）
POINT_SIZE_COMPARE_GT = 350     # 对比图真实点大小（原100→350）
POINT_SIZE_COMPARE_OLD = 320    # 对比图旧模型预测点大小（原90→320）
POINT_SIZE_COMPARE_NEW = 320    # 对比图新模型预测点大小（原90→320）

POINT_EDGE_WIDTH = 6.0          # 点的白色边框宽度（原2→6.0）
POINT_EDGE_WIDTH_REF = 5.0      # 参考真实点边框宽度（原1.5→5.0）
POINT_EDGE_WIDTH_COMPARE = 8.0  # 对比图点边框宽度（原3→8.0）

# ===================== 核心工具函数 =====================
def load_json_annotations(json_path):
    """加载JSON标注，返回：{图片相对路径: 标注信息}"""
    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    img_to_annot = {}
    for annot in annotations:
        img_rel_path = annot["path"]
        img_to_annot[img_rel_path] = annot
    return img_to_annot

def load_model(ckpt_path, model_name, device):
    """加载模型（通用函数）"""
    model, transform = get_gazelle_model(model_name)
    model.load_gazelle_state_dict(torch.load(ckpt_path, weights_only=True))
    model.to(device)
    model.eval()
    return model, transform

def calculate_l2_distance(gt_point, pred_point):
    """计算L2距离（欧氏距离，单位：像素）"""
    gt_x, gt_y = gt_point
    pred_x, pred_y = pred_point
    return np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)

def get_bbox_edge_intersection(bbox, center, target, linewidth=LINE_WIDTH_BBOX):
    """计算框边交点（用于绘制视线）"""
    xmin, ymin, xmax, ymax = bbox
    cx, cy = center
    tx, ty = target
    
    dx = tx - cx
    dy = ty - cy
    dist = np.sqrt(dx**2 + dy**2)
    if dist < 1e-6:
        return (cx, cy)
    unit_dx = dx / dist
    unit_dy = dy / dist
    
    t_list = []
    # 左边界
    if dx != 0:
        t_left = (xmin - cx) / dx
        if t_left > 0:
            y_left = cy + t_left * dy
            if ymin <= y_left <= ymax:
                t_list.append((t_left, xmin, y_left, 'left'))
    # 右边界
    if dx != 0:
        t_right = (xmax - cx) / dx
        if t_right > 0:
            y_right = cy + t_right * dy
            if ymin <= y_right <= ymax:
                t_list.append((t_right, xmax, y_right, 'right'))
    # 上边界
    if dy != 0:
        t_top = (ymin - cy) / dy
        if t_top > 0:
            x_top = cx + t_top * dx
            if xmin <= x_top <= xmax:
                t_list.append((t_top, x_top, ymin, 'top'))
    # 下边界
    if dy != 0:
        t_bottom = (ymax - cy) / dy
        if t_bottom > 0:
            x_bottom = cx + t_bottom * dx
            if xmin <= x_bottom <= xmax:
                t_list.append((t_bottom, x_bottom, ymax, 'bottom'))
    
    if not t_list:
        return (cx, cy)
    t_list.sort(key=lambda x: x[0])
    t, ix, iy, edge = t_list[0]
    
    offset = linewidth / 2
    if edge == 'left':
        ix -= offset
    elif edge == 'right':
        ix += offset
    elif edge == 'top':
        iy -= offset
    elif edge == 'bottom':
        iy += offset
    
    return (ix, iy)

def process_heatmap(heatmap_array):
    """统一热图处理逻辑（新旧模型共用）：归一化 + 高斯平滑"""
    # 归一化到0-1
    heatmap_norm = (heatmap_array - heatmap_array.min()) / (heatmap_array.max() - heatmap_array.min() + 1e-6)
    # 高斯平滑（消除方块感，让热图更润滑）
    heatmap_smoothed = gaussian_filter(heatmap_norm, sigma=GAUSSIAN_SIGMA)
    # 再次归一化（平滑后数值范围可能变化）
    heatmap_final = (heatmap_smoothed - heatmap_smoothed.min()) / (heatmap_smoothed.max() - heatmap_smoothed.min() + 1e-6)
    return heatmap_final

# ===================== 四张图的绘制函数 =====================
def draw_gt_image(img, bbox, gt_point, save_path, text):
    """图1：Ground Truth标注图（仅原始图+边界框+真实点，无任何文字和白边）"""
    # 设置画布大小与图片完全一致，无多余边距
    fig, ax = plt.subplots(1, 1, figsize=(img.width/100, img.height/100), dpi=100)
    ax.imshow(img)
    ax.axis('off')
    
    # 移除所有边距和留白
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax.margins(0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    
    # 绘制头部边界框（大幅加粗）
    xmin, ymin, xmax, ymax = bbox
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                                linewidth=LINE_WIDTH_BBOX, edgecolor=BBBOX_COLOR, facecolor='none'))
    
    # 绘制真实凝视点（大幅放大，加粗白色边框）
    ax.scatter(gt_point[0], gt_point[1], s=POINT_SIZE_GT, color=GT_POINT_COLOR, marker='o',
                edgecolors='white', linewidth=POINT_EDGE_WIDTH, alpha=1.0)
    
    # 保存图片（无白边，高分辨率）
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300, 
                facecolor='none', edgecolor='none')
    print(f"  - GT标注图已保存：{os.path.basename(save_path)}")

def draw_old_model_image(img, bbox, gt_point, old_pred_point, old_error, old_heatmap_array, save_path, text):
    """图2：旧模型可视化预测图（无文字、无白边、大幅加粗线条和点）"""
    # 设置画布大小与图片完全一致
    fig, ax = plt.subplots(1, 1, figsize=(img.width/100, img.height/100), dpi=100)
    
    # 统一热图处理：归一化 + 高斯平滑（消除方块感）
    old_heatmap_final = process_heatmap(old_heatmap_array)
    # 生成热图（和新模型用相同的可视化逻辑）
    heatmap_img = visualize_heatmap(img, old_heatmap_final)
    ax.imshow(heatmap_img)
    ax.axis('off')
    
    # 移除所有边距和留白
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax.margins(0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    
    xmin, ymin, xmax, ymax = bbox
    bbox_center = ((xmin+xmax)/2, (ymin+ymax)/2)
    
    # 绘制头部边界框（大幅加粗）
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                                linewidth=LINE_WIDTH_BBOX, edgecolor=BBBOX_COLOR, facecolor='none'))
    
    # 绘制真实点（参考用，大幅放大，加粗边框）
    ax.scatter(gt_point[0], gt_point[1], s=POINT_SIZE_GT_REF, color=GT_POINT_COLOR, marker='o',
                edgecolors='white', linewidth=POINT_EDGE_WIDTH_REF, alpha=0.7)
    
    # 绘制旧模型视线（大幅加粗）
    line_start = get_bbox_edge_intersection(bbox, bbox_center, old_pred_point, linewidth=LINE_WIDTH_GAZE_OLD)
    ax.plot([line_start[0], old_pred_point[0]], 
             [line_start[1], old_pred_point[1]], 
             color=OLD_LINE_COLOR, linewidth=LINE_WIDTH_GAZE_OLD, alpha=0.9)
    
    # 绘制旧模型预测点（大幅放大，加粗边框，方形区分）
    ax.scatter(old_pred_point[0], old_pred_point[1], s=POINT_SIZE_OLD_PRED, color=OLD_PRED_COLOR, marker='s',
                edgecolors='white', linewidth=POINT_EDGE_WIDTH, alpha=1.0)
    
    # 保存图片（无白边）
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300,
                facecolor='none', edgecolor='none')
    print(f"  - 旧模型预测图已保存：{os.path.basename(save_path)}")

def draw_new_model_image(img, bbox, gt_point, new_pred_point, new_error, new_heatmap_array, save_path, text):
    """图3：新模型可视化预测图（无文字、无白边、大幅加粗线条和点）"""
    # 设置画布大小与图片完全一致
    fig, ax = plt.subplots(1, 1, figsize=(img.width/100, img.height/100), dpi=100)
    
    # 统一热图处理：归一化 + 高斯平滑（和旧模型共用逻辑）
    new_heatmap_final = process_heatmap(new_heatmap_array)
    # 生成热图
    heatmap_img = visualize_heatmap(img, new_heatmap_final)
    ax.imshow(heatmap_img)
    ax.axis('off')
    
    # 移除所有边距和留白
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax.margins(0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    
    xmin, ymin, xmax, ymax = bbox
    bbox_center = ((xmin+xmax)/2, (ymin+ymax)/2)
    
    # 绘制头部边界框（大幅加粗）
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                                linewidth=LINE_WIDTH_BBOX, edgecolor=BBBOX_COLOR, facecolor='none'))
    
    # 绘制真实点（参考用，大幅放大，加粗边框）
    ax.scatter(gt_point[0], gt_point[1], s=POINT_SIZE_GT_REF, color=GT_POINT_COLOR, marker='o',
                edgecolors='white', linewidth=POINT_EDGE_WIDTH_REF, alpha=0.7)
    
    # 绘制新模型视线（大幅加粗）
    line_start = get_bbox_edge_intersection(bbox, bbox_center, new_pred_point, linewidth=LINE_WIDTH_GAZE_NEW)
    ax.plot([line_start[0], new_pred_point[0]], 
             [line_start[1], new_pred_point[1]], 
             color=NEW_LINE_COLOR, linewidth=LINE_WIDTH_GAZE_NEW, alpha=0.9)
    
    # 绘制新模型预测点（大幅放大，加粗边框，圆形区分）
    ax.scatter(new_pred_point[0], new_pred_point[1], s=POINT_SIZE_NEW_PRED, color=NEW_PRED_COLOR, marker='o',
                edgecolors='white', linewidth=POINT_EDGE_WIDTH, alpha=1.0)
    
    # 保存图片（无白边）
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300,
                facecolor='none', edgecolor='none')
    print(f"  - 新模型预测图已保存：{os.path.basename(save_path)}")

def draw_comparison_image(img, bbox, gt_point, old_pred_point, new_pred_point, 
                         old_error, new_error, error_reduction, save_path, text):
    """图4：双模型对比图（无文字、无白边、大幅加粗线条和点）"""
    # 设置画布大小与图片完全一致
    fig, ax = plt.subplots(1, 1, figsize=(img.width/100, img.height/100), dpi=100)
    ax.imshow(img)
    ax.axis('off')
    
    # 移除所有边距和留白
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    ax.margins(0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    
    xmin, ymin, xmax, ymax = bbox
    bbox_center = ((xmin+xmax)/2, (ymin+ymax)/2)
    
    # 绘制头部边界框（大幅加粗）
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                                linewidth=LINE_WIDTH_BBOX, edgecolor=BBBOX_COLOR, facecolor='none'))
    
    # 绘制真实点（大幅放大，加粗边框，突出显示）
    ax.scatter(gt_point[0], gt_point[1], s=POINT_SIZE_COMPARE_GT, color=GT_POINT_COLOR, marker='o',
                edgecolors='white', linewidth=POINT_EDGE_WIDTH_COMPARE, alpha=1.0)
    
    # 绘制旧模型预测（大幅加粗虚线视线+方形点）
    old_line_start = get_bbox_edge_intersection(bbox, bbox_center, old_pred_point, linewidth=LINE_WIDTH_COMPARE_OLD)
    ax.plot([old_line_start[0], old_pred_point[0]], 
             [old_line_start[1], old_pred_point[1]], 
             color=OLD_LINE_COLOR, linewidth=LINE_WIDTH_COMPARE_OLD, alpha=0.8, linestyle='--')
    ax.scatter(old_pred_point[0], old_pred_point[1], s=POINT_SIZE_COMPARE_OLD, color=OLD_PRED_COLOR, marker='s',
                edgecolors='white', linewidth=POINT_EDGE_WIDTH_COMPARE, alpha=1.0)
    
    # 绘制新模型预测（大幅加粗实线视线+圆形点）
    new_line_start = get_bbox_edge_intersection(bbox, bbox_center, new_pred_point, linewidth=LINE_WIDTH_COMPARE_NEW)
    ax.plot([new_line_start[0], new_pred_point[0]], 
             [new_line_start[1], new_pred_point[1]], 
             color=NEW_LINE_COLOR, linewidth=LINE_WIDTH_COMPARE_NEW, alpha=0.8, linestyle='-')
    ax.scatter(new_pred_point[0], new_pred_point[1], s=POINT_SIZE_COMPARE_NEW, color=NEW_PRED_COLOR, marker='o',
                edgecolors='white', linewidth=POINT_EDGE_WIDTH_COMPARE, alpha=1.0)
    
    # 保存图片（无白边）
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300,
                facecolor='none', edgecolor='none')
    print(f"  - 双模型对比图已保存：{os.path.basename(save_path)}")

# ===================== 主流程函数 =====================
@torch.no_grad()
def main():
    # 修复新模型推理时的变量错误（原代码使用了未定义的new_output）
    # 1. 初始化设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备：{device}")
    
    # 2. 加载两个模型
    print("加载旧模型...")
    old_model, old_transform = load_model(args.old_model_ckpt, args.model_name, device)
    print("加载新模型...")
    new_model, new_transform = load_model(args.new_model_ckpt, args.model_name, device)
    
    # 3. 加载标注和图片列表
    print("加载标注文件...")
    img_to_annot = load_json_annotations(args.json_path)
    img_rel_paths = list(img_to_annot.keys())
    print(f"测试集总样本数：{len(img_rel_paths)}")
    
    # 4. 遍历所有样本，计算误差
    sample_metrics = []
    print(f"开始遍历测试集，计算模型L2距离误差（旧模型最小误差阈值：{args.min_old_error}px）...")
    
    for img_rel_path in tqdm(img_rel_paths, desc="Processing samples"):
        annot = img_to_annot[img_rel_path]
        
        # 跳过无头部标注的样本
        if annot.get("num_heads", 0) == 0:
            continue
        # 取目标头部（默认第0个，与原逻辑一致）
        target_head = annot["heads"][0]
        
        # 计算真实凝视点（取所有有效标注点的平均值，更稳定）
        gazex_list = [x for x in target_head["gazex"] if x > 0]
        gazey_list = [y for y in target_head["gazey"] if y > 0]
        if not gazex_list or not gazey_list:
            continue
        # 真实点为所有有效标注点的平均（适配多标注点场景）
        gt_point = (np.mean(gazex_list), np.mean(gazey_list))
        
        # 拼接图片绝对路径
        img_abs_path = os.path.join(args.img_root, img_rel_path)
        if not os.path.exists(img_abs_path):
            print(f"警告：图片不存在 {img_abs_path}，跳过")
            continue
        
        # 加载图片
        try:
            img = Image.open(img_abs_path).convert("RGB")
            img_width, img_height = img.size
        except Exception as e:
            print(f"警告：加载图片失败 {img_abs_path}，错误：{e}，跳过")
            continue
        
        # 提取标注信息
        bbox = target_head["bbox"]  # [xmin, ymin, xmax, ymax]
        bbox_norm = target_head["bbox_norm"]
        # 关键修改：从目标头部target_head提取text，而非annot["text"]
        text = target_head.get("text", "")  # 若头部无text字段，返回空字符串
        bbox_center = ((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2)
        
        # ---------------------- 旧模型推理 ----------------------
        old_img_input = old_transform(img).unsqueeze(0).to(device)
        old_model_input = {
            "images": old_img_input,
            "bboxes": [[tuple(bbox_norm)]],
            "texts": [""]  # 空text适配模型代码
        }
        old_output = old_model(old_model_input)
        old_heatmap_array = old_output["heatmap"][0][0].detach().cpu().numpy()
        
        # 计算旧模型预测点
        old_pred_y, old_pred_x = np.unravel_index(np.argmax(old_heatmap_array), old_heatmap_array.shape)
        old_pred_x_abs = (old_pred_x / old_heatmap_array.shape[1]) * img_width
        old_pred_y_abs = (old_pred_y / old_heatmap_array.shape[0]) * img_height
        old_pred_point = (old_pred_x_abs, old_pred_y_abs)
        
        # 计算旧模型L2距离
        old_error = calculate_l2_distance(gt_point, old_pred_point)
        
        # ---------------------- 新模型推理 ----------------------
        new_img_input = new_transform(img).unsqueeze(0).to(device)
        new_model_input = {
            "images": new_img_input,
            "bboxes": [[tuple(bbox_norm)]],
            "texts": [text]  # 传入目标头部对应的text
        }
        # 修复：原代码错误使用new_output（未定义），应改为new_model输出
        new_output = new_model(new_model_input)
        new_heatmap_array = new_output["heatmap"][0][0].detach().cpu().numpy()
        
        # 计算新模型预测点
        new_pred_y, new_pred_x = np.unravel_index(np.argmax(new_heatmap_array), new_heatmap_array.shape)
        new_pred_x_abs = (new_pred_x / new_heatmap_array.shape[1]) * img_width
        new_pred_y_abs = (new_pred_y / new_heatmap_array.shape[0]) * img_height
        new_pred_point = (new_pred_x_abs, new_pred_y_abs)
        
        # 计算新模型L2距离
        new_error = calculate_l2_distance(gt_point, new_pred_point)
        
        # 保存该样本的关键信息
        sample_metrics.append({
            "img_abs_path": img_abs_path,
            "img_rel_path": img_rel_path,
            "bbox": bbox,
            "gt_point": gt_point,
            "old_pred_point": old_pred_point,
            "new_pred_point": new_pred_point,
            "old_error": old_error,
            "new_error": new_error,
            "error_reduction": old_error - new_error,
            "text": text,  # 保存的是目标头部的text
            "old_heatmap_array": old_heatmap_array,
            "new_heatmap_array": new_heatmap_array
        })
    
    # 5. 筛选最优样本
    print("筛选最优样本...")
    valid_samples = [s for s in sample_metrics if s["old_error"] > args.min_old_error and s["new_error"] < s["old_error"]]
    valid_samples.sort(key=lambda x: x["error_reduction"], reverse=True)
    best_samples = valid_samples[:args.top_k]
    
    if len(best_samples) < args.top_k:
        print(f"警告：仅找到 {len(best_samples)} 个符合条件的样本（需要旧模型L2误差>{args.min_old_error}px且新模型更优）")
        print(f"提示：可通过调整 --min_old_error 参数降低阈值（当前：{args.min_old_error}px）")
    
    # 6. 生成四张图（每个样本）
    print(f"\n开始生成可视化结果（每个样本4张图，共{len(best_samples)*4}张）...")
    for idx, sample in enumerate(best_samples, 1):
        sample_dir = os.path.join(args.save_dir, f"sample_{idx}")
        img = Image.open(sample["img_abs_path"]).convert("RGB")
        img_basename = os.path.splitext(os.path.basename(sample["img_abs_path"]))[0]
        
        print(f"\n样本 {idx}/{len(best_samples)} - {img_basename}:")
        
        # 图1：GT标注图
        gt_save_path = os.path.join(sample_dir, f"{img_basename}_1_gt_annotation.png")
        draw_gt_image(
            img=img,
            bbox=sample["bbox"],
            gt_point=sample["gt_point"],
            save_path=gt_save_path,
            text=sample["text"]
        )
        
        # 图2：旧模型预测图
        old_save_path = os.path.join(sample_dir, f"{img_basename}_2_old_model_pred.png")
        draw_old_model_image(
            img=img,
            bbox=sample["bbox"],
            gt_point=sample["gt_point"],
            old_pred_point=sample["old_pred_point"],
            old_error=sample["old_error"],
            old_heatmap_array=sample["old_heatmap_array"],
            save_path=old_save_path,
            text=sample["text"]
        )
        
        # 图3：新模型预测图
        new_save_path = os.path.join(sample_dir, f"{img_basename}_3_new_model_pred.png")
        draw_new_model_image(
            img=img,
            bbox=sample["bbox"],
            gt_point=sample["gt_point"],
            new_pred_point=sample["new_pred_point"],
            new_error=sample["new_error"],
            new_heatmap_array=sample["new_heatmap_array"],
            save_path=new_save_path,
            text=sample["text"]
        )
        
        # 图4：双模型对比图
        comp_save_path = os.path.join(sample_dir, f"{img_basename}_4_comparison.png")
        draw_comparison_image(
            img=img,
            bbox=sample["bbox"],
            gt_point=sample["gt_point"],
            old_pred_point=sample["old_pred_point"],
            new_pred_point=sample["new_pred_point"],
            old_error=sample["old_error"],
            new_error=sample["new_error"],
            error_reduction=sample["error_reduction"],
            save_path=comp_save_path,
            text=sample["text"]
        )
        
        print(f"  样本 {idx} 可视化完成，结果保存在：{sample_dir}")
    
    # 7. 保存最优样本信息汇总
    best_samples_info = [{
        "sample_rank": idx+1,
        "sample_dir": os.path.join(args.save_dir, f"sample_{idx+1}"),
        "img_abs_path": s["img_abs_path"],
        "old_model_L2_error(px)": s["old_error"],
        "new_model_L2_error(px)": s["new_error"],
        "error_reduction(px)": s["error_reduction"],
        "text_prompt": s["text"]  # 汇总中保存的是目标头部的text
    } for idx, s in enumerate(best_samples)]
    
    info_save_path = os.path.join(args.save_dir, "best_samples_summary.json")
    with open(info_save_path, "w", encoding="utf-8") as f:
        json.dump(best_samples_info, f, indent=2, ensure_ascii=False)
    
    print(f"\n最优样本汇总信息已保存至：{info_save_path}")
    print(f"\n所有可视化结果已保存至：{args.save_dir}")
    print(f"文件结构：")
    print(f"  {args.save_dir}/")
    print(f"    sample_1/  # 第1个最优样本")
    print(f"      xxx_1_gt_annotation.png    # 图1：GT标注图（无文字+无白边+粗线条+大尺寸点）")
    print(f"      xxx_2_old_model_pred.png   # 图2：旧模型预测图（无文字+无白边+粗线条+大尺寸点）")
    print(f"      xxx_3_new_model_pred.png   # 图3：新模型预测图（无文字+无白边+粗线条+大尺寸点）")
    print(f"      xxx_4_comparison.png       # 图4：双模型对比图（无文字+无白边+粗线条+大尺寸点）")
    print(f"    sample_2/  # 第2个最优样本（同上）")
    print(f"    sample_3/  # 第3个最优样本（同上）")
    print(f"    best_samples_summary.json    # 样本信息汇总")
    print(f"\n注意：")
    print(f"  - 所有图像已移除标题、误差、图例、文本提示等所有文字信息")
    print(f"  - 所有图像已移除白边，画布与原始图片尺寸完全一致")
    print(f"  - 线条宽度和点尺寸大幅放大（3-4倍），确保视觉清晰")
    print(f"  - 区分方式：旧模型=方形点+虚线，新模型=圆形点+实线（青柠绿统一风格）")
    print(f"  - Text提取路径已修正：从目标头部（target_head）获取，确保与头部一一对应")

if __name__ == "__main__":
    main()