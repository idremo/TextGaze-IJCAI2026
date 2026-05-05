import json
import requests
import base64
import os
from tqdm import tqdm  # 导入进度条库

def encode_image(image_path):
    """编码本地图片为base64格式"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# 配置参数
JSON_INPUT_PATH = '/data/sjh/EyeProject/DataSets/childplay/test_preprocessed.json'
JSON_OUTPUT_PATH = '/data/sjh/EyeProject/DataSets/childplay_Qwen_En/test_preprocessed_new.json'
IMAGE_ROOT = "/data/sjh/EyeProject/DataSets/childplay_Qwen_En/images"
VLLM_URL = 'http://localhost:8000/v1/chat/completions'

# 读取原始JSON数据
with open(JSON_INPUT_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 第一步：统计总任务数（所有frame的所有head数量）
total_tasks = 0
for d in data:
    for frame in d["frames"]:
        total_tasks += len(frame["heads"])

print(f"总任务数：{total_tasks} 个head需要生成描述")
print("-" * 50)

# 第二步：使用进度条遍历处理
with tqdm(total=total_tasks, desc="处理进度", unit="head") as pbar:
    for d in data:
        for frame in d["frames"]:
            image_path = os.path.join(IMAGE_ROOT, frame["path"])
            
            # 检查图像是否存在
            if not os.path.exists(image_path):
                print(f"\n警告：图像不存在，跳过所有相关head：{image_path}")
                pbar.update(len(frame["heads"]))  # 跳过的head也要更新进度条
                continue
            
            # 编码图像（每张图仅编码一次，多个head复用）
            base64_image = encode_image(image_path)
            
            # 为每个head单独发送请求
            for idx, head in enumerate(frame["heads"]):
                bbox = head["bbox"]
                head_bbox_x_min, head_bbox_y_min, head_bbox_x_max, head_bbox_y_max = bbox
                
                # 原始提示词不变
                prompt_text = (
                    f"The bbox coordinates of the target person's head in this image are ({head_bbox_x_min}, {head_bbox_y_min}, {head_bbox_x_max}, {head_bbox_y_max}). "
                    "There is only one target person in the picture.Use \"he/his\" or \"she/her\" to precisely refer to, rather than \"they/their\" Describe the appearance and actions of the target person in the image, as well as the direction in which the target person is looking relative to himself/herself. "
                    "Make sure that only the target person is the subject of your answer."
                    "Do not repeat the coordinates of the target person's head in the answer."
                    "Just describe the direction in which the target person is looking without detailing the object/person they are viewing."
                    "Example: The target person is wearing a white jersey, standing on the grass with his head tilted back, looking in the upper right direction relative to himself. "
                )
                
                # 构建请求体
                qa_payload = {
                    "model": "qwen",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an image understanding expert. For the target person in the image, answer my questions."
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                                {"type": "text", "text": prompt_text}
                            ]
                        }
                    ],
                    "temperature": 0.7,
                    "top_p": 0.7,
                    "repetition_penalty": 1.05,
                    "max_tokens": 2048
                }
                
                try:
                    # 发送请求
                    headers = {'Content-Type': 'application/json'}
                    response = requests.post(VLLM_URL, data=json.dumps(qa_payload), headers=headers, timeout=60)
                    response.raise_for_status()
                    
                    # 解析响应
                    qwen_text = response.json().get("choices", [])[0].get("message", {}).get("content", "").strip()
                    if not qwen_text:
                        qwen_text = "No valid description generated."
                    
                    head["text"] = qwen_text
                    
                except Exception as e:
                    # 异常处理
                    error_msg = f"生成描述失败：{str(e)}"
                    head["text"] = error_msg
                    print(f"\n警告：图像 {frame['path']} 的第 {idx+1} 个head {error_msg}")
                
                # 更新进度条
                pbar.update(1)

# 保存修改后的JSON文件
with open(JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("-" * 50)
print(f"处理完成！结果已保存至：{JSON_OUTPUT_PATH}")