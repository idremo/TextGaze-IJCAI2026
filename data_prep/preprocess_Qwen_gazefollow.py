import json
import requests
import base64
import os
import gc
import queue
import threading
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import sys

# 配置参数（简化批量提交，确保无残留）
JSON_INPUT_PATH = '/data/sjh/EyeProject/DataSets/gazefollow_extended/train_preprocessed.json'
JSON_OUTPUT_PATH = '/data/sjh/EyeProject/DataSets/gazefollow_extended_Qwen_En/train_preprocessed.json'
IMAGE_ROOT = "/data/sjh/EyeProject/DataSets/gazefollow_extended/"
VLLM_URL = 'http://localhost:8000/v1/chat/completions'

COLLECT_WORKERS = min(multiprocessing.cpu_count(), 16)  # 限制收集线程数，避免资源竞争
MAX_WORKERS = 16  # 推理并发数（根据VLLM承载能力调整）
TIMEOUT = 60  # 推理超时时间
QUEUE_MAX_SIZE = 1024  # 增大队列容量，避免提交阻塞
IMAGE_CACHE_SIZE = 2000  # 图像缓存大小
RETRY_FAILED = False  # 是否重新处理错误样本

def encode_image(image_path):
    """Linux/Mac兼容的图像编码"""
    with open(image_path, 'rb') as image_file:
        img_data = image_file.read()
    base64_str = base64.b64encode(img_data).decode('utf-8')
    del img_data
    return base64_str

def process_single_head(task, task_count, lock):
    """模型推理函数（增加任务计数锁，确保统计准确）"""
    item_path, head_idx, head_ref, base64_image = task
    bbox = head_ref["bbox"]
    x1, y1, x2, y2 = bbox

    prompt_text = (
        f"The bbox coordinates of the target person's head in this image are ({x1}, {y1}, {x2}, {y2}). "
        "There is only one target person in the picture.Use \"he/his\" or \"she/her\" to precisely refer to, rather than \"they/their\" Describe the appearance and actions of the target person in the image, as well as the direction in which the target person is looking relative to himself/herself. "
        "Make sure that only the target person is the subject of your answer."
        "Do not repeat the coordinates of the target person's head in the answer."
        "Just describe the direction in which the target person is looking without detailing the object/person they are viewing."
        "Example: The target person is wearing a white jersey, standing on the grass with his head tilted back, looking in the upper right direction relative to himself. "
    )

    qa_payload = {
        "model": "qwen",
        "messages": [
            {"role": "system", "content": "You are an image understanding expert. For the target person in the image, answer my questions."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": prompt_text}
            ]}
        ],
        "temperature": 0.7,
        "top_p": 0.7,
        "repetition_penalty": 1.05,
        "max_tokens": 2048
    }

    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            VLLM_URL,
            data=json.dumps(qa_payload),
            headers=headers,
            timeout=TIMEOUT,
            verify=False
        )
        response.raise_for_status()

        qwen_text = response.json().get("choices", [])[0].get("message", {}).get("content", "").strip()
        head_ref["text"] = qwen_text if qwen_text else "No valid description generated."
        del response
        return True

    except Exception as e:
        error_msg = f"Generate description failed: {str(e)[:100]}..."
        head_ref["text"] = error_msg
        with threading.Lock():
            print(f"\n警告：图像 {item_path} 的head {head_idx} {error_msg}", file=sys.stderr)
        return False
    finally:
        # 无论成功失败，都标记任务完成（原子操作）
        with lock:
            task_count["completed"] += 1

def consumer(task_queue, task_count, process_pbar, exit_event):
    """消费者线程：强制消费所有任务，直到队列空且任务完成"""
    # 推理线程池（固定大小，避免频繁创建销毁）
    with ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="infer") as executor:
        # 循环消费：直到退出信号或任务全部完成
        while not exit_event.is_set():
            # 1. 检查是否所有任务都已完成
            with task_count["lock"]:
                if task_count["completed"] >= task_count["total"]:
                    break

            # 2. 尝试从队列获取任务（非阻塞，避免卡死）
            try:
                task = task_queue.get_nowait()  # 立即获取，无任务则跳过
            except queue.Empty:
                threading.Event().wait(0.5)  # 无任务时休眠0.5秒，降低CPU占用
                continue

            # 3. 提交任务到推理线程池
            executor.submit(process_single_head, task, task_count, task_count["lock"])
            process_pbar.update(1)  # 进度条实时更新（提交即计数，避免统计延迟）
            task_queue.task_done()  # 标记队列任务完成

        # 4. 强制消费队列中所有残留任务（关键：确保队列空）
        print(f"\n开始处理队列残留任务（当前队列大小：{task_queue.qsize()}）", file=sys.stderr)
        while not task_queue.empty() and not exit_event.is_set():
            try:
                task = task_queue.get_nowait()
                executor.submit(process_single_head, task, task_count, task_count["lock"])
                process_pbar.update(1)
                task_queue.task_done()
            except queue.Empty:
                break

        # 5. 等待所有推理任务完成（确保提交的任务都执行完）
        print(f"\n等待所有推理任务完成（已完成：{task_count['completed']}/{task_count['total']}）", file=sys.stderr)
        while not exit_event.is_set():
            with task_count["lock"]:
                if task_count["completed"] >= task_count["total"]:
                    break
            threading.Event().wait(1)  # 每秒检查一次

    process_pbar.close()
    print(f"\n消费者线程退出（最终完成任务数：{task_count['completed']}/{task_count['total']}）", file=sys.stderr)

def load_existing_data():
    """加载已处理的结果，返回合并后的数据+需处理的任务列表"""
    merged_data = None

    # 读取原始数据
    with open(JSON_INPUT_PATH, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # 合并已处理结果（断点续传）
    if os.path.exists(JSON_OUTPUT_PATH):
        print(f"发现已存在结果文件，进行断点续传...")
        try:
            with open(JSON_OUTPUT_PATH, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            if len(existing_data) != len(raw_data):
                print(f"警告：已处理文件与原始文件长度不一致，重新处理所有样本！")
                merged_data = raw_data
            else:
                merged_data = raw_data
                for raw_item, existing_item in zip(raw_data, existing_data):
                    for raw_head, existing_head in zip(raw_item.get("heads", []), existing_item.get("heads", [])):
                        if "text" in existing_head:
                            raw_head["text"] = existing_head["text"]
        except Exception as e:
            print(f"加载已处理结果失败，重新处理所有样本：{e}")
            merged_data = raw_data
    else:
        merged_data = raw_data

    # 生成需处理的任务列表（直接返回任务，避免后续重复遍历）
    pending_tasks = []
    for item in merged_data:
        item_path = item["path"]
        image_abs_path = os.path.join(IMAGE_ROOT, item_path)
        heads = item.get("heads", [])
        for head_idx, head in enumerate(heads):
            # 未处理：无text或开启重试的错误样本
            if "text" not in head or (RETRY_FAILED and head["text"].startswith("Generate description failed")):
                pending_tasks.append((item, item_path, head_idx, head, image_abs_path))

    total_pending = len(pending_tasks)
    print(f"原始样本数：{len(merged_data)} 张 | 总head数：{sum(len(item.get('heads', [])) for item in merged_data)} 个")
    print(f"本次需处理的head数：{total_pending} 个")
    return merged_data, pending_tasks, total_pending

def collect_and_submit_task(task_info, image_cache, invalid_images, task_queue):
    """单任务收集+提交（取消批量，收集后立即提交）"""
    item, item_path, head_idx, head, image_abs_path = task_info

    # 跳过无效图片
    if image_abs_path in invalid_images:
        head["text"] = "Image not found: unable to generate description."
        return

    # 检查图片是否存在
    if not os.path.exists(image_abs_path):
        invalid_images.add(image_abs_path)
        head["text"] = "Image not found: unable to generate description."
        return

    # 图像编码（缓存优化）
    try:
        if image_abs_path in image_cache:
            base64_image = image_cache[image_abs_path]
        else:
            base64_image = encode_image(image_abs_path)
            # 缓存满时删除最早的条目
            if len(image_cache) >= IMAGE_CACHE_SIZE:
                oldest_key = next(iter(image_cache.keys()))
                del image_cache[oldest_key]
            image_cache[image_abs_path] = base64_image
    except Exception as e:
        invalid_images.add(image_abs_path)
        head["text"] = f"Image encode failed: {str(e)[:50]}..."
        return

    # 立即提交任务到队列（阻塞直到队列有空间）
    while True:
        try:
            task_queue.put((item_path, head_idx, head, base64_image), timeout=5)  # 5秒超时重试
            break
        except queue.Full:
            # 队列满时休眠1秒，避免CPU占用过高
            threading.Event().wait(1)
            continue

def main():
    # 加载数据和待处理任务
    data, pending_tasks, total_tasks = load_existing_data()
    if total_tasks == 0:
        print("所有样本已处理完成，无需继续运行！")
        return

    # 初始化任务计数（原子操作，避免多线程统计错误）
    task_count = {
        "total": total_tasks,
        "completed": 0,
        "lock": threading.Lock()
    }

    # 初始化队列、退出信号
    task_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)
    exit_event = threading.Event()
    invalid_images = set()
    image_cache = dict()

    print("\n初始化并行处理（无批量提交，确保无残留）...")
    print(f"收集线程数：{COLLECT_WORKERS} | 推理线程数：{MAX_WORKERS}")
    print(f"队列容量：{QUEUE_MAX_SIZE} | 图像缓存大小：{IMAGE_CACHE_SIZE}")
    print("-" * 50)

    # 进度条（按待处理任务总数计数）
    process_pbar = tqdm(total=total_tasks, desc="处理进度（按head）", unit="head")

    # 启动消费者线程（非守护线程）
    consumer_thread = threading.Thread(
        target=consumer,
        args=(task_queue, task_count, process_pbar, exit_event),
        daemon=False
    )
    consumer_thread.start()

    try:
        # 多线程收集并提交任务（收集后立即提交，无本地缓存）
        with ThreadPoolExecutor(max_workers=COLLECT_WORKERS, thread_name_prefix="collect") as collect_executor:
            # 提交所有收集任务（每个任务独立收集+提交）
            futures = [
                collect_executor.submit(collect_and_submit_task, task_info, image_cache, invalid_images, task_queue)
                for task_info in pending_tasks
            ]

            # 等待所有收集任务完成（确保所有任务都提交到队列）
            print(f"\n等待所有任务收集并提交到队列（共{len(futures)}个任务）", file=sys.stderr)
            for future in tqdm(as_completed(futures), total=len(futures), desc="收集并提交任务"):
                try:
                    future.result()
                except Exception as e:
                    with threading.Lock():
                        print(f"\n收集任务异常：{e}", file=sys.stderr)

        # 等待消费者线程处理完所有任务
        print(f"\n所有任务已提交到队列，等待处理完成...", file=sys.stderr)
        consumer_thread.join(timeout=43200)  # 12小时超时

        # 检查是否所有任务都已完成
        with task_count["lock"]:
            if task_count["completed"] < task_count["total"]:
                print(f"\n警告：超时未完成所有任务（已完成：{task_count['completed']}/{task_count['total']}）", file=sys.stderr)
            else:
                print(f"\n所有任务处理完成！", file=sys.stderr)

    except KeyboardInterrupt:
        print(f"\n\n用户中断程序，正在安全退出并保存结果...", file=sys.stderr)
        exit_event.set()
        consumer_thread.join(timeout=60)
    except Exception as e:
        print(f"\n错误：程序异常 -> {e}", file=sys.stderr)
        exit_event.set()
        consumer_thread.join(timeout=60)
        raise
    finally:
        # 关闭进度条
        process_pbar.close()

        # 清理根级text字段
        for item in data:
            if "text" in item:
                del item["text"]

        # 保存结果（覆盖已存在文件）
        with open(JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"\n已保存当前处理结果至：{JSON_OUTPUT_PATH}", file=sys.stderr)

        # 输出最终统计
        with task_count["lock"]:
            processed = task_count["completed"]
        print("-" * 50)
        print(f"处理完成统计：")
        print(f"本次需处理head数：{total_tasks}")
        print(f"实际处理完成数：{processed}")
        print(f"跳过无效图片：{len(invalid_images)} 张")
        print(f"未处理完成数：{total_tasks - processed}")

        if total_tasks - processed > 0:
            print(f"\n⚠️ 仍有未处理完成的样本，下次运行将自动续传！", file=sys.stderr)
        else:
            print(f"\n🎉 所有样本已100%处理完成！", file=sys.stderr)

if __name__ == "__main__":
    # 修复线程栈大小兼容问题
    try:
        threading.stack_size(1 << 27)
    except Exception as e:
        print(f"重置线程栈大小失败，使用默认值：{e}", file=sys.stderr)
    main()