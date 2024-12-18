import os
import cv2
import json
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from rouge_score import rouge_scorer

def LLM(frame, query):
    """使用 BLIP-2 生成基于图像和查询的预测"""
    try:
        if frame is None or not isinstance(frame, Image.Image):
            raise ValueError("Invalid frame: Ensure it is a valid PIL Image.")
        if not query or not isinstance(query, str):
            raise ValueError("Query is either empty or not a string")

        inputs = processor(images=frame, text=query, return_tensors="pt").to(device)
        
        if "input_ids" not in inputs or not inputs["input_ids"].size(0):
            raise ValueError("Processor output 'input_ids' is empty.")
        if "pixel_values" not in inputs or not inputs["pixel_values"].size(0):
            raise ValueError("Processor output 'pixel_values' is empty.")

        outputs = model.generate(**inputs)
        prediction = processor.decode(outputs[0], skip_special_tokens=True)
        return prediction
    except Exception as e:
        print(f"Error in LLM function: {e}")
        raise

def extract_frames(video_path, frames_per_second=1):
    """从视频中提取每秒若干帧，并返回帧的图像列表"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # 获取视频的帧率 (fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps == 0:
        print(f"Failed to get FPS for video: {video_path}")
        return frames

    frame_interval = int(fps // frames_per_second)  # 每帧间隔，确保每秒提取指定数量的帧

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_interval == 0:  # 只提取间隔帧
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
    
    cap.release()
    
    if not frames:
        print(f"No valid frames extracted from {video_path}")
    
    return frames

def SIM(prediction, answer):
    """计算预测结果与答案的 Rouge-1 分数"""
    try:
        if not prediction or not isinstance(prediction, str):
            raise ValueError("Prediction is either empty or not a string")
        if not answer or not isinstance(answer, str):
            raise ValueError("Answer is either empty or not a string")

        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        scores = scorer.score(answer, prediction)
        return scores["rouge1"].fmeasure
    except Exception as e:
        print(f"Error in SIM function (Rouge-1): {e}")
        raise

def pseudo_label_algorithm(frames, query, answer):
    """伪标签生成算法"""
    score_best = 0
    start = 0
    end = len(frames) - 1
    stack = []
    scores = []

    for frame in frames:
        prediction = LLM(frame, query)
        scores.append(SIM(prediction, answer))
    
    for i in range(len(scores)):
        while stack and scores[stack[-1]] > scores[i]:
            tmp = stack.pop()
            if stack:
                score_tmp = (i - stack[-1] - 1) * scores[tmp]
            else:
                score_tmp = i * scores[tmp]
            
            if score_tmp > score_best:
                score_best = score_tmp
                start = stack[-1] + 1 if stack else 0
                end = i - 2

        stack.append(i)

    return score_best, start, end

def process_videos_and_queries(video_folder, query_json_file, output_json_file):
    with open(query_json_file, 'r') as f:
        query_data = json.load(f)
    
    pseudo_labels = []

    for entry in query_data:
        video_id = entry["video_id"]
        query = entry["q"]
        answer = entry["a"]
        
        video_path = os.path.join(video_folder, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            continue
        
        frames = extract_frames(video_path)
        if not frames:
            print(f"No frames extracted for video: {video_id}")
            continue
        
        try:
            score_best, start_idx, end_idx = pseudo_label_algorithm(frames, query, answer)
            pseudo_labels.append({
                "video_id": video_id,
                "score_best": score_best,
                "start_index": start_idx,
                "end_index": end_idx,
                "query": query,
                "answer": answer
            })

        except Exception as e:
            print(f"Error processing video {video_id}: {e}")

    with open(output_json_file, 'w') as f:
        json.dump(pseudo_labels, f, indent=4)

if __name__ == "__main__":
    # 加载 BLIP-2 模型到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Blip2Processor.from_pretrained("/root/VideoTGB/ckpts/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("/root/VideoTGB/ckpts/blip2-flan-t5-xl").to(device)

    video_folder = "/root/VideoTGB/inputs/videoinstruct/video"
    query_json_file = "/root/VideoTGB/inputs/videoinstruct/test_data.json"
    output_json_file = "/root/VideoTGB/inputs/videoinstruct/pseudo_label2.json"

    process_videos_and_queries(video_folder, query_json_file, output_json_file)
