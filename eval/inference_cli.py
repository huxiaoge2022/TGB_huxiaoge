import os
import json
import argparse
import math
from tqdm import tqdm

import torch
from transformers import TextStreamer

import sys
print(sys.path)

from src.data.components.conversation import conv_templates, SeparatorStyle
from src.data.components.constants import DEFAULT_X_START_TOKEN, DEFAULT_X_TOKEN, DEFAULT_X_END_TOKEN, X_TOKEN_INDEX

from .utils.builder_utils import load_pretrained_model, get_frames, KeywordsStoppingCriteria




def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments

    '''
    # 原始
    parser.add_argument('--model_path', help='', default="ckpts/LSTP-FlanT5xl.ckpt")
    parser.add_argument('--cache_dir', help='', default="./cache_dir")
    parser.add_argument("--nframe", type=int, default=4)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default="ckpts/blip2-flan-t5-xl", type=str, required=False)
    parser.add_argument('--sampler_base', help='', default="ckpts/bert-base-uncased", type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)
    parser.add_argument("--lora", type=int, default=1)
    '''


    parser.add_argument('--model_path', help='', default="/root/VideoTGB/LSTP-Chat/LSTP-FlanT5xl.ckpt")
    parser.add_argument('--cache_dir', help='', default="./cache_dir")
    parser.add_argument("--nframe", type=int, default=4)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--device", type=str, required=False, default='cuda:0')
    parser.add_argument('--model_base', help='', default="/root/VideoTGB/ckpts/blip2-flan-t5-xl", type=str, required=False)
    parser.add_argument('--sampler_base', help='', default="/root/VideoTGB/ckpts/bert-base-uncased", type=str, required=False)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)
    parser.add_argument("--lora", type=int, default=1)


    return parser.parse_args()

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

def get_model_output(model, processor, sampler_processor, video_path, question, args):
    
    frames, flow_frames = get_frames(video_path, fps=2)
    # frames = frames.unsqueeze(0)
    flow_frames = flow_frames.unsqueeze(0)

    frames = frames.to(args.device)
    flow_frames = flow_frames.to(args.device)

    # prompt = "question: " + question + "short answer: "
    # prompt = "Question: " + question + "\nAnswer the question using a single word or phrase."
    prompt = "USER: <video>\n" + question + " ASSISTANT: "
    text_encoding = processor(
        text=prompt,
        padding="longest",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(args.device)
    sampler_text_encoding = sampler_processor(
        text=question,
        padding="longest",
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(args.device)

    if "vicuna" in processor.tokenizer.name_or_path:
        stopping_criteria = [KeywordsStoppingCriteria(['</s>'], processor.tokenizer, text_encoding.input_ids)]
        # stopping_criteria = None
    else:
        stopping_criteria = None

    with torch.inference_mode():
        output_ids, sampled_indices = model.generate(
            frames,
            flow_frames,
            args.nframe,
            text_encoding,
            sampler_text_encoding,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=128,
            use_cache=False,
            stopping_criteria=stopping_criteria,
        )
    if 'vicuna' in processor.tokenizer.name_or_path:
        outputs = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        if outputs.endswith('</s>'):
            outputs = outputs[:-len('</s>')]
    else:
        outputs = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    # print("question: ", question)
    # print("prediciton: ", outputs)
    return outputs

def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    # Initialize the model
    # model_name = get_model_name_from_path(args.model_path)
    # model, processor, sampler_processor = load_pretrained_model(args.model_path, args.model_base, args.sampler_base, args.device)
    model, processor, sampler_processor = load_pretrained_model(args.model_path, args.model_base, args.sampler_base, args.device, args.lora)
    model = model.to(args.device)
    
    
    
    question = "who opened the box that held an automatic weapon in a gun?"
    # video_path = "demo/examples/sample_demo_1.mp4"
    video_path = "/root/VideoTGB/demo/examples/sample_demo_1.mp4"
    
    output = get_model_output(model, processor, sampler_processor, video_path, question, args)
    
    print("question: ", question)
    print("predict: ", output)
    



if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
    