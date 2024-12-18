import os
import json
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoProcessor, InstructBlipProcessor, BlipImageProcessor

from .util import read_videos, read_videos_cv2, read_videos_av, sample_frames, flow_to_image
from src.data.components import conversation as conversation_lib

class VIDEOINSTRUCT(Dataset):
    
    def __init__(self,
            text_dir: str,  # 文本数据所在的目录，通常包括问题、答案和伪标签（pseudo_label.json）
            video_dir: str,  # 视频数据的目录，存储视频文件（例如 .mp4 格式）
            of_dir: str,  # 光流数据的目录，通常是 .npy 格式文件
            nframe: int,  # 从视频中采样的帧数
            split: str,  # 数据集划分标识（例如 train, val, test）
            processor: Optional[Callable] = None,
            sampler_processor: Optional[Callable] = None,
            video_transform: Optional[Callable] = None,
            image_transform: Optional[Callable] = None,
        ):
        
        # 将传入参数赋值给类的成员变量
        self.split = split
        self.video_transform = video_transform
        self.image_transform = image_transform
        self.processor = processor
        self.sampler_processor = sampler_processor
        
        self.video_dir = video_dir
        self.of_dir = of_dir
        self.text_dir = text_dir
        
        self.max_txt_len = 128  # 最大文本长度，限制文本输入的序列长度
        self.nframe = nframe
        self.sampling = 'uniform'  # 帧采样策略，默认为均匀采样
        
        self.data = self._load_data()  # 调用 _load_data 方法：读取 split 对应的 JSON 文件，提取问题、答案等数据(该数据能找着)

        # pseudo_label 从 pseudo_label.json 文件中读取伪标签，伪标签通常是动作的开始和结束帧，用于监督模型
        data_path = os.path.join(self.text_dir,  'pseudo_label.json')  # 伪标签要自己去生成 
        with open(data_path) as jp:
            self.pseudo_label = json.load(jp)
        
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, index):
        
        data_dct = self.data[index]  # 获取数据样本
        answer = data_dct["a"]  # 从样本的字典中提取问题 (q) 和答案 (a)
        question = data_dct["q"]
        question = "USER: " + question + "ASSISTANT: "  # 将问题格式化为 "USER: {问题} ASSISTANT: " 的形式
        answer = answer + " </s>"  # 答案末尾添加一个特殊的结束标记 </s>，这有助于模型理解回答的结束位置
        instruction = question + ' ' + answer  # 合成指令：将格式化后的问题和答案连接成一个完整的对话指令
        vid = data_dct["video_id"]  # 提取当前样本的视频 ID (video_id) 和索引 (idx)
        idx = data_dct['idx']
        
        # process video and of
        frames = self.get_frames(vid, self.video_dir) # TCHW  调用 self.get_frames() 方法，从视频文件中提取帧数据
        frames = frames.to(torch.float32) # 加
        of = self.get_of(vid, self.of_dir) # TCHW  调用 self.get_of() 方法，获取与视频对应的光流数据
        flows = []
        for flow in of:  # 遍历每一帧的光流数据 of
            flow = flow_to_image(np.transpose(flow, (1, 2, 0))) # HWC 将光流转换为图像格式
            flows.append(flow)
        of_rgb = torch.from_numpy(np.stack(flows, axis=0)) # THWC  # 将 NumPy 数组转换为 PyTorch 张量
        of_rgb = of_rgb.permute(3, 0, 1, 2).float() # CTHW 调整维度
        of_rgb = self.video_transform(of_rgb)  # 调用 self.video_transform() 对光流图像进行预处理，通常包括标准化、裁剪或缩放等
        of_rgb = of_rgb.permute(1,0,2,3) # TCHW 恢复原始维度
        # of_length = of.size(0) + 2
        of_length = of.shape[0]  # 计算光流长度：获取光流序列的长度，即时间轴上的帧数
        of = self.normalize_flow(of)  # 光流标准化
        of = torch.from_numpy(of)
        of = of.to(torch.float32)  # 加

        # pseudo_label
        # 伪标签处理：从 self.pseudo_label 中提取伪标签的开始和结束时间。
        # 伪标签存储为一个帧范围 [0, 31]，此处将其映射到实际光流帧数范围 [0, of_length-1]。
        # start = int(self.pseudo_label[idx][0] / 31 * (of_length-1))  # 原始
        # end = int(self.pseudo_label[idx][1] / 31 * (of_length-1))  # 原始
        
        pseudo = self.pseudo_label[idx]  # 改
        start = pseudo["start_index"]  # 改
        end = pseudo["end_index"]  # 改

        # of_length += 2


        return {'idx': idx, 'frames':frames, 'of':of, 'of_rgb':of_rgb, 'of_length':of_length, 'question':question, 'answer':answer, 'instruction':instruction, "idx": idx, "start": start, "end": end}
        '''
        返回样本：将所有处理后的数据打包成一个字典返回：
            idx：样本索引。
            frames：视频帧数据，形状为 T C H W。
            of：光流数据，形状为 T C H W。
            of_rgb：RGB格式的光流图像，形状为 T C H W。
            of_length：光流的帧数。
            question：格式化后的用户问题。
            answer：格式化后的助手回答。
            instruction：问题和回答的合成指令。
            start 和 end：伪标签的开始和结束帧。
        
        '''
    
    def collate(self, batch):
        
        idxs = [x['idx'] for x in batch]

        # frames = [x['frames'] for x in batch]
        frames = torch.cat([data['frames'] for data in batch]) # B*T, 3, 224, 224
        answers = [x['answer'] for x in batch]
        questions = [x['question'] for x in batch]
        instruction = [x['instruction'] for x in batch]

        # preprocess image and flow
        for i, x in enumerate(batch):  # 加
            print(f"Sample {i} shape: {x['of'].shape}")  # 加
            print(x['of'].dtype)  # 加
        

        of = pad_sequence([x['of'] for x in batch], batch_first=True)
        of_mask = torch.zeros(of.size(0), of.size(1)+2, dtype=torch.long)
        for i, data in enumerate(batch):
            of_mask[i, :data["of"].size(0)+2] = 1
        of_rgb = pad_sequence([x['of_rgb'] for x in batch], batch_first=True)
        of_rgb_mask = torch.zeros(of_rgb.size(0), of_rgb.size(1)+2, dtype=torch.long)
        for i, data in enumerate(batch):
            of_rgb_mask[i, :data["of_rgb"].size(0)+2] = 1
        of_lengths = [x['of_length'] for x in batch]
        starts = torch.tensor([x["start"] for x in batch], dtype=torch.long)
        ends = torch.tensor([x["end"] for x in batch], dtype=torch.long)

        # preprocess text
        sampler_question_encoding = self.sampler_processor(
            text=questions,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        )
        if 'vicuna' in self.processor.tokenizer.name_or_path:
            self.processor.padding_side = "right"
            self.processor.truncation_side = "left"
        question_encoding = self.processor(
            text=questions,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        )
        if 'vicuna' in self.processor.tokenizer.name_or_path:
            self.processor.truncation_side = "right" 
        answer_encoding = self.processor(
            text=answers,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        )
        instruction_encoding = self.processor(
            text=instruction,
            padding='longest',
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        )
        
        
        if "instructblip" in self.processor.tokenizer.name_or_path:
            return {"idxs": idxs,
                    "frames": frames, 
                    "of": of,
                    "of_mask": of_mask,
                    "of_rgb": of_rgb,
                    "of_rgb_mask": of_rgb_mask,
                    "sampler_question": sampler_question_encoding["input_ids"],
                    "sampler_question_attention_mask": sampler_question_encoding["attention_mask"],
                    "question": question_encoding["input_ids"],
                    "question_attention_mask": question_encoding["attention_mask"],
                    "qformer_text":question_encoding["qformer_input_ids"],
                    "qformer_text_attention_mask": question_encoding["qformer_attention_mask"],
                    "instruction": instruction_encoding["input_ids"],
                    "instruction_attention_mask": instruction_encoding["attention_mask"],
                    "answer": answer_encoding["input_ids"],
                    "answer_attention_mask": answer_encoding["attention_mask"],
                    "text_answer": answers,
                    "nframe": self.nframe,
                    "of_lengths": of_lengths,
                    "starts": starts,
                    "ends": ends,
                    }
        elif "blip2" in self.processor.tokenizer.name_or_path:
            return {"idxs": idxs,
                    "frames": frames, 
                    "of": of,
                    "of_mask": of_mask,
                    "of_rgb": of_rgb,
                    "of_rgb_mask": of_rgb_mask,
                    "sampler_question": sampler_question_encoding["input_ids"],
                    "sampler_question_attention_mask": sampler_question_encoding["attention_mask"],
                    "question": question_encoding["input_ids"],
                    "question_attention_mask": question_encoding["attention_mask"],
                    # "qformer_text":question_encoding["qformer_input_ids"],
                    # "qformer_text_attention_mask": question_encoding["qformer_attention_mask"],
                    "instruction": instruction_encoding["input_ids"],
                    "instruction_attention_mask": instruction_encoding["attention_mask"],
                    "answer": answer_encoding["input_ids"],
                    "answer_attention_mask": answer_encoding["attention_mask"],
                    "text_answer": answers,
                    "nframe": self.nframe,
                    "of_lengths": of_lengths,
                    "starts": starts,
                    "ends": ends,
                    }
            
    
    def _load_data(self):
        
        data_path = os.path.join(self.text_dir, self.split + '.json')  # 构建数据文件路径
        '''这里self.split应该是将含问题和答案的.json文件分别切成只含问题json文件和只含答案的json文件'''
        with open(data_path) as jp:
            data_dct = json.load(jp)  # 使用 json.load(jp) 将打开的文件内容解析为 Python 字典对象 data_dct
        data_lst = []
        # for idx, dct in data_dct.items():  # 原始
        for idx, dct in enumerate(data_dct):
            dct['idx'] = idx
            data_lst.append(dct)
        return data_lst
            
    @staticmethod
    def rescale(x,max_range,min_range):
        max_val = np.max(x)
        min_val = np.min(x)
        return (max_range-min_range)/(max_val-min_val)*(x-max_val)+max_range
    @staticmethod
    def normalize_flow(flow):
        # N, 2, H, W -> N, H, W, 2
        # flow_uv = np.transpose(flow, (0, 2, 3, 1))
        flow_uv = flow.transpose(0,2,3,1)
        u = flow_uv[:,:,:,0]
        v = flow_uv[:,:,:,1]
        rad = np.sqrt(np.square(u) + np.square(v))
        rad_max = np.max(rad)
        epsilon = 1e-5
        u = u / (rad_max + epsilon)
        v = v / (rad_max + epsilon)
        normalized_flow_uv = np.stack([u,v], axis=-1)
        # normalized_flow =np.transpose(normalized_flow_uv, (0, 3, 1, 2))
        normalized_flow = normalized_flow_uv.transpose(0, 3, 1, 2)
        return normalized_flow

    def get_of(self, video_name, of_path):  # 从光流数据路径中加载视频的光流数据
        of_path = os.path.join(of_path, video_name+'_raft.npy')  # 构建光流文件路径
        of = np.load(of_path) # num_of_frames, 2, H, W 加载光流数据 
        '''num_of_frames：光流数据的帧数（即视频中的帧数）
            2：光流数据的通道数，分别表示水平和垂直的光流分量
        '''
        # of_npy = self.normalize_flow(of_npy) # not compatible with rgb
        # of = torch.tensor(of, dtype=torch.float)

        # # way1. cut off
        if of.shape[0] > 64:
            fid = sample_frames(64, of.shape[0], self.sampling)
            of = of[fid]
        
        # # way2. fix length 
        # fid = list(range(of.size(0)))
        # vlen = len(fid)
        # while vlen < 32: # duplicate frames
        #     fid = [f for ind in fid for f in (ind, ind)]
        #     vlen = len(fid)
        # idx = sample_frames(32, vlen, self.sampling)
        # fid = [fid[x] for x in idx]
        # of = of[fid]

        return of

    def get_frames(self, video_name, video_path, keyframe=False, start_ratio=0.0, end_ratio=1.0):
        v = os.path.join(video_path, video_name+'.mp4')
        frames = read_videos_av(v, 32, self.sampling, 1., keyframe=keyframe, start_ratio=start_ratio, end_ratio=end_ratio) # C T H W
        # frames = read_videos(v, 32, self.sampling, 1., keyframe=keyframe, start_ratio=start_ratio, end_ratio=end_ratio)
        '''
            读取视频帧：调用 read_videos_av 函数读取视频文件，并提取帧数据。
                v：视频文件的完整路径。
                32：表示读取32帧视频。
                self.sampling：视频采样方式，可能是均匀采样或其他方式。
                1.：表示帧之间的时间间隔（默认为1秒）。
                keyframe：布尔值，决定是否提取关键帧。
                start_ratio 和 end_ratio：指定视频中提取帧的时间范围。

        '''
        frames = self.video_transform(frames)  # 调用 self.video_transform() 对提取的帧数据进行预处理
        frames = frames.permute(1,0,2,3) # T C H W  
        return frames
        