import os
import av
import decord
from decord import cpu

import numpy as np
import torch
from torchvision.transforms import Compose
from transformers import AutoProcessor, AutoTokenizer, StoppingCriteria

from peft import PeftModel, PeftConfig, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType

from src.data.components.util import sample_frames
from src.gadgets.transforms import RandomResizedCropVideo, ToTHWC, ToUint8, ToTensorVideo, NormalizeVideo, ResizeVideo
from .model import LSTP, LSTP_blip2


DEFAULT_X_PATCH_TOKEN = {'IMAGE': "<im_patch>", 'VIDEO': "<vi_patch>", 'AUDIO': "<au_patch>", 'THERMAL': "<th_patch>", 'DEPTH': "<de_patch>"}
# DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_X_START_TOKEN = {'IMAGE': "<im_start>", 'VIDEO': "<vi_start>", 'AUDIO': "<au_start>", 'THERMAL': "<th_start>", 'DEPTH': "<de_start>"}
# DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_X_END_TOKEN = {'IMAGE': "<im_end>", 'VIDEO': "<vi_end>", 'AUDIO': "<au_end>", 'THERMAL': "<th_end>", 'DEPTH': "<de_end>"}
# DEFAULT_IM_END_TOKEN = "<im_end>"

def read_videos_decord(video_path, num_frames=-1, sample='rand', trim=1., fix_start=-1, keyframe=False, start_ratio=0.0, end_ratio=1.0, fps=2):
    
    
    # video_reader = decord.VideoReader(video_path, height=224, width=224)
    # video_reader = decord.VideoReader(video_path, num_threads=0)
    video_reader = decord.VideoReader(video_path, ctx=cpu(0))
    
    if fps is not None:
        avg_fps = int(video_reader.get_avg_fps())
        indices = []
        if fps < avg_fps:
            step = avg_fps
            for idx in range(len(video_reader)):
                if idx % step == 0:
                    indices.append(idx)
        else:
            indices = list(range(len(video_reader)))
        flow_frames = video_reader.get_batch(indices).permute(3,0,1,2).float() # T H W C -> C H W T
        # vlen = len(indices)
        # ori_indices = indices
        return flow_frames 

    # else:
    #     vlen = len(video_reader)
    #     ori_indices = list(range(vlen))
        
    # if trim < 1.:
    #     remain = (1. - trim) / 2
    #     start, end = int(vlen * remain), int(vlen * (1 - remain))
    #     indices = ori_indices[start:end]
    # if keyframe:
    #     start, end = int(vlen*start_ratio), int(vlen*end_ratio)+1
    #     indices = ori_indices[start:end]

    # if num_frames > 0:
    #     while vlen < num_frames: # duplicate frames
    #         ori_indices = [f for ind in ori_indices for f in (ind, ind)]
    #         vlen = len(ori_indices)
    #     frame_ids = sample_frames(num_frames, vlen, sample, fix_start)
    #     indices = [ori_indices[ii] for ii in frame_ids]
    # frames = video_reader.get_batch(indices).permute(3,0,1,2).float() # T H W C -> C H W T
    # return frames, flow_frames

def read_videos_av(video_path, num_frames=-1, sample='rand', trim=1., fix_start=-1, keyframe=False, start_ratio=0.0, end_ratio=1.0, fps=2):
    
    
    video = av.open(video_path)
    
    if fps is not None:
        frames = []
        avg_fps = int(video.streams.video[0].average_rate)
        if fps <= avg_fps:
            step = avg_fps
            for idx, frame in enumerate(video.decode(video=0)):
                if idx % step == 0:
                    frame = frame.to_ndarray(format='rgb24')
                    frames.append(frame)
        else:
            for frame in enumerate(video.decode(video=0)):
                frame = frame.to_ndarry(format="rgb24")
                frames.append(frame)
        flow_frames = torch.from_numpy(np.stack(frames, axis=0)).permute(3,0,1,2).float() # T H W C -> C T H W
        return flow_frames
    # else:
    #     frames = []
    #     for frame in video.decode(video=0):
    #         frame = frame.to_ndarray(format='rgb24')
    #         frames.append(frame)
    # vlen = len(frames)
    # ori_indices = list(range(vlen))
    # indices = list(range(vlen))
    
    # if trim < 1.:
    #     remain = (1. - trim) / 2
    #     start, end = int(vlen * remain), int(vlen * (1 - remain))
    #     indices = ori_indices[start:end]
    # if keyframe:
    #     start, end = int(vlen*start_ratio), int(vlen*end_ratio)+1
    #     indices = ori_indices[start:end]

    # if num_frames > 0 and vlen > num_frames:
    #     while vlen < num_frames: # duplicate frames
    #         ori_indices = [f for ind in ori_indices for f in (ind, ind)]
    #         vlen = len(ori_indices)
    #     frame_ids = sample_frames(num_frames, vlen, sample, fix_start)
    #     indices = [ori_indices[ii] for ii in frame_ids]

    # frames = torch.from_numpy(np.stack([frames[x] for x in indices], axis=0)).permute(3,0,1,2).float() # T H W C -> C T H W
    # return frames, flow_frames



def get_frames(video_path, target_size=224, keyframe=False, start_ratio=0.0, end_ratio=1.0, fps=None):
    video_transform = Compose([
        ResizeVideo(target_size),
        ToUint8(),
        ToTHWC(),
        ToTensorVideo(),
        NormalizeVideo((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # C, T, H, W
    ])
    flow_frames = read_videos_av(video_path, 32, "uniform", 1., keyframe=keyframe, start_ratio=start_ratio, end_ratio=end_ratio, fps=fps)
    flow_frames = video_transform(flow_frames)
    # print(flow_frames.size())
    flow_frames = flow_frames.permute(1, 0, 2, 3) # C T H W -> T C H W
    # print(flow_frames.size())

    # uniform sampling
    vlen = flow_frames.shape[0]
    indices = list(range(vlen))
    while vlen < 32:
        indices = [f for ind in indices for f in (ind, ind)]
        vlen = len(indices)
    frame_ids = sample_frames(32, vlen, "uniform", 1.)
    indices = [indices[ii] for ii in frame_ids]
    frames = flow_frames[indices]

    # print(frames.size())
    # print(flow_frames.size())
    # frames = frames.permute(1,0,2,3) # T C H W
    return frames, flow_frames


def load_data(text, video, nframe, processor, sampler_processor):
    frames = get_frames(video)
    text_encoding = processor(
        text=text,
        padding=True,
        return_tensors="pt",
    )
    sampler_text_encoding = sampler_processor(
        text=text,
        padding=True,
        return_tensors="pt"
    )
    return frames, text_encoding, sampler_text_encoding

def dp_state_to_normal(state_dict):
    '''Converts a torch.DataParallel checkpoint to regular'''
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module'):
            new_state_dict[k.replace('module.', '')] = v
    return new_state_dict

# 原始
def load_pretrained_model(ckpt_path, base_model_path, base_sampler_path, device, lora=False):  # 原始
# def load_pretrained_model(ckpt_path, base_model_path, base_sampler_path, device, lora=True):  
    print("start to load model... real.")
    processor = AutoProcessor.from_pretrained(base_model_path)
    sampler_processor = AutoTokenizer.from_pretrained(base_sampler_path)

    if "instructblip" in base_model_path:
        model = LSTP(base_model_path, device, lora)
    elif "blip2" in base_model_path:
        model = LSTP_blip2(base_model_path, device, lora)
    state_dict = torch.load(ckpt_path, map_location='cpu')  # 原始
    #state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    # print(state_dict.keys())  # 打印文件中包含的所有键 # 加
    # 假设模型文件直接包含模型对象
    # state_dict = dp_state_to_normal(state_dict)  # 原始这条注释了
    # msg = model.load_state_dict(state_dict['state_dict'])  # 原始
    msg = model.load_state_dict(state_dict)  # 原始

    # msg = model.load_state_dict(state_dict, strict=False)  # trict=False忽略不匹配的键,只是忽略了，不代表不存在  # 加

    print(">>> Load checkpoint for LSTP from", ckpt_path)
    # 打印缺失和意外的键
    miss = set(m.split('.')[0] for m in msg.missing_keys)
    unexp = set(m.split('.')[0] for m in msg.unexpected_keys)
    print("Missing:", miss if len(miss) else "None")
    print("Unexpected:", unexp if len(unexp) else "None")
    # model = model.to(device)  # 加

    return model, processor, sampler_processor



'''
# 尝试修改1
def load_pretrained_model(ckpt_path, base_model_path, base_sampler_path, device, lora=False):
    print("start to load model...")
    processor = AutoProcessor.from_pretrained(base_model_path)
    sampler_processor = AutoTokenizer.from_pretrained(base_sampler_path)

    if "instructblip" in base_model_path:
        model = LSTP(base_model_path, device, lora)
    elif "blip2" in base_model_path:
        model = LSTP_blip2(base_model_path, device, lora)

    # 直接加载检查点文件
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # print(state_dict.keys())  # 打印文件中包含的所有键 

    # 假设检查点文件直接保存了模型
    model = state_dict  # 如果检查点包含模型本身
    model = model.to(device)

    print(">>> Load checkpoint for LSTP from", ckpt_path)

    return model, processor, sampler_processor
'''

'''
# 尝试修改4
def load_pretrained_model(ckpt_path, base_model_path, base_sampler_path, device, lora=False):
# def load_pretrained_model(ckpt_path, base_model_path, base_sampler_path, device, lora=True):
    print("start to load model...")

    # 加载处理器
    processor = AutoProcessor.from_pretrained(base_model_path)
    sampler_processor = AutoTokenizer.from_pretrained(base_sampler_path)

    # 根据不同模型路径选择相应模型架构
    if "instructblip" in base_model_path:
        model = LSTP(base_model_path, device, lora)
    elif "blip2" in base_model_path:
        model = LSTP_blip2(base_model_path, device, lora)

    # 加载检查点文件
    # state_dict = torch.load(ckpt_path, map_location='cpu')
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)

    # 如果检查点包含模型权重，加载到模型中
    if 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'], strict=False)
    else:
        # 如果检查点直接保存了权重字典
        model.load_state_dict(state_dict, strict=False)
    
    msg = model.load_state_dict(state_dict)
    miss = set(m.split('.')[0] for m in msg.missing_keys)
    unexp = set(m.split('.')[0] for m in msg.unexpected_keys)
    print("Missing:", miss if len(miss) else "None")
    print("Unexpected:", unexp if len(unexp) else "None")

    # 将模型移到指定设备（GPU/CPU）
    model = model.to(device)

    print(">>> Load checkpoint for LSTP from", ckpt_path)

    return model, processor, sampler_processor
'''





'''尝试修改2
def load_pretrained_model(ckpt_path, base_model_path, base_sampler_path, device, lora=False):
    print("start to load model...")
    processor = AutoProcessor.from_pretrained(base_model_path)
    sampler_processor = AutoTokenizer.from_pretrained(base_sampler_path)

    if "instructblip" in base_model_path:
        model = LSTP(base_model_path, device, lora)
    elif "blip2" in base_model_path:
        model = LSTP_blip2(base_model_path, device, lora)

    # 加载检查点
    state_dict = torch.load(ckpt_path, map_location='cpu')


    # 假设检查点文件中包含 'model' 键保存了模型
    if 'model' in state_dict:
        model.load_state_dict(state_dict['model'])
    else:
        # 如果没有 'model' 键，直接加载检查点（或者其他键名）
        model.load_state_dict(state_dict)

    print(">>> Load checkpoint for LSTP from", ckpt_path)

    miss = set(m.split('.')[0] for m in msg.missing_keys)
    unexp = set(m.split('.')[0] for m in msg.unexpected_keys)
    print("Missing:", miss if len(miss) else "None")
    print("Unexpected:", unexp if len(unexp) else "None")

    return model, processor, sampler_processor
'''

'''尝试修改3
def load_pretrained_model(ckpt_path, base_model_path, base_sampler_path, device, lora=False):
    print("start to load model...")
    processor = AutoProcessor.from_pretrained(base_model_path)
    sampler_processor = AutoTokenizer.from_pretrained(base_sampler_path)

    if "instructblip" in base_model_path:
        model = LSTP(base_model_path, device, lora)
    elif "blip2" in base_model_path:
        model = LSTP_blip2(base_model_path, device, lora)

    # 加载检查点
    state_dict = torch.load(ckpt_path, map_location='cpu')

    # print(state_dict.keys())  # 打印文件中包含的所有键

    # 根据检查点的结构加载模型权重
    if 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    elif 'model' in state_dict:
        model.load_state_dict(state_dict['model'])
    else:
        model.load_state_dict(state_dict)  # 如果直接包含模型

    print(">>> Load checkpoint for LSTP from", ckpt_path)

    miss = set(m.split('.')[0] for m in msg.missing_keys)
    unexp = set(m.split('.')[0] for m in msg.unexpected_keys)
    print("Missing:", miss if len(miss) else "None")
    print("Unexpected:", unexp if len(unexp) else "None")

    return model, processor, sampler_processor
'''



def load_huggingface_model(ckpt_path, base_model_path, base_sampler_path, device, lora=False):
    print("start to load model... ")
    processor = AutoProcessor.from_pretrained(base_model_path)
    sampler_processor = AutoTokenizer.from_pretrained(base_sampler_path)

    if "instructblip" in base_model_path:
        model = LSTP(base_model_path, device, lora)
    elif "blip2" in base_model_path:
        model = LSTP_blip2(base_model_path, device, lora)



    return model, processor, sampler_processor

# visualize flow
def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)



class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False