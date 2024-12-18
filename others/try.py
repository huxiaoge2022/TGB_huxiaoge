import os
import cv2
import json
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


    

if __name__ == '__main__':
    # 加载 BLIP-2 模型到设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Blip2Processor.from_pretrained("/root/VideoTGB/ckpts/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained('/root/VideoTGB/ckpts/instructblip-vicuna-7b').to(device)

    print('load success')

    from IPython import embed
    embed()