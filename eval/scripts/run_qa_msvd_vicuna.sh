

CKPT_NAME="LSTP-Instructblip-Vicuna"
model_base="/root/VideoTGB/ckpts/instructblip-vicuna-7b"
sampler_base="/root/VideoTGB/ckpts/bert-base-uncased"
model_path="/root/LSTP-Chat/LSTP-7B.ckpt"
cache_dir="./cache_dir"
GPT_Zero_Shot_QA="/root/VideoTGB/eval/GPT_Zero_Shot_QA"
video_dir="/root/VideoTGB/eval/GPT_Zero_Shot_QA/MSVD_Zero_Shot_QA/videos"
gt_file_question="/root/VideoTGB/eval/GPT_Zero_Shot_QA/MSVD_Zero_Shot_QA/test_q.json"
gt_file_answers="/root/VideoTGB/eval/GPT_Zero_Shot_QA/MSVD_Zero_Shot_QA/test_a.json"
output_dir="/root/VideoTGB/outputs"
nframe=4
lora=1

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


# 单 GPU 环境
CUDA_VISIBLE_DEVICES=0 python -m eval.inference \
    --model_base ${model_base} \
    --sampler_base ${sampler_base} \
    --model_path ${model_path} \
    --cache_dir ${cache_dir} \
    --video_dir ${video_dir} \
    --gt_file_question ${gt_file_question} \
    --gt_file_answers ${gt_file_answers} \
    --output_dir ${output_dir} \
    --output_name single_chunk \
    --nframe ${nframe} \
    --lora ${lora} \
    --num_chunks 1 \
    --chunk_idx 0

# 合并文件（此处仅有一个分块文件）
output_file=${output_dir}/merge.jsonl

# 确保目标文件为空
> "$output_file"

# 将单块文件写入目标文件
cat ${output_dir}/single_chunk.json >> "$output_file"

