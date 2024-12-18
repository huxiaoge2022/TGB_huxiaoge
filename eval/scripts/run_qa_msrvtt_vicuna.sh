

CKPT_NAME="LSTP-Instructblip-Vicuna-iv"
model_base="ckpts/instructblip-vicuna-7b"
sampler_base="ckpts/bert-base-uncased"
model_path="logs/ivinstruct/train/runs/2024-02-20/checkpoints/epoch_001.ckpt"
cache_dir="./cache_dir"
GPT_Zero_Shot_QA="eval/GPT_Zero_Shot_QA"
video_dir="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/videos/all"
gt_file_question="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/test_q.json"
gt_file_answers="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/test_a.json"
output_dir="${GPT_Zero_Shot_QA}/MSRVTT_Zero_Shot_QA/${CKPT_NAME}"
nframe=4
lora=0

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

'''
for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m eval.inference \
      --model_base ${model_base} \
      --sampler_base ${sampler_base} \
      --model_path ${model_path} \
      --cache_dir ${cache_dir} \
      --video_dir ${video_dir} \
      --gt_file_question ${gt_file_question} \
      --gt_file_answers ${gt_file_answers} \
      --output_dir ${output_dir} \
      --output_name ${CHUNKS}_${IDX} \
      --nframe ${nframe} \
      --lora ${lora} \
      --num_chunks $CHUNKS \
      --chunk_idx $IDX &
done

wait

output_file=${output_dir}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "$output_file"
done
'''

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

