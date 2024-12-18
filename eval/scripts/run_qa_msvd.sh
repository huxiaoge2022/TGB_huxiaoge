
# 设置基础变量
CKPT_NAME="LSTP-Blip2-flant5-xl-iv"
model_base="/root/VideoTGB/ckpts/blip2-flan-t5-xl"
sampler_base="/root/VideoTGB/ckpts/bert-base-uncased"
model_path="/root/VideoTGB/LSTP-Chat/LSTP-FlanT5xl.ckpt"

cache_dir="./cache_dir"
GPT_Zero_Shot_QA="/root/VideoTGB/eval/GPT_Zero_Shot_QA"
video_dir="/root/VideoTGB/eval/GPT_Zero_Shot_QA/MSVD_Zero_Shot_QA/videos"
gt_file_question="/root/VideoTGB/eval/GPT_Zero_Shot_QA/MSVD_Zero_Shot_QA/test_q.json"
gt_file_answers="/root/VideoTGB/eval/GPT_Zero_Shot_QA/MSVD_Zero_Shot_QA/test_a.json"
output_dir="/root/VideoTGB/outputs"
nframe=4
lora=1

# 单 GPU 推理
echo "Starting inference..."
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
echo "Inference finished."

# 合并结果（单 GPU 下只有一个输出文件）
output_file=${output_dir}/merge.jsonl

# 清空输出文件
> "$output_file"

# 将单个输出文件合并
cat ${output_dir}/single_chunk.json >> "$output_file"

