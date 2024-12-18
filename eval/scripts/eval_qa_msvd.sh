

GPT_Zero_Shot_QA="/root/VideoTGB/eval/GPT_Zero_Shot_QA"
output_name="LSTP-Blip2-flant5-xl-iv"
# output_name="LSTP-Instructblip-Vicuna-lora"
# output_name="VideoLLaVA"
pred_path="/root/VideoTGB/outputs/merge.jsonl"
output_dir="/root/VideoTGB/outputs"
output_json="/root/VideoTGB/outputs/results.json"
api_key="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"
api_base="https://api.openai.com/v1"
num_tasks=1



python -m eval.evaluate \
    --pred_path ${pred_path} \
    --output_dir ${output_dir} \
    --output_json ${output_json} \
    --api_key ${api_key} \
    --api_base ${api_base} \
    --num_tasks ${num_tasks}