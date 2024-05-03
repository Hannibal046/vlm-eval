dataset=text-vqa-full
model_family=deepseek_vl
model_id=deepseek-vl-7b-chat
model_dir=deepseek-ai/deepseek-vl-7b-chat

accelerate launch --num-processes 8 scripts/evaluate.py \
    --model_family ${model_family} \
    --model_id ${model_id} \
    --model_dir ${model_dir} \
    --dataset.type ${dataset}

python scripts/score.py \
    --model_id ${model_id} \
    --dataset.type ${dataset}