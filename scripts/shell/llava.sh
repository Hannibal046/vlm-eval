dataset=text-vqa-full
model_family=llava-v15
model_id=llava-hf-v1.5-7b
model_dir=llava-hf/llava-1.5-7b-hf

accelerate launch --num-processes 8 scripts/evaluate.py \
    --model_family ${model_family} \
    --model_id ${model_id} \
    --model_dir ${model_dir} \
    --dataset.type ${dataset}

python scripts/score.py \
    --model_id ${model_id} \
    --dataset.type ${dataset}