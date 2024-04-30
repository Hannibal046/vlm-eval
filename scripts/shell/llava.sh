dataset=text-vqa-slim
model_family=llava-v15
model_id=llava-v1.5-7b
model_dir=liuhaotian/llava-v1.5-7b

python scripts/evaluate.py \
    --model_family ${model_family} \
    --model_id ${model_id} \
    --model_dir ${model_dir} \
    --dataset.type ${dataset}

python scripts/score.py \
    --model_id ${model_id} \
    --dataset.type text-vqa-slim 