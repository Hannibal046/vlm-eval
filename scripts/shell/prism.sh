accelerate launch --num_processes 8  \
    scripts/evaluate.py \
    --model_id prism-dinosiglip+7b \
    --dataset.type text-vqa-slim

python scripts/score.py \
    --model_id prism-dinosiglip+7b \
    --dataset.type text-vqa-slim
