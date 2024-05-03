dataset=text-vqa-slim
model_dir=/mnt/xincheng/prismatic-vlms/runs/llama3+stage-finetune+x7
accelerate launch --num_processes 8  \
    scripts/evaluate.py \
    --model_dir ${model_dir} \
    --dataset.type ${dataset}

python scripts/score.py \
    --model_id prism-clip+7b \
    --dataset.type ${dataset}
