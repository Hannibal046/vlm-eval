
for dataset in pope-full ai2d-full nocaps-full vsr-full ocid-ref-full
do
    accelerate launch --num_processes 8  \
    scripts/evaluate.py \
    --model_id prism-dinosiglip+7b \
    --dataset.type ${dataset}

    python scripts/score.py \
        --model_id prism-dinosiglip+7b \
        --dataset.type ${dataset}

    model_family=deepseek_vl
    model_id=deepseek-vl-7b-chat
    model_dir=deepseek-ai/deepseek-vl-7b-chat

    accelerate launch  --num_processes 8  \
        scripts/evaluate.py \
        --model_family ${model_family} \
        --model_id ${model_id} \
        --model_dir ${model_dir} \
        --dataset.type ${dataset}

    python scripts/score.py \
        --model_id ${model_id} \
        --dataset.type ${dataset} 

    model_family=llava-v15
    model_id=llava-v1.5-7b
    model_dir=liuhaotian/llava-v1.5-7b

    accelerate launch  --num_processes 8  \
        scripts/evaluate.py \
        --model_family ${model_family} \
        --model_id ${model_id} \
        --model_dir ${model_dir} \
        --dataset.type ${dataset}

    python scripts/score.py \
        --model_id ${model_id} \
        --dataset.type ${dataset}
done

