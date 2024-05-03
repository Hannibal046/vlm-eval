
for dataset in vqa-v2 gqa vizwiz text-vqa refcoco tally-qa pope
do
    python scripts/datasets/prepare.py --dataset_family ${dataset}
done

python scripts/datasets/prepare.py --dataset_family vsr --create_slim_dataset False

## ocid-ref expired link 