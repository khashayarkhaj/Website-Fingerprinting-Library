#!/bin/bash
#SBATCH --job-name=holmes_tik_tok   # Job name
#SBATCH --gres=gpu:1                   # Request GPU resource
#SBATCH --nodes=1                      # Run all processes on a single node
#SBATCH --ntasks=1                     # Run a single task        
#SBATCH --cpus-per-task=8              # Number of CPU cores per task
#SBATCH --mem=32G                      # Job memory request
#SBATCH --time=30:00:00                # Time limit hrs:min:sec
#SBATCH --output=/home/kka151/projects/def-t55wang/kka151/WF-Representation-Learning/jobs/logs/early-detection/RF/holmes_tik_tok_%j.log  # Standard output and error log


# Load any modules and activate your conda environment here
module load StdEnv/2023
module load python/3.11.5
module load rust/1.76.0
module load gcc arrow/17.0.0
source /home/kka151/venvs/python_11_5/bin/activate


# Navigate to your project directory (optional)
cd /home/kka151/projects/def-t55wang/kka151/Website-Fingerprinting-Library



# Execute your deep learning script
dataset=Tik_Tok
attr_method=DeepLiftShap 

for filename in train valid
do 
    python3 -m exp.data_analysis.temporal_extractor \
      --dataset ${dataset} \
      --seq_len 10000 \
      --in_file ${filename} \
      -cc True
done

python3 -m exp.train
  --dataset ${dataset} \
  --model RF \
  --device cuda \
  --train_file temporal_train \
  --valid_file temporal_valid \
  --feature TAM \
  --seq_len 1000 \
  --train_epochs 30 \
  --batch_size 200 \
  --learning_rate 5e-4 \
  --optimizer Adam \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --save_name temporal \
  -cc True


python3 -m exp.data_analysis.feature_attr \
  --dataset ${dataset} \
  --model RF \
  --in_file temporal_valid \
  --device cpu \
  --feature TAM \
  --seq_len 1000 \
  --save_name temporal \
  --attr_method ${attr_method}


for filename in train valid
do 
    python3 -m exp.dataset_process.data_augmentation \
      --dataset ${dataset} \
      --model RF \
      --in_file ${filename} \
      --attr_method ${attr_method} \
      -cc True
done

for filename in aug_train aug_valid test
do 
    python3 -m exp.dataset_process.gen_taf \
      --dataset ${dataset} \
      --seq_len 10000 \
      --in_file ${filename} \
      -cc True
done

python3 -m exp.train \
  --dataset ${dataset} \
  --model Holmes \
  --device cuda \
  --train_file taf_aug_train \
  --valid_file taf_aug_valid \
  --feature TAF \
  --seq_len 2000 \
  --train_epochs 30 \
  --batch_size 256 \
  --learning_rate 5e-4 \
  --loss SupConLoss \
  --optimizer AdamW \
  --eval_metrics Accuracy Precision Recall F1-score \
  --save_metric F1-score \
  --save_name max_f1 \ 
  -cc True

python3 -m exp.data_analysis.spatial_analysis \
  --dataset ${dataset} \
  --model Holmes \
  --device cuda \
  --valid_file taf_aug_valid \
  --feature TAF \
  --seq_len 2000 \
  --batch_size 256 \
  --save_name max_f1 \
  -cc True

for percent in {20..100..10}
do
    python3 -m exp.dataset_process.gen_taf \
        --dataset ${dataset} \
        --seq_len 10000 \
        --in_file test_p${percent} \
        -cc True

    python3 -m exp.test
    --dataset ${dataset} \
    --model Holmes \
    --device cuda \
    --valid_file taf_aug_valid \
    --test_file taf_test_p${percent} \
    --feature TAF \
    --seq_len 2000 \
    --batch_size 256 \
    --eval_method Holmes \
    --eval_metrics Accuracy Precision Recall F1-score \
    --load_name max_f1 \
    --result_file test_p${percent} \
    -cc True
done