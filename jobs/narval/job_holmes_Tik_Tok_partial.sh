#!/bin/bash
#SBATCH --job-name=holmes_tik_tok   # Job name
#SBATCH --gres=gpu:1                   # Request GPU resource
#SBATCH --nodes=1                      # Run all processes on a single node
#SBATCH --ntasks=1                     # Run a single task        
#SBATCH --cpus-per-task=12              # Number of CPU cores per task
#SBATCH --mem=128G                      # Job memory request
#SBATCH --time=2:00:00                # Time limit hrs:min:sec
#SBATCH --output=/home/kka151/projects/def-t55wang/kka151/Website-Fingerprinting-Library/jobs/narval/logs/holmes_tik_tok_%j.log  # Standard output and error log


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


python3 -m exp.dataset_process.gen_early_traffic  --dataset Tik_Tok -cc True 
for percent in {20..100..1}
do
    python3 -m exp.dataset_process.gen_taf \
        --dataset ${dataset} \
        --seq_len 10000 \
        --in_file test_p${percent} \
        -cc True

    python3 -m exp.test \
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