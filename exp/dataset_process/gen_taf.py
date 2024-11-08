# Generates the traffic aggregation features. More details can be found in the paper: 
# Robust and Reliable Early-Stage Website Fingerprinting Attacks via Spatial-Temporal Distribution Analysis. CCS 2024.
import numpy as np
import os
import argparse
from typing import List
import time
import random
import torch
from tqdm import tqdm
from multiprocessing import Process
from WFlib.tools import data_processor, parser_utils
from .file_operations import predict_npz_file_size
import gc

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description='Feature extraction')
parser.add_argument("--dataset", type=str, required=True, default="Undefended", help="Dataset name")
parser.add_argument("--seq_len", type=int, default=5000, help="Input sequence length")
parser.add_argument("--in_file", type=str, default="train", help="input file")
parser.add_argument('-cc', '--compute_canada', type=parser_utils.str2bool, nargs='?', const=True, default=False,
                         help='Whether we are using compute canada')

# Parse arguments
args = parser.parse_args()
dataset_path = "./datasets"
if args.compute_canada:
    dataset_path = '/home/kka151/scratch/holmes/datasets'

in_path = os.path.join(dataset_path, args.dataset)
if not os.path.exists(in_path):
    raise FileNotFoundError(f"The dataset path does not exist: {in_path}")

# Define output file path
out_file = os.path.join(in_path, f"taf_{args.in_file}.npz")
print(f'starting gen taf script for {args.in_file}')
# If the output file does not exist, process the input file
if not os.path.exists(out_file):
    # Load dataset from the specified .npz file
    data = np.load(os.path.join(in_path, f"{args.in_file}.npz"))
    X = data["X"]
    y = data["y"]
    # Align the sequence length
    X_taf = data_processor.length_align(X, args.seq_len)
    del data
    del X
    gc.collect()
    # Extract the TAF
    X_taf = data_processor.extract_TAF(X_taf)
    # Print processing information
    print(f"{args.in_file} process done: X = {X_taf.shape}, y = {y.shape}")
    # Save the processed data into a new .npz file
    
    # again this file is huge and takes a lot of memmory
    # I will try to delete irrelevant stuff before it
    

    data_dict = {
    'X': X_taf,
    'y': y
    }
    # Predict file size
    predicted_size = predict_npz_file_size(data_dict, verbose=True)
    del data_dict
    gc.collect()

    
    np.savez(out_file, X = X_taf, y = y)
else:
    # Print a message if the output file already exists
    print(f"{out_file} has been generated.")

print(f'finished gen taf script for {args.in_file}')