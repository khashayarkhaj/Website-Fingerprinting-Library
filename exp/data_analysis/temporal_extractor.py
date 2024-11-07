import numpy as np
import os
import argparse
from tqdm import tqdm
from WFlib.tools import data_processor, parser_utils

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description='Temporal feature extraction of Holmes')

# Define command-line arguments
parser.add_argument("--dataset", type=str, required=True, default="Undefended", help="Dataset name")
parser.add_argument("--seq_len", type=int, default=5000, help="Input sequence length")
parser.add_argument("--in_file", type=str, default="train", help="Input file name")
parser.add_argument('-cc', '--compute_canada', type=parser_utils.str2bool, nargs='?', const=True, default=False,
                         help='Whether we are using compute canada')

# Parse arguments
args = parser.parse_args()
dataset_path = "./datasets"
if args.compute_canada:
    dataset_path = '/home/kka151/scratch/holmes/datasets'

# Construct the input path for the dataset
in_path = os.path.join(dataset_path, args.dataset)
if not os.path.exists(in_path):
    raise FileNotFoundError(f"The dataset path does not exist: {in_path}")

# Construct the output file path
out_file = os.path.join(in_path, f"temporal_{args.in_file}.npz")

# Check if the output file already exists
if not os.path.exists(out_file):
    # Load data from the specified input file
    X, y = data_processor.load_data(os.path.join(in_path, f"{args.in_file}.npz"), "Origin", args.seq_len)
    
    # Extract temporal features from the input data
    temporal_X = data_processor.extract_temporal_feature(X)
    
    # Print the shape of the extracted temporal features
    print("Shape of temporal_X:", temporal_X.shape)
    
    # Save the extracted features and labels to the output file
    np.savez(out_file, X=temporal_X, y=y)
else:
    # If the output file already exists, print a message indicating it has been generated
    print(f"{out_file} has been generated.")
