# Offline data augmentation method of Holmes.
# Details can be found in https://arxiv.org/pdf/2407.00918
import os
import argparse
import numpy as np
from tqdm import tqdm
import random
from WFlib.tools import parser_utils
import gc
from typing import Dict, Any

def predict_npz_file_size(data_dict: Dict[str, Any], verbose = True) -> Dict[str, int]:
    """
    Estimate the size of an NPZ file before saving it.
    
    Args:
        data_dict: Dictionary containing arrays to be saved
        
    Returns:
        Dictionary with size estimates in bytes
    """
    total_raw_size = 0
    array_sizes = {}
    
    for key, value in data_dict.items():
        # Convert to numpy array if it isn't already
        if not isinstance(value, np.ndarray):
            value = np.array(value)
            
        # Calculate size of this array
        array_size = value.nbytes
        array_sizes[key] = array_size
        total_raw_size += array_size
        
    # NPZ files use ZIP compression
    # Estimate compressed size (very rough estimate - assumes 50% compression)
    estimated_compressed = total_raw_size // 2
    
    # Add overhead for ZIP format and NumPy metadata
    zip_overhead = 1024  # Rough estimate for ZIP headers and metadata
    overhead_per_array = 256  # Rough estimate for NumPy array metadata
    total_overhead = zip_overhead + (overhead_per_array * len(data_dict))
    
    size_info = {
        'total_raw_bytes': total_raw_size,
        'estimated_compressed_bytes': estimated_compressed + total_overhead,
        'overhead_bytes': total_overhead,
        'array_sizes': array_sizes
    }

    if verbose:
        def bytes_to_human_readable(bytes_size: int) -> str:
            for unit in ['B', 'KB', 'MB', 'GB']:
                if bytes_size < 1024:
                    return f"{bytes_size:.2f} {unit}"
                bytes_size /= 1024
            return f"{bytes_size:.2f} TB"
        
        print("\nSize Analysis:")
        print("--------------")
        print(f"Total raw size: {bytes_to_human_readable(size_info['total_raw_bytes'])}")
        print(f"Estimated compressed size: {bytes_to_human_readable(size_info['estimated_compressed_bytes'])}")
        print(f"Overhead: {bytes_to_human_readable(size_info['overhead_bytes'])}")
        
        print("\nSize per array:")
        for array_name, size in size_info['array_sizes'].items():
            print(f"{array_name}: {bytes_to_human_readable(size)}")
    
    return size_info
def gen_augment(data, num_aug, effective_ranges, out_file):
    """
    Generate augmented data based on the provided dataset and save it to a file.
    
    Parameters:
    data (dict): Dictionary containing 'X' (features) and 'y' (labels) from the dataset.
    num_aug (int): Number of augmentations to generate per original sample.
    effective_ranges (dict): Dictionary specifying the effective ranges for each class.
    out_file (str): Path to the output file to save the augmented data.
    """
    X = data["X"]
    y = data["y"]

    new_X = []
    new_y = []
    abs_X = np.absolute(X)
    feat_length = X.shape[1]

    # Loop through each sample in the dataset
    for index in tqdm(range(abs_X.shape[0])):
        cur_abs_X = abs_X[index]
        cur_web = y[index]
        loading_time = cur_abs_X.max()

        # Generate augmentations for each sample
        for ii in range(num_aug):
            p = np.random.randint(effective_ranges[cur_web][0], effective_ranges[cur_web][1])
            threshold = loading_time * p / 100
            valid_X = cur_abs_X[cur_abs_X > 0]
            valid_X = valid_X[valid_X <= threshold]
            valid_length = valid_X.shape[0]
            new_X.append(np.pad(X[index][:valid_length], (0, feat_length - valid_length), "constant", constant_values=(0, 0)))
            new_y.append(cur_web)

        # Add the original sample
        new_X.append(X[index])
        new_y.append(cur_web)

    new_X = np.array(new_X)
    new_y = np.array(new_y)

    # The saving takes a lot of memmory
    # I will try to delete some temp variables
    # Save the augmented data to the specified output file
    del abs_X
    del X
    del y
    gc.collect()
    data_dict = {
    'X': new_X,
    'y': new_y
    }

    # Predict file size
    predicted_size = predict_npz_file_size(data_dict, verbose=True)
    del data_dict
    gc.collect()
    
    np.savez(out_file, X=new_X, y=new_y)
    print(f"Generate {out_file} done.")

# Set a fixed seed for reproducibility
fix_seed = 2024
random.seed(fix_seed)
np.random.seed(fix_seed)

# Argument parser for command-line options, arguments, and sub-commands
parser = argparse.ArgumentParser(description="WFlib")

# Define command-line arguments
parser.add_argument("--dataset", type=str, required=True, default="Undefended", help="Dataset name")
parser.add_argument("--model", type=str, required=True, default="DF", help="Model name")
parser.add_argument("--in_file", type=str, default="train", help="Input file name")
parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="Directory to save model checkpoints")
parser.add_argument("--attr_method", type=str, default="DeepLiftShap", 
                    help="Feature attribution method, options=[DeepLiftShap, GradientShap]")
parser.add_argument('-cc', '--compute_canada', type=parser_utils.str2bool, nargs='?', const=True, default=False,
                         help='Whether we are using compute canada')

# Parse arguments
args = parser.parse_args()
dataset_path = "./datasets"
if args.compute_canada:
    dataset_path = '/home/kka151/scratch/holmes/datasets'

print(f'starting data augmentation script for {args.in_file}')
# Construct the input path for the dataset
in_path = os.path.join(dataset_path, args.dataset)
data = np.load(os.path.join(in_path, f"{args.in_file}.npz"))

# Load the temporal attribution data
temporal_data = np.load(os.path.join(args.checkpoints, args.dataset, args.model, f"attr_{args.attr_method}.npz"))["attr_values"]

# Calculate effective ranges for each class based on the temporal attribution data
effective_ranges = {}
for web in range(temporal_data.shape[0]):
    cur_temporal = np.cumsum(temporal_data[web])
    cur_temporal /= cur_temporal.max()
    cur_lower = np.searchsorted(cur_temporal, 0.3, side="right") * 100 // temporal_data.shape[1]
    cur_upper = np.searchsorted(cur_temporal, 0.6, side="right") * 100 // temporal_data.shape[1]
    effective_ranges[web] = (cur_lower, cur_upper)

# Construct the output file path for the augmented data
out_file = os.path.join(in_path, f"aug_{args.in_file}.npz")

# Check if the output file already exists
if not os.path.exists(out_file):
    # Generate augmented data and save it to the output file
    gen_augment(data, 2, effective_ranges, out_file)
else:
    # If the output file already exists, print a message indicating it has been generated
    print(f"{out_file} has been generated.")

print(f'finished data augmentation script for {args.in_file}')