import numpy as np
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