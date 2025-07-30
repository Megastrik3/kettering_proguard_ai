"""
This code verifies if a GPU is available to use when exporting to tensorflow model
format.
"""
import torch
# Check if GPU is available
def check_gpu():
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
        return True
    else:
        print("No GPU available.")
        return False