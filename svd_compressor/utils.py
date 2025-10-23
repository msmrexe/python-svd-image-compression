# svd_compressor/utils.py

"""
Utility functions for image source resolution and performance metrics.
"""

import numpy as np
import urllib.request
import io
import os
from math import log10, sqrt
from skimage.metrics import peak_signal_noise_ratio as psnr

def resolve_image_source(address: str) -> tuple[str | io.BytesIO, str]:
    """
    Determines if the address is a URL or local path.
    Returns the source (path/BytesIO) and a base name for output.
    """
    if address.startswith("http://") or address.startswith("https://"):
        try:
            with urllib.request.urlopen(address) as url:
                image_data = url.read()
            image_bytes = io.BytesIO(image_data)
            # Extract filename from URL, handle cases without extension
            base_name = os.path.basename(urllib.parse.urlparse(address).path)
            name, _ = os.path.splitext(base_name)
            if not name: name = "image_from_url"
            return image_bytes, name
        except Exception as e:
            print(f"Error fetching image from URL: {e}")
            return None, None
    else:
        # Local file path
        if not os.path.exists(address):
            print(f"Error: Local file not found: {address}")
            return None, None
        name, _ = os.path.splitext(os.path.basename(address))
        return address, name

def calculate_compression_ratio(original_shape: tuple, k: int, svd_components: dict) -> float:
    """
    Calculates the compression ratio based on storing SVD components.
    Approximation: Size = (n*k + k + m*k) * num_channels * dtype_size
    """
    if not svd_components: return 0.0

    n, m = original_shape[0], original_shape[1]
    num_channels = len(svd_components) # 1 for gray, 3 for RGB

    # Calculate original size (assuming float64 for simplicity in calculation)
    original_pixels = n * m * num_channels
    original_size_bytes = original_pixels * 8 # float64 is 8 bytes

    # Calculate compressed size (storing U_k, s_k, Vh_k)
    # U is n x k, s is k, Vh is k x m
    compressed_elements = (n * k + k + m * k) * num_channels
    compressed_size_bytes = compressed_elements * 8 # float64

    if original_size_bytes == 0: return 0.0

    ratio = compressed_size_bytes / original_size_bytes
    return ratio

def calculate_psnr(original_matrix: np.ndarray, compressed_matrix: np.ndarray) -> float:
    """Calculates the Peak Signal-to-Noise Ratio (PSNR) in dB."""
    if original_matrix is None or compressed_matrix is None:
        return 0.0
    # PSNR works on the [0, 255] range
    original_uint8 = (original_matrix * 255).astype(np.uint8)
    compressed_uint8 = (compressed_matrix * 255).astype(np.uint8)

    try:
        # Use scikit-image's PSNR for correct calculation
        # data_range=255 assumes uint8 images
        return psnr(original_uint8, compressed_uint8, data_range=255)
    except Exception as e:
        print(f"Error calculating PSNR: {e}")
        return 0.0
