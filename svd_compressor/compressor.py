# svd_compressor/compressor.py

"""
Handles image loading, SVD compression/reconstruction, and saving.
Supports both grayscale and color images.
"""

import numpy as np
from PIL import Image
from .svd import power_iteration_svd

def load_image(image_source) -> np.ndarray | None:
    """Loads an image from a path or URL into a NumPy array."""
    try:
        img = Image.open(image_source)
        # Ensure image is in RGB for consistent channel handling
        img = img.convert("RGB")
        return np.array(img, dtype=float) / 255.0 # Normalize to [0, 1]
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_source}'")
        return None
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def reconstruct_image(U, s, Vh, k: int) -> np.ndarray:
    """
    Reconstructs the image matrix from truncated SVD components.
    """
    k = min(k, len(s)) # Ensure k is not out of bounds
    # S = np.diag(s[:k]) # sigma_truncated
    S = np.zeros((k, k))
    np.fill_diagonal(S, s[:k])
    
    U_k = U[:, :k]   # U_truncated
    Vh_k = Vh[:k, :] # V_truncated

    # Reconstruct: A_k = U_k @ S @ Vh_k
    reconstructed_matrix = U_k @ S @ Vh_k
    return reconstructed_matrix

def compress_channel(channel_matrix: np.ndarray, k: int, method: str = 'numpy') -> tuple:
    """
    Applies SVD to a single color channel.
    
    Returns: U, s, Vh, reconstructed_channel
    """
    if method == 'custom':
        U, s, Vh = power_iteration_svd(channel_matrix, k=k)
    elif method == 'numpy':
        U, s, Vh = np.linalg.svd(channel_matrix, full_matrices=False)
    else:
        raise ValueError("Invalid SVD method specified.")
        
    reconstructed = reconstruct_image(U, s, Vh, k)
    return U, s, Vh, reconstructed

def compress_image(img_matrix: np.ndarray, k: int, method: str = 'numpy') -> tuple[np.ndarray | None, dict | None]:
    """
    Compresses an image using SVD. Handles grayscale and color.

    Args:
        img_matrix: NumPy array of the image (normalized to [0, 1]).
        k: The number of singular values to keep.
        method: 'numpy' or 'custom'.

    Returns:
        A tuple (compressed_img_matrix, svd_components):
        - compressed_img_matrix: The reconstructed image matrix.
        - svd_components: Dict containing U, s, Vh for each channel.
                         Needed for calculating compression ratio. Returns None on error.
    """
    if img_matrix is None:
        return None, None
        
    original_shape = img_matrix.shape
    
    if len(original_shape) == 2: # Grayscale
        print(f"Compressing grayscale image with k={k} using {method} SVD...")
        try:
            U, s, Vh, reconstructed = compress_channel(img_matrix, k, method)
            # Clip values to [0, 1] range
            reconstructed = np.clip(reconstructed, 0, 1)
            svd_components = {'gray': (U, s, Vh)}
            return reconstructed, svd_components
        except Exception as e:
            print(f"Error during SVD compression: {e}")
            return None, None
            
    elif len(original_shape) == 3 and original_shape[2] == 3: # Color (RGB)
        print(f"Compressing color image with k={k} using {method} SVD...")
        reconstructed_channels = []
        svd_components = {}
        channel_names = ['R', 'G', 'B']
        
        try:
            for i in range(3): # Iterate through R, G, B channels
                print(f"  Processing {channel_names[i]} channel...")
                channel = img_matrix[:, :, i]
                U, s, Vh, reconstructed_ch = compress_channel(channel, k, method)
                reconstructed_channels.append(reconstructed_ch)
                svd_components[channel_names[i]] = (U, s, Vh)

            # Stack channels back together
            reconstructed_img = np.stack(reconstructed_channels, axis=2)
            # Clip values to [0, 1] range
            reconstructed_img = np.clip(reconstructed_img, 0, 1)
            return reconstructed_img, svd_components
        except Exception as e:
            print(f"Error during SVD compression: {e}")
            return None, None
    else:
        print(f"Error: Unsupported image shape {original_shape}")
        return None, None


def save_image(img_matrix: np.ndarray, output_path: str):
    """Saves a NumPy array (range [0, 1]) as an image file."""
    if img_matrix is None:
        print("Error: Cannot save None matrix.")
        return
    try:
        # Scale back to [0, 255] and convert to uint8
        img_array_uint8 = (img_matrix * 255).astype(np.uint8)
        img = Image.fromarray(img_array_uint8)
        img.save(output_path)
        print(f"Compressed image saved to: {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")
