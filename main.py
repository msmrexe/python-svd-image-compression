# main.py

"""
SVD Image Compression Tool - Command-Line Interface

Compresses an image using SVD, comparing a custom implementation
against NumPy's standard SVD.
"""

import argparse
import time
import os
import numpy as np
from svd_compressor.utils import resolve_image_source, calculate_compression_ratio, calculate_psnr
from svd_compressor.compressor import load_image, compress_image, save_image

def run_compression(image_source, k, method, output_dir):
    """ Loads, compresses, saves, and evaluates one method."""
    print(f"\n--- Running Compression: k={k}, Method='{method}' ---")

    # 1. Load Image
    img_matrix_original = load_image(image_source)
    if img_matrix_original is None:
        return None, None # Error handled in load_image

    # 2. Compress Image
    start_time = time.perf_counter()
    img_matrix_compressed, svd_components = compress_image(img_matrix_original, k, method)
    end_time = time.perf_counter()
    compression_time = end_time - start_time

    if img_matrix_compressed is None:
        return None, None # Error handled in compress_image

    # 3. Calculate Metrics
    comp_ratio = calculate_compression_ratio(img_matrix_original.shape[:2], k, svd_components)
    psnr_val = calculate_psnr(img_matrix_original, img_matrix_compressed)

    # 4. Save Image
    base_name = resolve_image_source(image_source)[1] # Get base name again for saving
    output_filename = f"{base_name}_k{k}_{method}.jpg"
    output_path = os.path.join(output_dir, output_filename)
    save_image(img_matrix_compressed, output_path)

    print(f"  Compression Time: {compression_time:.4f} seconds")
    print(f"  Compression Ratio (SVD components vs Original): {comp_ratio:.2%}")
    print(f"  PSNR: {psnr_val:.2f} dB")

    return compression_time, comp_ratio, psnr_val, output_path


def main():
    """ Parses CLI arguments and runs the compression process."""
    parser = argparse.ArgumentParser(
        description="Compress an image using SVD, comparing custom vs NumPy."
    )
    parser.add_argument(
        "image_source",
        help="Path or URL to the input image."
    )
    parser.add_argument(
        "-k",
        type=int,
        required=True,
        help="Number of singular values (rank) to keep for compression."
    )
    parser.add_argument(
        "--method",
        choices=['custom', 'numpy', 'both'],
        default='both',
        help="SVD implementation to use ('custom', 'numpy', or 'both'). Default: both"
    )
    parser.add_argument(
        "--output-dir",
        default="compressed_images",
        help="Directory to save compressed images (default: compressed_images)."
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve source once
    resolved_source, _ = resolve_image_source(args.image_source)
    if resolved_source is None:
        return # Error handled in resolve_image_source

    results = {}

    if args.method in ['custom', 'both']:
        stats = run_compression(resolved_source, args.k, 'custom', args.output_dir)
        if stats: results['custom'] = stats

    if args.method in ['numpy', 'both']:
         # Reload source if it's a BytesIO object (consumed by first run)
        if isinstance(resolved_source, io.BytesIO):
             resolved_source, _ = resolve_image_source(args.image_source)
        stats = run_compression(resolved_source, args.k, 'numpy', args.output_dir)
        if stats: results['numpy'] = stats

    print("\n--- Summary ---")
    if 'custom' in results:
        t, r, p, _ = results['custom']
        print(f"Custom SVD (k={args.k}): Time={t:.4f}s, Ratio={r:.2%}, PSNR={p:.2f}dB")
    if 'numpy' in results:
        t, r, p, _ = results['numpy']
        print(f"NumPy SVD  (k={args.k}): Time={t:.4f}s, Ratio={r:.2%}, PSNR={p:.2f}dB")

if __name__ == "__main__":
    main()
