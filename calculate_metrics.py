#!/usr/bin/env python3
"""
UISC Capstone Project - Sewar Image Quality Assessment Utility

This script calculates similarity metrics (MSE, PSNR, SSIM) between processed 
images (located in 'data/processed data') and their corresponding original 
images (located in 'data/raw data').

Requirements:
    - numpy
    - pillow
    - sewar
"""

import os
import csv
import sys
import argparse
import numpy as np
from PIL import Image
import sewar.full_ref as full_ref


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate image similarity metrics (MSE, PSNR, SSIM) using Sewar."
    )
    parser.add_argument(
        "--raw-dir",
        default=os.path.join("data", "raw data"),
        help="Path to the directory containing raw images (default: data/raw data)"
    )
    parser.add_argument(
        "--processed-dir",
        default=os.path.join("data", "processed data"),
        help="Path to the directory containing processed images (default: data/processed data)"
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join("data", "metrics_report.csv"),
        help="Path to output CSV report file (default: data/metrics_report.csv)"
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Skip image pairs that do not have matching resolutions instead of resizing processed images to match raw images."
    )
    return parser.parse_args()


def get_image_files(directory):
    """
    Returns a dictionary mapping base names (without extensions) to full file paths
    for all supported image files in the directory.
    """
    supported_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    image_map = {}
    if not os.path.exists(directory):
        return image_map

    for filename in os.listdir(directory):
        name, ext = os.path.splitext(filename)
        if ext.lower() in supported_exts:
            image_map[name.lower()] = os.path.join(directory, filename)
    return image_map


def match_files(raw_map, processed_map):
    """
    Matches processed images to raw images based on naming convention:
    [raw_name]_processed -> [raw_name]
    """
    matches = []
    for proc_name, proc_path in processed_map.items():
        # Look for '_processed' suffix
        if proc_name.endswith("_processed"):
            raw_base_name = proc_name[:-len("_processed")]
        else:
            # Fallback if user didn't use underscore but has "processed"
            if proc_name.endswith("processed"):
                raw_base_name = proc_name[:-len("processed")]
            else:
                raw_base_name = proc_name

        if raw_base_name in raw_map:
            matches.append((raw_map[raw_base_name], proc_path))
        else:
            print(f"Warning: No matching raw image found for processed file: '{os.path.basename(proc_path)}' (looked for prefix: '{raw_base_name}')")
    
    return matches


def process_image_pair(raw_path, proc_path, no_resize=False):
    """
    Loads raw and processed images, ensures they have matching dimensions,
    and computes MSE, PSNR, and SSIM.
    """
    try:
        # Load images
        raw_img = Image.open(raw_path)
        proc_img = Image.open(proc_path)
    except Exception as e:
        print(f"Error loading images {os.path.basename(raw_path)} or {os.path.basename(proc_path)}: {e}")
        return None

    # Handle transparency or invalid modes (e.g. RGBA -> RGB)
    if raw_img.mode in ("RGBA", "P"):
        raw_img = raw_img.convert("RGB")
    if proc_img.mode in ("RGBA", "P"):
        proc_img = proc_img.convert("RGB")

    # Match color channels / mode
    if raw_img.mode != proc_img.mode:
        print(f"Note: Color mode mismatch for {os.path.basename(raw_path)} ({raw_img.mode}) and {os.path.basename(proc_path)} ({proc_img.mode}). Converting processed image to {raw_img.mode}.")
        proc_img = proc_img.convert(raw_img.mode)

    # Check size match
    if raw_img.size != proc_img.size:
        if no_resize:
            print(f"Warning: Resolution mismatch between {os.path.basename(raw_path)} {raw_img.size} and {os.path.basename(proc_path)} {proc_img.size}. Skipping.")
            return None
        
        print(f"Warning: Resolution mismatch between {os.path.basename(raw_path)} {raw_img.size} and {os.path.basename(proc_path)} {proc_img.size}. Resizing processed image to match raw image.")
        try:
            # Pillow 10+ uses Resampling.LANCZOS
            resample_method = Image.Resampling.LANCZOS
        except AttributeError:
            resample_method = Image.LANCZOS
        
        proc_img = proc_img.resize(raw_img.size, resample_method)

    # Convert to NumPy arrays for Sewar
    raw_arr = np.array(raw_img)
    proc_arr = np.array(proc_img)

    # Compute metrics using Sewar
    try:
        mse_val = full_ref.mse(raw_arr, proc_arr)
        psnr_val = full_ref.psnr(raw_arr, proc_arr)
        # SSIM in sewar returns (ssim_score, cs_score)
        ssim_res = full_ref.ssim(raw_arr, proc_arr)
        ssim_val = ssim_res[0] if isinstance(ssim_res, tuple) else ssim_res
        
        return {
            "raw_filename": os.path.basename(raw_path),
            "proc_filename": os.path.basename(proc_path),
            "mse": float(mse_val),
            "psnr": float(psnr_val),
            "ssim": float(ssim_val)
        }
    except Exception as e:
        print(f"Error calculating metrics for pair {os.path.basename(raw_path)} and {os.path.basename(proc_path)}: {e}")
        return None


def main():
    args = parse_args()

    # Validate directories
    if not os.path.exists(args.raw_dir):
        print(f"Error: Raw images directory '{args.raw_dir}' does not exist.")
        sys.exit(1)
    if not os.path.exists(args.processed_dir):
        print(f"Error: Processed images directory '{args.processed_dir}' does not exist.")
        sys.exit(1)

    print("Scanning directories...")
    raw_map = get_image_files(args.raw_dir)
    processed_map = get_image_files(args.processed_dir)

    print(f"Found {len(raw_map)} files in raw data folder.")
    print(f"Found {len(processed_map)} files in processed data folder.")

    matches = match_files(raw_map, processed_map)
    print(f"Matched {len(matches)} pair(s) for processing.")

    if not matches:
        print("No matched image pairs found. Make sure names in 'processed data' end with '_processed'.")
        sys.exit(0)

    results = []
    print("\nCalculating metrics...")
    for raw_path, proc_path in matches:
        result = process_image_pair(raw_path, proc_path, args.no_resize)
        if result:
            results.append(result)
            print(f"Processed: {result['raw_filename']} <-> {result['proc_filename']}")

    if not results:
        print("No metrics computed (all pairs failed or were skipped due to resolution mismatch).")
        sys.exit(0)

    # Calculate averages
    avg_mse = sum(r["mse"] for r in results) / len(results)
    avg_psnr = sum(r["psnr"] for r in results) / len(results)
    avg_ssim = sum(r["ssim"] for r in results) / len(results)

    # Print nice ASCII-style report table
    print("\n" + "=" * 80)
    print(f"{'Image Pair Similarity Metrics Report':^80}")
    print("=" * 80)
    print(f"{'Raw Image':<25} | {'Processed Image':<25} | {'MSE':<8} | {'PSNR (dB)':<10} | {'SSIM':<6}")
    print("-" * 80)
    
    for r in results:
        # Truncate filename if it's too long
        raw_name = r["raw_filename"]
        if len(raw_name) > 23:
            raw_name = raw_name[:20] + "..."
        proc_name = r["proc_filename"]
        if len(proc_name) > 23:
            proc_name = proc_name[:20] + "..."
            
        print(f"{raw_name:<25} | {proc_name:<25} | {r['mse']:<8.2f} | {r['psnr']:<10.2f} | {r['ssim']:<6.4f}")
        
    print("-" * 80)
    print(f"{'AVERAGE':<25} | {'':<25} | {avg_mse:<8.2f} | {avg_psnr:<10.2f} | {avg_ssim:<6.4f}")
    print("=" * 80)

    # Write report to CSV
    try:
        os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Raw Image", "Processed Image", "MSE", "PSNR", "SSIM"])
            for r in results:
                writer.writerow([r["raw_filename"], r["proc_filename"], r["mse"], r["psnr"], r["ssim"]])
            writer.writerow([])
            writer.writerow(["AVERAGE", "", avg_mse, avg_psnr, avg_ssim])
        print(f"\nDetailed report saved to: {args.output_csv}")
    except Exception as e:
        print(f"Error saving CSV report: {e}")


if __name__ == "__main__":
    main()
