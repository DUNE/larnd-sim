#!/usr/bin/env python

import os

# Using ASCII escape codes for simplicity
def color_red(text):
    return f"\033[91m{text}\033[0m"
def color_green(text):
    return f"\033[92m{text}\033[0m"
def color_blue(text):
    return f"\033[94m{text}\033[0m"

import sys
import time
import argparse
import h5py
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

def compare_arrays(ref_arr, sim_arr, name, strict=False):
    """
    Helper to compare two numpy arrays.
    """
    if ref_arr.shape != sim_arr.shape:
        return False, color_red(f"Mismatching {name}: Shape mismatch!\n  Reference shape: {ref_arr.shape}\n  Simulation shape: {sim_arr.shape}")

    # Fast-path
    if strict:
        match = np.array_equal(ref_arr, sim_arr, equal_nan=True)
    elif np.issubdtype(ref_arr.dtype, np.number):
        match = np.allclose(ref_arr, sim_arr, equal_nan=True)
    else:
        match = np.array_equal(ref_arr, sim_arr)

    if match:
        return True, f"{name} match."

    # Slow-path
    if strict or not np.issubdtype(ref_arr.dtype, np.number):
        mask = (ref_arr != sim_arr)
    else:
        mask = ~np.isclose(ref_arr, sim_arr, equal_nan=True)

    indices = np.nonzero(mask)
    # Convert to lists of python ints to avoid NumPy dtype in output
    indices = [arr.tolist() for arr in indices]
    num_mismatches = int(np.sum(mask))

    res_msg = color_red(f"Mismatching {name}: Value mismatch!\n  Number of mismatched elements: {num_mismatches}")
    max_indices = 50 # Save time by not attempting to print everyting
    if num_mismatches <= max_indices:
        coords = list(zip(*indices))
        res_msg += f"\n  Index locations of mismatch: {coords}"
    else:
        sliced_indices = tuple(arr[:max_indices] for arr in indices)
        coords = list(zip(*sliced_indices))
        res_msg += f"\n  Index locations (first {max_indices}): {coords}"

    return False, res_msg

def compare_dataset_logic(ref_ds, sim_ds, name, strict=False):
    """
    Implements the logic for comparing a single dataset.
    """
    if ref_ds.dtype.names is not None:
        results = []
        all_match = True
        for field in ref_ds.dtype.names:
            field_name = f"{name}/{field}"
            match, msg = compare_arrays(ref_ds[field], sim_ds[field], field_name, strict)
            results.append(msg)
            if not match:
                all_match = False
        return all_match, "\n".join(results)
    else:
        return compare_arrays(ref_ds[:], sim_ds[:], name, strict)

def compare_dataset_worker(ref_file_path, sim_file_path, dataset_path, strict=False):
    """
    Worker function that opens files and compares a specific dataset.
    """
    try:
        with h5py.File(ref_file_path, 'r') as ref_file, h5py.File(sim_file_path, 'r') as sim_file:
            if dataset_path not in ref_file:
                return False, f"Error: Dataset {dataset_path} not found in reference file."
            if dataset_path not in sim_file:
                return False, f"Error: Dataset {dataset_path} not found in simulation file."

            ref_ds = ref_file[dataset_path]
            sim_ds = sim_file[dataset_path]

            return compare_dataset_logic(ref_ds, sim_ds, dataset_path, strict)
    except Exception as e:
        return False, f"Exception comparing {dataset_path}: {e}"

def get_all_datasets(group, path=""):
    """
    Recursively find all dataset paths in an HDF5 file.
    """
    datasets = []
    for key in group.keys():
        full_path = f"{path}/{key}" if path else key
        obj = group[key]
        if isinstance(obj, h5py.Group):
            datasets.extend(get_all_datasets(obj, full_path))
        elif isinstance(obj, h5py.Dataset):
            datasets.append(full_path)
    return datasets

def main():
    parser = argparse.ArgumentParser(description="Parallel compare of two larnd-sim HDF5 output files.")
    parser.add_argument('--ref_file', required=True, type=str, help='Path to reference simulation file')
    parser.add_argument('--new_file', required=True, type=str, help='Path to new simulation file')
    parser.add_argument('--strict', action='store_true', help='Enable strict equality comparisons')
    parser.add_argument('--datasets', nargs='+', help='Specific datasets to compare; compares all by default')
    parser.add_argument('--exclude', nargs='+', help='Datasets to exclude from comparison')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (defaults to a quarter of avaiable cores)')

    args = parser.parse_args()

    if args.workers is None:
        cpu_count = os.cpu_count() or 1
        args.workers = max(1, cpu_count // 4)

    np.set_printoptions(precision=3)
    print("-----------------------------------------")
    print("Comparing larnd-sim simulation outputs...")
    print(f"Reference file : {args.ref_file}")
    print(f"Simulation file: {args.new_file}")
    print("-----------------------------------------")

    try:
        with h5py.File(args.ref_file, 'r') as ref_file:
            if args.datasets:
                targets = args.datasets
            else:
                targets = get_all_datasets(ref_file)

        if args.exclude:
            exclude_set = set(args.exclude)
            targets = [t for t in targets if t not in exclude_set]
            print(f"Excluded datasets: {', '.join(args.exclude)}")

        if args.datasets:
            print(f"Targeting specific datasets: {', '.join(args.datasets)}")
        print(f"Comparing {len(targets)} datasets using {args.workers} workers...")
        print("-----------------------------------------")

        t_start = time.time()
        all_passed = True

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_ds = {
                executor.submit(compare_dataset_worker, args.ref_file, args.new_file, ds, args.strict): ds
                for ds in targets
            }

            results = {}
            for future in as_completed(future_to_ds):
                ds_path = future_to_ds[future]
                results[ds_path] = future.result()

            current_group = None
            for ds in sorted(targets):
                success, msg = results[ds]

                # Group delimiter
                group = ds.split('/')[0] if '/' in ds else ds
                if group != current_group:
                    print(color_blue(f"\n{'='*20} Group: {group} {'='*20}"))
                    current_group = group

                print(msg)
                if not success:
                    all_passed = False
        t_end = time.time()
        print("-----------------------------------------")
        print(f"Elapsed time: {t_end - t_start:.3f} s")

        if all_passed:
            print("Comparison PASSED.")
            sys.exit(0)
        else:
            print("Comparison FAILED.")
            sys.exit(121)

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
