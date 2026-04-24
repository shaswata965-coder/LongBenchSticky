import os
import sys
import glob
import argparse

# Ensure we can import from the Results directory
repo_root = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(repo_root, "Results")
if results_dir not in sys.path:
    sys.path.append(results_dir)

from npz_io import load_results_npz, save_results_npz

def process_directory(directory_path, target_length=None):
    npz_files = glob.glob(os.path.join(directory_path, "*.npz"))
    if not npz_files:
        print(f"No .npz files found in {directory_path}")
        return

    for filepath in npz_files:
        print(f"\nProcessing {filepath}...")
        try:
            results = load_results_npz(filepath)
            if not results:
                print("  File is empty, skipping.")
                continue
            
            # Extract generation lengths of all samples
            lengths = [len(r["metadata"]["generated_token_ids"]) for r in results]
            
            # Determine target length if not explicitly provided
            current_target = target_length if target_length is not None else max(lengths)
            print(f"  Target generation length for this file: {current_target}")

            filtered_results = []
            for i, r in enumerate(results):
                gen_len = len(r["metadata"]["generated_token_ids"])
                if gen_len >= current_target:
                    filtered_results.append(r)
                else:
                    print(f"  Removing sample {i} (length {gen_len} < target {current_target})")
            
            if len(filtered_results) < len(results):
                print(f"  Saving updated file with {len(filtered_results)} samples (removed {len(results) - len(filtered_results)})")
                # Overwrite the original npz file
                save_results_npz(filtered_results, filepath)
            else:
                print("  All samples meet the target length. No changes made.")

        except Exception as e:
            print(f"  Failed to process {filepath}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and remove samples that stopped early from .npz files.")
    parser.add_argument("directory", help="Path to the directory containing .npz files (e.g. p0_r5_o4_prefill100).")
    parser.add_argument("--expected-len", type=int, default=None, help="Optional: Hardcode expected generation length (e.g., 200). If omitted, the max length in each file is used.")
    args = parser.parse_args()

    # Ensure valid path
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory.")
        sys.exit(1)

    process_directory(args.directory, args.expected_len)
