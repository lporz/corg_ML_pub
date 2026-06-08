#!/usr/bin/env python3
"""
Script to find the best run by extracting RMSE values from all corg_train.log files
in the comprehensive_* directory and plotting them. Change directory accordingly.
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_rmse_from_log(log_file_path):
    """
    Extract RMSE value from a corg_train.log file.
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        float or None: RMSE value if found, None otherwise
    """
    try:
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
        
        # Look for RMSE line in the last few lines (where metrics are reported)
        # The pattern is: [timestamp][__main__][INFO] - RMSE: value
        rmse_pattern = r'\[.*\]\[__main__\]\[INFO\] - RMSE: ([\d.]+)'
        
        # Search from the end of the file backwards
        for line in reversed(lines):
            match = re.search(rmse_pattern, line)
            if match:
                return float(match.group(1))
        
        return None
    except Exception as e:
        print(f"Error reading {log_file_path}: {e}")
        return None

def main():
    # Base directory containing all the run folders
    base_dir = Path("./multirun/comprehensive_2026-06-05_17-04-46")
    
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist!")
        return
    
    # Collect RMSE values and run numbers
    rmse_values = []
    run_numbers = []
    successful_runs = 0
    failed_runs = 0
    
    print("Scanning runs for RMSE values...")
    
    # Iterate through all subdirectories
    for subdir in sorted(base_dir.iterdir()):
        if subdir.is_dir() and subdir.name.startswith('run_'):
            # Extract run number
            try:
                run_num = int(subdir.name.split('_')[1])
            except (IndexError, ValueError):
                print(f"Could not parse run number from {subdir.name}")
                continue
            
            # Check for corg_train.log file
            log_file = subdir / "corg_train.log"
            if log_file.exists():
                rmse_value = extract_rmse_from_log(log_file)
                if rmse_value is not None:
                    rmse_values.append(rmse_value)
                    run_numbers.append(run_num)
                    successful_runs += 1
                    print(f"Run {run_num}: RMSE = {rmse_value:.4f}")
                else:
                    print(f"Run {run_num}: Could not extract RMSE value")
                    failed_runs += 1
            else:
                print(f"Run {run_num}: No corg_train.log file found")
                failed_runs += 1
    
    print(f"\nSummary:")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Total runs processed: {successful_runs + failed_runs}")
    
    if not rmse_values:
        print("No RMSE values found!")
        return
    
    # Find the best run (lowest RMSE)
    best_idx = np.argmin(rmse_values)
    best_run = run_numbers[best_idx]
    best_rmse = rmse_values[best_idx]
    
    print(f"\nBest run: {best_run} with RMSE = {best_rmse:.4f}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(run_numbers, rmse_values, 'bo-', markersize=4, linewidth=1, alpha=0.7)
    
    # Highlight the best run
    plt.plot(best_run, best_rmse, 'ro', markersize=8, label=f'Best run: {best_run} (RMSE={best_rmse:.4f})')
    
    plt.xlabel('Run Number')
    plt.ylabel('RMSE')
    plt.title('RMSE Values Across All Runs\n(Lower is Better)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add some statistics
    mean_rmse = np.mean(rmse_values)
    std_rmse = np.std(rmse_values)
    plt.axhline(y=mean_rmse, color='orange', linestyle='--', alpha=0.7, 
                label=f'Mean RMSE: {mean_rmse:.4f}')
    
    # Update legend to include mean
    plt.legend()
    
    # Add text box with statistics
    stats_text = f'Statistics:\nMean: {mean_rmse:.4f}\nStd: {std_rmse:.4f}\nMin: {best_rmse:.4f}\nMax: {max(rmse_values):.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_file = base_dir.parent / "rmse_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    
    # Show the plot
    plt.show()
    
    # Print top 5 best runs
    print(f"\nTop 5 best runs:")
    sorted_indices = np.argsort(rmse_values)
    for i in range(min(5, len(rmse_values))):
        idx = sorted_indices[i]
        print(f"{i+1}. Run {run_numbers[idx]}: RMSE = {rmse_values[idx]:.4f}")

if __name__ == "__main__":
    main()
