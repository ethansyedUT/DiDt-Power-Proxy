import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from datetime import datetime
from scipy.optimize import curve_fit

def timestamp_to_ms(first_timestamp, current_timestamp):
    """Convert timestamps to elapsed milliseconds from start."""
    first_dt = datetime.strptime(first_timestamp, "%Y-%m-%d_%H-%M-%S.%f")
    current_dt = datetime.strptime(current_timestamp, "%Y-%m-%d_%H-%M-%S.%f")
    delta = current_dt - first_dt
    return delta.total_seconds() * 1000  # Convert to milliseconds

def fit_func(x, a, b, c):
    """Define the fitting function (polynomial of degree 2)."""
    return a * x**2 + b * x + c

def process_csv_files(directory, file_pattern, y_column, title, y_label):
    """Process all CSV files matching the pattern in the directory and create plots."""
    directory = Path(directory)
    
    # Find all matching CSV files
    csv_files = list(directory.glob(f"*_{file_pattern}.csv"))
    
    if not csv_files:
        print(f"No {file_pattern} files found in {directory}")
        return
    
    # Create Trace_Plots directory if it doesn't exist
    plots_dir = directory.parent / "Trace_Plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Process each file separately
    for file in csv_files:
        # Extract process ID from filename
        process_id = file.stem.split('_')[0]
        
        # Create individual plot
        plt.figure(figsize=(12, 6))    
        df = pd.read_csv(file)
        
        # Calculate correct memory usage if this is a memory usage file
        if 'MemoryUtilization_Percent' in df.columns and 'TotalMemory_MB' in df.columns:
            df['ActualMemoryUsed_MB'] = (df['MemoryUtilization_Percent'] / 100) * df['TotalMemory_MB']
            y_column = 'ActualMemoryUsed_MB'  # Override the y_column to use actual memory
        
        # Get first timestamp
        first_timestamp = df['Timestamp'].iloc[0]
        
        # Convert timestamps to elapsed milliseconds
        df['Elapsed_Time'] = df['Timestamp'].apply(
            lambda x: timestamp_to_ms(first_timestamp, x))
        
        # Sort by elapsed time to ensure sequential line is correct
        df = df.sort_values('Elapsed_Time')
        
        # Normalize time values for better fitting
        time_normalized = (df['Elapsed_Time'] - df['Elapsed_Time'].min()) / \
                         (df['Elapsed_Time'].max() - df['Elapsed_Time'].min())
        
        try:
            # Fit the curve
            popt, _ = curve_fit(fit_func, time_normalized, df[y_column], 
                              maxfev=5000)
            
            # Generate points for smooth curve
            x_smooth = np.linspace(0, 1, 1000)
            y_smooth = fit_func(x_smooth, *popt)
            
            # Convert x_smooth back to original time scale
            x_smooth_original = x_smooth * (df['Elapsed_Time'].max() - df['Elapsed_Time'].min()) + \
                              df['Elapsed_Time'].min()
            
            # Plot sequential line through actual data
            plt.plot(df['Elapsed_Time'], df[y_column], '-', 
                    label='Sequential Data', 
                    color='blue', alpha=0.5, linewidth=1)
            
            # Plot data points
            plt.plot(df['Elapsed_Time'], df[y_column], 'o', 
                    label='Data Points', 
                    color='blue', alpha=0.3, markersize=2)
            
            # Calculate and print R-squared value
            residuals = df[y_column] - fit_func(time_normalized, *popt)
            ss_res = np.sum(residuals** 2)
            ss_tot = np.sum((df[y_column] - np.mean(df[y_column]))** 2)
            r_squared = 1 - (ss_res / ss_tot)
            print(f"R-squared value for Process {process_id}: {r_squared:.4f}")
            
        except RuntimeError as e:
            print(f"Fitting failed for process {process_id}: {str(e)}")
            continue
        
        # Update title if using actual memory
        if y_column == 'ActualMemoryUsed_MB':
            title = 'GPU Memory Usage Over Time'
        
        # Customize plot
        plt.title(f"{title} - Process {process_id}")
        plt.xlabel('Elapsed Time (ms)')
        plt.ylabel(y_label)
        plt.grid(True, alpha=0.3)        
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        plt.tight_layout()
        plt.legend(loc='upper right')
        output_file = plots_dir / f"{process_id}_{file_pattern}.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Plot saved as {output_file}")

def main():
    base_dir = Path("logs")
    base_dir.mkdir(exist_ok=True)
    
    # Process memory usage data
    process_csv_files(
        f"{base_dir}/MemoryUsageTraces",
        "MemoryUsage",
        "MemoryUsed_MB",  # This will be overridden for memory usage files
        "GPU Memory Usage Over Time",
        "Memory Usage (MB)"
    )
    
    # Process power trace data
    process_csv_files(
        f"{base_dir}/PowerTraces",
        "PowerTrace",
        "PowerUsage_mW",
        "GPU Power Consumption Over Time",
        "Power Usage (mW)"
    )

if __name__ == "__main__":
    try:
        main()
        print("Analysis completed successfully!")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")