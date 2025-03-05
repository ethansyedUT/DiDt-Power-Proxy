import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set the style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams['font.size'] = 12

# Load the CSV data
def load_and_plot_data(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Create a figure with 3 subplots stacked vertically
    fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
    
    # ====================== Plot 1: All metrics ======================
    ax1 = axes[0]
    
    # Create twin axes for different scales
    ax1_power = ax1
    ax1_util = ax1.twinx()
    ax1_temp = ax1.twinx()
    ax1_clock = ax1.twinx()
    
    # Offset the right spines for better visibility
    ax1_temp.spines['right'].set_position(('outward', 60))
    ax1_clock.spines['right'].set_position(('outward', 120))
    
    # Plot the data with appropriate scales
    line_power = ax1_power.plot(df['ElapsedTime(s)'], df['Power(W)'], color='red', linewidth=2, label='Power (W)')
    line_util_gpu = ax1_util.plot(df['ElapsedTime(s)'], df['GPU_Util(%)'], color='blue', linewidth=2, label='GPU Util (%)')
    line_util_mem = ax1_util.plot(df['ElapsedTime(s)'], df['Mem_Util(%)'], color='green', linewidth=2, label='Mem Util (%)')
    line_temp = ax1_temp.plot(df['ElapsedTime(s)'], df['Temp(C)'], color='orange', linewidth=2, label='Temp (°C)')
    line_clock_sm = ax1_clock.plot(df['ElapsedTime(s)'], df['SM_Clock(MHz)'], color='purple', linewidth=2, label='SM Clock (MHz)')
    line_clock_mem = ax1_clock.plot(df['ElapsedTime(s)'], df['Mem_Clock(MHz)'], color='brown', linewidth=2, label='Mem Clock (MHz)')
    
    # Set y-axis limits
    ax1_power.set_ylim(0, 220)  # Max power 220W
    ax1_util.set_ylim(0, 110)   # Utilization 0-110% (to ensure 100% is visible)
    ax1_temp.set_ylim(20, 90)   # Temperature range
    ax1_clock.set_ylim(0, 2500) # Clock speeds in MHz
    
    # Set titles and labels
    ax1.set_title('GPU Performance Metrics (All)', fontsize=16)
    ax1_power.set_ylabel('Power (W)', color='red', fontsize=14)
    ax1_util.set_ylabel('Utilization (%)', color='blue', fontsize=14)
    ax1_temp.set_ylabel('Temperature (°C)', color='orange', fontsize=14)
    ax1_clock.set_ylabel('Clock Speed (MHz)', color='purple', fontsize=14)
    
    # Create a combined legend
    lines = line_power + line_util_gpu + line_util_mem + line_temp + line_clock_sm + line_clock_mem
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=True)
    
    # ====================== Plot 2: Excluding Clocks ======================
    ax2 = axes[1]
    
    # Create twin axes for different scales
    ax2_power = ax2
    ax2_util = ax2.twinx()
    ax2_temp = ax2.twinx()
    
    # Offset the right spine for better visibility
    ax2_temp.spines['right'].set_position(('outward', 60))
    
    # Plot the data
    line_power2 = ax2_power.plot(df['ElapsedTime(s)'], df['Power(W)'], color='red', linewidth=2, label='Power (W)')
    line_util_gpu2 = ax2_util.plot(df['ElapsedTime(s)'], df['GPU_Util(%)'], color='blue', linewidth=2, label='GPU Util (%)')
    line_util_mem2 = ax2_util.plot(df['ElapsedTime(s)'], df['Mem_Util(%)'], color='green', linewidth=2, label='Mem Util (%)')
    line_temp2 = ax2_temp.plot(df['ElapsedTime(s)'], df['Temp(C)'], color='orange', linewidth=2, label='Temp (°C)')
    
    # Set y-axis limits
    ax2_power.set_ylim(0, 220)  # Max power 220W
    ax2_util.set_ylim(0, 110)   # Utilization 0-110% (to ensure 100% is visible)
    ax2_temp.set_ylim(20, 90)   # Temperature range
    
    # Set titles and labels
    ax2.set_title('GPU Performance Metrics (Excluding Clocks)', fontsize=16)
    ax2_power.set_ylabel('Power (W)', color='red', fontsize=14)
    ax2_util.set_ylabel('Utilization (%)', color='blue', fontsize=14)
    ax2_temp.set_ylabel('Temperature (°C)', color='orange', fontsize=14)
    
    # Create a combined legend
    lines2 = line_power2 + line_util_gpu2 + line_util_mem2 + line_temp2
    labels2 = [line.get_label() for line in lines2]
    ax2.legend(lines2, labels2, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=True)
    
    # ====================== Plot 3: Power, GPU Util and Memory Util ======================
    ax3 = axes[2]
    
    # Create twin axes for different scales
    ax3_power = ax3
    ax3_util = ax3.twinx()
    
    # Plot the data
    line_power3 = ax3_power.plot(df['ElapsedTime(s)'], df['Power(W)'], color='red', linewidth=2.5, label='Power (W)')
    line_util_gpu3 = ax3_util.plot(df['ElapsedTime(s)'], df['GPU_Util(%)'], color='blue', linewidth=2.5, label='GPU Util (%)')
    line_util_mem3 = ax3_util.plot(df['ElapsedTime(s)'], df['Mem_Util(%)'], color='green', linewidth=2.5, label='Mem Util (%)')
    
    # Set y-axis limits
    ax3_power.set_ylim(0, 220)  # Max power 220W
    ax3_util.set_ylim(0, 110)   # Utilization 0-110% (to ensure 100% is visible)
    
    # Set titles and labels
    ax3.set_title('Power and Utilization', fontsize=16)
    ax3_power.set_ylabel('Power (W)', color='red', fontsize=14)
    ax3_util.set_ylabel('Utilization (%)', color='blue', fontsize=14)
    ax3.set_xlabel('Elapsed Time (s)', fontsize=14)
    
    # Create a combined legend
    lines3 = line_power3 + line_util_gpu3 + line_util_mem3
    labels3 = [line.get_label() for line in lines3]
    ax3.legend(lines3, labels3, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=True)
    
    # ====================== Global Settings ======================
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    # Save the figure
    output_file = csv_file.replace('.csv', '_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_file}")
    
    # Also save each plot separately for easier viewing
    for i, ax in enumerate(axes):
        fig_single = plt.figure(figsize=(14, 6))
        ax_new = fig_single.add_subplot(111)
        
        # Copy the contents from the subplot to the new figure
        if i == 0:
            plot_all_metrics(df, ax_new)
            title = "GPU Performance Metrics (All)"
        elif i == 1:
            plot_excluding_clocks(df, ax_new)
            title = "GPU Performance Metrics (Excluding Clocks)"
        else:
            plot_power_and_util(df, ax_new)
            title = "Power and Utilization"
        
        ax_new.set_title(title, fontsize=16)
        ax_new.set_xlabel('Elapsed Time (s)', fontsize=14)
        
        # Save the individual plot
        output_file_single = csv_file.replace('.csv', f'_plot{i+1}.png')
        fig_single.savefig(output_file_single, dpi=300, bbox_inches='tight')
        plt.close(fig_single)
        print(f"Individual plot saved as {output_file_single}")
    
    plt.close(fig)

# Helper functions for individual plots
def plot_all_metrics(df, ax):
    # Create twin axes
    ax_power = ax
    ax_util = ax.twinx()
    ax_temp = ax.twinx()
    ax_clock = ax.twinx()
    
    # Offset the right spines
    ax_temp.spines['right'].set_position(('outward', 60))
    ax_clock.spines['right'].set_position(('outward', 120))
    
    # Plot the data
    line_power = ax_power.plot(df['ElapsedTime(s)'], df['Power(W)'], color='red', linewidth=2, label='Power (W)')
    line_util_gpu = ax_util.plot(df['ElapsedTime(s)'], df['GPU_Util(%)'], color='blue', linewidth=2, label='GPU Util (%)')
    line_util_mem = ax_util.plot(df['ElapsedTime(s)'], df['Mem_Util(%)'], color='green', linewidth=2, label='Mem Util (%)')
    line_temp = ax_temp.plot(df['ElapsedTime(s)'], df['Temp(C)'], color='orange', linewidth=2, label='Temp (°C)')
    line_clock_sm = ax_clock.plot(df['ElapsedTime(s)'], df['SM_Clock(MHz)'], color='purple', linewidth=2, label='SM Clock (MHz)')
    line_clock_mem = ax_clock.plot(df['ElapsedTime(s)'], df['Mem_Clock(MHz)'], color='brown', linewidth=2, label='Mem Clock (MHz)')
    
    # Set limits
    ax_power.set_ylim(0, 220)
    ax_util.set_ylim(0, 110)  # Fixed to ensure 100% is visible
    ax_temp.set_ylim(20, 90)
    ax_clock.set_ylim(0, 2500)
    
    # Set labels
    ax_power.set_ylabel('Power (W)', color='red', fontsize=14)
    ax_util.set_ylabel('Utilization (%)', color='blue', fontsize=14)
    ax_temp.set_ylabel('Temperature (°C)', color='orange', fontsize=14)
    ax_clock.set_ylabel('Clock Speed (MHz)', color='purple', fontsize=14)
    
    # Create legend
    lines = line_power + line_util_gpu + line_util_mem + line_temp + line_clock_sm + line_clock_mem
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=True)

def plot_excluding_clocks(df, ax):
    # Create twin axes
    ax_power = ax
    ax_util = ax.twinx()
    ax_temp = ax.twinx()
    
    # Offset the right spine
    ax_temp.spines['right'].set_position(('outward', 60))
    
    # Plot the data
    line_power = ax_power.plot(df['ElapsedTime(s)'], df['Power(W)'], color='red', linewidth=2, label='Power (W)')
    line_util_gpu = ax_util.plot(df['ElapsedTime(s)'], df['GPU_Util(%)'], color='blue', linewidth=2, label='GPU Util (%)')
    line_util_mem = ax_util.plot(df['ElapsedTime(s)'], df['Mem_Util(%)'], color='green', linewidth=2, label='Mem Util (%)')
    line_temp = ax_temp.plot(df['ElapsedTime(s)'], df['Temp(C)'], color='orange', linewidth=2, label='Temp (°C)')
    
    # Set limits
    ax_power.set_ylim(0, 220)
    ax_util.set_ylim(0, 110)  # Fixed to ensure 100% is visible
    ax_temp.set_ylim(20, 90)
    
    # Set labels
    ax_power.set_ylabel('Power (W)', color='red', fontsize=14)
    ax_util.set_ylabel('Utilization (%)', color='blue', fontsize=14)
    ax_temp.set_ylabel('Temperature (°C)', color='orange', fontsize=14)
    
    # Create legend
    lines = line_power + line_util_gpu + line_util_mem + line_temp
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=True)

def plot_power_and_util(df, ax):
    # Create twin axes
    ax_power = ax
    ax_util = ax.twinx()
    
    # Plot the data
    line_power = ax_power.plot(df['ElapsedTime(s)'], df['Power(W)'], color='red', linewidth=2.5, label='Power (W)')
    line_util_gpu = ax_util.plot(df['ElapsedTime(s)'], df['GPU_Util(%)'], color='blue', linewidth=2.5, label='GPU Util (%)')
    line_util_mem = ax_util.plot(df['ElapsedTime(s)'], df['Mem_Util(%)'], color='green', linewidth=2.5, label='Mem Util (%)')
    
    # Set limits
    ax_power.set_ylim(0, 220)
    ax_util.set_ylim(0, 110)  # Fixed to ensure 100% is visible
    
    # Set labels
    ax_power.set_ylabel('Power (W)', color='red', fontsize=14)
    ax_util.set_ylabel('Utilization (%)', color='blue', fontsize=14)
    
    # Create legend
    lines = line_power + line_util_gpu + line_util_mem
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99), frameon=True)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = input("Enter the path to your CSV file: ")
    
    load_and_plot_data(csv_file)