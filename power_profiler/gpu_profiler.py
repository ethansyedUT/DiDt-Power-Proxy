#!/usr/bin/env python3
import subprocess
import time
import argparse
import csv
import signal
import os
import sys
import datetime
from threading import Thread, Event
import pynvml

class GPUProfiler:
    def __init__(self, interval=0.001, output_file=None, gpu_id=0):
        self.interval = interval  # Sampling interval in seconds
        
        # Generate filename based on current time if not provided
        if output_file is None:
            current_time = datetime.datetime.now()
            output_file = f"gpu_profile_{current_time.strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Handle existing file by renaming it
        self.output_file = self._get_unique_filename(output_file)
        
        self.gpu_id = gpu_id
        self.running = False
        self.csv_file = None
        self.csv_writer = None
        self.start_time = None
        self.stop_event = Event()  # For safe thread termination
    
    def _get_unique_filename(self, filename):
        """Generate a unique filename if the proposed name already exists"""
        if not os.path.exists(filename):
            return filename
            
        # File exists, so we need a new name
        base_name, extension = os.path.splitext(filename)
        counter = 1
        new_filename = f"{base_name}_{counter}{extension}"
        
        # Keep incrementing counter until we find a unique name
        while os.path.exists(new_filename):
            counter += 1
            new_filename = f"{base_name}_{counter}{extension}"
            
        print(f"File {filename} already exists. Using {new_filename} instead.")
        return new_filename
        
    def initialize_nvml(self):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        device_name = pynvml.nvmlDeviceGetName(self.handle)
        print(f"Monitoring GPU: {device_name}")
        
    def collect_metrics(self):
        try:
            # Power usage in milliwatts
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert to watts
            
            # Temperature in Celsius
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Utilization rates (GPU and Memory)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util = util.gpu
            mem_util = util.memory
            
            # Memory info in MiB
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            mem_used = mem_info.used / (1024 * 1024)
            mem_total = mem_info.total / (1024 * 1024)
            
            # Clock speeds in MHz
            sm_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
            mem_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
            
            # Current timestamp
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            return {
                "timestamp": current_time,
                "elapsed": elapsed,
                "power_watts": power,
                "temp_c": temp,
                "gpu_util": gpu_util,
                "mem_util": mem_util,
                "mem_used_mib": mem_used,
                "mem_total_mib": mem_total,
                "sm_clock_mhz": sm_clock,
                "mem_clock_mhz": mem_clock
            }
        except pynvml.NVMLError as e:
            print(f"Error collecting metrics: {e}")
            return None
            
    def start_monitoring(self):
        self.initialize_nvml()
        self.running = True
        self.start_time = time.time()
        self.stop_event.clear()
        
        # Setup CSV file
        self.csv_file = open(self.output_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write header
        # Timestamp,ElapsedTime(s),Power(W),GPU_Util(%),Mem_Util(%),Temp(C),MemUsed(MiB),MemTotal(MiB),SM_Clock(MHz),Mem_Clock(MHz)
        self.csv_writer.writerow([  
            "Timestamp", "ElapsedTime(s)", "Power(W)", 
            "GPU_Util(%)", "Mem_Util(%)", "Temp(C)",
            "MemUsed(MiB)", "MemTotal(MiB)",
            "SM_Clock(MHz)", "Mem_Clock(MHz)"
        ])
        
        while not self.stop_event.is_set():
            try:
                metrics = self.collect_metrics()
                if metrics:
                    self.csv_writer.writerow([
                        datetime.datetime.fromtimestamp(metrics["timestamp"]).strftime('%H:%M:%S.%f')[:-3],
                        f"{metrics['elapsed']:.6f}",
                        f"{metrics['power_watts']:.2f}",
                        metrics["gpu_util"],
                        metrics["mem_util"],
                        metrics["temp_c"],
                        f"{metrics['mem_used_mib']:.2f}",
                        f"{metrics['mem_total_mib']:.2f}",
                        metrics["sm_clock_mhz"],
                        metrics["mem_clock_mhz"]
                    ])
                # Use a small wait with timeout to check for stop event
                self.stop_event.wait(self.interval)
            except Exception as e:
                if not self.stop_event.is_set():  # Only print if not stopping intentionally
                    print(f"Error in monitoring loop: {e}")
                break
        
        # Clean up resources
        if self.csv_file and not self.csv_file.closed:
            self.csv_file.close()
            print(f"Profile data saved to {self.output_file}")
        
        try:
            pynvml.nvmlShutdown()
        except:
            pass  # Ignore errors during shutdown
    
    def stop_monitoring(self):
        # Signal the monitoring thread to stop
        self.stop_event.set()
        # Give the thread a moment to complete its current iteration
        time.sleep(0.1)
        # No need to close file here, it's handled in the monitoring thread

def run_command(command, profiler):
    try:
        # Start the profiler in a separate thread
        profiler_thread = Thread(target=profiler.start_monitoring)
        profiler_thread.daemon = True
        profiler_thread.start()
        
        # Run the command
        print(f"Running command: {command}")
        start_time = time.time()
        process = subprocess.run(command, shell=True)
        end_time = time.time()
        
        # Stop the profiler and wait for it to finish
        profiler.stop_monitoring()
        profiler_thread.join(timeout=2.0)  # Wait up to 2 seconds for thread to complete
        
        print(f"Command completed with return code: {process.returncode}")
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error running command: {e}")
        profiler.stop_monitoring()
        
def signal_handler(sig, frame):
    print("Received interrupt signal. Stopping...")
    if 'profiler' in globals():
        profiler.stop_monitoring()
    sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPU Profiler for CUDA applications')
    parser.add_argument('--interval', type=float, default=0.001, help='Sampling interval in seconds (default: 0.001)')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file (default: auto-generated based on current time)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to monitor (default: 0)')
    parser.add_argument('command', type=str, help='Command or shell script to run and profile')
    
    args = parser.parse_args()
    
    # Setup signal handler for clean exit
    signal.signal(signal.SIGINT, signal_handler)
    
    profiler = GPUProfiler(interval=args.interval, output_file=args.output, gpu_id=args.gpu)
    run_command(args.command, profiler)