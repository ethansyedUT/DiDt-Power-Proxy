

# Power Profiling Usage Guide

This guide provides detailed instructions on how to set up and use power profilers to monitor GPU power consumption while running your applications. We offer two profiler implementations: a Python-based profiler and a C-based profiler, each with different characteristics for power measurement.

## Power Measurement Methods

- **Python Profiler (`gpu_profiler.py`)**: Reports power as a sliding window average, which is the default behavior of NVML.
- **C Profiler (`nvml_gpu_profiler.c`)**: Measures instantaneous power usage by utilizing `NVML_FI_DEV_POWER_INSTANT 186`, providing more precise moment-to-moment power readings.

Both profilers generate CSV files with the same structure, allowing you to use our `plot_gpu_metrics.py` visualization tool with either output.

---

## 1. Monitoring Power During Application Runs

### Using the Python Profiler

Execute the profiler:

```bash
./gpu_profiler.py --interval 0.0005 --output gemm_profile.csv "./your_script.sh"
```

**Parameters explained:**
- `--interval 0.0005`: Sampling interval in seconds (e.g., `0.0005` = 0.5ms)
- `--output gemm_profile.csv`: Optional output filename. If omitted, a timestamp-based filename will be generated automatically.
- `"./your_script.sh"`: Required parameter specifying the command to execute and monitor. This can be a shell script or a direct command.

**Note:** Ensure your script has execution permissions:  
```bash
chmod +x your_script.sh
```

---

### Using the C Profiler

#### Compile the profiler:
```bash
nvcc nvml_gpu_profiler.c -o nvml_gpu_profiler -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lnvidia-ml -lpthread -lrt -Wno-deprecated-gpu-targets
```

#### Execute the profiler:
```bash
./nvml_gpu_profiler -d 0 -i 0.5 -c "./build/run_10.sh"
```
Or with a specified output filename:
```bash
./nvml_gpu_profiler -d 0 -i 0.5 -o tf32_gemm_profile.csv -c "./run_gemm.sh"
```

**Parameters explained:**
- `-d 0`: Specifies which GPU device to monitor (`0` for the first GPU)
- `-i 0.5`: Sampling interval in milliseconds (`0.5ms`)
- `-o tf32_gemm_profile.csv`: Optional parameter to specify the output CSV filename
- `-c "./run_gemm.sh"`: Required parameter specifying the command to execute and monitor

---

## 2. Generating Visualization Graphs

After collecting the power profile data, you can generate visualizations using our plotting script:

```bash
python plot_gpu_metrics.py gpu_profile_20250308_024956.csv
```

The script will create three plots:
1. Complete metrics including power, temperature, GPU utilization, memory utilization, and clock speeds.
2. Metrics excluding clock speeds.
3. Power and utilization metrics only.

The generated plots will be saved both as a combined image and as individual plot files for easier incorporation into documentation or presentations.

---

## CSV Output Format

Both profilers generate CSV files with the following columns:

| Column Name       | Description                                   |
|-------------------|-----------------------------------------------|
| **Timestamp**     | System timestamp in microseconds              |
| **ElapsedTime(s)** | Time elapsed since the start of monitoring   |
| **Power(W)**      | Power consumption in watts                    |
| **GPU_Util(%)**   | GPU utilization percentage                    |
| **Mem_Util(%)**   | Memory utilization percentage                 |
| **Temp(C)**       | GPU temperature in Celsius                    |
| **MemUsed(MiB)**  | GPU memory used in MiB                        |
| **MemTotal(MiB)** | Total GPU memory in MiB                       |
| **SM_Clock(MHz)** | GPU SM clock speed in MHz                     |
| **Mem_Clock(MHz)** | GPU memory clock speed in MHz                |

This standardized format ensures compatibility with our visualization tools regardless of which profiler you choose to use.
