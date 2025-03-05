#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>
#include <pthread.h>
#include <nvml.h>

#define CHECK_NVML(call) \
    do { \
        nvmlReturn_t result = call; \
        if (NVML_SUCCESS != result) { \
            fprintf(stderr, "NVML Error: %s\n", nvmlErrorString(result)); \
            exit(1); \
        } \
    } while (0)

typedef struct {
    nvmlDevice_t device;
    FILE* output_file;
    double interval_ms;
    volatile int should_stop;
} MonitorContext;

// Signal handler to stop monitoring gracefully
volatile int keep_running = 1;
void signal_handler(int sig) {
    printf("\nReceived signal %d, stopping...\n", sig);
    keep_running = 0;
}

// Get current timestamp in microseconds
unsigned long long get_timestamp_us() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (unsigned long long)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

// Monitoring thread function
void* monitor_gpu(void* arg) {
    MonitorContext* ctx = (MonitorContext*)arg;
    nvmlDevice_t device = ctx->device;
    FILE* output_file = ctx->output_file;
    double interval_ms = ctx->interval_ms;
    
    // Calculate sleep time in nanoseconds for more precise timing
    struct timespec sleep_time;
    sleep_time.tv_sec = (time_t)(interval_ms / 1000);
    sleep_time.tv_nsec = (long)((interval_ms - sleep_time.tv_sec * 1000) * 1000000);
    
    // Get device name
    char device_name[64];
    CHECK_NVML(nvmlDeviceGetName(device, device_name, sizeof(device_name)));
    
    printf("Monitoring GPU: %s\n", device_name);
    
    // Write CSV header
    fprintf(output_file, "Timestamp,ElapsedTime(s),Power(W),GPU_Util(%%),Mem_Util(%%),Temp(C),MemUsed(MiB),MemTotal(MiB),SM_Clock(MHz),Mem_Clock(MHz)\n");
    
    // Record start time for relative timestamps
    unsigned long long start_time = get_timestamp_us();
    unsigned long long last_time = start_time;
    
    while (keep_running && !ctx->should_stop) {
        unsigned long long current_time = get_timestamp_us();
        double elapsed_seconds = (current_time - start_time) / 1000000.0;
        
        // Power usage in milliwatts
        unsigned int power;
        CHECK_NVML(nvmlDeviceGetPowerUsage(device, &power));
        
        // GPU and Memory utilization
        nvmlUtilization_t utilization;
        CHECK_NVML(nvmlDeviceGetUtilizationRates(device, &utilization));
        
        // Temperature
        unsigned int temp;
        CHECK_NVML(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp));
        
        // Memory info
        nvmlMemory_t memory;
        CHECK_NVML(nvmlDeviceGetMemoryInfo(device, &memory));
        
        // Clock speeds
        unsigned int sm_clock, mem_clock;
        CHECK_NVML(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &sm_clock));
        CHECK_NVML(nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &mem_clock));
        
        // Write to CSV
        fprintf(output_file, "%llu,%.6f,%.3f,%u,%u,%u,%.2f,%.2f,%u,%u\n",
                current_time,
                elapsed_seconds,
                power / 1000.0,  // Convert to watts
                utilization.gpu,
                utilization.memory,
                temp,
                memory.used / (1024.0 * 1024.0),  // Convert to MiB
                memory.total / (1024.0 * 1024.0), // Convert to MiB
                sm_clock,
                mem_clock);
        
        // Flush to ensure data is written immediately
        fflush(output_file);
        
        // Calculate actual time since last measurement
        unsigned long long time_diff = current_time - last_time;
        last_time = current_time;
        
        // Sleep for the remainder of the interval if we haven't used it all
        unsigned long long sleep_us = (unsigned long long)(interval_ms * 1000) - time_diff;
        if (sleep_us > 0 && sleep_us < 1000000) {  // Sanity check on sleep time
            struct timespec sleep_time;
            sleep_time.tv_sec = sleep_us / 1000000;
            sleep_time.tv_nsec = (sleep_us % 1000000) * 1000;
            nanosleep(&sleep_time, NULL);
        } else {
            // Small delay to prevent tight loop
            usleep(1000);
        }
    }
    
    printf("Monitoring thread exiting\n");
    return NULL;
}

int main(int argc, char *argv[]) {
    int opt;
    unsigned int device_index = 0;
    char output_filename[256] = {0};
    double interval_ms = 1.0; // Default 1ms
    char command[1024] = {0};
    int has_command = 0;
    
    // Parse command line arguments
    while ((opt = getopt(argc, argv, "d:o:i:c:h")) != -1) {
        switch (opt) {
            case 'd':
                device_index = atoi(optarg);
                break;
            case 'o':
                strncpy(output_filename, optarg, sizeof(output_filename) - 1);
                break;
            case 'i':
                interval_ms = atof(optarg);
                if (interval_ms <= 0) interval_ms = 1.0;
                break;
            case 'c':
                strncpy(command, optarg, sizeof(command) - 1);
                has_command = 1;
                break;
            case 'h':
            default:
                printf("Usage: %s -d <device_index> -o <output_file> -i <interval_ms> -c <command>\n", argv[0]);
                printf("  -d <device_index>  : GPU device index (default: 0)\n");
                printf("  -o <output_file>   : Output CSV file (default: gpu_profile_YYYYMMDD_HHMMSS.csv)\n");
                printf("  -i <interval_ms>   : Sampling interval in milliseconds (default: 1.0)\n");
                printf("  -c <command>       : Command to execute and monitor\n");
                return 1;
        }
    }
    
    // Generate default output filename if not specified
    if (output_filename[0] == '\0') {
        time_t now = time(NULL);
        struct tm *tm_info = localtime(&now);
        strftime(output_filename, sizeof(output_filename), "gpu_profile_%Y%m%d_%H%M%S.csv", tm_info);
    }
    
    // Initialize NVML
    CHECK_NVML(nvmlInit());
    
    // Get device count
    unsigned int device_count;
    CHECK_NVML(nvmlDeviceGetCount(&device_count));
    if (device_index >= device_count) {
        fprintf(stderr, "Error: Invalid device index %u (found %u devices)\n", device_index, device_count);
        nvmlShutdown();
        return 1;
    }
    
    // Get device handle
    nvmlDevice_t device;
    CHECK_NVML(nvmlDeviceGetHandleByIndex(device_index, &device));
    
    // Open output file
    FILE* output_file = fopen(output_filename, "w");
    if (!output_file) {
        fprintf(stderr, "Error: Could not open output file %s\n", output_filename);
        nvmlShutdown();
        return 1;
    }
    
    printf("GPU monitoring started with interval %.2f ms\n", interval_ms);
    printf("Results will be saved to: %s\n", output_filename);
    
    // Set up signal handler for graceful termination
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Set up monitoring context
    MonitorContext ctx = {
        .device = device,
        .output_file = output_file,
        .interval_ms = interval_ms,
        .should_stop = 0
    };
    
    // Start monitoring thread
    pthread_t monitor_thread;
    if (pthread_create(&monitor_thread, NULL, monitor_gpu, &ctx) != 0) {
        fprintf(stderr, "Error creating monitoring thread\n");
        fclose(output_file);
        nvmlShutdown();
        return 1;
    }
    
    // If a command was specified, execute it
    if (has_command) {
        printf("Executing command: %s\n", command);
        int ret = system(command);
        printf("Command completed with return code: %d\n", ret);
        
        // Signal monitoring thread to stop
        ctx.should_stop = 1;
        keep_running = 0;  // Also set the global flag
        
        printf("Waiting for monitoring thread to finish...\n");
        
        // Add a small delay to allow thread to notice the stop signal
        usleep(100000);  // 100ms
        
        // Use a timeout approach with standard functions
        time_t start_time = time(NULL);
        int thread_joined = 0;
        
        while (difftime(time(NULL), start_time) < 2.0) {  // 2 second timeout
            // Check if the thread has updated any shared data
            // This is a heuristic to detect if the thread is still running
            int old_value = ctx.should_stop;
            ctx.should_stop = 3;  // Set to a value different from 1
            usleep(50000);       // 50ms to let the thread see the new value
            
            if (ctx.should_stop == 3) {
                // Value wasn't changed back, thread is likely not running
                thread_joined = 1;
                break;
            }
            ctx.should_stop = old_value;  // Restore value
        }
        
        if (!thread_joined) {
            printf("Monitoring thread didn't exit in time, forcibly terminating...\n");
            pthread_cancel(monitor_thread);
        }
        
        // Always join to clean up resources
        pthread_join(monitor_thread, NULL);
    } else {
        // Wait for user to interrupt with Ctrl+C
        printf("Press Ctrl+C to stop monitoring...\n");
        while (keep_running) {
            sleep(1);
        }
        
        // Signal thread to stop and wait for it
        ctx.should_stop = 1;
        pthread_join(monitor_thread, NULL);
    }

    // Clean up
    printf("Cleaning up resources...\n");
    fclose(output_file);
    CHECK_NVML(nvmlShutdown());

    printf("Monitoring finished\n");
    return 0;
}