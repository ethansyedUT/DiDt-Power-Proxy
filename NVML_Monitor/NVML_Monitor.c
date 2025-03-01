#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <time.h>
#include <direct.h>
#include <errno.h>
#include <stdbool.h>

#define CHECK_NVML(call) \
do { \
    nvmlReturn_t result = call; \
    if (result != NVML_SUCCESS && result != NVML_ERROR_NOT_FOUND) { \
        fprintf(logFile, "NVML Error: %s\n", nvmlErrorString(result)); \
        if (result == NVML_ERROR_UNINITIALIZED) { \
            return -1; \
        } \
    } \
} while(0)

#define MAX_PROCESSES 128

typedef struct {
    unsigned int pid;
    FILE* powerFile;
    FILE* memoryFile;
    FILE* timeFile;
    bool active;
} ProcessFiles;

typedef struct {
    DWORD samplingIntervalMs;
    LARGE_INTEGER performanceFrequency;
    LARGE_INTEGER lastSampleTime;
    ProcessFiles processes[MAX_PROCESSES];
    int processCount;
} MonitorConfig;

void getTimeString(char* buffer, size_t size) {
    SYSTEMTIME st;
    GetLocalTime(&st);
    snprintf(buffer, size, "%04d-%02d-%02d_%02d-%02d-%02d.%03d",
        st.wYear, st.wMonth, st.wDay,
        st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
}

void printMemorySize(FILE* file, unsigned long long bytes) {
    if (bytes == 18446744073709551615ULL) {
        fprintf(file, "Unknown");
    }
    else {
        fprintf(file, "%llu bytes (%.2f MB)", bytes, bytes / (1024.0 * 1024.0));
    }
}

double getElapsedMs(const LARGE_INTEGER* start, const LARGE_INTEGER* end, const LARGE_INTEGER* freq) {
    LARGE_INTEGER elapsed;
    elapsed.QuadPart = end->QuadPart - start->QuadPart;
    return (elapsed.QuadPart * 1000.0) / freq->QuadPart;
}

int createDirectoryIfNotExists(const char* path) {
    int result = _mkdir(path);
    if (result != 0 && errno != EEXIST) {
        printf("Error creating directory %s. Error code: %d\n", path, errno);
        return -1;
    }
    return 0;
}

void initProcessFiles(ProcessFiles* proc, unsigned int pid) {
    char filename[256];

    // Create power trace file
    sprintf_s(filename, sizeof(filename), "logs/PowerTraces/%u_PowerTrace.csv", pid);
    fopen_s(&proc->powerFile, filename, "w");
    if (proc->powerFile) fprintf(proc->powerFile, "Timestamp,PowerUsage_mW\n");

    // Create memory usage file
    sprintf_s(filename, sizeof(filename), "logs/MemoryUsageTraces/%u_MemoryUsage.csv", pid);
    fopen_s(&proc->memoryFile, filename, "w");
    if (proc->memoryFile) fprintf(proc->memoryFile, "Timestamp,MemoryUsed_MB,TotalMemory_MB,MemoryUtilization_Percent\n");
    // if (proc->memoryFile) fprintf(proc->memoryFile, "Timestamp,ProcessMemoryUsed_MB,TotalMemory_MB,ProcessMemoryUtilization_Percent\n");

    // Create time log file
    sprintf_s(filename, sizeof(filename), "logs/SamplingTiming/%u_TimeLog.csv", pid);
    fopen_s(&proc->timeFile, filename, "w");
    if (proc->timeFile) fprintf(proc->timeFile, "Timestamp,SampleInterval_ms,Temperature_C,GPUUtilization_Percent\n");

    proc->pid = pid;
    proc->active = true;
}

void closeProcessFiles(ProcessFiles* proc) {
    if (proc->powerFile) fclose(proc->powerFile);
    if (proc->memoryFile) fclose(proc->memoryFile);
    if (proc->timeFile) fclose(proc->timeFile);
    proc->active = false;
}

ProcessFiles* findOrCreateProcessEntry(MonitorConfig* config, unsigned int pid) {
    // Look for existing process
    for (int i = 0; i < config->processCount; i++) {
        if (config->processes[i].pid == pid && config->processes[i].active) {
            return &config->processes[i];
        }
    }

    // Create new process entry
    if (config->processCount < MAX_PROCESSES) {
        ProcessFiles* proc = &config->processes[config->processCount++];
        initProcessFiles(proc, pid);
        return proc;
    }

    return NULL;
}

int main(int argc, char* argv[]) {
    nvmlDevice_t device;
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    char timeStr[26];
    FILE* logFile;

    // Initialize monitoring configuration
    MonitorConfig config = { 0 };
    config.samplingIntervalMs = (argc > 1) ? atoi(argv[1]) : 1;
    QueryPerformanceFrequency(&config.performanceFrequency);

    if (config.samplingIntervalMs < 1) {
        printf("Warning: Sampling interval too low. Setting to 1ms minimum.\n");
        config.samplingIntervalMs = 1;
    }

    if (_mkdir("logs") != 0 && errno != EEXIST) {
        printf("Error creating logs directory. Error code: %d\n", errno);
        return -1;
    }

        // Create main logs directory
    if (createDirectoryIfNotExists("logs") != 0) {
        printf("Error creating main logs directory\n");
        return -1;
    }

    // Create subdirectories for specific log types
    if (createDirectoryIfNotExists("logs/PowerTraces") != 0) {
        printf("Error creating PowerTraces directory\n");
        return -1;
    }

    if (createDirectoryIfNotExists("logs/MemoryUsageTraces") != 0) {
        printf("Error creating MemoryUsageTraces directory\n");
        return -1;
    }

    if (createDirectoryIfNotExists("logs/SamplingTiming") != 0) {
        printf("Error creating SamplingTiming directory\n");
        return -1;
    }

    getTimeString(timeStr, sizeof(timeStr));
    char filename[256];
    sprintf_s(filename, sizeof(filename), "logs/gpu_monitor_%s.log", timeStr);

    errno_t err = fopen_s(&logFile, filename, "w");
    if (err != 0 || logFile == NULL) {
        char errMsg[100];
        strerror_s(errMsg, sizeof(errMsg), err);
        printf("Error opening log file: %s\nAttempted path: %s\n", errMsg, filename);
        return -1;
    }

    printf("Initializing NVIDIA GPU monitor...\n");
    printf("Logging to file: %s\n", filename);
    printf("Sampling interval: %lu ms\n", config.samplingIntervalMs);
    fprintf(logFile, "Initializing NVIDIA GPU monitor...\n");
    fprintf(logFile, "Sampling interval: %lu ms\n", config.samplingIntervalMs);

    CHECK_NVML(nvmlInit());
    CHECK_NVML(nvmlDeviceGetHandleByIndex(0, &device));
    CHECK_NVML(nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE));

    fprintf(logFile, "Monitoring GPU: %s\n", name);
    fprintf(logFile, "Monitoring started at: %s\n\n", timeStr);
    printf("Monitoring started. Check log file for details.\n");
    fflush(logFile);

    bool hadProcess = false;
    QueryPerformanceCounter(&config.lastSampleTime);

    while (1) {
        LARGE_INTEGER currentTime;
        QueryPerformanceCounter(&currentTime);

        double elapsedMs = getElapsedMs(&config.lastSampleTime, &currentTime, &config.performanceFrequency);

        if (elapsedMs >= config.samplingIntervalMs) {
            getTimeString(timeStr, sizeof(timeStr));

            unsigned int processCount = 0;
            nvmlProcessInfo_t processes[10];
            processCount = 10;

            nvmlReturn_t result = nvmlDeviceGetComputeRunningProcesses(device, &processCount, processes);
            bool currentlyHasProcess = (processCount > 0);

            if (currentlyHasProcess) {
                bool fail_read = false;
                nvmlMemory_t memInfo;
                result = nvmlDeviceGetMemoryInfo(device, &memInfo);
                if (result != NVML_SUCCESS) {
                    fprintf(logFile, "----------------------------------------");
                    fprintf(logFile, "BAD MEMORY DATA");
                    fprintf(logFile, "----------------------------------------");
                    fail_read = true;
                }


                nvmlUtilization_t utilInfo;
                result = nvmlDeviceGetUtilizationRates(device, &utilInfo);
                if (result != NVML_SUCCESS) {
                    utilInfo.gpu = 0;
                    utilInfo.memory = 0;
                    fprintf(logFile, "----------------------------------------");
                    fprintf(logFile, "BAD DEVICE UTILIZATION DATA");
                    fprintf(logFile, "----------------------------------------");
                    fail_read = true;
                }

                unsigned int power = 0;
                result = nvmlDeviceGetPowerUsage(device, &power);
                if (result != NVML_SUCCESS) {
                    fprintf(logFile, "----------------------------------------");
                    fprintf(logFile, "BAD POWER DATA");
                    fprintf(logFile, "----------------------------------------");
                    fail_read = true;
                }

                unsigned int temp = 0;
                result = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);
                if (result != NVML_SUCCESS) {
                    fprintf(logFile, "----------------------------------------");
                    fprintf(logFile, "BAD TEMPERATURE DATA");
                    fprintf(logFile, "----------------------------------------");
                    fail_read = true;
                }
                if(!fail_read){
                    // Write to verbose log
                    fprintf(logFile, "----------------------------------------\n");
                    fprintf(logFile, "Timestamp: %s (Sample interval: %.2f ms)\n", timeStr, elapsedMs);
                    fprintf(logFile, "Power Usage: %d mW\n", power);
                    fprintf(logFile, "Active CUDA Processes: %u\n", processCount);

                    // Process each active CUDA process
                    for (unsigned int i = 0; i < processCount; i++) {
                        ProcessFiles* proc = findOrCreateProcessEntry(&config, processes[i].pid);
                        if (proc) {
                            // Write to process-specific CSV files
                            fprintf(proc->powerFile, "%s,%u\n", timeStr, power);
                            fprintf(proc->memoryFile, "%s,%llu,%llu,%u\n",
                                timeStr,
                                memInfo.used / (1024 * 1024),
                                memInfo.total / (1024 * 1024),
                                utilInfo.memory);
                            fprintf(proc->timeFile, "%s,%.2f,%u,%u\n",
                                timeStr,
                                elapsedMs,
                                temp,
                                utilInfo.gpu);

                            // Flush the files to ensure data is written
                            fflush(proc->powerFile);
                            fflush(proc->memoryFile);
                            fflush(proc->timeFile);
                        }

                        fprintf(logFile, "  Process ID: %u, Memory Used: ", processes[i].pid);
                        printMemorySize(logFile, processes[i].usedGpuMemory);
                        fprintf(logFile, "\n");
                    }

                    fprintf(logFile, "\nGPU Memory: Used %llu MB / Total %llu MB\n",
                        memInfo.used / (1024 * 1024),
                        memInfo.total / (1024 * 1024));
                    fprintf(logFile, "GPU Utilization: %d%%\n", utilInfo.gpu);
                    fprintf(logFile, "Memory Utilization: %d%%\n", utilInfo.memory);
                    fprintf(logFile, "Temperature: %d°C\n", temp);

                    hadProcess = true;
                    fflush(logFile);
                }
            }
            else if (hadProcess) {
                fprintf(logFile, "----------------------------------------\n");
                fprintf(logFile, "Timestamp: %s\n", timeStr);
                fprintf(logFile, "No active CUDA processes. Waiting for new processes...\n");
                fflush(logFile);
                hadProcess = false;

                // Close files for all active processes
                for (int i = 0; i < config.processCount; i++) {
                    if (config.processes[i].active) {
                        closeProcessFiles(&config.processes[i]);
                    }
                }
            }

            config.lastSampleTime = currentTime;
            Sleep(100);
        }

    }

    // Cleanup
    for (int i = 0; i < config.processCount; i++) {
        if (config.processes[i].active) {
            closeProcessFiles(&config.processes[i]);
        }
    }

    fclose(logFile);
    nvmlShutdown();
    return 0;
}