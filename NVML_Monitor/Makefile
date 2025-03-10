CC = nvcc
CFLAGS = -lnvml
TARGET = NVML_Monitor
SRC = NVML_Monitor.c
OS = linux
LOG_DIR = logs
PTX_FLAGS = -O0 -Xptxas -O0 -arch=sm_80 -ptx

ifeq ($(OS),windows)
    RM = del /Q /F
    RM_CMD = $(RM) $(TARGET).exe
    RUN_CMD = $(TARGET).exe
    RMDIR_CMD = rmdir /S /Q
    PROMPT_CMD = powershell -Command "$$response = Read-Host 'Are you sure you want to delete all logs? (y/n)'; if($$response -eq 'y'){exit 0}else{exit 1}"
    LOG_DEL_CMD = $(RM) $(LOG_DIR)\*.* /S /Q
else
    RM = rm -f
    RM_CMD = $(RM) $(TARGET) $(TARGET).exe
    RUN_CMD = ./$(TARGET)
    RMDIR_CMD = rm -rf
    PROMPT_CMD = read -p "Are you sure you want to delete all logs? (y/n) " yn; [ "$$yn" = "y" ]
    LOG_DEL_CMD = $(RM) $(LOG_DIR)/*
endif

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(SRC) -o $(TARGET) $(CFLAGS)

# Build PTX file from CUDA source
ptx:
	@if [ -z "$(filepath)" ]; then \
		echo "Usage: make ptx filepath=/path/to/file.cu"; \
	else \
		$(CC) $(PTX_FLAGS) $(filepath); \
	fi

clean:
	$(RM_CMD)

# Create logs directory if it doesn't exist
$(LOG_DIR):
	@mkdir $(LOG_DIR) 2>nul || echo "Log directory already exists"

# Clear logs with confirmation prompt
clearlogs: $(LOG_DIR)
	@echo WARNING: This will delete all log files in the $(LOG_DIR) directory
	@$(PROMPT_CMD) && $(LOG_DEL_CMD) && echo Logs cleared || echo Log deletion cancelled

run: $(TARGET)
	$(RUN_CMD)

.PHONY: all clean clearlogs run ptx