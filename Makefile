INC_DIR = ./src
SRC_DIR = ./src

BUILD_DIR = ./build
DEST_DIR = ./fastqaoa/ctypes


CFLAGS ?= -I$(INC_DIR) -O3 -march=native -mtune=native

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
TARGET = libqaoa.so
TARGET32 = libqaoa32.so
DYLIB = -shared
CFLAGS += -fPIC
else
ARCH = $(shell arch)
ifeq ($(ARCH),arm64)
CFLAGS += -I/opt/homebrew/include/
LDFLAGS += -L/opt/homebrew/lib/
else
CFLAGS += -I/usr/local/include/
LDFLAGS += -L/usr/local/lib/
endif
TARGET = libqaoa.dylib
TARGET32 = libqaoa32.dylib
DYLIB = -dynamiclib
endif

SRCS := $(shell find $(SRC_DIR) -name *.c)

OBJS := $(SRCS:%.c=$(BUILD_DIR)/%64.o)
OBJS32 := $(SRCS:%.c=$(BUILD_DIR)/%32.o)

all: $(DEST_DIR)/$(TARGET) $(DEST_DIR)/$(TARGET32)

$(DEST_DIR)/$(TARGET): $(OBJS)
	@$(MKDIR_P) $(dir $@)
	$(CC) $(OBJS) $(DYLIB) -o $@ $(LDFLAGS) -llbfgs

$(BUILD_DIR)/%64.o: %.c
	@$(MKDIR_P) $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(DEST_DIR)/$(TARGET32): $(OBJS32)
	@$(MKDIR_P) $(dir $@)
	$(CC) $(OBJS32) $(DYLIB) -o $@ $(LDFLAGS) -llbfgs

$(BUILD_DIR)/%32.o: %.c
	@$(MKDIR_P) $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@ -DUSE_FLOAT32


.PHONY: all clean test

clean:
	$(RM) -r $(BUILD_DIR)
	$(RM) $(DEST_DIR)/$(TARGET)
	$(RM) $(DEST_DIR)/$(TARGET32)

MKDIR_P = mkdir -p
