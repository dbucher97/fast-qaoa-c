INC_DIR = ./src
SRC_DIR = ./src

BUILD_DIR = ./build


CFLAGS ?= -I$(INC_DIR) -O2 -mtune=native -march=native
# -march=native 

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
TARGET = libqaoa.so
TARGET32 = libqaoa32.so
DYLIB = -shared
CFLAGS += -fPIC
else
CFLAGS += -I/opt/homebrew/include/
LDFLAGS += -L/opt/homebrew/lib/
TARGET = libqaoa.dylib
TARGET32 = libqaoa32.dylib
DYLIB = -dynamiclib
endif

SRCS := $(shell find $(SRC_DIR) -name *.c)

OBJS := $(SRCS:%.c=$(BUILD_DIR)/%64.o)
OBJS32 := $(SRCS:%.c=$(BUILD_DIR)/%32.o)

all: $(BUILD_DIR)/$(TARGET) $(BUILD_DIR)/$(TARGET32)

$(BUILD_DIR)/$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(DYLIB) -o $@ $(LDFLAGS) -llbfgs

$(BUILD_DIR)/%64.o: %.c
	@$(MKDIR_P) $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/$(TARGET32): $(OBJS32)
	$(CC) $(OBJS32) $(DYLIB) -o $@ $(LDFLAGS) -llbfgs

$(BUILD_DIR)/%32.o: %.c
	@$(MKDIR_P) $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@ -DUSE_FLOAT32


.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)

MKDIR_P = mkdir -p
