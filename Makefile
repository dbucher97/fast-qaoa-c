INC_DIR = ./src
SRC_DIR = ./src

BUILD_DIR = ./build

CFLAGS ?= -I$(INC_DIR) -O3 -I/opt/homebrew/include/ -DACCELERATE_NEW_LAPACK

SRCS := $(shell find $(SRC_DIR) -name *.c)
OBJS := $(SRCS:%.c=$(BUILD_DIR)/%.o)

TARGET = libqaoa.dylib

LDFLAGS += -L/opt/homebrew/lib/ -framework Accelerate

$(BUILD_DIR)/$(TARGET): $(OBJS)
	$(CC) $(OBJS) -dynamiclib -o $@ $(LDFLAGS) -llbfgs


$(BUILD_DIR)/%.o: %.c
	@$(MKDIR_P) $(dir $@)
	$(CC) $(CFLAGS)  -c $< -o $@


.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)

MKDIR_P = mkdir -p
