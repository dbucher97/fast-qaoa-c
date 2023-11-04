INC_DIR = ./src
SRC_DIR = ./src

BUILD_DIR = ./build


CFLAGS ?= -I$(INC_DIR) -O3 -march=native

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
TARGET = libqaoa.so
DYLIB = -fPIC -shared
else
CFLAGS += -I/opt/homebrew/include/
LDFLAGS += -L/opt/homebrew/lib/
TARGET = libqaoa.dylib
DYLIB = -dynamiclib
endif

SRCS := $(shell find $(SRC_DIR) -name *.c)
OBJS := $(SRCS:%.c=$(BUILD_DIR)/%.o)



$(BUILD_DIR)/$(TARGET): $(OBJS)
	$(CC) $(OBJS) $(DYLIB) -o $@ $(LDFLAGS) -llbfgs


$(BUILD_DIR)/%.o: %.c
	@$(MKDIR_P) $(dir $@)
	$(CC) $(CFLAGS)  -c $< -o $@


.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)

MKDIR_P = mkdir -p
