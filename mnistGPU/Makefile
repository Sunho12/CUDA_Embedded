# Compiler to use
CC=g++
NVCC=nvcc

# Compiler flags
#CFLAGS = -Wall -g
CFLAGS= -g

# Directories
SRCDIR = src
OBJDIR = obj

# Source files
SRCS = main.cu $(SRCDIR)/dataReader.h $(SRCDIR)/model.h 

# Object files (replace .c with .o and put in obj directory)
OBJS = $(OBJDIR)/main.o $(OBJDIR)/dataReader.o $(OBJDIR)/model.o 

# Name of the executable
EXEC = mnist.out

# Default target
all: $(EXEC)

# Rule to create the executable by linking object files
$(EXEC): $(OBJS)
	$(NVCC) $(OBJS) -o $(EXEC)

# Rule to compile main.cu file to obj/main.o
$(OBJDIR)/main.o: main.cu
	@mkdir -p $(OBJDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

# Rule to compile src/*.c files to obj/*.o
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Rule to compile src/*.cu files to obj/*.o
$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(OBJDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

# Clean up object files and the executable
clean:
	rm -rf $(OBJDIR) $(EXEC)
