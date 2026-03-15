HIPCC    ?= /opt/rocm/bin/hipcc
ARCH     ?= gfx950
CXXFLAGS := -std=c++17

.PHONY: all clean run

all: kernel_vadd.co launch_hsaco

# Step 1: Compile kernel source to HSACO (.co)
kernel_vadd.co: kernel_vadd.cpp
	$(HIPCC) --genco --offload-arch=$(ARCH) -o $@ $<

# Step 2: Build the host launcher
launch_hsaco: launch_hsaco.cpp
	$(HIPCC) $(CXXFLAGS) -o $@ $<

# Run the test
run: all
	./launch_hsaco kernel_vadd.co vector_add

clean:
	rm -f kernel_vadd.co launch_hsaco
