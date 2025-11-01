#==========================================================================
# Makefile
#==========================================================================
# @brief: A makefile the compiles and runs the bnn program
#
# @desc: 1. "make" or "make sw" runs software execution by default
#        2. "make fpga" invokes the HW accelerator
#        3. "make clean" cleans up the directory


INC_PATH=/usr/include/vivado_hls
CFLAGS = -I${INC_PATH} -DHLS_NO_XIL_FPO_LIB -O3

.PHONY: all sw fpga

all: sw

bnn-arm: bnn.cpp bnn_test.cpp
	g++ ${CFLAGS} $^ -o $@

result/bnn_arm_sim.txt: bnn-arm
	@echo "Compiling & executing bnn software program on ARM ..."
	mkdir -p result
	./$< | tee $@

sw: result/bnn_arm_sim.txt
	@echo "Result saved to $@"

bnn-fpga: host.cpp
	@echo "Compiling host program"
	g++ ${CFLAGS} $^ -o $@
	@echo "Make sure bitstream is loaded!"

fpga: bnn-fpga
	@echo "Running bnn accelerator ..."
	./$<

# Clean up the directory before submission
clean:
	@echo "Clean up output files"
	rm -rf bnn-arm vivado_hls.log *.prj result out.dat *~
	rm -rf bnn-fpga
