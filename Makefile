SRCS = $(wildcard ecelinux/*.cpp)
SRCS += $(wildcard ecelinux/*.h)
SRCS += $(wildcard ecelinux/*.sh)
SRCS += $(wildcard ecelinux/*.tcl)
SRCS += $(wildcard ecelinux/Makefile)
SRCS += $(wildcard ecelinux/data)
SRCS += $(wildcard zedboard/*.cpp)
SRCS += $(wildcard zedboard/*.h)
SRCS += $(wildcard zedboard/*.sh)
SRCS += $(wildcard zedboard/*.tcl)
SRCS += $(wildcard zedboard/Makefile)
SRCS += $(wildcard zedboard/data)

.PHONY: all

all: bnn.zip

bnn.zip: report.pdf $(SRCS)
	zip -r $@ $^
