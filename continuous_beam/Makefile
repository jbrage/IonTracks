# Use the bash shell
SHELL = /usr/bin/env bash

##############################
# Specification of filenames #
##############################
# Modules which should be cythonized and compiled
pyfiles=distribute_and_evolve_pulsed
# Filename of the .pyx preprocessor script
pyxpp = pyxpp.py


################################################
# Environment information from the .paths file #
################################################
python        = python3
python_config = python3-config


########################################
# Settings for compilation and linking #
########################################
# Options passed when cynthonizing .pyx files
cythonflags = -3 -a
# Includes
includes    = $(shell $(python_config) --includes)
# Compiler options
CC = gcc
python_cflags = $(shell $(python_config) --cflags)
other_cflags  = -pthread -std=c99 -fno-strict-aliasing -fPIC
CFLAGS    = $(python_cflags) $(other_cflags) $(includes)
# Libraries to link
LDLIBS = -L$(python_dir)/lib -Wl,"-rpath=$(python_dir)/lib" $(shell $(python_config) --libs)
# Linker options
python_ldflags = $(shell $(python_config) --ldflags)
other_ldflags  = -shared
LDFLAGS    = $(python_ldflags) $(other_ldflags)

###################
# Primary targets #
###################
# The below targets are responsible for the
# .py --> (.pyx --> .pxd) --> .c --> .o --> .so build chain. For each,
# module, a heading are printed at the beginning of its build process.
# The 'build_status' variable is used as a flag, controlling these
# headings.

# Make everything
all: $(addsuffix .so, $(pyfiles))
	@# This suppresses "make: Nothing to be done for `all'."

# Link object filed into shared object Python modules
$(addsuffix .so, $(pyfiles)): %.so: %.o
	@$(python) -c "print('\nBuilding the $(basename $@) module') if '$(build_status)' != 'running' else ''"
	$(eval build_status = running)
	$(CC) $< -o $@ -fPIC $(LDFLAGS) $(LDLIBS)
	$(eval build_status = finsihed)

# Compile c source files into object files
$(addsuffix .o, $(pyfiles)): %.o: %.c
	@$(python) -c "print('\nBuilding the $(basename $@) module') if '$(build_status)' != 'running' else ''"
	$(eval build_status = running)
	$(CC) $(CFLAGS) -fPIC -c -o $@ $<

# Cythonize .pyx and .pxd files into c source files
$(addsuffix .c, $(pyfiles)): %.c: %.pyx %.pxd
	@$(python) -c "print('\nBuilding the $(basename $@) module') if '$(build_status)' != 'running' else ''"
	$(eval build_status = running)
	$(python) -m cython $(cythonflags) $<

# Write .pxd files from .pyx files using the pyxpp script
$(addsuffix .pxd, $(pyfiles)): %.pxd: %.pyx $(pyxpp)
	@$(python) -c "print('\nBuilding the $(basename $@) module') if '$(build_status)' != 'running' else ''"
	$(eval build_status = running)
	$(python) $(pyxpp) $<

# Write .pyx files from .py files using the pyxpp script
$(addsuffix .pyx, $(pyfiles)): %.pyx: %.py $(pyxpp) $(MAKEFILE_LIST)
	@$(python) -c "print('\nBuilding the $(basename $@) module') if '$(build_status)' != 'running' else ''"
	$(eval build_status = running)
	$(python) $(pyxpp) $<



###################
# Cleanup targets #
###################
# Remove all compiled files
clean:
	$(RM) $(foreach ext, pyx pxd c o so html,$(addsuffix .$(ext), $(pyfiles)))

