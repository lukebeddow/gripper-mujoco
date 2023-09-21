# ----- description ----- #

# This Makefile compiles code which interfaces with mujoco physics simulator.
# There are two key targets, firstly a c++ compilation and secondly a python
# compilation. The c++ compilation results in an executable aimed at testing.
# The python compilation results in a python module that can be imported and
# used from within python.

# Useful resources:
#   https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html
#   https://www.gnu.org/software/make/manual/html_node/Static-Usage.html#Static-Usage
#   https://www.hiroom2.com/2016/09/03/makefile-header-dependencies/
#   https://stackoverflow.com/questions/16924333/makefile-compiling-from-directory-to-another-directory

# This file does not need to be edited to build on a new machine. Instead, the
# file 'buildsettings.mk' should be edited, as explained in the readme file.

# ----- user defined variables ----- #

# define the targets (must compile from a .cpp file with same name in SOURCEDIR)
TARGET_LIST_CPP := test mysimulate
TARGET_LIST_PY := bind

# define directory structure
SOURCEDIR := src
BUILDDIR := build
BUILDPY := $(BUILDDIR)/py
BUILDCPP := $(BUILDDIR)/cpp
BUILDDEP := $(BUILDDIR)/depends
OUTCPP := bin
OUTPY := rl/env/mjpy

# default object set name
DEFAULT_OBJECTSET = set6_fullset_800_50i

# where is the gripper-description submodule (if we have it)
DESCRIPTION_MODULE := description

# where are we storing mjcf files
MJCF_FOLDER := mjcf

# do we want to prevent any rendering libraries from compiling (1=True, 0=false)
PREVENT_RENDERING = 0

# ----- conditional compilation with user defined options ----- #

# are we compiling in debug mode if so add symbols and turn off optimisations
ifeq ($(filter debug, $(MAKECMDGOALS)), debug)
OPTIM = -Og -g -pg
else
OPTIM = -O2
endif

# define library locations - this file contains user specified options
DEPENDS_BOOST = 0 # by default do not depend on boost
include buildsettings.mk

# ----- compilation settings ----- #

COMMON = $(OPTIM) -std=c++11 -mavx -pthread $(DEFINE_VAR) \
		 -Wl,-rpath='$(MUJOCO_LIB)' \
		 -DLUKE_DEFAULTOBJECTS='"$(DEFAULT_OBJECTSET)"' \
		 -I$(MUJOCO_PATH)/include \
		 -I$(PYBIND_PATH)/include \
		 -I$(ARMA_PATH)/include \
		 -I$(PYTHON_INCLUDE) \
		 -I$(RENDER_PATH)/include

PYBIND = $(COMMON) -fPIC -Wall -shared

# ----- automatically generated variables ----- #

# get every source file and each corresponding dependecy file
SOURCES := $(wildcard $(SOURCEDIR)/*.cpp)

# are we going to prevent any rendering libraries compiling
ifeq ($(PREVENT_RENDERING), 1)
SOURCES := $(filter-out $(SOURCEDIR)/rendering.cpp, $(SOURCES))
else
UITOOLS = $(BUILDDIR)/uitools.o
endif

# if we don't depend on boost, remove boost from src files
ifeq ($(DEPENDS_BOOST), 0)
SOURCES := $(filter-out $(SOURCEDIR)/boostdep.cpp, $(SOURCES))
endif

# get the dependencies of each source file
DEPENDS := $(patsubst $(SOURCEDIR)/%.cpp, $(BUILDDEP)/%.d, $(SOURCES))

# define targets in the output directory
CPPTARGETS := $(patsubst %, $(OUTCPP)/%, $(TARGET_LIST_CPP))
PYTARGETS := $(patsubst %, $(OUTPY)/%.so, $(TARGET_LIST_PY))

# seperate source files for targets and those for shared objects
CPPTARGETSRC := $(patsubst %, $(SOURCEDIR)/%.cpp, $(TARGET_LIST_CPP))
PYTARGETSRC := $(patsubst %, $(SOURCEDIR)/%.cpp, $(TARGET_LIST_PY))
SOURCES := $(filter-out $(CPPTARGETSRC), $(SOURCES))
SOURCES := $(filter-out $(PYTARGETSRC), $(SOURCES))

# define target object files and shared object files
CPPTARGETOBJ := $(patsubst %, $(BUILDDIR)/%.o, $(TARGET_LIST_CPP))
PYTARGETOBJ := $(patsubst %, $(BUILDDIR)/%.o, $(TARGET_LIST_PY))
CPPSHAREDOBJ := $(patsubst $(SOURCEDIR)/%.cpp, $(BUILDCPP)/%.o, $(SOURCES))
PYSHAREDOBJ := $(patsubst $(SOURCEDIR)/%.cpp, $(BUILDPY)/%.o, $(SOURCES))

# create the directories if they don't exist
DIRS := $(BUILDPY) $(BUILDCPP) $(BUILDDEP) $(OUTCPP) $(OUTPY)
$(info $(shell mkdir -p $(DIRS)))

# ----- start of make ----- #

all: cpp py
cpp: $(CPPTARGETS) $(DEPENDS)
py: $(PYTARGETS) $(DEPENDS)

.PHONY: debug
debug: cpp

.PHONY: cluster
cluster: py $(OUTCPP)/test

.PHONY: pc
pc: cpp py lab

.PHONY: sets
sets:
	$(info $(shell mkdir -p $(MJCF_FOLDER)))
	$(MAKE) -C $(DESCRIPTION_MODULE) sets $(ARGS) EXTRA_COPY_TO="../../$(MJCF_FOLDER)" \
		MUJOCO_PATH=$(MUJOCO_PATH) EXTRA_COPY_YES_TO_ALL=yes \
		EXTRA_COPY_TO_OVERRIDE_EXISTING=yes \
		PYTHON=$(PYTHON_EXE)

# compile the uitools object file which is used by both cpp and python targets
# ADDED -fPIC FOR CLUSTER TO WORK
$(BUILDDIR)/uitools.o:
	gcc -c -O2 -mavx -fPIC -I$(RENDER_PATH)/include -I$(MUJOCO_PATH)/include \
		$(MUJOCO_PATH)/include/uitools.c -o $@

# build object files
$(CPPSHAREDOBJ): $(BUILDCPP)/%.o : $(SOURCEDIR)/%.cpp
	g++ $(COMMON) -c $< -o $@
$(PYSHAREDOBJ): $(BUILDPY)/%.o : $(SOURCEDIR)/%.cpp
	g++ $(COMMON) -c -fPIC $< -o $@
$(CPPTARGETOBJ): $(BUILDDIR)/%.o : $(SOURCEDIR)/%.cpp
	g++ $(COMMON) -c $< -o $@
$(PYTARGETOBJ): $(BUILDDIR)/%.o : $(SOURCEDIR)/%.cpp
	g++ $(PYBIND) -c $< -o $@

# build targets
$(CPPTARGETS): $(OUTCPP)% : $(BUILDDIR)%.o $(UITOOLS) $(CPPSHAREDOBJ)
	g++ $(COMMON) $^ $(CORE_LIBS) $(RENDER_LIBS) -o $@
$(PYTARGETS): $(OUTPY)%.so : $(BUILDDIR)%.o $(UITOOLS) $(PYSHAREDOBJ)
	g++ $(PYBIND) $^ $(CORE_LIBS) $(RENDER_LIBS) -o $@

# if not cleaning, declare the dependencies of each object file (headers and source)
ifneq ($(filter clean, $(MAKECMDGOALS)), clean)
include $(DEPENDS)
endif

# generate dependency files (-M for all, -MM for exclude system dependencies)
$(BUILDDEP)/%.d: $(SOURCEDIR)/%.cpp
	@set -e; rm -f $@; \
		g++ -MM $(COMMON) $< > $@.$$$$; \
		sed 's,\($*\)\.o[ :]*,$(BUILDDIR)/\1.o $(BUILDCPP)/\1.o $(BUILDPY)/\1.o $@ : ,g' \
			< $@.$$$$ > $@;
		rm -f $@.*

clean:
	rm -f $(BUILDDIR)/*.o 
	rm -f $(BUILDCPP)/*.o 
	rm -f $(BUILDPY)/*.o 
	rm -f $(BUILDDEP)/*.d
	rm -f $(CPPTARGETS)
	rm -f $(PYTARGETS)
