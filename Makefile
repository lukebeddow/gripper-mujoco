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
DEFAULT_OBJECTSET = set1_nocuboid_525

# for generating models, non-essential feature
MODELBASH := generate_models.sh
MODELDIR := /home/luke/gripper_repo_ws/src/gripper_v2/gripper_description/urdf/mujoco

# ----- conditional compilation with user defined options ----- #

# are we compiling in debug mode if so add symbols and turn off optimisations
ifeq ($(filter debug, $(MAKECMDGOALS)), debug)
OPTIM = -O0 -g
else
OPTIM = -O2
endif

# define library locations - this file contains user specified options
include buildsettings.mk

# ----- compilation settings ----- #

# define compiler flags and libraries
COMMON = $(OPTIM) -std=c++11 -mavx -pthread -Wl,-rpath,'$$ORIGIN' $(DEFINE_VAR) \
		 -DLUKE_DEFAULTOBJECTS='"$(DEFAULT_OBJECTSET)"' \
		 -I$(MUJOCO_PATH)/include \
		 -I$(PYBIND_PATH)/include \
		 -I$(ARMA_PATH)/include \
		 -I$(PYTHON_PATH) \
		 -I$(RENDER_PATH)/include

PYBIND = $(COMMON) -fPIC -Wall -shared #-I$(PYTHON_PATH) $(PY_LIBS)

# ----- automatically generated variables ----- #

# get every source file and each corresponding dependecy file
SOURCES := $(wildcard $(SOURCEDIR)/*.cpp)
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

.PHONY: everything
everything: cpp py models

.PHONY: debug
debug: cpp

.PHONY: cluster
cluster: py

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
$(CPPTARGETS): $(OUTCPP)% : $(BUILDDIR)%.o $(BUILDDIR)/uitools.o $(CPPSHAREDOBJ)
	g++ $(COMMON) $^ $(CORE_LIBS) $(RENDER_LIBS) -o $@
$(PYTARGETS): $(OUTPY)%.so : $(BUILDDIR)%.o $(BUILDDIR)/uitools.o $(PYSHAREDOBJ)
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

# run the bash script that re-generates the model files
.PHONY: models
models:
	./$(MODELBASH)

# run the bash script, but only recompile mjcf
.PHONY: xml
xml:
	$(MAKE) -C $(MODELDIR)

clean:
	rm -f $(BUILDDIR)/*.o 
	rm -f $(BUILDCPP)/*.o 
	rm -f $(BUILDPY)/*.o 
	rm -f $(BUILDDEP)/*.d
	rm -f $(CPPTARGETS)
	rm -f $(PYTARGETS)
