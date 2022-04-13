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

# ----- user defined variables ----- #

# define the targets (must compile from a .cpp file with same name in SOURCEDIR)
TARGET_LIST_CPP := test mysimulate
TARGET_LIST_PY := bind

# define directory structure (these folders must exist)
SOURCEDIR := src
BUILDDIR := build
BUILDPY := $(BUILDDIR)/py
BUILDCPP := $(BUILDDIR)/cpp
BUILDDEP := $(BUILDDIR)/depends
OUTCPP := bin
OUTPY := rl/env/mjpy

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

# library locations, different for on and off the cluster
ifeq ($(filter cluster, $(MAKECMDGOALS)), cluster)

# cluster mjcf files location (model files like gripper/objects)
# MJCF_PATH = /home/lbeddow/mjcf/
MJCF_PATH = /home/lbeddow/mymujoco/mjcf/object_set_1

# cluster library locations
PYTHON_PATH = /share/apps/python-3.6.9/include/python3.6m
PYBIND_PATH = /home/lbeddow/pybind11
ARMA_PATH = /home/lbeddow/clusterlibs/armadillo-code
MUJOCO_PATH = /home/lbeddow/.mujoco/mujoco210
CORE_LIBS = -lmujoco210 -lblas -llapack
RENDER_LIBS = -lGL -lglew	$(MUJOCO_PATH)/bin/libglfw.so.3
DEFINE_VAR = -DLUKE_CLUSTER -DARMA_DONT_USE_WRAPPER -DLUKE_MJCF_PATH='"$(MJCF_PATH)"'

else

# mjcf files location (model files like gripper/objects)
MJCF_PATH = /home/luke/mymujoco/mjcf/object_set_1

# local machine library locations
PYTHON_PATH = /usr/include/python3.6m
PYBIND_PATH = /home/luke/pybind11
ARMA_PATH = # none, use system library
MUJOCO_PATH = /home/luke/.mujoco/mujoco210
CORE_LIBS = -lmujoco210 -larmadillo
RENDER_LIBS = -lGL -lglew	$(MUJOCO_PATH)/bin/libglfw.so.3
DEFINE_VAR = -DLUKE_MJCF_PATH='"$(MJCF_PATH)"'

# extras
MAKEFLAGS += -j8 # jN => use N parallel cores

endif

# ----- compilation settings ----- #

# define compiler flags and libraries
COMMON = $(OPTIM) -std=c++11 -mavx -pthread -Wl,-rpath,'$$ORIGIN' $(DEFINE_VAR) \
         -I$(MUJOCO_PATH)/include \
         -I$(PYBIND_PATH)/include \
				 -I$(ARMA_PATH)/include \
				 -L$(MUJOCO_PATH)/bin
PYBIND = $(COMMON) -fPIC -Wall -shared -I$(PYTHON_PATH)

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

# ----- start of make ----- #

all: cpp py
cpp: $(CPPTARGETS) $(DEPENDS)
py: $(PYTARGETS) $(DEPENDS)

.PHONY: everything
everything: cpp py models

.PHONY: cluster
cluster: py

.PHONY: debug
debug: cpp

# compile the uitools object file which is used by both cpp and python targets
# ADDED -fPIC FOR CLUSTER TO WORK
$(BUILDDIR)/uitools.o:
	gcc -c -O2 -mavx -fPIC -I ~/.mujoco/mujoco210/include \
		~/.mujoco/mujoco210/include/uitools.c -o $@

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
