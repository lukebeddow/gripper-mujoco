# This file details the paths used for compiling. In order to compile
# on your computer you will need to add your specific paths and the
# locations of the libraries. Copy one of the ifeq...endif blocks,
# change the name eg cluster->yourname, then edit the paths.

# IMPORTANT NOTE: no trailing whitespace after path variables allowed
#	eg -> MUJOCO_PATH = /home/luke/mujoco210   # trailing whitespace
#		 -> -I $(MUJOCO_PATH)/include
#		 -> -I /home/luke/mujoco210   /include   # whitespace breaks compile command
#		 -> g++: error: /include: No such file or directory

# ----- default settings, overwritten by any of the below options ----- #

# what machine are we compiling for
MACHINE = luke-laptop

# mjcf files location (model files like gripper/objects)
MJCF_PATH = /home/luke/mymujoco/mjcf

# local machine library locations
PYTHON_PATH = /usr/include/python3.6m
PYBIND_PATH = /home/luke/repo/pybind11
ARMA_PATH = # none, use system library
MUJOCO_PATH = /home/luke/repo/mujoco/mujoco-2.1.5
MUJOCO_LIB = $(MUJOCO_PATH)/lib
RENDER_PATH = # none, use system library
CORE_LIBS = -L$(MUJOCO_LIB) -lmujoco -larmadillo
RENDER_LIBS = -lglfw
DEFINE_VAR = -DLUKE_MJCF_PATH='"$(MJCF_PATH)"' \
						 -DLUKE_MACHINE='"$(MACHINE)"'

# extras
MAKEFLAGS += -j4 # jN => use N parallel cores

# ----- compiling on lukes laptop ----- #
ifeq ($(findstring luke, $(MAKECMDGOALS)), luke)

# set this command goal as a phony target (important)
.PHONY: luke

# what machine are we compiling for
MACHINE = luke-laptop

# mjcf files location (model files like gripper/objects)
MJCF_PATH = /home/luke/mymujoco/mjcf

# local machine library locations
PYTHON_PATH = /usr/include/python3.6m
PYBIND_PATH = /home/luke/repo/pybind11
ARMA_PATH = # none, use system library
MUJOCO_PATH = /home/luke/repo/mujoco/mujoco-2.1.5
MUJOCO_LIB = $(MUJOCO_PATH)/lib

# are we compiling a specific mujoco version
ifeq ($(findstring 220, $(MAKECMDGOALS)), 220)
.PHONY: 220
MUJOCO_PATH = /home/luke/repo/mujoco/src/mujoco-2.2.0
MUJOCO_LIB = $(MUJOCO_PATH)/build/lib
ifeq ($(findstring debug, $(MAKECMDGOALS)), debug)
MUJOCO_PATH = /home/luke/repo/mujoco/src/mujoco-2.2.0-debug
MUJOCO_LIB = $(MUJOCO_PATH)/build/lib
endif
endif
ifeq ($(findstring 237, $(MAKECMDGOALS)), 237)
.PHONY: 237
MUJOCO_PATH = /home/luke/repo/mujoco/mujoco-2.3.7
MUJOCO_LIB = $(MUJOCO_PATH)/lib
endif

RENDER_PATH = # none, use system library
CORE_LIBS = -L$(MUJOCO_LIB) -lmujoco -larmadillo
RENDER_LIBS = -lglfw
DEFINE_VAR = -DLUKE_MJCF_PATH='"$(MJCF_PATH)"' \
						 -DLUKE_MACHINE='"$(MACHINE)"'

# extras
MAKEFLAGS += -j4 # jN => use N parallel cores

endif

# ----- compiling on the cluster ----- #
ifeq ($(filter cluster, $(MAKECMDGOALS)), cluster)

# phony target for cluster is defined in Makefile

# what machine are we compiling for
MACHINE = cluster

# cluster mjcf files location (model files like gripper/objects)
MJCF_PATH = /home/lbeddow/mymujoco/mjcf

# cluster library locations
PYTHON_PATH = /share/apps/python-3.6.9/include/python3.6m
PYBIND_PATH = /home/lbeddow/clusterlibs/pybind11
ARMA_PATH = /home/lbeddow/clusterlibs/armadillo-code
MUJOCO_PATH = /home/lbeddow/clusterlibs/mujoco/mujoco-2.1.5
MUJOCO_LIB = $(MUJOCO_PATH)/lib
RENDER_PATH = /home/lbeddow/clusterlibs/glfw
CORE_LIBS = -L$(MUJOCO_LIB) -lmujoco -lblas -llapack
RENDER_LIBS = # none, no rendering
DEFINE_VAR = -DLUKE_CLUSTER -DARMA_DONT_USE_WRAPPER \
						 -DLUKE_MJCF_PATH='"$(MJCF_PATH)"' \
						 -DLUKE_MACHINE='"$(MACHINE)"'

# we do not want to compile any rendering files
PREVENT_RENDERING := 1

endif

# ----- compiling on the lab desktop PC ----- #
ifeq ($(filter lab, $(MAKECMDGOALS)), lab)

# set this command goal as a phony target (important)
.PHONY: lab

# what machine are we compiling for
MACHINE = luke-PC

# mjcf files location (model files like gripper/objects)
MJCF_PATH = /home/luke/mymujoco/mjcf

# local machine library locations
PYTHON_PATH = /usr/include/python3.6m
PYBIND_PATH = /home/luke/mymujoco/libs/pybind11
ARMA_PATH = # none, use system library
MUJOCO_PATH = /home/luke/mymujoco/libs/mujoco/mujoco-2.1.5
MUJOCO_LIB = $(MUJOCO_PATH)/lib
RENDER_PATH = # none, use system library
CORE_LIBS = -L$(MUJOCO_LIB) -lmujoco -larmadillo 
RENDER_LIBS = -lglfw
DEFINE_VAR = -DLUKE_MJCF_PATH='"$(MJCF_PATH)"' \
						 -DLUKE_MACHINE='"$(MACHINE)"'

# extras
MAKEFLAGS += -j8 # jN => use N parallel cores

endif

# ----- compiling on the lab operator PC ----- #
ifeq ($(filter lab-op, $(MAKECMDGOALS)), lab-op)

# set this command goal as a phony target (important)
.PHONY: lab-op

# what machine are we compiling for
MACHINE = operator-PC

# mjcf files location (model files like gripper/objects)
MJCF_PATH = /home/luke/luke-gripper-mujoco/mjcf

# local machine library locations
PYTHON_PATH = /usr/include/python3.8
PYBIND_PATH = /home/luke/luke-gripper-mujoco/libs/pybind11
ARMA_PATH = # none, use system library
MUJOCO_PATH = /home/luke/luke-gripper-mujoco/libs/mujoco/mujoco-2.1.5
MUJOCO_LIB = $(MUJOCO_PATH)/lib
RENDER_PATH = # none, use system library
CORE_LIBS = -L$(MUJOCO_LIB) -lmujoco -larmadillo 
RENDER_LIBS = -lglfw
DEFINE_VAR = -DLUKE_MJCF_PATH='"$(MJCF_PATH)"' \
						 -DLUKE_MACHINE='"$(MACHINE)"'

# extras
MAKEFLAGS += -j8 # jN => use N parallel cores

endif

# ----- compiling on the lab zotac PC ----- #
ifeq ($(filter zotac, $(MAKECMDGOALS)), zotac)

# set this command goal as a phony target (important)
.PHONY: zotac

# what machine are we compiling for
MACHINE = zotac-PC

# mjcf files location (model files like gripper/objects)
MJCF_PATH = /home/luke/mymujoco/mjcf

# local machine library locations
PYTHON_PATH = /usr/include/python3.6m
PYBIND_PATH = /home/luke/mymujoco/libs/pybind11
ARMA_PATH = # none, use system library
MUJOCO_PATH = /home/luke/mymujoco/libs/mujoco/mujoco-2.1.5
MUJOCO_LIB = $(MUJOCO_PATH)/lib
RENDER_PATH = # none, use system library
CORE_LIBS = -L$(MUJOCO_LIB) -lmujoco -larmadillo 
RENDER_LIBS = -lglfw
DEFINE_VAR = -DLUKE_MJCF_PATH='"$(MJCF_PATH)"' \
			 -DLUKE_MACHINE='"$(MACHINE)"'

# extras
MAKEFLAGS += -j8 # jN => use N parallel cores

endif