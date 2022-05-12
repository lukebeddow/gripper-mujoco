# This file details the paths used for compiling. In order to compile
# on your computer you will need to add your specific paths and the
# locations of the libraries. Copy one of the ifeq...endif blocks,
# change the name eg cluster->yourname, then edit the paths.

# ----- default settings, overwritten by any of the below options ----- #

# what machine are we compiling for
MACHINE = luke-laptop

# mjcf files location (model files like gripper/objects)
MJCF_PATH = /home/luke/mymujoco/mjcf

# local machine library locations
PYTHON_PATH = /usr/include/python3.6m
PYBIND_PATH = /home/luke/pybind11
ARMA_PATH = # none, use system library
MUJOCO_PATH = /home/luke/.mujoco/mujoco210
RENDER_PATH = # none, use system library
CORE_LIBS = -L$(MUJOCO_PATH)/bin -lmujoco210 -larmadillo
RENDER_LIBS = -lGL -lglew $(MUJOCO_PATH)/bin/libglfw.so.3
DEFINE_VAR = -DLUKE_MJCF_PATH='"$(MJCF_PATH)"' \
						 -DLUKE_MACHINE='"$(MACHINE)"'

# ----- compiling on the cluster with the old mujoco version ----- #
ifeq ($(filter cluster-old, $(MAKECMDGOALS)), cluster-old)

.PHONY: cluster-old
cluster-old: cluster

# what machine are we compiling for
MACHINE = cluster

# cluster mjcf files location (model files like gripper/objects)
MJCF_PATH = /home/lbeddow/mymujoco/mjcf

# cluster library locations
PYTHON_PATH = /share/apps/python-3.6.9/include/python3.6m
PYBIND_PATH = /home/lbeddow/clusterlibs/pybind11
ARMA_PATH = /home/lbeddow/clusterlibs/armadillo-code
MUJOCO_PATH = /home/lbeddow/clusterlibs/mujoco/mujoco210
RENDER_PATH = # none, use system library
CORE_LIBS = -L$(MUJOCO_PATH)/bin -lmujoco210 -lblas -llapack
RENDER_LIBS = -lGL -lglew $(MUJOCO_PATH)/bin/libglfw.so.3
DEFINE_VAR = -DLUKE_CLUSTER -DARMA_DONT_USE_WRAPPER \
						 -DLUKE_MJCF_PATH='"$(MJCF_PATH)"' \
						 -DLUKE_MACHINE='"$(MACHINE)"'

endif

# ----- compiling on the cluster, new mujoco version ----- #
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
RENDER_PATH = /home/lbeddow/clusterlibs/glfw
CORE_LIBS = -L$(MUJOCO_PATH)/lib/ -lmujoco -lblas -llapack
RENDER_LIBS =
# RENDER_LIBS = -lGL -lglew $(MUJOCO_PATH)/bin/libglfw.so.3
DEFINE_VAR = -DLUKE_CLUSTER -DARMA_DONT_USE_WRAPPER \
						 -DLUKE_MJCF_PATH='"$(MJCF_PATH)"' \
						 -DLUKE_MACHINE='"$(MACHINE)"'

# we do not want to compile any rendering files
PREVENT_RENDERING := 1

endif

# ----- compiling on lukes laptop, old mujoco version ----- #
ifeq ($(filter luke-old, $(MAKECMDGOALS)), luke-old)

# set this command goal as a phony target (important)
.PHONY: luke-old

# what machine are we compiling for
MACHINE = luke-laptop

# mjcf files location (model files like gripper/objects)
MJCF_PATH = /home/luke/mymujoco/mjcf

# local machine library locations
PYTHON_PATH = /usr/include/python3.6m
PYBIND_PATH = /home/luke/pybind11
ARMA_PATH = # none, use system library
MUJOCO_PATH = /home/luke/.mujoco/mujoco210
RENDER_PATH = # none, use system library
CORE_LIBS = -L$(MUJOCO_PATH)/bin -lmujoco210 -larmadillo
RENDER_LIBS = -lGL -lglew $(MUJOCO_PATH)/bin/libglfw.so.3
DEFINE_VAR = -DLUKE_MJCF_PATH='"$(MJCF_PATH)"' \
						 -DLUKE_MACHINE='"$(MACHINE)"'

# extras
MAKEFLAGS += -j8 # jN => use N parallel cores

endif

# ----- compiling on lukes laptop, new mujoco version ----- #
ifeq ($(filter luke, $(MAKECMDGOALS)), luke)

# set this command goal as a phony target (important)
.PHONY: luke

# what machine are we compiling for
MACHINE = luke-laptop

# mjcf files location (model files like gripper/objects)
MJCF_PATH = /home/luke/mymujoco/mjcf

# local machine library locations
PYTHON_PATH = /usr/include/python3.6m
PYBIND_PATH = /home/luke/pybind11
ARMA_PATH = # none, use system library
MUJOCO_PATH = /home/luke/mujoco-2.1.5
RENDER_PATH = #/home/luke/clusterlibs/glfw
CORE_LIBS = -L$(MUJOCO_PATH)/lib -lmujoco -larmadillo
RENDER_LIBS = -lglfw
DEFINE_VAR = -DLUKE_MJCF_PATH='"$(MJCF_PATH)"' \
						 -DLUKE_MACHINE='"$(MACHINE)"'

# extras
MAKEFLAGS += -j8 # jN => use N parallel cores

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
PYBIND_PATH = /home/luke/pybind11
ARMA_PATH = # none, use system library
MUJOCO_PATH = /home/luke/mujoco-2.1.5
RENDER_PATH = # none, use system library
CORE_LIBS = -L$(MUJOCO_PATH)/lib -lmujoco -larmadillo 
RENDER_LIBS = -lglfw #-lGL -lGLU
DEFINE_VAR = -DLUKE_MJCF_PATH='"$(MJCF_PATH)"' \
						 -DLUKE_MACHINE='"$(MACHINE)"'

# extras
MAKEFLAGS += -j8 # jN => use N parallel cores

endif