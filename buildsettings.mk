# This file details the paths used for compiling. In order to compile
# on your computer you will need to add your specific paths and the
# locations of the libraries. Copy one of the ifeq...endif blocks,
# change the name eg cluster->yourname, then edit the paths. In the full
# Makefile you should then add a PHONY target called yourname.

# ----- default settings, easily overwritten ----- #

# mjcf files location (model files like gripper/objects)
MJCF_PATH = /home/luke/mymujoco/mjcf/object_set_1

# local machine library locations
PYTHON_PATH = /usr/include/python3.6m
PYBIND_PATH = /home/luke/pybind11
ARMA_PATH = # none, use system library
MUJOCO_PATH = /home/luke/.mujoco/mujoco210
CORE_LIBS = -lmujoco210 -larmadillo
RENDER_LIBS = -lGL -lglew $(MUJOCO_PATH)/bin/libglfw.so.3
DEFINE_VAR = -DLUKE_MJCF_PATH='"$(MJCF_PATH)"'

# ----- compiling on the cluster ----- #
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
RENDER_LIBS = -lGL -lglew $(MUJOCO_PATH)/bin/libglfw.so.3
DEFINE_VAR = -DLUKE_CLUSTER -DARMA_DONT_USE_WRAPPER -DLUKE_MJCF_PATH='"$(MJCF_PATH)"'

endif

# ----- compiling on lukes laptop ----- #
ifeq ($(filter luke, $(MAKECMDGOALS)), luke)

# mjcf files location (model files like gripper/objects)
MJCF_PATH = /home/luke/mymujoco/mjcf/object_set_1

# local machine library locations
PYTHON_PATH = /usr/include/python3.6m
PYBIND_PATH = /home/luke/pybind11
ARMA_PATH = # none, use system library
MUJOCO_PATH = /home/luke/.mujoco/mujoco210
CORE_LIBS = -lmujoco210 -larmadillo
RENDER_LIBS = -lGL -lglew $(MUJOCO_PATH)/bin/libglfw.so.3
DEFINE_VAR = -DLUKE_MJCF_PATH='"$(MJCF_PATH)"'

# extras
MAKEFLAGS += -j8 # jN => use N parallel cores

endif

# ----- compiling on the lab desktop PC ----- #
ifeq ($(filter lab, $(MAKECMDGOALS)), lab)

# mjcf files location (model files like gripper/objects)
MJCF_PATH = /home/luke/luke-gripper-mujoco/mjcf/object_set_1

# local machine library locations
PYTHON_PATH = /usr/include/python3.6m
PYBIND_PATH = /home/luke/pybind11
ARMA_PATH = # none, use system library
MUJOCO_PATH = /home/luke/mujoco-2.1.5
CORE_LIBS = -larmadillo -L$(MUJOCO_PATH)/lib/ -lmujoco
RENDER_LIBS = -lGL -lglfw -lGLU
DEFINE_VAR = -DLUKE_MJCF_PATH='"$(MJCF_PATH)"'

# extras
MAKEFLAGS += -j8 # jN => use N parallel cores

endif