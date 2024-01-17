#ifndef RENDERING_H_
#define RENDERING_H_

#include "mjxmacro.h"
#include "uitools.h"
#include "stdio.h"
#include "string.h"
#include "stdlib.h"
#include "GLFW/glfw3.h"

#include <thread>
#include <mutex>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>

// forward declaration to prevent circular dependency
#include "mjclass.h"
class MjClass;

namespace render
{

// functions
void mouse_button(GLFWwindow* window, int button, int act, int mods);
void mouse_move(GLFWwindow* window, double xpos, double ypos);
void scroll(GLFWwindow* window, double xoffset, double yoffset);
void init_camera(MjClass& myMjClass);
void init_window(MjClass& myMjClass);
void resize_camera_window(int width, int height);
void create_camera_window(int width, int height);
void uiLayout(mjuiState* state);
void reload_for_rendering(MjClass& myMjClass);
bool render_window();
bool render_camera();
bool render_camera_with_seg_mask();
void render_rgbd_feed();
void finish_window();
void finish_camera();

void lukesensorfigsinit();
void lukesensorfigsupdate();
void lukesensorfigshow(mjrRect rect);

luke::RGBD read_rgbd();
luke::RGBD read_mask();

}

#endif // RENDERING_H_