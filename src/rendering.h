#ifndef RENDERING_H_
#define RENDERING_H_

#include "mjxmacro.h"
#include "uitools.h"
#include "stdio.h"
#include "string.h"
// #include "glfw3.h"
#include "stdlib.h"
#include "GLFW/glfw3.h"

#include <thread>
#include <mutex>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>

#include "mjclass.h"
class MjClass;

namespace render
{

// functions
void mouse_button(GLFWwindow* window, int button, int act, int mods);
void mouse_move(GLFWwindow* window, double xpos, double ypos);
void scroll(GLFWwindow* window, double xoffset, double yoffset);
void init(mjModel* model, mjData* data);
void uiLayout(mjuiState* state);
void reload_for_rendering(mjModel* model, mjData* data);
bool render(MjClass& myMjClass);
void finish();

void lukesensorfigsinit();
void lukesensorfigsupdate(MjClass& myMjClass);
void lukesensorfigshow(mjrRect rect);

}

#endif // RENDERING_H_