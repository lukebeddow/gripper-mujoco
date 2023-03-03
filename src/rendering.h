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

// forward declaration to prevent circular dependency
#include "mjclass.h"
class MjClass;

namespace render
{

// functions
void mouse_button(GLFWwindow* window, int button, int act, int mods);
void mouse_move(GLFWwindow* window, double xpos, double ypos);
void scroll(GLFWwindow* window, double xoffset, double yoffset);
void init(MjClass& myMjClass);
void uiLayout(mjuiState* state);
void reload_for_rendering(MjClass& myMjClass);
bool render();
void finish();

void lukesensorfigsinit();
void lukesensorfigsupdate();
void lukesensorfigshow(mjrRect rect);

}

#endif // RENDERING_H_