#include "rendering.h"

namespace render
{

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

// mjclass for plotting sensor data
MjClass* MjPtr = NULL;

// for plotting
bool plot_sensors = true;
mjvFigure figgauges;
mjvFigure figbendgauge;
mjvFigure figaxialgauge;
mjvFigure figpalm;
mjvFigure figwrist;
mjvFigure figmotors;
mjuiState uistate;
mjUI ui0;

// window
GLFWwindow* window = NULL;

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;

// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}

// initialise window
void init(MjClass& myMjClass)
{
    MjPtr = &myMjClass;

    // update our global m/d to this most recent update
    m = MjPtr->model;
    d = MjPtr->data;

    // init GLFW
    if( !glfwInit() )
        mju_error("Could not initialize GLFW");

    // create window, make OpenGL context current, request v-sync
    window = glfwCreateWindow(1200, 900, "luke-gripper-mujoco", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // install GLFW mouse and keyboard callbacks
    // glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    // init state and uis
    memset(&uistate, 0, sizeof(mjuiState));
    memset(&ui0, 0, sizeof(mjUI));
    ui0.spacing = mjui_themeSpacing(0);
    ui0.color = mjui_themeColor(0);
    // ui0.predicate = uiPredicate;
    ui0.rectid = 1;
    ui0.auxid = 0;

    uiLayout(&uistate);

    if (plot_sensors) lukesensorfigsinit();
}

// set window layout
void uiLayout(mjuiState* state)
{
    mjrRect* rect = state->rect;

    // set number of rectangles
    state->nrect = 4;

    // rect 0: entire framebuffer
    rect[0].left = 0;
    rect[0].bottom = 0;
    glfwGetFramebufferSize(window, &rect[0].width, &rect[0].height);

    // rect 1: UI 0
    rect[1].left = 0;
    rect[1].width = 1; //settings.ui0 ? ui0.width : 0;
    rect[1].bottom = 0;
    rect[1].height = rect[0].height;

    // rect 2: UI 1
    rect[2].width = 1; //settings.ui1 ? ui1.width : 0;
    rect[2].left = mjMAX(0, rect[0].width - rect[2].width);
    rect[2].bottom = 0;
    rect[2].height = rect[0].height;

    // rect 3: 3D plot (everything else is an overlay)
    rect[3].left = rect[1].width;
    rect[3].width = mjMAX(0, rect[0].width - rect[1].width - rect[2].width);
    rect[3].bottom = 0;
    rect[3].height = rect[0].height;
}

void reload_for_rendering(MjClass& myMjClass)
{
    /* reload the scene after the model and data have changed */

    MjPtr = &myMjClass;

    m = MjPtr->model;
    d = MjPtr->data;

    // re-create scene and context
    int maxgeom = 5000;
    mjv_makeScene(m, &scn, maxgeom);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // // re-render
    // render(model, data);
}

// render the scene
bool render()
{
    if (not m or not d) {
        mju_error_s("Error: %s", "Render has been called without first running init");
    }

    // // update our global m/d to this most recent update
    // m = myMjClass.model;
    // d = myMjClass.data;


    if (!glfwWindowShouldClose(window)) {

        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);

        // std::cout << "before crash\n";
        mjr_render(viewport, &scn, &con);
        // std::cout << "after crash\n";

        // added - render UIs
        if (plot_sensors) {
            mjrRect rect = uistate.rect[3];
            mjui_render(&ui0, &uistate, &con);
            lukesensorfigsupdate();
            lukesensorfigshow(rect);
        }

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();

        return true;
    }

    return false;
}

// before closing
void finish()
{
    // free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);
}

void lukesensorfigsinit(void)
{
    // create default figures
    mjv_defaultFigure(&figbendgauge);
    mjv_defaultFigure(&figaxialgauge);
    mjv_defaultFigure(&figpalm);
    mjv_defaultFigure(&figwrist);
    mjv_defaultFigure(&figmotors);

    // what figures are we initialising
    std::vector<mjvFigure*> myfigs {
        &figbendgauge, &figaxialgauge, &figpalm, &figwrist, &figmotors
    };

    // initialise all figures the same
    for (int i = 0; i < myfigs.size(); i++) {

        // create a default figure
        mjv_defaultFigure(myfigs[i]);

        // set flags
        myfigs[i]->flg_legend = 1;
        myfigs[i]->flg_extend = 1;
        myfigs[i]->flg_symmetric = 1;

        // y-tick number format
        strcpy(myfigs[i]->yformat, "%.0f");

        // grid size
        myfigs[i]->gridsize[0] = 2;
        myfigs[i]->gridsize[1] = 3;

        // minimum range
        myfigs[i]->range[0][0] = 0;
        myfigs[i]->range[0][1] = 0;
        myfigs[i]->range[1][0] = -1;
        myfigs[i]->range[1][1] = 1;
    }

    // add titles
    strcpy(figbendgauge.title, "Bending gauges");
    strcpy(figaxialgauge.title, "Axial gauges");
    strcpy(figpalm.title, "Palm sensor");
    strcpy(figwrist.title, "Wrist sensor");
    strcpy(figmotors.title, "Motor states");

    // add legends
    strcpy(figbendgauge.linename[0], "1");
    strcpy(figbendgauge.linename[1], "2");
    strcpy(figbendgauge.linename[2], "3");
    strcpy(figaxialgauge.linename[0], "1");
    strcpy(figaxialgauge.linename[1], "2");
    strcpy(figaxialgauge.linename[2], "3");
    strcpy(figpalm.linename[0], "P");
    strcpy(figwrist.linename[0], "X");
    strcpy(figwrist.linename[1], "Y");
    strcpy(figwrist.linename[2], "Z");
    strcpy(figmotors.linename[0], "X");
    strcpy(figmotors.linename[1], "Y");
    strcpy(figmotors.linename[2], "Z");
    strcpy(figmotors.linename[3], "H");
}

void lukesensorfigsupdate()
{
    if (not MjPtr) 
        throw std::runtime_error("lukesensorfigsupdate called with MjPtr=NULL");

    // amount of data we extract for each sensor
    int gnum = MjPtr->gauge_buffer_size;

    // check we can plot this amount of data
    if (gnum > mjMAXLINEPNT) {
        std::cout << "gnum exceeds mjMAXLINEPNT in gaugefigupdate()\n";
        gnum = mjMAXLINEPNT;
    }

    // read the data    
    std::vector<luke::gfloat> b1data = MjPtr->sim_sensors_.finger1_gauge.read(gnum);
    std::vector<luke::gfloat> b2data = MjPtr->sim_sensors_.finger2_gauge.read(gnum);
    std::vector<luke::gfloat> b3data = MjPtr->sim_sensors_.finger3_gauge.read(gnum);
    std::vector<luke::gfloat> p1data = MjPtr->sim_sensors_.palm_sensor.read(gnum);
    std::vector<luke::gfloat> a1data = MjPtr->sim_sensors_.finger1_axial_gauge.read(gnum);
    std::vector<luke::gfloat> a2data = MjPtr->sim_sensors_.finger2_axial_gauge.read(gnum);
    std::vector<luke::gfloat> a3data = MjPtr->sim_sensors_.finger3_axial_gauge.read(gnum);
    std::vector<luke::gfloat> wXdata = MjPtr->sim_sensors_.wrist_X_sensor.read(gnum);
    std::vector<luke::gfloat> wYdata = MjPtr->sim_sensors_.wrist_Y_sensor.read(gnum);
    std::vector<luke::gfloat> wZdata = MjPtr->sim_sensors_.wrist_Z_sensor.read(gnum);
    std::vector<luke::gfloat> mXdata = MjPtr->sim_sensors_.x_motor_position.read(gnum);
    std::vector<luke::gfloat> mYdata = MjPtr->sim_sensors_.y_motor_position.read(gnum);
    std::vector<luke::gfloat> mZdata = MjPtr->sim_sensors_.z_motor_position.read(gnum);
    std::vector<luke::gfloat> mHdata = MjPtr->sim_sensors_.z_base_position.read(gnum);

    // get the corresponding timestamps
    std::vector<float> btdata = MjPtr->gauge_timestamps.read(gnum);
    std::vector<float> atdata = MjPtr->axial_timestamps.read(gnum);
    std::vector<float> ptdata = MjPtr->palm_timestamps.read(gnum);
    std::vector<float> wtdata = MjPtr->wristZ_timestamps.read(gnum);
    std::vector<float> mtdata = MjPtr->step_timestamps.read(gnum);

    // package sensor data pointers in iterable vectors
    std::vector<std::vector<luke::gfloat>*> bdata {
        &b1data, &b2data, &b3data
    };
    std::vector<std::vector<luke::gfloat>*> adata {
        &a1data, &a2data, &a3data
    };
    std::vector<std::vector<luke::gfloat>*> pdata {
        &p1data
    };
    std::vector<std::vector<luke::gfloat>*> wdata {
        &wXdata, &wYdata, &wZdata
    };
    std::vector<std::vector<luke::gfloat>*> mdata {
        &mXdata, &mYdata, &mZdata, &mHdata  
    };

    // package figures into iterable vector
    std::vector<mjvFigure*> myfigs {
        &figbendgauge, &figaxialgauge, &figpalm, &figwrist, &figmotors
    };

    // package sensor data in same order as figures
    std::vector< std::vector<std::vector<luke::gfloat>*>* > sensordata {
        &bdata, &adata, &pdata, &wdata, &mdata
    };
    std::vector<std::vector<float>*> timedata {
        &btdata, &atdata, &ptdata, &wtdata, &mtdata
    };

    // maximum number of lines on a figure
    static const int maxline = 10;

    // loop through each figure/sensor type
    for (int i = 0; i < sensordata.size(); i++) {

        // start with line 0
        int lineid = 0;

        // loop over each set of sensor data (eg finger gauges 1, 2, then 3)
        for (int n = 0; n < sensordata[i]->size(); n++)
        {
            lineid = n;

            if (lineid >= maxline) 
                throw std::runtime_error("max number of figure lines exceeded");
        
            // data pointer in line (don't need this as sliding window does it for us)
            int p = myfigs[i]->linepnt[lineid];

            // loop over the data itself and save it in the figure
            for (int g = 0; g < gnum; g++) {
                // figgauges.linedata[lineid][2 * g] = tdata[g];
                // figgauges.linedata[lineid][2 * g + 1] = (*fdata[n])[g];

                myfigs[i]->linedata[lineid][2 * g] = (*timedata[i])[g];
                myfigs[i]->linedata[lineid][2 * g + 1] = (*(*sensordata[i])[n])[g];
            }

            // update linepnt (index of last data point)
            myfigs[i]->linepnt[lineid] = gnum;
        }
    }
}

void lukesensorfigshow(mjrRect rect)
{
    // what figures are showing
    std::vector<mjvFigure*> myfigs {
        &figbendgauge, &figpalm, &figwrist, &figmotors //, &figaxialgauge
    };

    // constant width with and without profiler
    int width = rect.width / myfigs.size();
    int show = 0;

    for (int i = 0; i < myfigs.size(); i++) {

        // another figure is showing
        show += 1;

        // render figure on the right
        mjrRect viewport = {
            rect.left + rect.width - width * show,
            rect.bottom,
            width,
            rect.height/3
        };

        mjr_figure(viewport, myfigs[i], &con);
    }
}



} // namespace render