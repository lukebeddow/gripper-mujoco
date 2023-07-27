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
GLFWwindow* camera_window = NULL;

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;

// will we put items into the render window
bool plot_sensors = true;
bool plot_camera_feed = true;

// what will we render to the GUI screen
bool render_rgb_flag = false;
bool render_depth_flag = false;

// what is initialised
bool camera_initialised = false;
bool window_initialised = false;

// window size
int window_width = 1200;
int window_height = 900;
int camera_width = 640;
int camera_height = 480;

// data storage
unsigned char* rgb_ = NULL;
float* depth_ = NULL;

// global viewport object
mjrRect window_viewport = {0, 0, 0, 0};
mjrRect camera_viewport = {0, 0, 0, 0};

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

void init_camera(MjClass& myMjClass)
{
    /* initialise the rendering backend (eg for read_RGBD_pixels)*/

    MjPtr = &myMjClass;

    // update our global m/d to this most recent update
    m = MjPtr->model;
    d = MjPtr->data;
    
    // init GLFW
    if( !glfwInit() ) {
        mju_error("Could not initialize GLFW");
    }

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // set the window hidden by default, create with default size
    create_camera_window(camera_width, camera_height);
    
    // // set rendering to offscreen buffer
    // mjr_setBuffer(mjFB_OFFSCREEN, &con);
    // if (con.currentBuffer!=mjFB_OFFSCREEN) {
    //     std::printf("Warning: offscreen rendering not supported, using default/window framebuffer\n");
    // }

    // set to use the fixed camera in xml
    cam.type = mjCAMERA_FIXED;
    cam.fixedcamid = 0;

    camera_initialised = true;
}

// initialise window
void init_window(MjClass& myMjClass)
{
    /* initialise the viewing window as a GUI for rendered scenes */

    MjPtr = &myMjClass;

    // update our global m/d to this most recent update
    m = MjPtr->model;
    d = MjPtr->data;

    // init GLFW
    if( !glfwInit() ) {
        mju_error("Could not initialize GLFW");
    }

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // set a new window size and make it visible
    int window_height = 900;
    int window_width = 1200;
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);

    // create window, make OpenGL context current, request v-sync
    window = glfwCreateWindow(window_width, window_height, "luke-gripper-mujoco", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // create the viewport for this window size
    window_viewport =  mjr_maxViewport(&con);

    // ensure the window is shown onscreen
    glfwShowWindow(window);

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

    window_initialised = true;
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
bool render_window()
{
    if (not m or not d) {
        mju_error_s("Error: %s", "Render has been called without first running init");
    }

    if (not window_initialised) {
        std::runtime_error("render_window() called without first calling init_window()");
    }

    if (!glfwWindowShouldClose(window)) {

        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);

        // this stops a crash, but nothing new renders on screen
        // // swap OpenGL buffers (blocking call due to v-sync)
        // glfwSwapBuffers(window);
  
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

// render the scene
void render_camera()
{

    if (not m or not d) {
        mju_error_s("Error: %s", "Render has been called without first running init");
    }

    if (not camera_initialised) {
        std::runtime_error("render_camera() called without first calling init_camera()");
    }

    // update scene and render
    mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
    mjr_render(camera_viewport, &scn, &con);

    // // swap OpenGL buffers (blocking call due to v-sync)
    // glfwSwapBuffers(camera_window);

    // glfwPollEvents();
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

void resize_camera(int width, int height)
{
    /* resize the camera window */

    if (camera_window == NULL) {
        throw std::runtime_error("render::resize_camera() found NULL camera pointer");
    }

    glfwDestroyWindow(camera_window);
    create_camera_window(width, height);
}

void create_camera_window(int width, int height)
{
    /* create a graphical window, and indicate if it is visible. Also allocate
    the rgb and depth buffers for if we want to use this window as a camera */

    // save global indicators of height and width
    camera_width = width;
    camera_height = height;

    // glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);

    // we don't want the camera window to be visibile
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    // create window, make OpenGL context current, request v-sync
    camera_window = glfwCreateWindow(width, height, "mj-camera-window", NULL, NULL);
    if (not camera_window) {
        mju_error("Could not create GLFW camera window");
    }
    glfwMakeContextCurrent(camera_window);
    // glfwSwapInterval(1);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // // set rendering to offscreen buffer
    // mjr_setBuffer(mjFB_OFFSCREEN, &con);
    // if (con.currentBuffer!=mjFB_OFFSCREEN) {
    //     std::printf("Warning: offscreen rendering not supported, using default/window framebuffer\n");
    // }

    // create the viewport for the camera
    camera_viewport =  mjr_maxViewport(&con);

    // reallocate data buffers
    if (rgb_ != NULL) std::free(rgb_);
    if (depth_ != NULL) std::free(depth_);

    int X = height * width;
    rgb_ = (unsigned char*)std::malloc(3 * X);
    depth_ = (float*)std::malloc(sizeof(float) * X);
}

luke::RGBD read_rgbd()
{
    /* get an rgbd image out of the simulation */
    
    if (not camera_initialised and (rgb_ == NULL or depth_ == NULL)) {
        throw std::runtime_error("render::read_rgbd() called but rendering not initialised");
    }

    int H = camera_height;
    int W = camera_width;

    // mjrRect cam_viewport = {0, 0, W, H};

    // I should make this better, so rgb/depth dont have to be copied everytime
    luke::RGBD output;

    if (!rgb_ || !depth_) {
        mju_error("render::read_rgbd() failed, could not allocate buffers for rgb_ or depth_");
    }

    // read the depth camera using mujoco
    mjr_readPixels(rgb_, depth_, camera_viewport, &con);

    for (int i = 0;  i < 3*W*H; i++) {
        output.rgb.push_back((luke::rgbint)rgb_[i]);
    }

    for (int i = 0;  i < W*H; i++) {
        output.depth.push_back(depth_[i]);
    }

    // // this code overwrites the rgb_ data, which we don't want
    // if (render_depth_flag) {

    //     std::cout << "2.1\n";

    //     const int NS = 3;           // depth_ image sub-sampling
    //     for (int r=0; r<H; r+=NS)
    //         for (int c=0; c<W; c+=NS) {
    //             int adr = (r/NS)*W + c/NS;
    //             rgb_[3*adr] = rgb_[3*adr+1] = rgb_[3*adr+2] = (unsigned char)((1.0f-depth_[r*W+c])*255.0f);
    //         }

    //     // if we have a window, draw the pixels on screen
    //     if (window_visible) {
    //         mjrRect viewport2 =  rect;
    //         viewport2.height = rect.height/2;
    //         viewport2.width = rect.width/2;
    //         mjr_drawPixels(rgb_, NULL, viewport2, &con);
    //     }
    // }

    // if (render_rgb_flag) {

    //     std::cout << "2.2\n";

    //     const int NS = 3;           // depth_ image sub-sampling
    //     for (int r=0; r<H; r+=NS)
    //         for (int c=0; c<W; c+=NS) {
    //             int adr = (r/NS)*W + c/NS;
    //             rgb_[3*adr] = (unsigned char)((1.0f-rgb_[3*r*W+c])*255.0f);
    //             rgb_[3*adr+1] = (unsigned char)((1.0f-rgb_[3*r*W+c + 1])*255.0f);
    //             rgb_[3*adr+2] = (unsigned char)((1.0f-rgb_[3*r*W+c + 2])*255.0f);

    //             // rgb_[3*adr] = rgb_[3*adr+1] = rgb_[3*adr+2] = (unsigned char)((1.0f-depth_[r*W+c])*255.0f);
    //         }

    //     // if we have a window, draw the pixels on screen
    //     if (window_visible) {
    //         mjrRect viewport3 =  rect;
    //         viewport3.bottom = rect.height/2;
    //         viewport3.height = rect.height/2;
    //         viewport3.width = rect.width/2;
    //         mjr_drawPixels(rgb_, NULL, viewport3, &con);
    //     }
    // }

    return output;
}

void render_rgb(bool set_as)
{
    render_rgb_flag = set_as;
}

void render_depth(bool set_as)
{
    render_depth_flag = set_as;
}

} // namespace render