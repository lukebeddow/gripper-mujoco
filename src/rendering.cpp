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

// for rgbd camera
mjvCamera cam_rgbd;
mjvOption opt_rgbd;
mjvScene scn_rgbd;
mjrContext con_rgbd;

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

// will we put items into the render window GUI
bool plot_sensors = true;
bool plot_rgbd = true;

// what is initialised
bool window_initialised = false;
bool camera_initialised = false;

// size defaults
int window_width = 1200;
int window_height = 900;
int camera_width = 848;
int camera_height = 480;

// data to return out of read_rgbd()
luke::RGBD rgbd_data;

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
    mjv_defaultCamera(&cam_rgbd);
    mjv_defaultOption(&opt_rgbd);
    mjv_defaultScene(&scn_rgbd);
    mjr_defaultContext(&con_rgbd);

    cam_rgbd.type = mjCAMERA_FIXED;
    cam_rgbd.fixedcamid = 0;

    // create a new invisible window for the camera
    create_camera_window(camera_width, camera_height);

    camera_initialised = true;
}

void init_window(MjClass& myMjClass)
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
    
    // set window size and visibility
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);

    // create window, make OpenGL context current, request v-sync
    window = glfwCreateWindow(window_width, window_height, "luke-gripper-mujoco", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

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
    if (not m or not d or not window_initialised) {
        mju_error_s("Error: %s", "Render has been called without first running init");
    }

    if (!glfwWindowShouldClose(window)) {

        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // added - render UIs
        if (plot_sensors) {
            mjrRect rect = uistate.rect[3];
            mjui_render(&ui0, &uistate, &con);
            lukesensorfigsupdate();
            lukesensorfigshow(rect);
        }

        if (plot_rgbd and camera_initialised) {
            render_rgbd_feed();
        }

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();

        return true;
    }

    return false;
}

void render_rgbd_feed()
{
    /* render the rgbd images in the regular rendering window */

    // scale the depth values to show more colour variety, assumes depth never exceeds 1.0/x metres
    constexpr float depthstretch = 4;

    int W = camera_width;
    int H = camera_height;

    int Wt = 300;
    int scale = Wt / camera_width;

    int X = camera_height * camera_width * scale * scale;
    luke::rgbint* rgb_ = (luke::rgbint*)std::malloc(3 * X);
    luke::rgbint* depth_ = (luke::rgbint*)std::malloc(3 * X);

    for (int r = 0; r < H; r++) {
        for (int c = 0; c < W; c++) {
            
            // pixel location in our original image
            int adr_old = r*W + c;

            // loop and copy that pixel in a grid to make the image larger
            for (int i = 0; i < scale; i++) {
                for (int j = 0; j < scale; j++) {

                    // copy the current value into the same position but on new rows just below
                    int new_row = r*scale + i;
                    int new_col = c*scale + j;
                    int new_adr = new_row*W*scale + new_col;

                    rgb_[3 * new_adr + 0] = rgbd_data.rgb[3 * adr_old + 0];
                    rgb_[3 * new_adr + 1] = rgbd_data.rgb[3 * adr_old + 1];
                    rgb_[3 * new_adr + 2] = rgbd_data.rgb[3 * adr_old + 2];

                    depth_[3 * new_adr + 0] = (luke::rgbint)((1.0f - rgbd_data.depth[adr_old] * depthstretch) * 255.0f);
                    depth_[3 * new_adr + 1] = (luke::rgbint)((1.0f - rgbd_data.depth[adr_old] * depthstretch) * 255.0f);
                    depth_[3 * new_adr + 2] = (luke::rgbint)((1.0f - rgbd_data.depth[adr_old] * depthstretch) * 255.0f);
                }
            }
        }
    }

    // draw the rgb and depth onto the viewing window
    mjrRect rgb_viewport =  {0, window_height - H*scale, W*scale, H*scale};
    mjr_drawPixels(rgb_, NULL, rgb_viewport, &con);

    mjrRect depth_viewport =  {0, window_height - H*scale*2, W*scale, H*scale};
    mjr_drawPixels(depth_, NULL, depth_viewport, &con);

    // // this code overwrites the rgb_ data, which we don't want
    // const int NS = 3;           // depth_ image sub-sampling
    // for (int r=0; r<H; r+=NS)
    //     for (int c=0; c<W; c+=NS) {
    //         int adr = (r/NS)*W + c/NS;
    //         rgb_[3*adr] = rgb_[3*adr+1] = rgb_[3*adr+2] = (unsigned char)((1.0f-depth_[r*W+c])*255.0f);
    //     }

    // // if we have a window, draw the pixels on screen
    // mjrRect viewport2 =  {0, 0, 0, 0};
    // viewport2.height = rect.height/2;
    // viewport2.width = rect.width/2;
    // mjr_drawPixels(rgb_, NULL, viewport2, &con);

    // const int NS = 3;           // depth_ image sub-sampling
    // for (int r=0; r<H; r+=NS)
    //     for (int c=0; c<W; c+=NS) {
    //         int adr = (r/NS)*W + c/NS;
    //         rgb_[3*adr] = (unsigned char)((1.0f-rgb_[3*r*W+c])*255.0f);
    //         rgb_[3*adr+1] = (unsigned char)((1.0f-rgb_[3*r*W+c + 1])*255.0f);
    //         rgb_[3*adr+2] = (unsigned char)((1.0f-rgb_[3*r*W+c + 2])*255.0f);

    //         // rgb_[3*adr] = rgb_[3*adr+1] = rgb_[3*adr+2] = (unsigned char)((1.0f-depth_[r*W+c])*255.0f);
    //     }

    // // if we have a window, draw the pixels on screen
    // mjrRect viewport3 =  rect;
    // viewport3.bottom = rect.height/2;
    // viewport3.height = rect.height/2;
    // viewport3.width = rect.width/2;
    // mjr_drawPixels(rgb_, NULL, viewport3, &con);

    // reallocate data buffers
    if (rgb_ != NULL) std::free(rgb_);
    if (depth_ != NULL) std::free(depth_);
}

// render the camera
bool render_camera()
{
    if (not m or not d or not camera_initialised) {
        mju_error_s("Error: %s", "render_camera() has been called without first running init");
    }

    if (!glfwWindowShouldClose(camera_window)) {

        // get framebuffer viewport
        mjrRect viewport = {0, 0, camera_width, camera_height};
        // glfwGetFramebufferSize(camera_window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt_rgbd, NULL, &cam_rgbd, mjCAT_ALL, &scn_rgbd);
        mjr_render(viewport, &scn_rgbd, &con_rgbd);

        return true;
    }

    return false;
}

// before closing
void finish_window()
{
    // free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);
}

void finish_camera()
{
    // free visualization storage
    mjv_freeScene(&scn_rgbd);
    mjr_freeContext(&con_rgbd);
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
    int gnum = 50;

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

void resize_camera_window(int width, int height)
{
    /* resize a window */

    if (camera_window == NULL) {
        throw std::runtime_error("render::resize_window found NULL camera_window pointer");
    }

    glfwDestroyWindow(camera_window);
    create_camera_window(width, height);
}

void create_camera_window(int width, int height)
{
    /* create a graphical window, and indicate if it is visible. Also allocate
    the rgb and depth buffers for if we want to use this window as a camera */

    // save the window size to global variables
    camera_width = width;
    camera_height = height;

    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    // offscreen rendering requires to set the buffer size, default is 640x480
    /* In the XML: <visual> <global offwidth="640" offheight="480"/></visual> */
    // offscreen buffer appears to be slower than regular
    // glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);

    // create window, make OpenGL context current, request v-sync
    camera_window = glfwCreateWindow(camera_width, camera_height, "rgbd-camera-window", NULL, NULL);
    glfwMakeContextCurrent(camera_window);

    // create scene and context
    mjv_makeScene(m, &scn_rgbd, 2000);
    mjr_makeContext(m, &con_rgbd, mjFONTSCALE_150);

    // // set rendering to offscreen buffer
    // mjr_setBuffer(mjFB_OFFSCREEN, &con_rgbd);
    // if (con_rgbd.currentBuffer!=mjFB_OFFSCREEN) {
    //     std::printf("Warning: offscreen rendering not supported, using default/window framebuffer\n");
    // }

    // clear and resize vectors storing rgbd data
    int X = camera_width * camera_height;
    rgbd_data.rgb.clear();
    rgbd_data.depth.clear();
    rgbd_data.rgb.resize(3 * X);
    rgbd_data.depth.resize(X);
}

luke::RGBD read_rgbd()
{
    /* get an rgbd image out of the simulation */
    
    if (not camera_initialised) {
        throw std::runtime_error("render::read_rgbd() called but camera not initialised");
    }

    int W = camera_width;
    int H = camera_height;

    mjrRect rect = {0, 0, W, H};

    // read depth camera directly into std::vector by passing pointer
    mjr_readPixels(&rgbd_data.rgb[0], &rgbd_data.depth[0], rect, &con_rgbd);

    return rgbd_data;
}

} // namespace render