#include "mjxmacro.h"
#include "uitools.h"
#include "stdio.h"
#include "string.h"

#include <thread>
#include <mutex>
#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <functional>

#include "myfunctions.h"
#include "mjclass.h"

/* This file replaces the standard mj_step(...) function with my own functions, which
are in the namespace "luke". See myfunctions.cpp */


//-------------------------------- global -----------------------------------------------

// constants
const int maxgeom = 5000;           // preallocated geom array in mjvScene
const double syncmisalign = 0.1;    // maximum time mis-alignment before re-sync
const double refreshfactor = 0.7;   // fraction of refresh available for simulation

// model and data
mjModel* m = NULL;
mjData* d = NULL;
char filename[1000] = "";

// added by luke
MjClass myMjClass;
int manual_steps = 0; // added for recording number or arrow key press steps, for testing

// sim thread synchronization
std::mutex mtx;

// abstract visualization
mjvScene scn;
mjvCamera cam;
mjvOption vopt;
mjvPerturb pert;
mjvFigure figconstraint;
mjvFigure figcost;
mjvFigure figtimer;
mjvFigure figsize;
mjvFigure figsensor;

// added by luke
mjvFigure figgauges;
mjvFigure figbendgauge;
mjvFigure figaxialgauge;
mjvFigure figpalm;
mjvFigure figwrist;
mjvFigure figmotors;
mjvFigure figstepperx;
mjvFigure figsteppery;
mjvFigure figstepperz;
mjvFigure figbasex;
mjvFigure figbasey;
mjvFigure figbasez;
mjvFigure figbaserotz;

// OpenGL rendering and UI
GLFWvidmode vmode;
int windowpos[2];
int windowsize[2];
mjrContext con;
GLFWwindow* window = NULL;
mjuiState uistate;
mjUI ui0, ui1;


// UI settings not contained in MuJoCo structures
struct
{
    // file
    int exitrequest = 0;

    // option
    int spacing = 0;
    int color = 0;
    int font = 0;
    int ui0 = 1;
    int ui1 = 1;
    int help = 0;
    int info = 0;
    int profiler = 0;
    int sensor = 0;
    int fullscreen = 0;
    int vsync = 1;
    int busywait = 0;

    int object_int = 0;            // added by luke
    int env_steps = 1;             // added by luke
    int object_x_noise_mm = 0;     // added by luke
    int object_y_noise_mm = 0;     // added by luke
    int object_z_rot_deg = 0;      // added by luke
    int complete_action_steps = 1; // added by luke
    double finger_thickness = 0.9;  // added by luke
    int seg_num_for_frc = 1;       // added by luke
    double seg_force = 0.0;         // added by luke
    double seg_moment = 0.0;        // added by luke
    int force_style = 0;            // added by luke
    int all_sensors_use_noise = 1;  // added by luke
    int scene_objects = 0;
    double scene_x = 0.5;
    double scene_y = 0.5;
    double motor_x = 1e3 * luke::Gripper::xy_home;
    double motor_y = 1e3 * luke::Gripper::xy_home;
    double motor_z = 1e3 * luke::Gripper::z_home;

    // action flags
    mjtNum action_motor_mm = 2;
    mjtNum action_motor_rad = 0.01;
    mjtNum action_base_mm = 2;
    mjtNum action_base_rad = 0.01;
    int gram_force = 0;               // added by luke

    // figure show flags
    int bendgauge = 0;             // added by luke
    int axialgauge = 0;            // added by luke
    int palmsensor = 0;            // added by luke
    int wristsensor = 0;           // added by luke
    int statesensor = 0;           // added by luke
    int allsensors = 0;            // added by luke
    int xstepfig = 0;              // added by luke
    int ystepfig = 0;              // added by luke
    int zstepfig = 0;              // added by luke
    int xbasefig = 0;              // added by luke
    int ybasefig = 0;              // added by luke
    int zbasefig = 0;              // added by luke
    int zbaserotfig = 0;
    int allactuators = 0;          // added by luke
    int use_SI_sensors = 0;        // added by luke
    int render_rgb = 0;
    int render_depth = 0;

    float sensor_mag = 0;
    float sensor_mu = 0;
    float sensor_std = 0;

    // simulation
    int run = 1;
    int key = 0;
    int loadrequest = 0;

    // watch
    char field[mjMAXUITEXT] = "qpos";
    int index = 0;

    // physics: need sync
    int disable[mjNDISABLE];
    int enable[mjNENABLE];

    // rendering: need sync
    int camera = 0;
} settings;


// section ids
enum
{
    // left ui
    SECT_FILE   = 0,
    SECT_OPTION,
    SECT_SIMULATION,
    SECT_WATCH,
	SECT_PHYSICS,
    SECT_RENDERING,
    SECT_GROUP,
    NSECT0,

    // right ui
    SECT_JOINT = 0,
    SECT_CONTROL,
    SECT_GRIPPER,   // added by luke
    SECT_OBJECT,    // added by luke
    SECT_ACTION,    // added by luke
    SECT_SETTINGS,  // added by luke
    NSECT1
};


// file section of UI
const mjuiDef defFile[] =
{
    {mjITEM_SECTION,   "File",          1, NULL,                    "AF"},
    {mjITEM_BUTTON,    "Save xml",      2, NULL,                    ""},
    {mjITEM_BUTTON,    "Save mjb",      2, NULL,                    ""},
    {mjITEM_BUTTON,    "Print model",   2, NULL,                    "CM"},
    {mjITEM_BUTTON,    "Print data",    2, NULL,                    "CD"},
    {mjITEM_BUTTON,    "Quit",          1, NULL,                    "CQ"},
    {mjITEM_END}
};


// option section of UI
const mjuiDef defOption[] =
{
    {mjITEM_SECTION,   "Option",        1, NULL,                    "AO"},
    {mjITEM_SELECT,    "Spacing",       1, &settings.spacing,       "Tight\nWide"},
    {mjITEM_SELECT,    "Color",         1, &settings.color,         "Default\nOrange\nWhite\nBlack"},
    {mjITEM_SELECT,    "Font",          1, &settings.font,          "50 %\n100 %\n150 %\n200 %\n250 %\n300 %"},
    {mjITEM_CHECKINT,  "Left UI (Tab)", 1, &settings.ui0,           " #258"},
    {mjITEM_CHECKINT,  "Right UI",      1, &settings.ui1,           "S#258"},
    {mjITEM_CHECKINT,  "Help",          2, &settings.help,          " #290"},
    {mjITEM_CHECKINT,  "Info",          2, &settings.info,          " #291"},
    {mjITEM_CHECKINT,  "Profiler",      2, &settings.profiler,      " #292"},
    {mjITEM_CHECKINT,  "Sensor",        2, &settings.sensor,        " #293"},
#ifdef __APPLE__
    {mjITEM_CHECKINT,  "Fullscreen",    0, &settings.fullscreen,    " #294"},
#else
    {mjITEM_CHECKINT,  "Fullscreen",    1, &settings.fullscreen,    " #294"},
#endif
    {mjITEM_CHECKINT,  "Vertical Sync", 1, &settings.vsync,         " #295"},
    {mjITEM_CHECKINT,  "Busy Wait",     1, &settings.busywait,      " #296"},
    {mjITEM_CHECKINT,  "Bend gauge",    2, &settings.bendgauge,     " #401"}, // added by luke
    {mjITEM_CHECKINT,  "Axial gauge",   2, &settings.axialgauge,    " #402"}, // added by luke
    {mjITEM_CHECKINT,  "Palm sensor",   2, &settings.palmsensor,    " #403"}, // added by luke
    {mjITEM_CHECKINT,  "Wrist sensor",  2, &settings.wristsensor,   " #404"}, // added by luke
    {mjITEM_CHECKINT,  "Motor sensor",  2, &settings.statesensor,   " #405"}, // added by luke
    {mjITEM_CHECKINT,  "All sensors",   2, &settings.allsensors,    " #406"}, // added by luke
    {mjITEM_CHECKINT,  "Use noise",     2, &settings.all_sensors_use_noise, " #407"},  // added by luke
    {mjITEM_SLIDERNUM, "sensor mag",    2, &myMjClass.s_.sensor_noise_mag,   "0.0 1.0"},   // added by luke
    {mjITEM_SLIDERNUM, "sensor mean",   2, &myMjClass.s_.sensor_noise_mu,    "0 1.0"},  // added by luke
    {mjITEM_SLIDERNUM, "sensor std",    2, &myMjClass.s_.sensor_noise_std,   "-0.1 1.0"},  // added by luke
    {mjITEM_SLIDERNUM, "state mag",     2, &myMjClass.s_.state_noise_mag,   "0.0 1.0"},   // added by luke
    {mjITEM_SLIDERNUM, "state mean",    2, &myMjClass.s_.state_noise_mu,    "0 1.0"},  // added by luke
    {mjITEM_SLIDERNUM, "state std",     2, &myMjClass.s_.state_noise_std,   "-0.1 1.0"},  // added by luke
    {mjITEM_CHECKINT,  "x stepper",     2, &settings.xstepfig,      " #408"}, // added by luke
    {mjITEM_CHECKINT,  "y stepper",     2, &settings.ystepfig,      " #409"}, // added by luke
    {mjITEM_CHECKINT,  "z stepper",     2, &settings.zstepfig,      " #410"}, // added by luke
    {mjITEM_CHECKINT,  "x base",        2, &settings.xbasefig,      " #411"}, // added by luke
    {mjITEM_CHECKINT,  "y base",        2, &settings.ybasefig,      " #411"}, // added by luke
    {mjITEM_CHECKINT,  "z base",        2, &settings.zbasefig,      " #411"}, // added by luke
    {mjITEM_CHECKINT,  "z base rot",    2, &settings.zbaserotfig,   " #411"}, // added by luke    
    {mjITEM_CHECKINT,  "all actuators", 2, &settings.allactuators,  " #412"}, // added by luke
    {mjITEM_CHECKINT,  "SI values",     2, &settings.use_SI_sensors," #413"}, // added by luke
    {mjITEM_CHECKINT,  "Render rgb",    2, &settings.render_rgb,    " #414"}, // added by luke
    {mjITEM_CHECKINT,  "Render depth",  2, &settings.render_depth,  " #415"}, // added by luke
    {mjITEM_END}
};

// simulation section of UI
const mjuiDef defSimulation[] =
{
    {mjITEM_SECTION,   "Simulation",    1, NULL,                    "AS"},
    {mjITEM_RADIO,     "",              2, &settings.run,           "Pause\nRun"},
    {mjITEM_BUTTON,    "Reset",         2, NULL,                    " #259"},
    {mjITEM_BUTTON,    "Reload",        2, NULL,                    "CL"},
    {mjITEM_BUTTON,    "Align",         2, NULL,                    "CA"},
    {mjITEM_BUTTON,    "Copy pose",     2, NULL,                    "CC"},
    {mjITEM_SLIDERINT, "Key",           3, &settings.key,           "0 0"},
    {mjITEM_BUTTON,    "Reset to key",  3},
    {mjITEM_BUTTON,    "Set key",       3},
    {mjITEM_END}
};


// watch section of UI
const mjuiDef defWatch[] =
{
    {mjITEM_SECTION,   "Watch",         0, NULL,                    "AW"},
    {mjITEM_EDITTXT,   "Field",         2, settings.field,          "qpos"},
    {mjITEM_EDITINT,   "Index",         2, &settings.index,         "1"},
    {mjITEM_STATIC,    "Value",         2, NULL,                    " "},
    {mjITEM_END}
};


// help strings
const char help_content[] =
"Alt mouse button\n"
"UI right hold\n"
"UI title double-click\n"
"Space\n"
"Esc\n"
"Right arrow\n"
"Left arrow\n"
"Down arrow\n"
"Up arrow\n"
"Page Up\n"
"Double-click\n"
"Right double-click\n"
"Ctrl Right double-click\n"
"Scroll, middle drag\n"
"Left drag\n"
"[Shift] right drag\n"
"Ctrl [Shift] drag\n"
"Ctrl [Shift] right drag";

const char help_title[] =
"Swap left-right\n"
"Show UI shortcuts\n"
"Expand/collapse all  \n"
"Pause\n"
"Free camera\n"
"Step forward\n"
"Step back\n"
"Step forward 100\n"
"Step back 100\n"
"Select parent\n"
"Select\n"
"Center\n"
"Track camera\n"
"Zoom\n"
"View rotate\n"
"View translate\n"
"Object rotate\n"
"Object translate";


// info strings
char info_title[1000];
char info_content[1000];

//----------------------- profiler, sensor, info, watch ---------------------------------

// init profiler figures
void profilerinit(void)
{
    int i, n;

    // set figures to default
    mjv_defaultFigure(&figconstraint);
    mjv_defaultFigure(&figcost);
    mjv_defaultFigure(&figtimer);
    mjv_defaultFigure(&figsize);

    // titles
    strcpy(figconstraint.title, "Counts");
    strcpy(figcost.title, "Convergence (log 10)");
    strcpy(figsize.title, "Dimensions");
    strcpy(figtimer.title, "CPU time (msec)");

    // x-labels
    strcpy(figconstraint.xlabel, "Solver iteration");
    strcpy(figcost.xlabel, "Solver iteration");
    strcpy(figsize.xlabel, "Video frame");
    strcpy(figtimer.xlabel, "Video frame");

    // y-tick nubmer formats
    strcpy(figconstraint.yformat, "%.0f");
    strcpy(figcost.yformat, "%.1f");
    strcpy(figsize.yformat, "%.0f");
    strcpy(figtimer.yformat, "%.2f");

    // colors
    figconstraint.figurergba[0] =   0.1f;
    figcost.figurergba[2] =         0.2f;
    figsize.figurergba[0] =         0.1f;
    figtimer.figurergba[2] =        0.2f;
    figconstraint.figurergba[3] =   0.5f;
    figcost.figurergba[3] =         0.5f;
    figsize.figurergba[3] =         0.5f;
    figtimer.figurergba[3] =        0.5f;

    // legends
    strcpy(figconstraint.linename[0], "total");
    strcpy(figconstraint.linename[1], "active");
    strcpy(figconstraint.linename[2], "changed");
    strcpy(figconstraint.linename[3], "evals");
    strcpy(figconstraint.linename[4], "updates");
    strcpy(figcost.linename[0], "improvement");
    strcpy(figcost.linename[1], "gradient");
    strcpy(figcost.linename[2], "lineslope");
    strcpy(figsize.linename[0], "dof");
    strcpy(figsize.linename[1], "body");
    strcpy(figsize.linename[2], "constraint");
    strcpy(figsize.linename[3], "sqrt(nnz)");
    strcpy(figsize.linename[4], "contact");
    strcpy(figsize.linename[5], "iteration");
    strcpy(figtimer.linename[0], "total");
    strcpy(figtimer.linename[1], "collision");
    strcpy(figtimer.linename[2], "prepare");
    strcpy(figtimer.linename[3], "solve");
    strcpy(figtimer.linename[4], "other");

    // grid sizes
    figconstraint.gridsize[0] = 5;
    figconstraint.gridsize[1] = 5;
    figcost.gridsize[0] = 5;
    figcost.gridsize[1] = 5;
    figsize.gridsize[0] = 3;
    figsize.gridsize[1] = 5;
    figtimer.gridsize[0] = 3;
    figtimer.gridsize[1] = 5;

    // minimum ranges
    figconstraint.range[0][0] = 0;
    figconstraint.range[0][1] = 20;
    figconstraint.range[1][0] = 0;
    figconstraint.range[1][1] = 80;
    figcost.range[0][0] = 0;
    figcost.range[0][1] = 20;
    figcost.range[1][0] = -15;
    figcost.range[1][1] = 5;
    figsize.range[0][0] = -200;
    figsize.range[0][1] = 0;
    figsize.range[1][0] = 0;
    figsize.range[1][1] = 100;
    figtimer.range[0][0] = -200;
    figtimer.range[0][1] = 0;
    figtimer.range[1][0] = 0;
    figtimer.range[1][1] = 0.4f;

    // init x axis on history figures (do not show yet)
    for( n=0; n<6; n++ )
        for( i=0; i<mjMAXLINEPNT; i++ )
        {
            figtimer.linedata[n][2*i] = (float)-i;
            figsize.linedata[n][2*i] = (float)-i;
        }
}



// update profiler figures
void profilerupdate(void)
{
    int i, n;

    // update constraint figure
    figconstraint.linepnt[0] = mjMIN(mjMIN(d->solver_iter, mjNSOLVER), mjMAXLINEPNT);
    for( i=1; i<5; i++ )
        figconstraint.linepnt[i] = figconstraint.linepnt[0];
    if( m->opt.solver==mjSOL_PGS )
    {
        figconstraint.linepnt[3] = 0;
        figconstraint.linepnt[4] = 0;
    }
    if( m->opt.solver==mjSOL_CG )
        figconstraint.linepnt[4] = 0;
    for( i=0; i<figconstraint.linepnt[0]; i++ )
    {
        // x
        figconstraint.linedata[0][2*i] = (float)i;
        figconstraint.linedata[1][2*i] = (float)i;
        figconstraint.linedata[2][2*i] = (float)i;
        figconstraint.linedata[3][2*i] = (float)i;
        figconstraint.linedata[4][2*i] = (float)i;

        // y
        figconstraint.linedata[0][2*i+1] = (float)d->nefc;
        figconstraint.linedata[1][2*i+1] = (float)d->solver[i].nactive;
        figconstraint.linedata[2][2*i+1] = (float)d->solver[i].nchange;
        figconstraint.linedata[3][2*i+1] = (float)d->solver[i].neval;
        figconstraint.linedata[4][2*i+1] = (float)d->solver[i].nupdate;
    }

    // update cost figure
    figcost.linepnt[0] = mjMIN(mjMIN(d->solver_iter, mjNSOLVER), mjMAXLINEPNT);
    for( i=1; i<3; i++ )
        figcost.linepnt[i] = figcost.linepnt[0];
    if( m->opt.solver==mjSOL_PGS )
    {
        figcost.linepnt[1] = 0;
        figcost.linepnt[2] = 0;
    }

    for( i=0; i<figcost.linepnt[0]; i++ )
    {
        // x
        figcost.linedata[0][2*i] = (float)i;
        figcost.linedata[1][2*i] = (float)i;
        figcost.linedata[2][2*i] = (float)i;

        // y
        figcost.linedata[0][2*i+1] = (float)mju_log10(mju_max(mjMINVAL, d->solver[i].improvement));
        figcost.linedata[1][2*i+1] = (float)mju_log10(mju_max(mjMINVAL, d->solver[i].gradient));
        figcost.linedata[2][2*i+1] = (float)mju_log10(mju_max(mjMINVAL, d->solver[i].lineslope));
    }

    // get timers: total, collision, prepare, solve, other
    mjtNum total = d->timer[mjTIMER_STEP].duration;
    int number = d->timer[mjTIMER_STEP].number;
    if( !number )
    {
        total = d->timer[mjTIMER_FORWARD].duration;
        number = d->timer[mjTIMER_FORWARD].number;
    }
    number = mjMAX(1, number);
    float tdata[5] = {
        (float)(total/number),
        (float)(d->timer[mjTIMER_POS_COLLISION].duration/number),
        (float)(d->timer[mjTIMER_POS_MAKE].duration/number) +
            (float)(d->timer[mjTIMER_POS_PROJECT].duration/number),
        (float)(d->timer[mjTIMER_CONSTRAINT].duration/number),
        0
    };
    tdata[4] = tdata[0] - tdata[1] - tdata[2] - tdata[3];

    // update figtimer
    int pnt = mjMIN(201, figtimer.linepnt[0]+1);
    for( n=0; n<5; n++ )
    {
        // shift data
        for( i=pnt-1; i>0; i-- )
            figtimer.linedata[n][2*i+1] = figtimer.linedata[n][2*i-1];

        // assign new
        figtimer.linepnt[n] = pnt;
        figtimer.linedata[n][1] = tdata[n];
    }

    // get sizes: nv, nbody, nefc, sqrt(nnz), ncont, iter
    float sdata[6] = {
        (float)m->nv,
        (float)m->nbody,
        (float)d->nefc,
        (float)mju_sqrt((mjtNum)d->solver_nnz),
        (float)d->ncon,
        (float)d->solver_iter
    };

    // update figsize
    pnt = mjMIN(201, figsize.linepnt[0]+1);
    for( n=0; n<6; n++ )
    {
        // shift data
        for( i=pnt-1; i>0; i-- )
            figsize.linedata[n][2*i+1] = figsize.linedata[n][2*i-1];

        // assign new
        figsize.linepnt[n] = pnt;
        figsize.linedata[n][1] = sdata[n];
    }
}



// show profiler figures
void profilershow(mjrRect rect)
{
    mjrRect viewport = {
        rect.left + rect.width - rect.width/4,
        rect.bottom,
        rect.width/4,
        rect.height/4
    };
    mjr_figure(viewport, &figtimer, &con);
    viewport.bottom += rect.height/4;
    mjr_figure(viewport, &figsize, &con);
    viewport.bottom += rect.height/4;
    mjr_figure(viewport, &figcost, &con);
    viewport.bottom += rect.height/4;
    mjr_figure(viewport, &figconstraint, &con);
}



// init sensor figure
void sensorinit(void)
{
    // set figure to default
    mjv_defaultFigure(&figsensor);
    figsensor.figurergba[3] = 0.5f;

    // set flags
    figsensor.flg_extend = 1;
    figsensor.flg_barplot = 1;
    figsensor.flg_symmetric = 1;

    // title
    strcpy(figsensor.title, "Sensor data");

    // y-tick nubmer format
    strcpy(figsensor.yformat, "%.0f");

    // grid size
    figsensor.gridsize[0] = 2;
    figsensor.gridsize[1] = 3;

    // minimum range
    figsensor.range[0][0] = 0;
    figsensor.range[0][1] = 0;
    figsensor.range[1][0] = -1;
    figsensor.range[1][1] = 1;
}



// update sensor figure
void sensorupdate(void)
{
    static const int maxline = 10;

    // clear linepnt
    for( int i=0; i<maxline; i++ )
        figsensor.linepnt[i] = 0;

    // start with line 0
    int lineid = 0;

    // loop over sensors
    for( int n=0; n<m->nsensor; n++ )
    {
        // go to next line if type is different
        if( n>0 && m->sensor_type[n]!=m->sensor_type[n-1] )
            lineid = mjMIN(lineid+1, maxline-1);

        // get info about this sensor
        mjtNum cutoff = (m->sensor_cutoff[n]>0 ? m->sensor_cutoff[n] : 1);
        int adr = m->sensor_adr[n];
        int dim = m->sensor_dim[n];

        // data pointer in line
        int p = figsensor.linepnt[lineid];

        // fill in data for this sensor
        for( int i=0; i<dim; i++ )
        {
            // check size
            if( (p+2*i)>=mjMAXLINEPNT/2 )
                break;

            // x
            figsensor.linedata[lineid][2*p+4*i] = (float)(adr+i);
            figsensor.linedata[lineid][2*p+4*i+2] = (float)(adr+i);

            // y
            figsensor.linedata[lineid][2*p+4*i+1] = 0;
            figsensor.linedata[lineid][2*p+4*i+3] = (float)(d->sensordata[adr+i]/cutoff);
        }

        // update linepnt
        figsensor.linepnt[lineid] = mjMIN(mjMAXLINEPNT-1,
                                          figsensor.linepnt[lineid]+2*dim);
    }
}



// show sensor figure
void sensorshow(mjrRect rect)
{
    // constant width with and without profiler
    int width = settings.profiler ? rect.width/3 : rect.width/4;

    // render figure on the right
    mjrRect viewport = {
        rect.left + rect.width - width,
        rect.bottom,
        width,
        rect.height/3
    };
    mjr_figure(viewport, &figsensor, &con);
}


/* ----- added by luke ----- */

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
    strcpy(figmotors.linename[0], "gX");
    strcpy(figmotors.linename[1], "gY");
    strcpy(figmotors.linename[2], "gZ");
    strcpy(figmotors.linename[3], "bZ");
    strcpy(figmotors.linename[4], "bX");
    strcpy(figmotors.linename[5], "bY");
    strcpy(figmotors.linename[5], "byaw");
    
}


void lukesensorfigsupdate(void)
{
    // amount of data we extract for each sensor
    int gnum = 50;

    // check we can plot this amount of data
    if (gnum > mjMAXLINEPNT) {
        std::cout << "gnum exceeds mjMAXLINEPNT in gaugefigupdate()\n";
        gnum = mjMAXLINEPNT;
    }

    MjType::SensorData* data_ptr;

    if (settings.use_SI_sensors) {
        data_ptr = &myMjClass.sim_sensors_SI_;
    }
    else {
        data_ptr = &myMjClass.sim_sensors_;
    }

    bool base_xyz = luke::use_base_xyz();
    bool base_z_rot = luke::use_base_z_rot();

    // read the data    
    std::vector<luke::gfloat> g1data = data_ptr->finger1_gauge.read(gnum);
    std::vector<luke::gfloat> g2data = data_ptr->finger2_gauge.read(gnum);
    std::vector<luke::gfloat> g3data = data_ptr->finger3_gauge.read(gnum);
    std::vector<luke::gfloat> p1data = data_ptr->palm_sensor.read(gnum);
    std::vector<luke::gfloat> a1data = data_ptr->finger1_axial_gauge.read(gnum);
    std::vector<luke::gfloat> a2data = data_ptr->finger2_axial_gauge.read(gnum);
    std::vector<luke::gfloat> a3data = data_ptr->finger3_axial_gauge.read(gnum);
    std::vector<luke::gfloat> wXdata = data_ptr->wrist_X_sensor.read(gnum);
    std::vector<luke::gfloat> wYdata = data_ptr->wrist_Y_sensor.read(gnum);
    std::vector<luke::gfloat> wZdata = data_ptr->wrist_Z_sensor.read(gnum);
    std::vector<luke::gfloat> mXdata = data_ptr->x_motor_position.read(gnum);
    std::vector<luke::gfloat> mYdata = data_ptr->y_motor_position.read(gnum);
    std::vector<luke::gfloat> mZdata = data_ptr->z_motor_position.read(gnum);
    std::vector<luke::gfloat> bXdata = data_ptr->x_base_position.read(gnum);
    std::vector<luke::gfloat> bYdata = data_ptr->y_base_position.read(gnum);
    std::vector<luke::gfloat> bZdata = data_ptr->z_base_position.read(gnum);
    std::vector<luke::gfloat> byawdata = data_ptr->yaw_base_rotation.read(gnum);

    // get the corresponding timestamps
    std::vector<float> btdata = myMjClass.gauge_timestamps.read(gnum);
    std::vector<float> atdata = myMjClass.axial_timestamps.read(gnum);
    std::vector<float> ptdata = myMjClass.palm_timestamps.read(gnum);
    std::vector<float> wtdata = myMjClass.wristZ_timestamps.read(gnum);
    std::vector<float> mtdata = myMjClass.step_timestamps.read(gnum);

    // package sensor data pointers in iterable vectors
    std::vector<std::vector<luke::gfloat>*> bdata {
        &g1data, &g2data, &g3data
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
        &mXdata, &mYdata, &mZdata, &bZdata
    };
    if (base_xyz) {
        mdata.push_back(&bXdata);
        mdata.push_back(&bYdata);
        if (base_z_rot) {
            mdata.push_back(&byawdata);
        }
    }

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
        &figbendgauge, &figaxialgauge, &figpalm, &figwrist, &figmotors
    };
    // what settings determine if these are showing
    std::vector<int> flags {
        settings.bendgauge, 
        settings.axialgauge, 
        settings.palmsensor,
        settings.wristsensor,
        settings.statesensor
    };

    // how many graphs do we need to fit
    int num = myfigs.size();
    int to_show = 0;

    if (settings.allsensors) {
        to_show = num;
    }
    else {
        for (int i = 0; i < myfigs.size(); i++) {
            if (flags[i] != 0) {
                to_show += 1;
            }
        }
    }

    // maximum size
    if (to_show < 3) to_show = 3;

    // constant width with and without profiler
    int width = settings.profiler ? rect.width / 3 : rect.width / to_show;
    int show = 0;

    for (int i = 0; i < myfigs.size(); i++) {

        // is this figure set to display
        if (flags[i] != 0 or settings.allsensors) {

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
}


void lukestepperfigsinit(void)
{
    bool use_xyz = luke::use_base_xyz();

    mjv_defaultFigure(&figstepperx);
    mjv_defaultFigure(&figsteppery);
    mjv_defaultFigure(&figstepperz);
    mjv_defaultFigure(&figbasex);
    mjv_defaultFigure(&figbasey);
    mjv_defaultFigure(&figbasez);
    mjv_defaultFigure(&figbaserotz);

    // what figures are we initialising
    std::vector<mjvFigure*> myfigs {
        &figstepperx, &figsteppery, &figstepperz, &figbasex, &figbasey, &figbasez,
        &figbaserotz
    };
    
    // what are the figure titles
    std::vector<std::string> titles {
        "Stepper x / um", "Stepper y / um", "Stepper z / um", 
        "Base x / um", "Base y / um", "Base z / um", "Base z rot / urad"
    };

    // // figure limits
    // std::vector<float> mins {
    //     luke::Gripper::xy_min,
    //     luke::Gripper::xy_min,
    //     luke::Gripper::z_min,
    //     luke::Target::base_z_min
    // };
    // std::vector<float> maxs {
    //     luke::Gripper::xy_max,
    //     luke::Gripper::xy_max,
    //     luke::Gripper::z_max,
    //     luke::Target::base_z_max
    // };

    // initialise all figures the same
    for (int i = 0; i < myfigs.size(); i++) {

        // create a default figure
        mjv_defaultFigure(myfigs[i]);

        // set flags
        myfigs[i]->flg_legend = 1;
        myfigs[i]->flg_extend = 1;
        myfigs[i]->flg_symmetric = 0;

        // y-tick number format
        strcpy(myfigs[i]->yformat, "%.0f");

        // grid size
        myfigs[i]->gridsize[0] = 2;
        myfigs[i]->gridsize[1] = 3;

        // minimum range
        myfigs[i]->range[0][0] = 0;
        myfigs[i]->range[0][1] = 0;
        myfigs[i]->range[1][0] = 0;
        myfigs[i]->range[1][1] = 0;

        // all figs have same two lines, add them
        strcpy(myfigs[i]->linename[0], "Target");
        strcpy(myfigs[i]->linename[1], "Actual");

        // add figure title
        strcpy(myfigs[i]->title, titles[i].c_str());
    }
}

void lukestepperfigsupdate(void)
{
    // amount of data we extract for each sensor
    int gnum = luke::target_.datanum;

    // check we can plot this amount of data
    if (gnum > mjMAXLINEPNT) {
        std::cout << "gnum exceeds mjMAXLINEPNT in stepperfigupdate()\n";
        gnum = mjMAXLINEPNT;
    }

    bool base_xyz = luke::use_base_xyz();
    bool base_z_rot = luke::use_base_z_rot();

    // read the data    
    std::vector<luke::gfloat> txsdata = luke::target_.target_stepperx.read(gnum);
    std::vector<luke::gfloat> tysdata = luke::target_.target_steppery.read(gnum);
    std::vector<luke::gfloat> tzsdata = luke::target_.target_stepperz.read(gnum);
    std::vector<luke::gfloat> txbdata = luke::target_.target_basex.read(gnum);
    std::vector<luke::gfloat> tybdata = luke::target_.target_basey.read(gnum);
    std::vector<luke::gfloat> tzbdata = luke::target_.target_basez.read(gnum);
    std::vector<luke::gfloat> tzrotbdata = luke::target_.target_baseyaw.read(gnum);
    std::vector<luke::gfloat> axsdata = luke::target_.actual_stepperx.read(gnum);
    std::vector<luke::gfloat> aysdata = luke::target_.actual_steppery.read(gnum);
    std::vector<luke::gfloat> azsdata = luke::target_.actual_stepperz.read(gnum);
    std::vector<luke::gfloat> axbdata = luke::target_.actual_basex.read(gnum);
    std::vector<luke::gfloat> aybdata = luke::target_.actual_basey.read(gnum);
    std::vector<luke::gfloat> azbdata = luke::target_.actual_basez.read(gnum);
    std::vector<luke::gfloat> azrotbdata = luke::target_.actual_baseyaw.read(gnum);
    std::vector<luke::gfloat> timedata = luke::target_.timedata.read(gnum);

    // package sensor data pointers in iterable vectors
    std::vector<std::vector<luke::gfloat>*> sxdata {
        &txsdata, &axsdata
    };
    std::vector<std::vector<luke::gfloat>*> sydata {
        &tysdata, &aysdata
    };
    std::vector<std::vector<luke::gfloat>*> szdata {
        &tzsdata, &azsdata
    };
    std::vector<std::vector<luke::gfloat>*> bxdata {
        &txbdata, &axbdata
    };
    std::vector<std::vector<luke::gfloat>*> bydata {
        &tybdata, &aybdata
    };
    std::vector<std::vector<luke::gfloat>*> bzdata {
        &tzbdata, &azbdata
    };
    std::vector<std::vector<luke::gfloat>*> bzrotdata {
        &tzrotbdata, &azrotbdata
    };

    // package figures into iterable vector
    std::vector<mjvFigure*> myfigs {
        &figstepperx, &figsteppery, &figstepperz, &figbasez
    };
    if (base_xyz) {
        myfigs[3] = &figbasex;
        myfigs.push_back(&figbasey);
        myfigs.push_back(&figbasez);
        if (base_z_rot) {
            myfigs.push_back(&figbaserotz);
        }
    }

    // package sensor data in same order as figures
    std::vector< std::vector<std::vector<luke::gfloat>*>* > sensordata {
        &sxdata, &sydata, &szdata, &bzdata
    };
    if (base_xyz) {
        sensordata[3] = &bxdata;
        sensordata.push_back(&bydata);
        sensordata.push_back(&bzdata);
        if (base_z_rot) {
            sensordata.push_back(&bzrotdata);
        }
    }

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

                myfigs[i]->linedata[lineid][2 * g] = (timedata)[g];
                myfigs[i]->linedata[lineid][2 * g + 1] = (*(*sensordata[i])[n])[g];
            }

            // update linepnt (index of last data point)
            myfigs[i]->linepnt[lineid] = gnum;
        }
    }
}

void lukestepperfigshow(mjrRect rect)
{
    bool base_xyz = luke::use_base_xyz();
    bool base_z_rot = luke::use_base_z_rot();

    // what figures are showing
    std::vector<mjvFigure*> myfigs {
        &figstepperx, &figsteppery, &figstepperz, &figbasez
    };
    if (base_xyz) {
        myfigs[3] = &figbasex;
        myfigs.push_back(&figbasey);
        myfigs.push_back(&figbasez);
        if (base_z_rot) {
            myfigs.push_back(&figbaserotz);
        }
    }
    // what settings determine if these are showing
    std::vector<int> flags {
        settings.xstepfig, 
        settings.ystepfig,
        settings.zstepfig,
        settings.zbasefig
    };
    if (base_xyz) {
        flags[3] = settings.xbasefig;
        flags.push_back(settings.ybasefig);
        flags.push_back(settings.zbasefig);
        if (base_z_rot) {
            flags.push_back(settings.zbaserotfig);
        }
    }

    // how many graphs do we need to fit
    int num = myfigs.size();
    int to_show = 0;

    if (settings.allactuators) {
        to_show = num;
    }
    else {
        for (int i = 0; i < myfigs.size(); i++) {
            if (flags[i] != 0) {
                to_show += 1;
            }
        }
    }

    // maximum size
    if (to_show < 3) to_show = 3;

    // constant width with and without profiler
    int width = settings.profiler ? rect.width / 3 : rect.width / to_show;
    int show = 0;

    for (int i = 0; i < myfigs.size(); i++) {

        // is this figure set to display
        if (flags[i] != 0 or settings.allactuators) {

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
}

/* ----- end of added by luke ----- */

// prepare info text
void infotext(char* title, char* content, double interval)
{
    char tmp[20];

    // compute solver error
    mjtNum solerr = 0;
    if( d->solver_iter )
    {
        int ind = mjMIN(d->solver_iter-1,mjNSOLVER-1);
        solerr = mju_min(d->solver[ind].improvement, d->solver[ind].gradient);
        if( solerr==0 )
            solerr = mju_max(d->solver[ind].improvement, d->solver[ind].gradient);
    }
    solerr = mju_log10(mju_max(mjMINVAL, solerr));

    // prepare info text
    strcpy(title, "Time\nSize\nCPU\nSolver   \nFPS\nstack\nconbuf\nefcbuf");
    sprintf(content, "%-20.3f\n%d  (%d con)\n%.3f\n%.1f  (%d it)\n%.0f\n%.3f\n%.3f\n%.3f",
            d->time,
            d->nefc, d->ncon,
            settings.run ?
                d->timer[mjTIMER_STEP].duration / mjMAX(1, d->timer[mjTIMER_STEP].number) :
                d->timer[mjTIMER_FORWARD].duration / mjMAX(1, d->timer[mjTIMER_FORWARD].number),
            solerr, d->solver_iter,
            1/interval,
            d->maxuse_stack/(double)d->nstack,
            d->maxuse_con/(double)m->nconmax,
            d->maxuse_efc/(double)m->njmax);

    // add Energy if enabled
    if( mjENABLED(mjENBL_ENERGY) )
    {
        sprintf(tmp, "\n%.3f", d->energy[0]+d->energy[1]);
        strcat(content, tmp);
        strcat(title, "\nEnergy");
    }

    // add FwdInv if enabled
    if( mjENABLED(mjENBL_FWDINV) )
    {
        sprintf(tmp, "\n%.1f %.1f",
            mju_log10(mju_max(mjMINVAL,d->solver_fwdinv[0])),
            mju_log10(mju_max(mjMINVAL,d->solver_fwdinv[1])));
        strcat(content, tmp);
        strcat(title, "\nFwdInv");
    }
}


// sprintf forwarding, to avoid compiler warning in x-macro
void printfield(char* str, void* ptr)
{
    sprintf(str, "%g", *(mjtNum*)ptr);
}



// update watch
void watch(void)
{
    // clear
    ui0.sect[SECT_WATCH].item[2].multi.nelem = 1;
    strcpy(ui0.sect[SECT_WATCH].item[2].multi.name[0], "invalid field");

    // prepare constants for NC
    int nv = m->nv;
    int njmax = m->njmax;

    // find specified field in mjData arrays, update value
    #define X(TYPE, NAME, NR, NC)                                           \
        if( !strcmp(#NAME, settings.field) && !strcmp(#TYPE, "mjtNum") )    \
        {                                                                   \
            if( settings.index>=0 && settings.index<m->NR*NC )              \
                printfield(ui0.sect[SECT_WATCH].item[2].multi.name[0],      \
                           d->NAME + settings.index);                       \
            else                                                            \
                strcpy(ui0.sect[SECT_WATCH].item[2].multi.name[0],          \
                       "invalid index");                                    \
            return;                                                         \
        }

        MJDATA_POINTERS
    #undef X
}



//-------------------------------- UI construction --------------------------------------

// make physics section of UI
void makephysics(int oldstate)
{
    int i;

    mjuiDef defPhysics[] =
    {
        {mjITEM_SECTION,   "Physics",       oldstate, NULL,                 "AP"},
        {mjITEM_SELECT,    "Integrator",    2, &(m->opt.integrator),        "Euler\nRK4"},
        {mjITEM_SELECT,    "Collision",     2, &(m->opt.collision),         "All\nPair\nDynamic"},
        {mjITEM_SELECT,    "Cone",          2, &(m->opt.cone),              "Pyramidal\nElliptic"},
        {mjITEM_SELECT,    "Jacobian",      2, &(m->opt.jacobian),          "Dense\nSparse\nAuto"},
        {mjITEM_SELECT,    "Solver",        2, &(m->opt.solver),            "PGS\nCG\nNewton"},
        {mjITEM_SEPARATOR, "Algorithmic Parameters", 1},
        {mjITEM_EDITNUM,   "Timestep",      2, &(m->opt.timestep),          "1 0 1"},
        {mjITEM_EDITINT,   "Iterations",    2, &(m->opt.iterations),        "1 0 1000"},
        {mjITEM_EDITNUM,   "Tolerance",     2, &(m->opt.tolerance),         "1 0 1"},
        {mjITEM_EDITINT,   "Noslip Iter",   2, &(m->opt.noslip_iterations), "1 0 1000"},
        {mjITEM_EDITNUM,   "Noslip Tol",    2, &(m->opt.noslip_tolerance),  "1 0 1"},
        {mjITEM_EDITINT,   "MRR Iter",      2, &(m->opt.mpr_iterations),    "1 0 1000"},
        {mjITEM_EDITNUM,   "MPR Tol",       2, &(m->opt.mpr_tolerance),     "1 0 1"},
        {mjITEM_EDITNUM,   "API Rate",      2, &(m->opt.apirate),           "1 0 1000"},
        {mjITEM_SEPARATOR, "Physical Parameters", 1},
        {mjITEM_EDITNUM,   "Gravity",       2, m->opt.gravity,              "3"},
        {mjITEM_EDITNUM,   "Wind",          2, m->opt.wind,                 "3"},
        {mjITEM_EDITNUM,   "Magnetic",      2, m->opt.magnetic,             "3"},
        {mjITEM_EDITNUM,   "Density",       2, &(m->opt.density),           "1"},
        {mjITEM_EDITNUM,   "Viscosity",     2, &(m->opt.viscosity),         "1"},
        {mjITEM_EDITNUM,   "Imp Ratio",     2, &(m->opt.impratio),          "1"},
        {mjITEM_SEPARATOR, "Disable Flags", 1},
        {mjITEM_END}
    };
    mjuiDef defEnableFlags[] =
    {
        {mjITEM_SEPARATOR, "Enable Flags", 1},
        {mjITEM_END}
    };
    mjuiDef defOverride[] =
    {
        {mjITEM_SEPARATOR, "Contact Override", 1},
        {mjITEM_EDITNUM,   "Margin",        2, &(m->opt.o_margin),          "1"},
        {mjITEM_EDITNUM,   "Sol Imp",       2, &(m->opt.o_solimp),          "5"},
        {mjITEM_EDITNUM,   "Sol Ref",       2, &(m->opt.o_solref),          "2"},
        {mjITEM_END}
    };

    // add physics
    mjui_add(&ui0, defPhysics);

    // add flags programmatically
    mjuiDef defFlag[] =
    {
        {mjITEM_CHECKINT,  "", 2, NULL, ""},
        {mjITEM_END}
    };
    for( i=0; i<mjNDISABLE; i++ )
    {
        strcpy(defFlag[0].name, mjDISABLESTRING[i]);
        defFlag[0].pdata = settings.disable + i;
        mjui_add(&ui0, defFlag);
    }
    mjui_add(&ui0, defEnableFlags);
    for( i=0; i<mjNENABLE; i++ )
    {
        strcpy(defFlag[0].name, mjENABLESTRING[i]);
        defFlag[0].pdata = settings.enable + i;
        mjui_add(&ui0, defFlag);
    }

    // add contact override
    mjui_add(&ui0, defOverride);
}



// make rendering section of UI
void makerendering(int oldstate)
{
    int i;
    unsigned int j;

    mjuiDef defRendering[] =
    {
        {mjITEM_SECTION,    "Rendering",        oldstate, NULL,             "AR"},
        {mjITEM_SELECT,     "Camera",           2, &(settings.camera),      "Free\nTracking"},
        {mjITEM_SELECT,     "Label",            2, &(vopt.label),
            "None\nBody\nJoint\nGeom\nSite\nCamera\nLight\nTendon\nActuator\nConstraint\nSkin\nSelection\nSel Pnt\nForce"},
        {mjITEM_SELECT,     "Frame",            2, &(vopt.frame),
            "None\nBody\nGeom\nSite\nCamera\nLight\nWorld"},
        {mjITEM_SEPARATOR,  "Model Elements",   1},
        {mjITEM_END}
    };
    mjuiDef defOpenGL[] =
    {
        {mjITEM_SEPARATOR, "OpenGL Effects", 1},
        {mjITEM_END}
    };

    // add model cameras, up to UI limit
    for( i=0; i<mjMIN(m->ncam, mjMAXUIMULTI-2); i++ )
    {
        // prepare name
        char camname[mjMAXUITEXT] = "\n";
        if( m->names[m->name_camadr[i]] )
            strcat(camname, m->names+m->name_camadr[i]);
        else
            sprintf(camname, "\nCamera %d", i);

        // check string length
        if( strlen(camname) + strlen(defRendering[1].other)>=mjMAXUITEXT-1 )
            break;

        // add camera
        strcat(defRendering[1].other, camname);
    }

    // add rendering standard
    mjui_add(&ui0, defRendering);

    // add flags programmatically
    mjuiDef defFlag[] =
    {
        {mjITEM_CHECKBYTE,  "", 2, NULL, ""},
        {mjITEM_END}
    };
    for( i=0; i<mjNVISFLAG; i++ )
    {
        // set name, remove "&"
        strcpy(defFlag[0].name, mjVISSTRING[i][0]);
        for( j=0; j<strlen(mjVISSTRING[i][0]); j++ )
            if( mjVISSTRING[i][0][j]=='&' )
            {
                strcpy(defFlag[0].name+j, mjVISSTRING[i][0]+j+1);
                break;
            }

        // set shortcut and data
        sprintf(defFlag[0].other, " %s", mjVISSTRING[i][2]);
        defFlag[0].pdata = vopt.flags + i;
        mjui_add(&ui0, defFlag);
    }
    mjui_add(&ui0, defOpenGL);
    for( i=0; i<mjNRNDFLAG; i++ )
    {
        strcpy(defFlag[0].name, mjRNDSTRING[i][0]);
        sprintf(defFlag[0].other, " %s", mjRNDSTRING[i][2]);
        defFlag[0].pdata = scn.flags + i;
        mjui_add(&ui0, defFlag);
    }
}



// make group section of UI
void makegroup(int oldstate)
{
    mjuiDef defGroup[] =
    {
        {mjITEM_SECTION,    "Group enable",     oldstate, NULL,             "AG"},
        {mjITEM_SEPARATOR,  "Geom groups",  1},
        {mjITEM_CHECKBYTE,  "Geom 0",           2, vopt.geomgroup,          " 0"},
        {mjITEM_CHECKBYTE,  "Geom 1",           2, vopt.geomgroup+1,        " 1"},
        {mjITEM_CHECKBYTE,  "Geom 2",           2, vopt.geomgroup+2,        " 2"},
        {mjITEM_CHECKBYTE,  "Geom 3",           2, vopt.geomgroup+3,        " 3"},
        {mjITEM_CHECKBYTE,  "Geom 4",           2, vopt.geomgroup+4,        " 4"},
        {mjITEM_CHECKBYTE,  "Geom 5",           2, vopt.geomgroup+5,        " 5"},
        {mjITEM_SEPARATOR,  "Site groups",  1},
        {mjITEM_CHECKBYTE,  "Site 0",           2, vopt.sitegroup,          "S0"},
        {mjITEM_CHECKBYTE,  "Site 1",           2, vopt.sitegroup+1,        "S1"},
        {mjITEM_CHECKBYTE,  "Site 2",           2, vopt.sitegroup+2,        "S2"},
        {mjITEM_CHECKBYTE,  "Site 3",           2, vopt.sitegroup+3,        "S3"},
        {mjITEM_CHECKBYTE,  "Site 4",           2, vopt.sitegroup+4,        "S4"},
        {mjITEM_CHECKBYTE,  "Site 5",           2, vopt.sitegroup+5,        "S5"},
        {mjITEM_SEPARATOR,  "Joint groups", 1},
        {mjITEM_CHECKBYTE,  "Joint 0",          2, vopt.jointgroup,         ""},
        {mjITEM_CHECKBYTE,  "Joint 1",          2, vopt.jointgroup+1,       ""},
        {mjITEM_CHECKBYTE,  "Joint 2",          2, vopt.jointgroup+2,       ""},
        {mjITEM_CHECKBYTE,  "Joint 3",          2, vopt.jointgroup+3,       ""},
        {mjITEM_CHECKBYTE,  "Joint 4",          2, vopt.jointgroup+4,       ""},
        {mjITEM_CHECKBYTE,  "Joint 5",          2, vopt.jointgroup+5,       ""},
        {mjITEM_SEPARATOR,  "Tendon groups",    1},
        {mjITEM_CHECKBYTE,  "Tendon 0",         2, vopt.tendongroup,        ""},
        {mjITEM_CHECKBYTE,  "Tendon 1",         2, vopt.tendongroup+1,      ""},
        {mjITEM_CHECKBYTE,  "Tendon 2",         2, vopt.tendongroup+2,      ""},
        {mjITEM_CHECKBYTE,  "Tendon 3",         2, vopt.tendongroup+3,      ""},
        {mjITEM_CHECKBYTE,  "Tendon 4",         2, vopt.tendongroup+4,      ""},
        {mjITEM_CHECKBYTE,  "Tendon 5",         2, vopt.tendongroup+5,      ""},
        {mjITEM_SEPARATOR,  "Actuator groups", 1},
        {mjITEM_CHECKBYTE,  "Actuator 0",       2, vopt.actuatorgroup,      ""},
        {mjITEM_CHECKBYTE,  "Actuator 1",       2, vopt.actuatorgroup+1,    ""},
        {mjITEM_CHECKBYTE,  "Actuator 2",       2, vopt.actuatorgroup+2,    ""},
        {mjITEM_CHECKBYTE,  "Actuator 3",       2, vopt.actuatorgroup+3,    ""},
        {mjITEM_CHECKBYTE,  "Actuator 4",       2, vopt.actuatorgroup+4,    ""},
        {mjITEM_CHECKBYTE,  "Actuator 5",       2, vopt.actuatorgroup+5,    ""},
        {mjITEM_END}
    };

    // add section
    mjui_add(&ui0, defGroup);
}

/* ----- added by luke ----- */

// make a custom gripper section of the ui
void makeGripperUI(int oldstate)
{
    int i;

    mjuiDef defGripper[] =
    {
        {mjITEM_SECTION, "Gripper", oldstate,   NULL,   "AJ"},
        {mjITEM_END}
    };
    mjuiDef defSlider[] =
    {
        {mjITEM_SLIDERNUM, "", 2, NULL, "0 1"},
        {mjITEM_END}
    };
    mjui_add(&ui1, defGripper);

    bool base_xyz = luke::use_base_xyz();

    if (not base_xyz) {

        defSlider[0].state = 4;

        std::vector<std::string> slider_names {
            "Prismatic", "Revolute", "Palm", "Base Z"
        };
        std::vector<void*> slider_values {
            &luke::target_.end.x, &luke::target_.end.th, &luke::target_.end.z,
            &luke::target_.base.z
        };
        std::vector<std::string> slider_ranges {
            "0.05 0.14", "-0.6 0.6", "0.0 0.16", "-0.1 0.1"
        };

        for (i = 0; i < slider_names.size(); i++) {

            // set the data address
            defSlider[0].pdata = slider_values[i];

            mju_strncpy(defSlider[0].name, slider_names[i].c_str(), mjMAXUINAME);
            mju_strncpy(defSlider[0].other, slider_ranges[i].c_str(), mjMAXUINAME);

            // add
            mjui_add(&ui1, defSlider);
        }
    }
    else if (base_xyz) {
        defSlider[0].state = 6;

        std::vector<std::string> slider_names {
            "Prismatic", "Revolute", "Palm", "Base X", "Base Y", "Base Z"
        };
        std::vector<void*> slider_values {
            &luke::target_.end.x, &luke::target_.end.th, &luke::target_.end.z,
            &luke::target_.base.x, &luke::target_.base.y, &luke::target_.base.z
        };
        std::vector<std::string> slider_ranges {
            "0.05 0.14", "-0.6 0.6", "0.0 0.16", "-0.1 0.1", "-0.1 0.1", "-0.1 0.1"
        };

        for (i = 0; i < slider_names.size(); i++) {

            // set the data address
            defSlider[0].pdata = slider_values[i];

            mju_strncpy(defSlider[0].name, slider_names[i].c_str(), mjMAXUINAME);
            mju_strncpy(defSlider[0].other, slider_ranges[i].c_str(), mjMAXUINAME);

            // add
            mjui_add(&ui1, defSlider);
        }
    }

    // for (i = 0; i < slider_names.size(); i++) {

    //     // set the data address
    //     defSlider[0].pdata = slider_values[i];

    //     mju_strncpy(defSlider[0].name, slider_names[i].c_str(), mjMAXUINAME);
    //     mju_strncpy(defSlider[0].other, slider_ranges[i].c_str(), mjMAXUINAME);

    //     // add
    //     mjui_add(&ui1, defSlider);
    // }
}

void makeActionsUI(int oldstate)
{
    mjuiDef defActions[] =
    {
        {mjITEM_SECTION, "Actions",    oldstate,  NULL,   " #303"},
        {mjITEM_BUTTON, "Action 1",         2,  NULL,   " #304"},
        {mjITEM_BUTTON, "Action 2",         2,  NULL,   " #305"},
        {mjITEM_BUTTON, "Action 3",        2,  NULL,   " #306"},
        {mjITEM_BUTTON, "Action 4",        2,  NULL,   " #307"},
        {mjITEM_BUTTON, "Action 5",         2,  NULL,   " #308"},
        {mjITEM_BUTTON, "Action 6",         2,  NULL,   " #309"},
        {mjITEM_BUTTON, "Action 7",        2,  NULL,   " #310"},
        {mjITEM_BUTTON, "Action 8",        2,  NULL,   " #310"},
        {mjITEM_BUTTON, "Action 9",         2,  NULL,   " #304"},
        {mjITEM_BUTTON, "Action 10",         2,  NULL,   " #305"},
        {mjITEM_BUTTON, "Action 11",            2,  NULL,   " #306"},
        {mjITEM_BUTTON, "Action 12",            2,  NULL,   " #307"},
        {mjITEM_BUTTON, "Action 13",            2,  NULL,   " #308"},
        {mjITEM_BUTTON, "Action 14",            2,  NULL,   " #309"},
        {mjITEM_BUTTON, "Action 15",            2,  NULL,   " #310"},
        {mjITEM_BUTTON, "Action 16",            2,  NULL,   " #310"},
        {mjITEM_SLIDERNUM, "Motor mm",        2,  &settings.action_motor_mm,        "0.0 20.0"},
        {mjITEM_SLIDERNUM, "Motor rad",       2,  &settings.action_motor_rad,       "0.0 0.2"},
        {mjITEM_SLIDERNUM, "Base mm",         2,  &settings.action_base_mm,         "0.0 20.0"},
        {mjITEM_SLIDERNUM, "Base rad",        2,  &settings.action_base_rad,        "0.0 0.2"},
        {mjITEM_BUTTON, "Reward",             2,  NULL,   " #311"},
        {mjITEM_BUTTON, "Print actions",            2,  NULL,   " #310"},
        {mjITEM_CHECKINT, "Debug",            2,  &myMjClass.s_.debug,   " #312"},
        {mjITEM_CHECKINT, "Env steps",        2,  &settings.env_steps, " #313"},
        {mjITEM_CHECKINT, "Full step",        2,  &settings.complete_action_steps, " #314"},
        {mjITEM_END}
    };

    mjui_add(&ui1, defActions);
}

void makeSettingsUI(int oldstate)
{
    mjuiDef defActions[] =
    {
        {mjITEM_SECTION,  "Sim Settings",      oldstate,  NULL,   " #500"},
        {mjITEM_CHECKINT, "Debug",             2, &myMjClass.s_.debug,             " #600"},
        {mjITEM_SLIDERNUM,"mj timestep",       2, &myMjClass.s_.mujoco_timestep,   "0.00001 0.003"},
        {mjITEM_CHECKINT, "curve_validation",  2, &myMjClass.s_.curve_validation,  " #601"},
        {mjITEM_SLIDERNUM,"tip_force",         2, &myMjClass.s_.tip_force_applied, "-10 10"},
        {mjITEM_CHECKINT, "randomise_colours", 2, &myMjClass.s_.randomise_colours, " #602"},
        {mjITEM_SLIDERNUM,"finger_thickness",  2, &settings.finger_thickness,  "0.5 1.5"},
        {mjITEM_BUTTON,   "apply thickness",   2, NULL,                            " #602"},
        {mjITEM_SLIDERINT,"state_prev_n",      2, &myMjClass.s_.state_n_prev_steps,  "0 5"},
        {mjITEM_SLIDERINT,"sensor_prev_n",     2, &myMjClass.s_.sensor_n_prev_steps, "0 5"},
        {mjITEM_SLIDERINT,"state_mode",        2, &myMjClass.s_.state_sample_mode,   "0 5"},
        {mjITEM_SLIDERINT,"sensor_mode",       2, &myMjClass.s_.sensor_sample_mode,  "0 5"},
        {mjITEM_SLIDERNUM,"time4action",       2, &myMjClass.s_.time_for_action,     "0.1 1.0"},
        {mjITEM_SLIDERNUM,"solimp_d0",         2, &myMjClass.model->opt.o_solimp[0],  "0 1.0"},
        {mjITEM_SLIDERNUM,"solimp_dw",         2, &myMjClass.model->opt.o_solimp[1],  "0 1.0"},
        {mjITEM_SLIDERNUM,"solimp_wdth",       2, &myMjClass.model->opt.o_solimp[2],  "0 0.01"},
        {mjITEM_SLIDERNUM,"solimp_mpt",        2, &myMjClass.model->opt.o_solimp[3],  "0 1.0"},
        {mjITEM_SLIDERNUM,"solimp_pow",        2, &myMjClass.model->opt.o_solimp[4],  "0 6.0"},
        {mjITEM_CHECKINT, "cal gauges",        2, &myMjClass.s_.auto_calibrate_gauges,  " #700"},
        {mjITEM_END}
    };

    mjui_add(&ui1, defActions);
}

void makeObjectUI(int oldstate)
{
    int i;

    // doesn't work...
    std::vector<std::string> object_names = luke::get_objects();
    int num_objects = object_names.size();
    std::string str_num = std::to_string(num_objects);
    char lims[] = "0 00";
    lims[4] = '0'; // overwrite null terminator to 0

    if (str_num.size() == 1) {
        lims[4] = str_num[0];
    }
    else if (str_num.size() == 2) {
        lims[3] = str_num[0];
        lims[4] = str_num[1];
    }
    else if (str_num.size() == 3) {
        lims[2] = str_num[0];
        lims[3] = str_num[1];
        lims[4] = str_num[2];
    }
    else {
        throw std::runtime_error("num_objects string size not 1,2,or 3");
    }

    // lims[2] = '0' + num_objects + '\0';

    // std::cout << "lims is " << lims << '\n';

    mjuiDef defObjectUI[] =
    {
        {mjITEM_SECTION,   "Objects",       1, NULL,                    " #300"},
        {mjITEM_BUTTON,    "Reset",         2, NULL,                    " #301"},
        {mjITEM_BUTTON,    "Respawn",       2, NULL,                    " #302"},
        {mjITEM_BUTTON,    "Print forces",  2, NULL,                    " #303"},
        {mjITEM_BUTTON,    "Ground forces",  2, NULL,                   " #303"},
        {mjITEM_BUTTON,    "Object forces",  2, NULL,                   " #303"},
        {mjITEM_BUTTON,    "All forces",     2, NULL,                   " #303"},
        {mjITEM_BUTTON,    "Print curve fit",2, NULL,                   " #304"},
        {mjITEM_BUTTON,    "Wipe curve fit", 2, NULL,                   " #305"},
        {mjITEM_BUTTON,    "Validate regime",2, NULL,                   " #311"},
        {mjITEM_BUTTON,    "obj. rgb rand",  2, NULL,                   " #306"},
        {mjITEM_BUTTON,    "gnd rgb rand",  2, NULL,                    " #307"},
        {mjITEM_BUTTON,    "fing. rgb rand", 2, NULL,                   " #308"},
        {mjITEM_BUTTON,    "all rgb rand",  2, NULL,                    " #309"},
        {mjITEM_BUTTON,    "none rgb rand", 2, NULL,                    " #310"},
        {mjITEM_BUTTON,   "find timestep",     2, NULL,                 " #311"},
        {mjITEM_BUTTON,   "cal. gauges",     2, NULL,                   " #312"},
        {mjITEM_BUTTON,   "print stiffness",   2, NULL,                 " #313"},
        {mjITEM_BUTTON,   "apply seg frc",   2, NULL,                   " #314"},
        {mjITEM_BUTTON,   "apply UDL",   2, NULL,                   " #315"},
        {mjITEM_BUTTON,   "apply tip frc",   2, NULL,                   " #316"},
        {mjITEM_BUTTON,   "wipe seg frc",    2, NULL,                   " #317"},
        {mjITEM_BUTTON,   "apply gram frc",    2, NULL,                   " #317"},
        {mjITEM_SLIDERINT,   "seg num",           2, &settings.seg_num_for_frc, "0 10"},
        {mjITEM_SLIDERNUM,  "force",         2, &settings.seg_force, "0 10"},
        {mjITEM_SLIDERNUM,  "moment",         2, &settings.seg_moment, "0 2"},
        {mjITEM_SLIDERINT,   "frc style",     2, &settings.force_style, "0 2"},
        {mjITEM_SLIDERINT,   "gram frc",      2, &settings.gram_force, "0 5"},

        // {mjITEM_BUTTON,    "Copy pose",     2, NULL,                    " #304"},
        {mjITEM_SLIDERINT, "Live Object",   3, &settings.object_int,    "0 19"},
        {mjITEM_SLIDERINT, "x noise",       3, &settings.object_x_noise_mm, "-10 10"},
        {mjITEM_SLIDERINT, "y noise",       3, &settings.object_y_noise_mm, "-10 10"},
        {mjITEM_SLIDERINT, "z rotation",    3, &settings.object_z_rot_deg,  "0 360"},
        // {mjITEM_BUTTON,    "Reset to key",  3},
        // {mjITEM_BUTTON,    "Set key",       3},
        {mjITEM_BUTTON,   "visibility",    2, NULL,                   " #317"},
        {mjITEM_BUTTON,   "spawn scene",   2, NULL,                   " #318"},
        {mjITEM_BUTTON,   "spawn into",   2, NULL,                   " #318"},
        {mjITEM_SLIDERINT, "scene obj",   3, &settings.scene_objects,    "1 20"},
        {mjITEM_SLIDERNUM,  "scene X",         2, &settings.scene_x, "0 1"},
        {mjITEM_SLIDERNUM,  "scene Y",         2, &settings.scene_y, "0 1"},
        {mjITEM_BUTTON,   "random base",    2, NULL,                 " #317"},
        {mjITEM_BUTTON,   "set base XY",    2, NULL,                 " #317"},
        {mjITEM_BUTTON,   "random_yaw",     2, NULL,                 " #317"},
        {mjITEM_BUTTON,   "gripper visible",     2, NULL,                 " #317"},
        {mjITEM_BUTTON,   "f-end/palm xyz",  2, NULL,                " #317"},
        {mjITEM_BUTTON,   "MAT reopen",  2, NULL,                " #317"},
        {mjITEM_BUTTON,   "set target",  2, NULL,                " #318"},
        {mjITEM_SLIDERNUM,  "motor X",         2, &settings.motor_x, "49 140"},
        {mjITEM_SLIDERNUM,  "motor Y",         2, &settings.motor_y, "49 140"},
        {mjITEM_SLIDERNUM,  "motor Z",         2, &settings.motor_z, "0 165"},
        {mjITEM_BUTTON,     "print frc",       2, NULL,     " #300"},
        {mjITEM_BUTTON,     "msr constrict",       2, NULL,     " #300"},
        {mjITEM_BUTTON,     "msr tilt",       2, NULL,     " #300"},
        {mjITEM_BUTTON,     "msr palm",       2, NULL,     " #300"},
        {mjITEM_END}
    };

    mjui_add(&ui1, defObjectUI);
}

// make joint section of UI
void makejoint(int oldstate)
{
    int i;

    mjuiDef defJoint[] =
    {
        {mjITEM_SECTION, "Joint", oldstate, NULL, "AJ"},
        {mjITEM_END}
    };
    mjuiDef defSlider[] =
    {
        {mjITEM_SLIDERNUM, "", 2, NULL, "0 1"},
        {mjITEM_END}
    };

    // add section
    mjui_add(&ui1, defJoint);
    defSlider[0].state = 4;

    // add scalar joints, exit if UI limit reached
    int itemcnt = 0;
    for( i=0; i<m->njnt && itemcnt<mjMAXUIITEM; i++ )
        if( (m->jnt_type[i]==mjJNT_HINGE || m->jnt_type[i]==mjJNT_SLIDE) )
        {
            // skip if joint group is disabled
            if( !vopt.jointgroup[mjMAX(0, mjMIN(mjNGROUP-1, m->jnt_group[i]))] )
                continue;

            // set data and name
            defSlider[0].pdata = d->qpos + m->jnt_qposadr[i];
            if( m->names[m->name_jntadr[i]] )
                mju_strncpy(defSlider[0].name, m->names+m->name_jntadr[i],
                            mjMAXUINAME);
            else
                sprintf(defSlider[0].name, "joint %d", i);

            // set range
            if( m->jnt_limited[i] )
                sprintf(defSlider[0].other, "%.4g %.4g",
                    m->jnt_range[2*i], m->jnt_range[2*i+1]);
            else if( m->jnt_type[i]==mjJNT_SLIDE )
                strcpy(defSlider[0].other, "-1 1");
            else
                strcpy(defSlider[0].other, "-3.1416 3.1416");

            // add and count
            mjui_add(&ui1, defSlider);
            itemcnt++;
        }
}

/* ----- end of added by luke ----- */

// make control section of UI
void makecontrol(int oldstate)
{
    int i;

    mjuiDef defControl[] =
    {
        {mjITEM_SECTION, "Control", oldstate, NULL, "AC"},
        {mjITEM_BUTTON,  "Clear all", 2},
        {mjITEM_END}
    };
    mjuiDef defSlider[] =
    {
        {mjITEM_SLIDERNUM, "", 2, NULL, "0 1"},
        {mjITEM_END}
    };

    // add section
    mjui_add(&ui1, defControl);
    defSlider[0].state = 2;

    // add controls, exit if UI limit reached (Clear button already added)
    int itemcnt = 1;
    for( i=0; i<m->nu && itemcnt<mjMAXUIITEM; i++ )
    {
        // skip if actuator group is disabled
        if( !vopt.actuatorgroup[mjMAX(0, mjMIN(mjNGROUP-1, m->actuator_group[i]))] )
            continue;

        // set data and name
        defSlider[0].pdata = d->ctrl + i;
        if( m->names[m->name_actuatoradr[i]] )
            mju_strncpy(defSlider[0].name, m->names+m->name_actuatoradr[i],
                        mjMAXUINAME);
        else
            sprintf(defSlider[0].name, "control %d", i);

        // set range
        if( m->actuator_ctrllimited[i] )
            sprintf(defSlider[0].other, "%.4g %.4g",
                m->actuator_ctrlrange[2*i], m->actuator_ctrlrange[2*i+1]);
        else
            strcpy(defSlider[0].other, "-1 1");

        // add and count
        mjui_add(&ui1, defSlider);
        itemcnt++;
    }
}



// make model-dependent UI sections
void makesections(void)
{
    int i;

    // get section open-close state, UI 0
    int oldstate0[NSECT0];
    for( i=0; i<NSECT0; i++ )
    {
        oldstate0[i] = 0;
        if( ui0.nsect>i )
            oldstate0[i] = ui0.sect[i].state;
    }

    // get section open-close state, UI 1
    int oldstate1[NSECT1];
    for( i=0; i<NSECT1; i++ )
    {
        oldstate1[i] = 0;
        if( ui1.nsect>i )
            oldstate1[i] = ui1.sect[i].state;
    }

    // clear model-dependent sections of UI
    ui0.nsect = SECT_PHYSICS;
    ui1.nsect = 0;

    // make
    makephysics(oldstate0[SECT_PHYSICS]);
    makerendering(oldstate0[SECT_RENDERING]);
    makegroup(oldstate0[SECT_GROUP]);
    makejoint(oldstate1[SECT_JOINT]);
    makecontrol(oldstate1[SECT_CONTROL]);

    // added by luke
    makeGripperUI(oldstate1[SECT_GRIPPER]);
    makeObjectUI(oldstate1[SECT_OBJECT]);
    makeActionsUI(oldstate1[SECT_ACTION]);
    makeSettingsUI(oldstate1[SECT_SETTINGS]);
}



//-------------------------------- utility functions ------------------------------------

// align and scale view
void alignscale(void)
{
    // autoscale
    cam.lookat[0] = m->stat.center[0];
    cam.lookat[1] = m->stat.center[1];
    cam.lookat[2] = m->stat.center[2];
    cam.distance = 1.5 * m->stat.extent;

    // set to free camera
    cam.type = mjCAMERA_FREE;
}



// copy qpos to clipboard as key
void copykey(void)
{
    char clipboard[5000] = "<key qpos='";
    char buf[200];

    // prepare string
    for( int i=0; i<m->nq; i++ )
    {
        sprintf(buf, i==m->nq-1 ? "%g" : "%g ", d->qpos[i]);
        strcat(clipboard, buf);
    }
    strcat(clipboard, "'/>");

    // copy to clipboard
    glfwSetClipboardString(window, clipboard);
}



// millisecond timer, for MuJoCo built-in profiler
mjtNum timer(void)
{
    return (mjtNum)(1000*glfwGetTime());
}



// clear all times
void cleartimers(void)
{
    for( int i=0; i<mjNTIMER; i++ )
    {
        d->timer[i].duration = 0;
        d->timer[i].number = 0;
    }
}



// update UI 0 when MuJoCo structures change (except for joint sliders)
void updatesettings(void)
{
    int i;

    // physics flags
    for( i=0; i<mjNDISABLE; i++ )
        settings.disable[i] = ((m->opt.disableflags & (1<<i)) !=0 );
    for( i=0; i<mjNENABLE; i++ )
        settings.enable[i] = ((m->opt.enableflags & (1<<i)) !=0 );

    // camera
    if( cam.type==mjCAMERA_FIXED )
        settings.camera = 2 + cam.fixedcamid;
    else if( cam.type==mjCAMERA_TRACKING )
        settings.camera = 1;
    else
        settings.camera = 0;

    // update UI
    mjui_update(-1, -1, &ui0, &uistate, &con);
}



// drop file callback
void drop(GLFWwindow* window, int count, const char** paths)
{
    // make sure list is non-empty
    if( count>0 )
    {
        mju_strncpy(filename, paths[0], 1000);
        settings.loadrequest = 1;
    }
}



// load mjb or xml model
void loadmodel(void)
{
    // clear request
    settings.loadrequest = 0;

    /* original code
    // make sure filename is not empty
    if( !filename[0]  )
        return;

    // load and compile
    char error[500] = "";
    mjModel* mnew = 0;
    if( strlen(filename)>4 && !strcmp(filename+strlen(filename)-4, ".mjb") )
    {
        mnew = mj_loadModel(filename, NULL);
        if( !mnew )
            strcpy(error, "could not load binary model");
    }
    else
        mnew = mj_loadXML(filename, NULL, error, 500);
    if( !mnew )
    {
        printf("%s\n", error);
        return;
    }

    // compiler warning: print and pause
    if( error[0] )
    {
        // mj_forward() below will print the warning message
        printf("Model compiled, but simulation warning (paused):\n  %s\n\n",
                error);
        settings.run = 0;
    }

    // delete old model, assign new
    mj_deleteData(d);
    mj_deleteModel(m);
    m = mnew;
    d = mj_makeData(m);
    mj_forward(m, d);

    // added by luke
    // initialise the model
    luke::init(m, d);
    */

    std::cout << "Loading model from: " << filename << "\n";
    char error[500] = "";
    m = mj_loadXML(filename, 0, error, 500);
    if (error[0] != '\0') std::cout << "Load error: " << error << '\n';
    d = mj_makeData(m);

    // initialise my simulation class with the pointers
    myMjClass.init(m, d);

    // added by luke, old version, assign new pointers
    // myMjClass.load(filename);
    // m = myMjClass.model;
    // d = myMjClass.data;

    // re-create scene and context
    mjv_makeScene(m, &scn, maxgeom);
    mjr_makeContext(m, &con, 50*(settings.font+1));

    // clear perturbation state
    pert.active = 0;
    pert.select = 0;
    pert.skinselect = -1;

    // align and scale view, update scene
    alignscale();
    mjv_updateScene(m, d, &vopt, &pert, &cam, mjCAT_ALL, &scn);

    // set window title to model name
    if( window && m->names )
    {
        char title[200] = "Simulate : ";
        strcat(title, m->names);
        glfwSetWindowTitle(window, title);
    }

    // set keyframe range and divisions
    ui0.sect[SECT_SIMULATION].item[5].slider.range[0] = 0;
    ui0.sect[SECT_SIMULATION].item[5].slider.range[1] = mjMAX(0, m->nkey - 1);
    ui0.sect[SECT_SIMULATION].item[5].slider.divisions = mjMAX(1, m->nkey - 1);

    // rebuild UI sections
    makesections();

    // full ui update
    uiModify(window, &ui0, &uistate, &con);
    uiModify(window, &ui1, &uistate, &con);
    updatesettings();
}



//--------------------------------- UI hooks (for uitools.c) ----------------------------

// determine enable/disable item state given category
int uiPredicate(int category, void* userdata)
{
    switch( category )
    {
    case 2:                 // require model
        return (m!=NULL);

    case 3:                 // require model and nkey
        return (m && m->nkey);

    case 4:                 // require model and paused
        return (m && !settings.run);

    default:
        return 1;
    }
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
    rect[1].width = settings.ui0 ? ui0.width : 0;
    rect[1].bottom = 0;
    rect[1].height = rect[0].height;

    // rect 2: UI 1
    rect[2].width = settings.ui1 ? ui1.width : 0;
    rect[2].left = mjMAX(0, rect[0].width - rect[2].width);
    rect[2].bottom = 0;
    rect[2].height = rect[0].height;

    // rect 3: 3D plot (everything else is an overlay)
    rect[3].left = rect[1].width;
    rect[3].width = mjMAX(0, rect[0].width - rect[1].width - rect[2].width);
    rect[3].bottom = 0;
    rect[3].height = rect[0].height;
}



// handle UI event
void uiEvent(mjuiState* state)
{
    int i;
    char err[200];

    // call UI 0 if event is directed to it
    if( (state->dragrect==ui0.rectid) ||
        (state->dragrect==0 && state->mouserect==ui0.rectid) ||
        state->type==mjEVENT_KEY )
    {
        // process UI event
        mjuiItem* it = mjui_event(&ui0, state, &con);

        // file section
        if( it && it->sectionid==SECT_FILE )
        {
            switch( it->itemid )
            {
            case 0:             // Save xml
                if( !mj_saveLastXML("mjmodel.xml", m, err, 200) )
                    printf("Save XML error: %s", err);
                break;

            case 1:             // Save mjb
                mj_saveModel(m, "mjmodel.mjb", NULL, 0);
                break;

            case 2:             // Print model
                mj_printModel(m, "MJMODEL.TXT");
                break;

            case 3:             // Print data
                mj_printData(m, d, "MJDATA.TXT");
                break;

            case 4:             // Quit
                settings.exitrequest = 1;
                break;
            }
        }

        // option section
        else if( it && it->sectionid==SECT_OPTION )
        {
            switch( it->itemid )
            {
            case 0:             // Spacing
                ui0.spacing = mjui_themeSpacing(settings.spacing);
                ui1.spacing = mjui_themeSpacing(settings.spacing);
                break;

            case 1:             // Color
                ui0.color = mjui_themeColor(settings.color);
                ui1.color = mjui_themeColor(settings.color);
                break;

            case 2:             // Font
                mjr_changeFont(50*(settings.font+1), &con);
                break;

            case 9:             // Full screen
                if( glfwGetWindowMonitor(window) )
                {
                    // restore window from saved data
                    glfwSetWindowMonitor(window, NULL, windowpos[0], windowpos[1],
                                         windowsize[0], windowsize[1], 0);
                }

                // currently windowed: switch to full screen
                else
                {
                    // save window data
                    glfwGetWindowPos(window, windowpos, windowpos+1);
                    glfwGetWindowSize(window, windowsize, windowsize+1);

                    // switch
                    glfwSetWindowMonitor(window, glfwGetPrimaryMonitor(), 0, 0,
                                         vmode.width, vmode.height, vmode.refreshRate);
                }

                // reinstante vsync, just in case
                glfwSwapInterval(settings.vsync);
                break;

            case 10:            // Vertical sync
                glfwSwapInterval(settings.vsync);
                break;
            

            case 18:            // apply noise
            case 19:            // mag slider
            case 20:            // mean slider
            case 21:            // std slider
                // myMjClass.tick();
                // while (myMjClass.tock() < 1) {};
                // throw std::runtime_error("");
                // std::cout << it->itemid << '\n';
                myMjClass.s_.set_use_noise(settings.all_sensors_use_noise);
                myMjClass.s_.apply_noise_params(myMjClass.uniform_dist);
                break;

            //     // do nothing
            //     break;
            }

            // modify UI
            uiModify(window, &ui0, state, &con);
            uiModify(window, &ui1, state, &con);
        }

        // simulation section
        else if( it && it->sectionid==SECT_SIMULATION )
        {
            switch( it->itemid )
            {
            case 1:             // Reset
                if( m )
                {
                    /* original code replaced by luke::reset
                    mj_resetData(m, d);
                    mj_forward(m, d);
                    */
                    // luke::reset(m, d);
                    myMjClass.reset();
                    profilerupdate();
                    sensorupdate();
                    updatesettings();
                }
                break;

            case 2:             // Reload
                settings.loadrequest = 1;
                break;

            case 3:             // Align
                alignscale();
                updatesettings();
                break;

            case 4:             // Copy pose
                copykey();
                break;

            case 5:             // Adjust key
            case 6:             // Reset to key
                i = settings.key;
                /* original code replaced by luke::keyframe
                d->time = m->key_time[i];
                mju_copy(d->qpos, m->key_qpos+i*m->nq, m->nq);
                mju_copy(d->qvel, m->key_qvel+i*m->nv, m->nv);
                mju_copy(d->act, m->key_act+i*m->na, m->na);
				mju_copy(d->mocap_pos, m->key_mpos+i*3*m->nmocap, 3*m->nmocap);
				mju_copy(d->mocap_quat, m->key_mquat+i*4*m->nmocap, 4*m->nmocap);
                */
                luke::keyframe(m, d, i);
                mj_forward(m, d);
                profilerupdate();
                sensorupdate();
                updatesettings();
                break;

            case 7:             // Set key
                i = settings.key;
                m->key_time[i] = d->time;
                mju_copy(m->key_qpos+i*m->nq, d->qpos, m->nq);
                mju_copy(m->key_qvel+i*m->nv, d->qvel, m->nv);
                mju_copy(m->key_act+i*m->na, d->act, m->na);
				mju_copy(m->key_mpos+i*3*m->nmocap, d->mocap_pos, 3*m->nmocap);
				mju_copy(m->key_mquat+i*4*m->nmocap, d->mocap_quat, 4*m->nmocap);
                break;
            }
        }

        // physics section
        else if( it && it->sectionid==SECT_PHYSICS )
        {
            // update disable flags in mjOption
            m->opt.disableflags = 0;
            for( i=0; i<mjNDISABLE; i++ )
                if( settings.disable[i] )
                    m->opt.disableflags |= (1<<i);

            // update enable flags in mjOption
            m->opt.enableflags = 0;
            for( i=0; i<mjNENABLE; i++ )
                if( settings.enable[i] )
                    m->opt.enableflags |= (1<<i);
        }

        // rendering section
        else if( it && it->sectionid==SECT_RENDERING )
        {
            // set camera in mjvCamera
            if( settings.camera==0 )
                cam.type = mjCAMERA_FREE;
            else if( settings.camera==1 )
            {
                if( pert.select>0 )
                {
                    cam.type = mjCAMERA_TRACKING;
                    cam.trackbodyid = pert.select;
                    cam.fixedcamid = -1;
                }
                else
                {
                    cam.type = mjCAMERA_FREE;
                    settings.camera = 0;
                    mjui_update(SECT_RENDERING, -1, &ui0, &uistate, &con);
                }
            }
            else
            {
                cam.type = mjCAMERA_FIXED;
                cam.fixedcamid = settings.camera - 2;
            }
        }

        // group section
        else if( it && it->sectionid==SECT_GROUP )
        {
            // remake joint section if joint group changed
            if( it->name[0]=='J' && it->name[1]=='o' )
            {
                ui1.nsect = SECT_JOINT;
                makejoint(ui1.sect[SECT_JOINT].state);
                ui1.nsect = NSECT1;
                uiModify(window, &ui1, state, &con);
            }

            // remake control section if actuator group changed
            if( it->name[0]=='A' && it->name[1]=='c' )
            {
                ui1.nsect = SECT_CONTROL;
                makecontrol(ui1.sect[SECT_CONTROL].state);
                ui1.nsect = NSECT1;
                uiModify(window, &ui1, state, &con);
            }
        }

        // stop if UI processed event
        if( it!=NULL || (state->type==mjEVENT_KEY && state->key==0) )
            return;
    }

    // call UI 1 if event is directed to it
    if( (state->dragrect==ui1.rectid) ||
        (state->dragrect==0 && state->mouserect==ui1.rectid) ||
        state->type==mjEVENT_KEY )
    {
        // process UI event
        mjuiItem* it = mjui_event(&ui1, state, &con);

        // control section
        if( it && it->sectionid==SECT_CONTROL )
        {
            // clear controls
            if( it->itemid==0 )
            {
                mju_zero(d->ctrl, m->nu);
                mjui_update(SECT_CONTROL, -1, &ui1, &uistate, &con);
            }
        }

        // added by luke
        // gripper section
        else if (it and it->sectionid == SECT_GRIPPER)
        {
            // std::cout << "case " << it->itemid << '\n';
            switch (it->itemid)
            {
                // changing the gripper slider
                default: {
                    luke::target_.end.update_x_th_z();
                    break;
                }
            }
        }

        else if (it and it->sectionid == SECT_SETTINGS)
        {
            // std::cout << "it->itemid is " << it->itemid << '\n';

            switch (it->itemid)
            {
                case 6:
                    std::cout << "Applying finger thickness of " << settings.finger_thickness << " mm\n";
                    myMjClass.set_finger_thickness(settings.finger_thickness * 1e-3);
                    myMjClass.reset();
                    break;
            }
        }

        else if (it and it->sectionid == SECT_ACTION)
        {
            // // for testing
            // std::cout << "case " << it->itemid << '\n';
            switch (it->itemid)
            {
                case 0: case 1: case 2: case 3: case 4: case 5: case 6: case 7: 
                case 8: case 9: case 10: case 11: case 12: case 13: case 14: case 15: {
                    if (luke::get_num_live_objects() == 0) {
                        myMjClass.spawn_object(settings.object_int, 
                            settings.object_x_noise_mm * 1e-3,
                            settings.object_y_noise_mm * 1e-3,
                            settings.object_z_rot_deg * (3.1416 / 180.0));
                    }
                    std::vector<float> obs = myMjClass.set_discrete_action(it->itemid);
                    if (settings.complete_action_steps and obs.size() > 0) {
                        myMjClass.action_step();
                    }
                    std::vector<std::vector<double>> xyrel = luke::get_object_XY_relative_to_gripper(myMjClass.model, myMjClass.data);
                    for (int i = 0; i < xyrel.size(); i++) {
                        luke::print_vec(xyrel[i], "Object relative XY");
                    }
                    break;
                }

                case 16: case 17: case 18: case 19: {

                    myMjClass.s_.gripper_X.value = settings.action_motor_mm * 1e-3;
                    myMjClass.s_.gripper_prismatic_X.value = settings.action_motor_mm * 1e-3;
                    myMjClass.s_.gripper_Y.value = settings.action_motor_mm * 1e-3;
                    myMjClass.s_.gripper_revolute_Y.value = settings.action_motor_rad;
                    myMjClass.s_.gripper_Z.value = settings.action_motor_mm * 1e-3;

                    myMjClass.s_.base_X.value = settings.action_base_mm * 1e-3;
                    myMjClass.s_.base_Y.value = settings.action_base_mm * 1e-3;
                    myMjClass.s_.base_Z.value = settings.action_base_mm * 1e-3;
                    myMjClass.s_.base_roll.value = settings.action_base_rad;
                    myMjClass.s_.base_pitch.value = settings.action_base_rad;
                    myMjClass.s_.base_yaw.value = settings.action_base_rad;
                    break;
                }

                case 20: {
                    double reward = myMjClass.reward();
                    std::cout << "Reward is " << reward << '\n';
                    std::cout << "Cumulative reward is "
                        << myMjClass.env_.cumulative_reward << '\n';
                    break;
                }
                case 21: {
                    std::cout << "Printing the possible actions:\n";
                    myMjClass.print_actions();
                    break;
                }
            }

            // if we are set to full environment steps
            if (settings.env_steps and it->itemid < 15) {
                std::vector<luke::gfloat> obs = 
                    myMjClass.get_observation();
                double reward = myMjClass.reward();
                bool done = myMjClass.is_done();

                std::cout << "Action taken: " << it->itemid << '\n';
                myMjClass.debug_observation(obs, true);
                // luke::print_vec(obs, "Observation");
                if (myMjClass.s_.use_HER)
                    luke::print_vec(myMjClass.assess_goal(), "Goal performance");
                std::cout << "Reward is " << reward << '\n';
                std::cout << "Cumulative reward is "
                    << myMjClass.env_.cumulative_reward << '\n';
                std::cout << "is_done = " << (done ? "true\n" : "false\n");
                std::cout << '\n';
            }
        }

        else if (it and it->sectionid == SECT_OBJECT)
        {
            switch( it->itemid )
            {
            case 0: {            // Reset
                myMjClass.reset_object();
                break;
            }
            case 1: {            // Respawn
                myMjClass.spawn_object(settings.object_int, 
                    settings.object_x_noise_mm * 1e-3,
                    settings.object_y_noise_mm * 1e-3,
                    settings.object_z_rot_deg * (3.1416 / 180.0));

                break;
            }
            case 2: {            // Print forces
                luke::Forces_faster f = luke::get_object_forces_faster(myMjClass.model, myMjClass.data);
                f.print();
                break;
            }
            case 3: {            // Print ground forces
                luke::Forces_faster f = luke::get_object_forces_faster(myMjClass.model, myMjClass.data);
                f.print_gnd_global();
                f.print_gnd_local(); 
                break;
            }
            case 4: {            // Print object forces
                luke::Forces_faster f = luke::get_object_forces_faster(myMjClass.model, myMjClass.data);
                f.print_obj_global();
                f.print_obj_local(); 
                break;
            }
            case 5: {            // Print all forces
                luke::Forces_faster f = luke::get_object_forces_faster(myMjClass.model, myMjClass.data);
                f.print_all_global();
                f.print_all_local(); 
                break;
            }
            case 6: {            // Print curve fit validation
                std::cout << "Printing curve validation data\n";
                myMjClass.validate_curve();
                myMjClass.curve_validation_data_.print();
                break;
            }
            case 7: {
                std::cout << "Wiping curve validation data\n";
                myMjClass.curve_validation_data_.entries.clear();
                break;
            }
            case 8: {
                std::cout << "Running curve validation regime\n";
                bool print = true;
                myMjClass.curve_validation_regime(print, settings.force_style);
                std::cout << "force style was " << settings.force_style << '\n';
                break;
            }
            case 9: {           // randomise object colour
                myMjClass.randomise_object_colour();
                break;
            }
            case 10: {           // randomise ground colour
                myMjClass.randomise_ground_colour();
                break;
            }
            case 11: {          // randomise finger colour
                myMjClass.randomise_finger_colours();
                break;
            }
            case 12: {          // randomise all colours
                luke::randomise_all_object_colours(myMjClass.model, MjType::generator);
                myMjClass.randomise_ground_colour();
                myMjClass.randomise_finger_colours();
                break;
            }
            case 13: {          // restore default colours
                luke::default_colours(myMjClass.model);
                break;
            }
            case 14: {          // find timestep which is stable
                float timestep = myMjClass.find_highest_stable_timestep();
                std::cout << "The highest stable timestep is " << timestep << '\n';
                break;
            }
            case 15: {          // calibrate gauge readings
                float yield = myMjClass.yield_load();
                float factor = myMjClass.s_.saturation_yield_factor;
                float sat_load = yield * factor;
                std::cout << "The saturation load for calibration is " << sat_load << '\n';
                myMjClass.calibrate_simulated_sensors(sat_load);
                break;
            }
            case 16: {          // print stiffness
                luke::print_stiffnesses();
                break;
            }
            case 17: {          // apply segment force
                if (not myMjClass.s_.curve_validation) {
                    myMjClass.s_.curve_validation = true;
                }
                // static bool first_call = true;
                // if (first_call) luke::get_segment_matrices(m, d);
                // first_call = false;
                luke::set_segment_force(settings.seg_num_for_frc, true, settings.seg_force);
                luke::set_segment_moment(settings.seg_num_for_frc, true, settings.seg_moment);
                std::cout << "Applying force of " << settings.seg_force
                    << "N on segment " << settings.seg_num_for_frc << "\n";
                std::cout << "Applying moment of " << settings.seg_moment
                    << "N on segment " << settings.seg_num_for_frc << "\n";
                break;
            }
            case 18: {          // apply UDL
                if (not myMjClass.s_.curve_validation) {
                        myMjClass.s_.curve_validation = true;
                }
                // static bool first_call = true;
                // if (first_call) luke::get_segment_matrices(m, d);
                // first_call = false;
                luke::apply_UDL_force_per_joint(settings.seg_force);
                std::cout << "Applying UDL with a force on each joint of " << settings.seg_force << "N\n";
                break;
            }
            case 19: {          // apply tip force
                if (not myMjClass.s_.curve_validation) {
                    myMjClass.s_.curve_validation = true;
                }
                // static bool first_call = true;
                // if (first_call) luke::get_segment_matrices(m, d);
                // first_call = false;
                luke::apply_tip_force(settings.seg_force);
                std::cout << "Applying tip force of " << settings.seg_force << "N\n";
                break;
            }
            case 20: {          // wipe segment forces
                if (myMjClass.s_.curve_validation) {
                    myMjClass.s_.curve_validation = false;
                }
                luke::wipe_segment_forces();
                std::cout << "Wiping all segment forces\n";
                break;
            }

            case 21: {          // apply gram force
                if (not myMjClass.s_.curve_validation) {
                        myMjClass.s_.curve_validation = true;
                }
                float force = settings.gram_force * 0.981;
                luke::apply_tip_force(force);
                std::cout << "Applying tip force of " << settings.gram_force * 100 << "grams\n";
                break;
            }
            
            case 31: {          // set object visibility
                static bool visible = false; // default case is false
                visible = not visible;
                luke::set_object_visibility(myMjClass.model, visible);
                std::cout << "Setting hidden object visibility to: " << visible << "\n";
                break;
            }
            case 32: {          // spawn object scene
                std::cout << "Spawning a scene, goal is " << settings.scene_objects << " objects, ";
                myMjClass.reset_object();
                int num_spawned = myMjClass.spawn_scene(settings.scene_objects, settings.scene_x, settings.scene_y, 0.0);
                std::cout << "spawned " << num_spawned << " objects\n";
                break;
            }
            case 33: {          // spawn into object scene
                std::cout << "Spawning into scene object num " << settings.object_int << "\n";
                bool success = myMjClass.spawn_into_scene(settings.object_int, settings.scene_x, settings.scene_y,
                    settings.object_z_rot_deg, settings.object_x_noise_mm * 1e-3, settings.object_y_noise_mm * 1e-3,
                    M_PI);
                std::cout << "success flag is " << success << "\n";
                break;
            }
            case 37: {         // random base movement
                std::cout << "Performaning a random base movement with noise " << myMjClass.s_.base_position_noise << "\n";
                double z = myMjClass.random_base_Z_movement(myMjClass.s_.base_position_noise);
                std::cout << "Amount of noise applied was " << z << "\n";
                break;
            }
            case 38: {         // set base XY position
                std::cout << "Setting base position to (0.1, 0.1)\n";
                double z = myMjClass.set_new_base_XY(0.1, 0.1);
                break;
            }
            case 39: {         // random base yaw
                std::cout << "Setting base to random yaw = ";
                double z = myMjClass.random_base_yaw(myMjClass.base_max_[5]);
                std::cout << z << "\n";
                break;
            }
            case 40: {         // toggle gripper visible
                std::cout << "Toggle gripper visibility\n";
                luke::toggle_gripper_visibility(myMjClass.model);
                break;
            }
            case 41: {         // print f-end/palm xyz fingerend palm cartesian positions
                std::vector<double> finger_forces_SI(3);
                finger_forces_SI[0] = myMjClass.sim_sensors_SI_.finger1_gauge.read_element();
                finger_forces_SI[1] = myMjClass.sim_sensors_SI_.finger2_gauge.read_element();
                finger_forces_SI[2] = myMjClass.sim_sensors_SI_.finger3_gauge.read_element();
                std::vector<luke::Vec3> xyz = luke::get_fingerend_and_palm_xyz(finger_forces_SI);
                std::cout << "Printing fingerend and palm cartesian xyz positions\n";
                for (int i = 0; i < 4; i++) {
                    std::cout << "Finger " << i + 1 << " (4=palm)\n";
                    std::cout << "\tx = " << xyz[i].x << "\n";
                    std::cout << "\ty = " << xyz[i].y << "\n";
                    std::cout << "\tz = " << xyz[i].z << "\n";
                }
                break;
            }
            case 42: {         // MAT reopen

                std::uniform_real_distribution<double> distribution(-3.14, 3.14);
                double rand_angle = distribution(*MjType::generator);

                std::cout << "Performing MAT reopen with random angle = "
                    << rand_angle << " rad (" << rand_angle * (180.0/3.141592)
                    << " degrees)\n";
                myMjClass.MAT_reopen(rand_angle);
                break;
            }
            case 43: {         // gripper target set

                std::cout << "Setting gripper target as (x, y, z)mm: ("
                    << settings.motor_x*1e-3 << ", "
                    << settings.motor_y*1e-3 << ", "
                    << settings.motor_z*1e-3 << ")\n";
                luke::set_gripper_target_m(settings.motor_x*1e-3, 
                    settings.motor_y*1e-3, settings.motor_z*1e-3);
                break;
            }
            case 47: {         // print finger forces

                std::cout << "Printing gripper finger forces:\n"
                    << "Finger 1: " << myMjClass.sim_sensors_SI_.read_finger1_gauge() << " N\n"
                    << "Finger 2: " << myMjClass.sim_sensors_SI_.read_finger2_gauge() << " N\n"
                    << "Finger 3: " << myMjClass.sim_sensors_SI_.read_finger3_gauge() << " N\n"
                    << "Palm: " << myMjClass.sim_sensors_SI_.read_palm_sensor() << " N\n"
                    << "Wrist Z: " << myMjClass.sim_sensors_SI_.read_wrist_Z_sensor() << " N\n";
                break;
            }
            case 48: {         // measure constrict

                std::cout << "Running a force measurement program (constrict)\n";
                myMjClass.reset_object();
                myMjClass.spawn_object(0);
                std::vector<double> pos;
                for (int i = 130; i >= 58; i -= 2) pos.push_back(i * 1e-3);
                std::vector<std::vector<double>> force_matrix;
                // go to start position
                luke::set_gripper_target_m(pos[0], pos[0], 5e-3);
                for (int i = 0; i < 10; i++) myMjClass.action_step();
                for (int i = 0; i < pos.size(); i++) {
                    // set the gripper target
                    luke::set_gripper_target_m(pos[i], pos[i], 5e-3);
                    // step the simulation to reach it
                    for (int j = 0; j < 7; j++) myMjClass.action_step();
                    // record the forces
                    std::vector<double> forces(3);
                    forces[0] = myMjClass.sim_sensors_SI_.read_finger1_gauge();
                    forces[1] = myMjClass.sim_sensors_SI_.read_finger2_gauge();
                    forces[2] = myMjClass.sim_sensors_SI_.read_finger3_gauge();
                    force_matrix.push_back(forces);
                }
                // print out the final result
                luke::Vec3 objbox = luke::get_object_xyz_bounding_box(0);
                std::cout << "Sphere xyz bounding box is: "
                    << objbox.x << ", " << objbox.y << ", " << objbox.z << "\n";
                std::cout << "XY pos | Gauge1 | Gauge2 | Gauge3 | Avg\n";
                for (int i = 0; i < force_matrix.size(); i++) {
                    double avg = (1/3.0) * (force_matrix[i][0] + force_matrix[i][1]
                        + force_matrix[i][2]);
                    std::cout << pos[i]*1e3 << " | "
                        << force_matrix[i][0] << " | "
                        << force_matrix[i][1] << " | "
                        << force_matrix[i][2] << " | "
                        << avg << "\n";
                }
                std::cout << "Finished. Sphere diameter was: "
                    << objbox.x * 1e3 << " mm\n";

                break;
            }
            case 49: {         // measure tilt

                std::cout << "Running a force measurement program (tilt)\n";
                std::cout << "WARNING: to run this set j_.ctrl.num_steps = 1 "
                    "and j_.ctrl.pulses_per_s = 5000. Then run the program twice\n";
                myMjClass.reset_object();
                myMjClass.spawn_object(0);
                std::vector<double> pos;
                std::vector<double> y_actual;
                double start = 100;
                for (double i = start; i > start - 6; i -= 0.25) 
                    pos.push_back(i * 1e-3);
                std::vector<std::vector<double>> force_matrix;
                // go to start position
                luke::set_gripper_target_m(pos[0], pos[0], 5e-3);
                for (int i = 0; i < 100; i++) myMjClass.action_step();
                for (int i = 0; i < pos.size(); i++) {
                    // set the gripper target
                    luke::set_gripper_target_m(pos[0], pos[i], 5e-3);
                    // step the simulation to reach it
                    for (int j = 0; j < 5; j++) myMjClass.action_step();
                    // record the forces
                    std::vector<double> forces(3);
                    forces[0] = myMjClass.sim_sensors_SI_.read_finger1_gauge();
                    forces[1] = myMjClass.sim_sensors_SI_.read_finger2_gauge();
                    forces[2] = myMjClass.sim_sensors_SI_.read_finger3_gauge();
                    force_matrix.push_back(forces);
                    // record the actual y motor position
                    luke::Vec3 actual = luke::get_gripper_target_actual();
                    y_actual.push_back(actual.y);
                }
                // print out the final result
                luke::Vec3 objbox = luke::get_object_xyz_bounding_box(0);
                std::cout << "Sphere xyz bounding box is: "
                    << objbox.x << ", " << objbox.y << ", " << objbox.z << "\n";
                std::cout << "XY pos | Y actual | Gauge1 | Gauge2 | Gauge3 | Avg\n";
                for (int i = 0; i < force_matrix.size(); i++) {
                    double avg = (1/3.0) * (force_matrix[i][0] + force_matrix[i][1]
                        + force_matrix[i][2]);
                    std::cout << pos[i]*1e3 << " | "
                        << y_actual[i]*1e3 << " | "
                        << force_matrix[i][0] << " | "
                        << force_matrix[i][1] << " | "
                        << force_matrix[i][2] << " | "
                        << avg << "\n";
                }
                std::cout << "Finished. Sphere diameter was: "
                    << objbox.x * 1e3 << " mm\n";

                break;
            }
            case 50: {         // measure palm

                std::cout << "Running a force measurement program (palm)\n";
                std::cout << "WARNING: to run this set j_.ctrl.num_steps = 1 "
                    "and j_.ctrl.pulses_per_s = 5000. Then run the program twice\n";
                myMjClass.reset_object();
                myMjClass.spawn_object(0);
                std::vector<double> pos;
                std::vector<double> z_actual;
                double start = 60;
                for (double i = start; i < start + 25; i += 0.25) 
                    pos.push_back(i * 1e-3);
                std::vector<double> force_vector;
                // go to start position
                luke::set_gripper_target_m(130e-3, 130e-3, pos[0]);
                for (int i = 0; i < 100; i++) myMjClass.action_step();
                for (int i = 0; i < pos.size(); i++) {
                    // set the gripper target
                    luke::set_gripper_target_m(130e-3, 130e-3, pos[i]);
                    // step the simulation to reach it
                    for (int j = 0; j < 5; j++) myMjClass.action_step();
                    // record the force
                    double force = myMjClass.sim_sensors_SI_.read_palm_sensor();
                    force_vector.push_back(force);
                    // record the actual y motor position
                    luke::Vec3 actual = luke::get_gripper_target_actual();
                    z_actual.push_back(actual.z);
                }
                // print out the final result
                luke::Vec3 objbox = luke::get_object_xyz_bounding_box(0);
                std::cout << "Sphere xyz bounding box is: "
                    << objbox.x << ", " << objbox.y << ", " << objbox.z << "\n";
                std::cout << "XY pos | Z actual | Force\n";
                for (int i = 0; i < force_vector.size(); i++) {
                    std::cout << pos[i]*1e3 << " | "
                        << z_actual[i]*1e3 << " | "
                        << force_vector[i] << "\n";
                }
                std::cout << "Finished. Sphere diameter was: "
                    << objbox.x * 1e3 << " mm\n";

                break;
            }
            }
        }

        // stop if UI processed event
        if( it!=NULL || (state->type==mjEVENT_KEY && state->key==0) )
            return;
    }

    // shortcut not handled by UI
    if( state->type==mjEVENT_KEY && state->key!=0 )
    {
        switch( state->key )
        {
        case ' ':                   // Mode
            if( m )
            {
                settings.run = 1 - settings.run;
                pert.active = 0;
                mjui_update(-1, -1, &ui0, state, &con);
            }
            break;

        case mjKEY_RIGHT:           // step forward
            if( m && !settings.run )
            {
                cleartimers();
                myMjClass.step();
                // luke::step(m, d);
                profilerupdate();
                sensorupdate();
                updatesettings();

                manual_steps += 1;
                std::cout << "Manual step count: " << manual_steps << '\n';
            }
            break;

        case mjKEY_LEFT:            // step back
            if( m && !settings.run )
            {
                std::cout << "stepping backwards has not been implemented\n";
                // m->opt.timestep = -m->opt.timestep;
                // cleartimers();
                // myMjClass.step();
                // // luke::step(m, d);
                // m->opt.timestep = -m->opt.timestep;
                // profilerupdate();
                // sensorupdate();
                // updatesettings();

                std::cout << "Wiped the counter of manual key press steps\n";
                manual_steps = 0;
            }
            break;

        case mjKEY_DOWN:            // step forward 100
            if( m && !settings.run )
            {
                cleartimers();
                for( i=0; i<100; i++ )
                    myMjClass.step();
                    // luke::step(m, d);
                profilerupdate();
                sensorupdate();
                updatesettings();

                manual_steps += 100;
                std::cout << "Manual step count: " << manual_steps << '\n';
            }
            break;

        case mjKEY_UP:              // step back 100
            if( m && !settings.run )
            {
                std::cout << "stepping backwards has not been implemented\n";
                // m->opt.timestep = -m->opt.timestep;
                // cleartimers();
                // for( i=0; i<100; i++ )
                //     myMjClass.step();
                //     // luke::step(m, d);
                // m->opt.timestep = -m->opt.timestep;
                // profilerupdate();
                // sensorupdate();
                // updatesettings();
            }
            break;

        case mjKEY_PAGE_UP:         // select parent body
            if( m && pert.select>0 )
            {
                pert.select = m->body_parentid[pert.select];
                pert.skinselect = -1;

                // stop perturbation if world reached
                if( pert.select<=0 )
                    pert.active = 0;
            }

            break;

        case mjKEY_ESCAPE:          // free camera
            cam.type = mjCAMERA_FREE;
            settings.camera = 0;
            mjui_update(SECT_RENDERING, -1, &ui0, &uistate, &con);
            break;
        }

        return;
    }

    // 3D scroll
    if( state->type==mjEVENT_SCROLL && state->mouserect==3 && m )
    {
        // emulate vertical mouse motion = 5% of window height
        mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*state->sy, &scn, &cam);

        return;
    }

    // 3D press
    if( state->type==mjEVENT_PRESS && state->mouserect==3 && m )
    {
        // set perturbation
        int newperturb = 0;
        if( state->control && pert.select>0 )
        {
            // right: translate;  left: rotate
            if( state->right )
                newperturb = mjPERT_TRANSLATE;
            else if( state->left )
                newperturb = mjPERT_ROTATE;

            // perturbation onset: reset reference
            if( newperturb && !pert.active )
                mjv_initPerturb(m, d, &scn, &pert);
        }
        pert.active = newperturb;

        // handle double-click
        if( state->doubleclick )
        {
            // determine selection mode
            int selmode;
            if( state->button==mjBUTTON_LEFT )
                selmode = 1;
            else if( state->control )
                selmode = 3;
            else
                selmode = 2;

            // find geom and 3D click point, get corresponding body
            mjrRect r = state->rect[3];
            mjtNum selpnt[3];
            int selgeom, selskin;
            int selbody = mjv_select(m, d, &vopt,
                                     (mjtNum)r.width/(mjtNum)r.height,
                                     (mjtNum)(state->x-r.left)/(mjtNum)r.width,
                                     (mjtNum)(state->y-r.bottom)/(mjtNum)r.height,
                                     &scn, selpnt, &selgeom, &selskin);

            // set lookat point, start tracking is requested
            if( selmode==2 || selmode==3 )
            {
                // copy selpnt if anything clicked
                if( selbody>=0 )
                    mju_copy3(cam.lookat, selpnt);

                // switch to tracking camera if dynamic body clicked
                if( selmode==3 && selbody>0 )
                {
                    // mujoco camera
                    cam.type = mjCAMERA_TRACKING;
                    cam.trackbodyid = selbody;
                    cam.fixedcamid = -1;

                    // UI camera
                    settings.camera = 1;
                    mjui_update(SECT_RENDERING, -1, &ui0, &uistate, &con);
                }
            }

            // set body selection
            else
            {
                if( selbody>=0 )
                {
                    // record selection
                    pert.select = selbody;
                    pert.skinselect = selskin;

                    // compute localpos
                    mjtNum tmp[3];
                    mju_sub3(tmp, selpnt, d->xpos+3*pert.select);
                    mju_mulMatTVec(pert.localpos, d->xmat+9*pert.select, tmp, 3, 3);
                }
                else
                {
                    pert.select = 0;
                    pert.skinselect = -1;
                }
            }

            // stop perturbation on select
            pert.active = 0;
        }

        return;
    }

    // 3D release
    if( state->type==mjEVENT_RELEASE && state->dragrect==3 && m )
    {
        // stop perturbation
        pert.active = 0;

        return;
    }

    // 3D move
    if( state->type==mjEVENT_MOVE && state->dragrect==3 && m )
    {
        // determine action based on mouse button
        mjtMouse action;
        if( state->right )
            action = state->shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
        else if( state->left )
            action = state->shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
        else
            action = mjMOUSE_ZOOM;

        // move perturb or camera
        mjrRect r = state->rect[3];
        if( pert.active )
            mjv_movePerturb(m, d, action, state->dx/r.height, -state->dy/r.height,
                            &scn, &pert);
        else
            mjv_moveCamera(m, action, state->dx/r.height, -state->dy/r.height,
                           &scn, &cam);

        return;
    }
}



//--------------------------- rendering and simulation ----------------------------------


// prepare to render
void prepare(void)
{
    // data for FPS calculation
    static double lastupdatetm = 0;

    // update interval, save update time
    double tmnow = glfwGetTime();
    double interval = tmnow - lastupdatetm;
    interval = mjMIN(1, mjMAX(0.0001, interval));
    lastupdatetm = tmnow;

    // no model: nothing to do
    if( !m )
        return;

    // update scene
    mjv_updateScene(m, d, &vopt, &pert, &cam, mjCAT_ALL, &scn);

    // update watch
    if( settings.ui0 && ui0.sect[SECT_WATCH].state )
    {
		watch();
		mjui_update(SECT_WATCH, -1, &ui0, &uistate, &con);
    }

    // ipdate joint
    if( settings.ui1 && ui1.sect[SECT_JOINT].state )
            mjui_update(SECT_JOINT, -1, &ui1, &uistate, &con);

    // update info text
    if( settings.info )
        infotext(info_title, info_content, interval);

    // update profiler
    if( settings.profiler && settings.run )
        profilerupdate();

    // update sensor
    if( settings.sensor && settings.run )
        sensorupdate();

    // added by luke
    lukesensorfigsupdate();
    lukestepperfigsupdate();

    // clear timers once profiler info has been copied
    cleartimers();
}



// render im main thread (while simulating in background thread)
void render_MS(GLFWwindow* window)
{
    // get 3D rectangle and reduced for profiler
    mjrRect rect = uistate.rect[3];
    mjrRect smallrect = rect;
    if( settings.profiler )
        smallrect.width = rect.width - rect.width/4;

    // no model
    if( !m )
    {
        // blank screen
        mjr_rectangle(rect, 0.2f, 0.3f, 0.4f, 1);

        // label
        if( settings.loadrequest )
            mjr_overlay(mjFONT_BIG, mjGRID_TOPRIGHT, smallrect,
                        "loading", NULL, &con);
        else
            mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, rect,
                        "Drag-and-drop model file here", 0, &con);

        // render uis
        if( settings.ui0 )
            mjui_render(&ui0, &uistate, &con);
        if( settings.ui1 )
            mjui_render(&ui1, &uistate, &con);

        // finalize
        glfwSwapBuffers(window);

        return;
    }

    // render scene
    mjr_render(rect, &scn, &con);

    // show pause/loading label
    if( !settings.run || settings.loadrequest )
        mjr_overlay(mjFONT_BIG, mjGRID_TOPRIGHT, smallrect,
                    settings.loadrequest ? "loading" : "pause", NULL, &con);

    // show ui 0
    if( settings.ui0 )
        mjui_render(&ui0, &uistate, &con);

    // show ui 1
    if( settings.ui1 )
        mjui_render(&ui1, &uistate, &con);

    // show help
    if( settings.help )
        mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, rect, help_title, help_content, &con);

    // show info
    if( settings.info )
        mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, rect,
                    info_title, info_content, &con);

    // show profiler
    if( settings.profiler )
        profilershow(rect);

    // show sensor
    if( settings.sensor )
        sensorshow(smallrect);

    // added by luke
    if (settings.render_depth or settings.render_rgb) {
        // // render::render_rgb(settings.render_rgb);
        // // render::render_depth(settings.render_depth);
        // static bool first_call = true;
        // if (first_call) {
        //     render::init_camera(myMjClass);
        //     // render::init_window(myMjClass);
        //     first_call = false;
        // }
        // render::read_rgbd();
        // render::render_rgbd_feed();
        std::cout << "render_depth and render_rgb are disabled\n";
    }

    // added by luke
    lukesensorfigshow(smallrect);
    lukestepperfigshow(smallrect);

    // finalize
    glfwSwapBuffers(window);
}



// simulate in background thread (while rendering in main thread)
void simulate(void)
{
    // cpu-sim syncronization point
    double cpusync = 0;
    mjtNum simsync = 0;

    // run until asked to exit
    while( !settings.exitrequest )
    {
        // sleep for 1 ms or yield, to let main thread run
        //  yield results in busy wait - which has better timing but kills battery life
        if( settings.run && settings.busywait )
            std::this_thread::yield();
        else
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // start exclusive access
        mtx.lock();

        // run only if model is present
        if( m )
        {
            // record start time
            // double startwalltm = glfwGetTime();

            // running
            if( settings.run )
            {
                // record cpu time at start of iteration
                double tmstart = glfwGetTime();

                // out-of-sync (for any reason)
                if( d->time<simsync || tmstart<cpusync || cpusync==0 ||
                    mju_abs((d->time-simsync)-(tmstart-cpusync))>syncmisalign )
                {
                    // re-sync
                    cpusync = tmstart;
                    simsync = d->time;


                    /* original code
                    // clear old perturbations, apply new
                    mju_zero(d->xfrc_applied, 6*m->nbody);
                    mjv_applyPerturbPose(m, d, &pert, 0);  // move mocap bodies only
                    mjv_applyPerturbForce(m, d, &pert);

                    // run single step, let next iteration deal with timing
                    mj_step(m, d);
                    */

                    // // added by luke
                    // luke::before_step(m, d);
                    // luke::step(m, d);
                    // luke::after_step(m, d);

                    myMjClass.step();
                }

                // in-sync
                else
                {
                    // step while simtime lags behind cputime, and within safefactor
                    while( (d->time-simsync)<(glfwGetTime()-cpusync) &&
                           (glfwGetTime()-tmstart)<refreshfactor/vmode.refreshRate )
                    {
                        /* original code
                        // clear old perturbations, apply new
                        mju_zero(d->xfrc_applied, 6*m->nbody);
                        mjv_applyPerturbPose(m, d, &pert, 0);  // move mocap bodies only
                        mjv_applyPerturbForce(m, d, &pert);
                        */

                        // run mj_step
                        mjtNum prevtm = d->time;
                        // mj_step(m, d);

                        // // added by luke - should this have before_step(...)?
                        // luke::step(m, d);
                        // luke::after_step(m, d);
                        myMjClass.step();

                        // break on reset
                        if( d->time<prevtm )
                            break;
                    }
                }
            }

            // paused
            else
            {
                // apply pose perturbation
                mjv_applyPerturbPose(m, d, &pert, 1);      // move mocap and dynamic bodies

                // run mj_forward, to update rendering and joint sliders
                mj_forward(m, d);
            }
        }

        // end exclusive access
        mtx.unlock();
    }
}



//-------------------------------- init and main ----------------------------------------

// added by luke
void luke_openGL_error_callback(int error, const char *msg)
{
    std::string s;
    s = " [" + std::to_string(error) + "] " + msg + '\n';
    std::cerr << s << std::endl;
}

// initalize everything
void init(void)
{
    // print version, check compatibility
    printf("MuJoCo Pro version %.2lf\n", 0.01*mj_version());
    if( mjVERSION_HEADER!=mj_version() )
        mju_error("Headers and library have different versions");

    // init GLFW, set timer callback (milliseconds)
    if (!glfwInit())
        mju_error("could not initialize GLFW");
    mjcb_time = timer;

    // // added by luke for debugging, then commented out as it caused errors
    // glfwSetErrorCallback( luke_openGL_error_callback );
    // glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 3 );
    // glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 2 );

    // multisampling
    glfwWindowHint(GLFW_SAMPLES, 4);
    glfwWindowHint(GLFW_VISIBLE, 1);

    // get videomode and save
    vmode = *glfwGetVideoMode(glfwGetPrimaryMonitor());

    // create window
    window = glfwCreateWindow((2*vmode.width)/3, (2*vmode.height)/3,
                              "Simulate", NULL, NULL);
    if( !window )
    {
        glfwTerminate();
        mju_error("could not create window");
    }

    // save window position and size
    glfwGetWindowPos(window, windowpos, windowpos+1);
    glfwGetWindowSize(window, windowsize, windowsize+1);

    // make context current, set v-sync
    glfwMakeContextCurrent(window);
    glfwSwapInterval(settings.vsync);

    // init abstract visualization
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&vopt);
    profilerinit();
    sensorinit();
    lukesensorfigsinit();   // added by luke
    lukestepperfigsinit();  // added by luke

    // make empty scene
    mjv_defaultScene(&scn);
    mjv_makeScene(NULL, &scn, maxgeom);

    // select default font
    int fontscale = uiFontScale(window);
    settings.font = fontscale/50 - 1;

    // make empty context
    mjr_defaultContext(&con);
    mjr_makeContext(NULL, &con, fontscale);

    // set GLFW callbacks
    uiSetCallback(window, &uistate, uiEvent, uiLayout);
    glfwSetWindowRefreshCallback(window, render_MS);
    glfwSetDropCallback(window, drop);

    // init state and uis
    memset(&uistate, 0, sizeof(mjuiState));
    memset(&ui0, 0, sizeof(mjUI));
    memset(&ui1, 0, sizeof(mjUI));
    ui0.spacing = mjui_themeSpacing(settings.spacing);
    ui0.color = mjui_themeColor(settings.color);
    ui0.predicate = uiPredicate;
    ui0.rectid = 1;
    ui0.auxid = 0;
    ui1.spacing = mjui_themeSpacing(settings.spacing);
    ui1.color = mjui_themeColor(settings.color);
    ui1.predicate = uiPredicate;
    ui1.rectid = 2;
    ui1.auxid = 1;

    // populate uis with standard sections
    mjui_add(&ui0, defFile);
    mjui_add(&ui0, defOption);
    mjui_add(&ui0, defSimulation);
    mjui_add(&ui0, defWatch);
    uiModify(window, &ui0, &uistate, &con);
    uiModify(window, &ui1, &uistate, &con);
}

// run event loop
int main(int argc, char** argv)
{

    printf("Started\n");

    // initialize everything
    init();

    // temporary mjclass object to parse command line
    MjClass temp;
    std::string filepath = temp.file_from_from_command_line(argc, argv);

    // echo file input
    printf("mysimulate loading mjcf from: %s\n", filepath.c_str());

    // mju_strncpy(filename, argv[1], 1000);
    mju_strncpy(filename, filepath.c_str(), 1000);
    settings.loadrequest = 2;

    // start simulation thread
    std::thread simthread(simulate);

    // event loop
    while( !glfwWindowShouldClose(window) && !settings.exitrequest )
    {
        // start exclusive access (block simulation thread)
        mtx.lock();

        // load model (not on first pass, to show "loading" label)
        if( settings.loadrequest==1 )
            loadmodel();
        else if( settings.loadrequest>1 )
            settings.loadrequest = 1;

        // handle events (calls all callbacks)
        glfwPollEvents();

        // prepare to render
        prepare();

        // end exclusive access (allow simulation thread to run)
        mtx.unlock();

        // render while simulation is running
        render_MS(window);
    }

    // stop simulation thread
    settings.exitrequest = 1;
    simthread.join();

    // delete everything we allocated
    uiClearCallback(window);

    /* dont need to delete due to myMjClass destructor */
    // mj_deleteData(d);
    // mj_deleteModel(m);
    
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // terminate GLFW (crashes with Linux NVidia drivers)
    #if defined(__APPLE__) || defined(_WIN32)
        glfwTerminate();
    #endif

    return 0;
}


