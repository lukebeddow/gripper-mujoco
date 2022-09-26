#include <boost/math/special_functions.hpp>

namespace luke_boost {

struct ArcPoint {
  double s;           // arc length along beam
  double x;           // cartesian x position
  double y;           // cartesian y position
  double xi;          // local co-ord, parallel to force
  double eta;         // local co-ord, perpendicular to force
  double phi;         // angle from the horizontal
  double kappa;       // beam curvature
};

double eval_xi(double s, double omega, double C, double k);
double eval_eta(double s, double omega, double C, double k);
double jacobi_arcsn(double x, double k);
ArcPoint get_point(double s, double P, double M0, double L, double EI, double alpha);
ArcPoint get_point(double s, double omega, double C, double k, double gamma);

};