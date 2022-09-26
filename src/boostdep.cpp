#include "boostdep.h"

namespace luke_boost {

double eval_xi(double s, double omega, double C, double k)
{
  /* find xi(s) */

  double K = boost::math::ellint_1(k);
  double E = boost::math::ellint_2(k);
  double Z1 = boost::math::jacobi_zeta(k, omega + C);
  double Z2 = boost::math::jacobi_zeta(k, omega * s + C);

  double xi = (((2 * E) / K) - 1) * (1 - s) + (2 / omega) * (Z1 - Z2);

  return xi;
}

double eval_eta(double s, double omega, double C, double k)
{
  /* find eta(s) */

  double cn1 = boost::math::jacobi_cn(k, omega + C);
  double cn2 = boost::math::jacobi_cn(k, omega * s + C);

  double eta = -((2 * k) / omega) * (cn1 - cn2);

  return eta;
}

double jacobi_arcsn(double x, double k)
{
  /* use elliptic integrals for the inverse of the jacobi sine function */

  return boost::math::ellint_1(k, asin(x));
}

ArcPoint get_point(double s, double P, double M0, double L, double EI, double alpha)
{
  /* find a point along the arc length s given the loading parameters:
        P - applied free end force
        M0 - applied free end moment
        L - length of beam
        EI - flexural rigidity
        alpha - angle from applied force to cantilever base curve (eg vertical force gives alpha = pi/2)
        gamma - angle from applied force to the horizontal axis
  */

  double kappa_0 = (M0 * L / EI);
  double omega = sqrt((P * pow(L, 2)) / EI);
  double k = sqrt(pow(sin(alpha * 0.5), 2) + pow(kappa_0 / (2 * omega), 2));
  double C;

  double x = sin((alpha / 2) / k); 

  if (kappa_0 < 0) {
    C = jacobi_arcsn(x, k);
  }
  else if (kappa_0 > 0) {
    double K = boost::math::ellint_1(k);
    C = 2 * K - jacobi_arcsn(x, k);
  }
  else {
    double K = boost::math::ellint_1(k);
    C = K;
  }

  double gamma = 2 * asin(k * boost::math::jacobi_sn(k, omega + C));
  gamma = -M_PI_2;

  // // adjust alpha to stay independent of gamma
  // alpha += gamma;

  return get_point(s, omega, C, k, gamma);
}

ArcPoint get_point(double s, double omega, double C, double k, double gamma)
{
  /* evaluate a point on the curve */

  ArcPoint point;

  point.s = s;
  point.xi = eval_xi(s, omega, C, k);
  point.eta = eval_eta(s, omega, C, k);
  point.x = point.xi * cos(gamma) + point.eta * sin(gamma);
  point.y = -point.xi * sin(gamma) + point.eta * cos(gamma);

  // calculate angle (phi) and curvature (kappa)
  double sn = boost::math::jacobi_sn(omega * s + C, k);
  double cn = boost::math::jacobi_cn(omega * s + C, k);
  point.phi = 2 * asin(k * sn);
  point.kappa = - 2 * omega * k * cn;

  return point;
}

}; // namespace