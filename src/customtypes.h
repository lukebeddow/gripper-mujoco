#ifndef CUSTOMTYPES_H_
#define CUSTOMTYPES_H_

#include "mujoco.h"

#include <vector>
#include <iostream>
#include <array>

// include large custom types (all in namespace luke already)
#include "slidingwindow.h"
#include "gripper.h"
#include "mynum.h"

// next we will define some more smaller types in the namespace
namespace luke
{

/* ----- type definitions ----- */

// define the precision of the gauges
typedef float gfloat;

// define type for RGB image data
typedef uint8_t rgbint;

/* ----- custom types ----- */

struct Vec3 {
  double x {};
  double y {};
  double z {};
};

class Box2d {
private:

  /* corners must be in counter-clockwise order, ie: 
    
    ^+ve y
    
    x4,y4          x3,y3
  
  
  
    x1, y1         x2, y2     -> +ve x
  */
  double x[4]; // Array to store X coordinates of corners
  double y[4]; // Array to store Y coordinates of corners

public:
  Box2d() {
    // Initialize corners with zeros
    for (int i = 0; i < 4; ++i) {
      x[i] = 0.0;
      y[i] = 0.0;
    }
  }

  // Function to initialize the region based on a center point (cx, cy) and X and Y dimensions
  void initCentre(double cx, double cy, double width, double height) {
    double halfWidth = width / 2.0;
    double halfHeight = height / 2.0;

    // Set the corners based on the center point and dimensions
    x[0] = cx - halfWidth; y[0] = cy - halfHeight; // Bottom-left corner
    x[1] = cx + halfWidth; y[1] = cy - halfHeight; // Bottom-right corner
    x[2] = cx + halfWidth; y[2] = cy + halfHeight; // Top-right corner
    x[3] = cx - halfWidth; y[3] = cy + halfHeight; // Top-left corner
  }

  // Function to initialize the region based on the bottom-left (x1, y1) and top-right (x2, y2) corners
  void initCorners(double x1, double y1, double x2, double y2) {
    // Set the corners based on the provided coordinates
    x[0] = x1; y[0] = y1; // Bottom-left corner
    x[1] = x2; y[1] = y1; // Bottom-right corner
    x[2] = x2; y[2] = y2; // Top-right corner
    x[3] = x1; y[3] = y2; // Top-left corner
  }


  // // Function to check if this region overlaps with another region
  // bool overlapsWith(const Box2d& other) {
  //   for (int i = 0; i < 4; ++i) {
  //     double x1 = x[i];
  //     double y1 = y[i];
  //     double x2 = x[(i + 1) % 4];
  //     double y2 = y[(i + 1) % 4];

  //     for (int j = 0; j < 4; ++j) {
  //       double x3 = other.x[j];
  //       double y3 = other.y[j];
  //       double x4 = other.x[(j + 1) % 4];
  //       double y4 = other.y[(j + 1) % 4];

  //       double crossProduct1 = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
  //       double crossProduct2 = (x2 - x1) * (y4 - y1) - (y2 - y1) * (x4 - x1);
  //       double crossProduct3 = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3);
  //       double crossProduct4 = (x4 - x3) * (y2 - y3) - (y4 - y3) * (x2 - x3);

  //       // Check if the edges intersect
  //       if (((crossProduct1 * crossProduct2) <= 0) && ((crossProduct3 * crossProduct4) <= 0)) {
  //         return true; // There is an intersection
  //       }
  //     }
  //   }

  //   return false; // No intersection found
  // }

 // Function to calculate the dot product of two vectors
    static double dotProduct(double x1, double y1, double x2, double y2) {
        return x1 * x2 + y1 * y2;
    }

    // Member function to check if two rotated regions overlap using SAT with a minimum separation distance
    bool overlapsWith(const Box2d& other, double minDistance) const {
        for (int i = 0; i < 4; ++i) {
            int j = (i + 1) % 4; // Get the index of the next point

            // Calculate the edge vector for this region
            double edgeX1 = x[j] - x[i];
            double edgeY1 = y[j] - y[i];

            // Calculate the perpendicular vector to the edge
            double perpX1 = -edgeY1;
            double perpY1 = edgeX1;

            // Normalize the perpendicular vector
            double length1 = std::sqrt(dotProduct(perpX1, perpY1, perpX1, perpY1));
            perpX1 /= length1;
            perpY1 /= length1;

            // Project the corners of both regions onto the perpendicular vector
            double min1 = dotProduct(x[0], y[0], perpX1, perpY1);
            double max1 = min1;

            double min2 = dotProduct(other.x[0], other.y[0], perpX1, perpY1);
            double max2 = min2;

            for (int k = 1; k < 4; ++k) {
                double projection1 = dotProduct(x[k], y[k], perpX1, perpY1);
                double projection2 = dotProduct(other.x[k], other.y[k], perpX1, perpY1);

                if (projection1 < min1) min1 = projection1;
                if (projection1 > max1) max1 = projection1;
                if (projection2 < min2) min2 = projection2;
                if (projection2 > max2) max2 = projection2;
            }

            // Check for overlap on this axis with minimum separation distance
            if (max1 + minDistance < min2 || max2 + minDistance < min1) {
                return false; // Separating axis found
            }
        }

        return true; // No separating axis found, overlap detected
    }
  

  // bool overlapsWith(const Box2d& other, double minDistance) {
  //   for (int i = 0; i < 4; ++i) {
  //       // Get the vertices of this region
  //       double x1 = x[i];
  //       double y1 = y[i];

  //       // Get the next vertex
  //       int nextIdx = (i + 1) % 4;
  //       double x2 = x[nextIdx];
  //       double y2 = y[nextIdx];

  //       // Calculate the edge vector
  //       double edgeX = x2 - x1;
  //       double edgeY = y2 - y1;

  //       // Calculate the perpendicular vector
  //       double perpX = -edgeY;
  //       double perpY = edgeX;

  //       // Normalize the perpendicular vector
  //       double length = sqrt(perpX * perpX + perpY * perpY);
  //       perpX /= length;
  //       perpY /= length;

  //       // Project the vertices of both regions onto the perpendicular vector
  //       double minThis = 1e9;
  //       double maxThis = -1e9;
  //       double minOther = 1e9;
  //       double maxOther = -1e9;

  //       for (int j = 0; j < 4; ++j) {
  //           double dotThis = (x[j] - x1) * perpX + (y[j] - y1) * perpY;
  //           minThis = std::min(minThis, dotThis);
  //           maxThis = std::max(maxThis, dotThis);

  //           double dotOther = (other.x[j] - x1) * perpX + (other.y[j] - y1) * perpY;
  //           minOther = std::min(minOther, dotOther);
  //           maxOther = std::max(maxOther, dotOther);
  //       }

  //       // Check for overlap on the projected axis with minimum separation distance
  //       if (!(maxThis + minDistance >= minOther && maxOther + minDistance >= minThis)) {
  //           return false; // No overlap with the specified minimum distance
  //       }
  //   }

  //   return true; // Overlap with the specified minimum distance on all axes
  // }

  bool inbounds(double xmin, double ymin, double xmax, double ymax)
  {
    /* check if the region is within certain bounds */

    for (int i = 0; i < 4; i++) {
      if (x[i] < xmin or x[i] > xmax or
          y[i] < ymin or y[i] > ymax) {
        return false;
      }
    }

    return true;
  }

// Function to rotate the region by a specified angle (in radians) around its center
  void rotate(double theta) {
    double centerX = (x[0] + x[1] + x[2] + x[3]) / 4.0;
    double centerY = (y[0] + y[1] + y[2] + y[3]) / 4.0;

    for (int i = 0; i < 4; ++i) {
      double newX = centerX + (x[i] - centerX) * cos(theta) - (y[i] - centerY) * sin(theta);
      double newY = centerY + (x[i] - centerX) * sin(theta) + (y[i] - centerY) * cos(theta);
      x[i] = newX;
      y[i] = newY;
    }
  }

  // Function to display the coordinates of the corners
  void printCorners() {
    for (int i = 0; i < 4; ++i) {
      std::cout << "Corner " << (i + 1) << ": (" << x[i] << ", " << y[i] << ")\n";
    }
  }
};

// class Box2d {

// private:
//   double x1 {}, y1 {}; // Bottom-left corner
//   double x2 {}, y2 {}; // Bottom-right corner
//   double x3 {}, y3 {}; // Top-right corner
//   double x4 {}, y4 {}; // Top-left corner

// public:
//   // Constructor to initialize the region's corners
//   Region() {}
//   Region(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4)
//     : x1(x1), y1(y1), x2(x2), y2(y2), x3(x3), y3(y3), x4(x4), y4(y4) {}

//   void initCentre(double x, double y, double xlength, double ylength)
//   {
//     /* initialise a Box2d given a centre point, and the length of the two sides */

//     x1 = x - 0.5 * xlength;
//     x2 = x + 0.5 * xlength;
//     x3 = x + 0.5 * xlength;
//     x4 = x - 0.5 * xlength;

//     y1 = y - 0.5 * ylength;
//     y2 = y - 0.5 * ylength;
//     y3 = y + 0.5 * ylength;
//     y4 = y + 0.5 * ylength;
//   }

//   void initCorners(double x1_, double y1_, double x2_, double y2_)
//   {
//     /* initialise a Box2d given two opposite corners */

//     x1 = x1_;
//     x2 = x2_;
//     x3 = x2_;
//     x4 = x1_;

//     y1 = y1_;
//     y2 = y1_;
//     y3 = y2_;
//     y4 = y2_;
//   }

//   // Function to rotate the region about its center by a given angle (in radians)
//   void rotate(double theta) {
//     // Calculate the center of the region
//     double centerX = (x1 + x2 + x3 + x4) / 4.0;
//     double centerY = (y1 + y2 + y3 + y4) / 4.0;

//     // Rotate each corner about the center
//     double cosTheta = cos(theta);
//     double sinTheta = sin(theta);

//     double newX1 = centerX + (x1 - centerX) * cosTheta - (y1 - centerY) * sinTheta;
//     double newY1 = centerY + (x1 - centerX) * sinTheta + (y1 - centerY) * cosTheta;

//     double newX2 = centerX + (x2 - centerX) * cosTheta - (y2 - centerY) * sinTheta;
//     double newY2 = centerY + (x2 - centerX) * sinTheta + (y2 - centerY) * cosTheta;

//     double newX3 = centerX + (x3 - centerX) * cosTheta - (y3 - centerY) * sinTheta;
//     double newY3 = centerY + (x3 - centerX) * sinTheta + (y3 - centerY) * cosTheta;

//     double newX4 = centerX + (x4 - centerX) * cosTheta - (y4 - centerY) * sinTheta;
//     double newY4 = centerY + (x4 - centerX) * sinTheta + (y4 - centerY) * cosTheta;

//     // Update the corner positions
//     x1 = newX1; y1 = newY1;
//     x2 = newX2; y2 = newY2;
//     x3 = newX3; y3 = newY3;
//     x4 = newX4; y4 = newY4;
//   }

//   // Function to check if this region overlaps with another region
//   bool overlapsWith(const Box2d& other) {
//     for (int i = 0; i < 4; ++i) {
//       int j = (i + 1) % 4; // Get the index of the next point for this region
//       int k = (i + 2) % 4; // Get the index of the next-next point for this region

//       double crossProduct1 = (x2 - x1) * (other.y1 - y1) - (y2 - y1) * (other.x1 - x1);
//       double crossProduct2 = (x2 - x1) * (other.y2 - y1) - (y2 - y1) * (other.x2 - x1);
//       double crossProduct3 = (other.x2 - other.x1) * (y1 - other.y1) - (other.y2 - other.y1) * (x1 - other.x1);
//       double crossProduct4 = (other.x2 - other.x1) * (y2 - other.y1) - (other.y2 - other.y1) * (x2 - other.x1);

//       // Check if the edges intersect
//       if (((crossProduct1 * crossProduct2) <= 0) && ((crossProduct3 * crossProduct4) <= 0)) {
//         return true; // There is an intersection
//       }
//     }

//     return false; // No intersection found
//   }
// };

// struct Box2d {

//   double x1 {};
//   double x2 {};
//   double y1 {};
//   double y2 {};

//   struct Region {
//     Vec3 points[4]; // Four corner points of the region
//   };

//   // Function to check if two rotated 2D regions are overlapping
//   bool areRotatedRegionsOverlapping(const Region &region1, const Region &region2) 
//   {
//     /* chatGPT function to determine if two regions overlap */

//     for (int i = 0; i < 4; ++i) {

//       int j = (i + 1) % 4; // Get the index of the next point

//       // Check if any of the edges of region1 intersect with any of the edges of region2
//       double x1 = region1.points[i].x;
//       double y1 = region1.points[i].y;
//       double x2 = region1.points[j].x;
//       double y2 = region1.points[j].y;

//       for (int k = 0; k < 4; ++k) {

//         int l = (k + 1) % 4; // Get the index of the next point in the other region

//         double x3 = region2.points[k].x;
//         double y3 = region2.points[k].y;
//         double x4 = region2.points[l].x;
//         double y4 = region2.points[l].y;

//         // Calculate the cross products to check if the edges intersect
//         double crossProduct1 = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
//         double crossProduct2 = (x2 - x1) * (y4 - y1) - (y2 - y1) * (x4 - x1);

//         double crossProduct3 = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3);
//         double crossProduct4 = (x4 - x3) * (y2 - y3) - (y4 - y3) * (x2 - x3);

//         // Check if the edges intersect
//         if (((crossProduct1 * crossProduct2) < 0) && ((crossProduct3 * crossProduct4) < 0)) {
//           return true; // There is an intersection
//         }
//       }
//     }

//     return false; // No intersection found
//   }

//   bool overlap(Box2d& box2)
//   {
//     /* determine whether two boxes overlap*/

//     // convert each box into a region

//     Region r1;
//     Region r2;

//     r1.points[0].x = x1;
//     r1.points[0].y = y1;
//     r1.points[1].x = x1;
//     r1.points[1].y = y2;
//     r1.points[2].x = x2;
//     r1.points[2].y = y2;
//     r1.points[3].x = x2;
//     r1.points[3].y = y1;

//     r2.points[0].x = box2.x1;
//     r2.points[0].y = box2.y1;
//     r2.points[1].x = box2.x1;
//     r2.points[1].y = box2.y2;
//     r2.points[2].x = box2.x2;
//     r2.points[2].y = box2.y2;
//     r2.points[3].x = box2.x2;
//     r2.points[3].y = box2.y1;

//     return areRotatedRegionsOverlapping(r1, r2);
//   }

//   void rotate(double theta_rad)
//   {
//     /* rotate the box about its centre point by theta in radians */

//     double centre_x = 0.5 * (x2 + x1);
//     double centre_y = 0.5 * (y2 + y1);

//     double local_x1 = x1 - centre_x;
//     double local_x2 = x2 - centre_x;
//     double local_y1 = y1 - centre_y;
//     double local_y2 = y2 - centre_y;

//     double cth = cos(theta_rad);
//     double sth = sin(theta_rad);

//     // do the rotation
//     x1 = cth * local_x1 - sth * local_y1 + centre_x;
//     y1 = sth * local_x1 + cth * local_y1 + centre_y;
//     x2 = cth * local_x2 - sth * local_y2 + centre_x;
//     y2 = sth * local_x2 + cth * local_y2 + centre_y;
//   }

//   void print()
//   {
//     /* print out the coordinates of the box */

//     std::cout << "(x1, y1, x2, y2) >> ("
//       << x1 << ", " << y1 << ", "
//       << x2 << ", " << y2 << ")\n";
//   }
// };

struct Forces {

  /* Here we will save forces extracted from the simulation.

  For the local finger frame: 
    x = axial, y = tangential, z = lateral, all +ve with object contact

  For the local palm frame:
    x = axial, -ve with object contact
  */

  bool empty = false;

  // forces applied on the object
  struct Obj {

    // global frame
    myNum net;           // net force on object, almost always [0,0,weight,0,0,0]
    myNum sum;           // sum of all contact forces, does not always = net or 0
    myNum finger1;
    myNum finger2;
    myNum finger3;
    myNum palm;
    myNum ground;

    // forces felt from the object in their local frame
    myNum finger1_local;  
    myNum finger2_local;
    myNum finger3_local;
    myNum palm_local;

    // need to initialise to zero to enable += running totals
    Obj() : net(6,1), sum(6,1), finger1(6,1), finger2(6,1), finger3(6,1),
            palm(6,1), ground(6,1), finger1_local(3,1), finger2_local(3,1),
            finger3_local(3,1), palm_local(3,1) {}

  } obj;

  // all forces involved (excluding unnamed geoms!)
  struct All {

    // global frame
    myNum finger1;
    myNum finger2;
    myNum finger3;
    myNum palm;

    // local frame
    myNum finger1_local;
    myNum finger2_local;
    myNum finger3_local;
    myNum palm_local;

    // need to initialise to zero to enable += running totals
    All() : finger1(6,1), finger2(6,1), finger3(6,1), palm(6,1), finger1_local(3,1),
            finger2_local(3,1), finger3_local(3,1), palm_local(6,1) {}

  } all;

  // ground forces on gripper fingers
  struct Gnd {

    // global frame
    myNum finger1;
    myNum finger2;
    myNum finger3;

    // local frame
    myNum finger1_local;
    myNum finger2_local;
    myNum finger3_local;

    // need to initialise to zero to enable += running totals
    Gnd() : finger1(6,1), finger2(6,1), finger3(6,1), finger1_local(3,1),
            finger2_local(3,1), finger3_local(3,1) {}

  } gnd;

  void print() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    print_gnd_global();
    print_gnd_local();    
    print_obj_global();
    print_obj_local();
    print_all_global();
    print_all_local();
  }

  void print_obj_global() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces on object in global frame:\n";
    std::cout << "net force (mag = " << obj.net.magnitude3() << "):\n"; obj.net.print();
    std::cout << "sum force (mag = " << obj.sum.magnitude3() << "):\n"; obj.sum.print();
    std::cout << "ground force (mag = " << obj.ground.magnitude3() << "):\n"; obj.ground.print();
    std::cout << "finger1 force (mag = " << obj.finger1.magnitude3() << "):\n"; obj.finger1.print();
    std::cout << "finger2 force (mag = " << obj.finger2.magnitude3() << "):\n"; obj.finger2.print();
    std::cout << "finger3 force (mag = " << obj.finger3.magnitude3() << "):\n"; obj.finger3.print();
    std::cout << "palm force (mag = " << obj.palm.magnitude3() << "):\n"; obj.palm.print();
  }

  void print_obj_local() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces from object in local frames:\n";
    std::cout << "finger1 local force (mag = " << obj.finger1_local.magnitude3() << "):\n"; obj.finger1_local.print();
    std::cout << "finger2 local force (mag = " << obj.finger2_local.magnitude3() << "):\n"; obj.finger2_local.print();
    std::cout << "finger3 local force (mag = " << obj.finger3_local.magnitude3() << "):\n"; obj.finger3_local.print();
    std::cout << "palm local force (mag = " << obj.palm.magnitude3() << "):\n"; obj.palm_local.print();
  }

  void print_all_global() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces from all named geoms in global frame:\n";
    std::cout << "finger1 force (mag = " << all.finger1.magnitude3() << "):\n"; all.finger1.print();
    std::cout << "finger2 force (mag = " << all.finger2.magnitude3() << "):\n"; all.finger2.print();
    std::cout << "finger3 force (mag = " << all.finger3.magnitude3() << "):\n"; all.finger3.print();
    std::cout << "palm force (mag = " << all.palm.magnitude3() << "):\n"; all.palm.print();
  }

  void print_all_local() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces from all named geoms on the gripper fingers:\n";
    std::cout << "finger1 local force (mag = " << all.finger1_local.magnitude3() << "):\n"; all.finger1_local.print();
    std::cout << "finger2 local force (mag = " << all.finger2_local.magnitude3() << "):\n"; all.finger2_local.print();
    std::cout << "finger3 local force (mag = " << all.finger3_local.magnitude3() << "):\n"; all.finger3_local.print();
    std::cout << "palm local force (mag = " << all.palm.magnitude3() << "):\n"; all.palm_local.print();
  }

  void print_gnd_global() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces from gnd named geoms in global frame:\n";
    std::cout << "finger1 force (mag = " << gnd.finger1.magnitude3() << "):\n"; gnd.finger1.print();
    std::cout << "finger2 force (mag = " << gnd.finger2.magnitude3() << "):\n"; gnd.finger2.print();
    std::cout << "finger3 force (mag = " << gnd.finger3.magnitude3() << "):\n"; gnd.finger3.print();
  }

  void print_gnd_local() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces from gnd named geoms in local frames:\n";
    std::cout << "finger1 local force (mag = " << gnd.finger1_local.magnitude3() << "):\n"; gnd.finger1_local.print();
    std::cout << "finger2 local force (mag = " << gnd.finger2_local.magnitude3() << "):\n"; gnd.finger2_local.print();
    std::cout << "finger3 local force (mag = " << gnd.finger3_local.magnitude3() << "):\n"; gnd.finger3_local.print();
 }

};

struct Forces_faster {

  /* Here we will save forces extracted from the simulation.

  For the local finger frame: 
    x = axial, y = tangential, z = lateral, all +ve with object contact

  For the local palm frame:
    x = axial, -ve with object contact
  */

  bool empty = false;

  // forces applied on the object
  struct Obj {

    // global frame
    rawNum net;           // net force on object, almost always [0,0,weight,0,0,0]
    rawNum sum;           // sum of all contact forces, does not always = net or 0
    rawNum finger1;
    rawNum finger2;
    rawNum finger3;
    rawNum palm;
    rawNum ground;

    // forces felt from the object in their local frame
    rawNum finger1_local;  
    rawNum finger2_local;
    rawNum finger3_local;
    rawNum palm_local;

    // need to initialise to zero to enable += running totals
    Obj() : net(6,1), sum(6,1), finger1(6,1), finger2(6,1), finger3(6,1),
            palm(6,1), ground(6,1), finger1_local(3,1), finger2_local(3,1),
            finger3_local(3,1), palm_local(3,1) {}

  };

  std::vector<Obj> obj;

  // all forces involved (excluding unnamed geoms!)
  struct All {

    // global frame
    rawNum finger1;
    rawNum finger2;
    rawNum finger3;
    rawNum palm;

    // local frame
    rawNum finger1_local;
    rawNum finger2_local;
    rawNum finger3_local;
    rawNum palm_local;

    // need to initialise to zero to enable += running totals
    All() : finger1(6,1), finger2(6,1), finger3(6,1), palm(6,1), finger1_local(3,1),
            finger2_local(3,1), finger3_local(3,1), palm_local(6,1) {}

  } all;

  // ground forces on gripper fingers
  struct Gnd {

    // global frame
    rawNum finger1;
    rawNum finger2;
    rawNum finger3;

    // local frame
    rawNum finger1_local;
    rawNum finger2_local;
    rawNum finger3_local;

    // need to initialise to zero to enable += running totals
    Gnd() : finger1(6,1), finger2(6,1), finger3(6,1), finger1_local(3,1),
            finger2_local(3,1), finger3_local(3,1) {}

  } gnd;

  void print() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    print_gnd_global();
    print_gnd_local();    
    print_obj_global();
    print_obj_local();
    print_all_global();
    print_all_local();
  }

  void print_obj_global() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    for (uint i = 0; i < obj.size(); i++) {
      std::cout << "Printing forces on object " << i << " in global frame:\n";
      std::cout << "net force (mag = " << obj[i].net.magnitude3() << "):\n"; obj[i].net.print();
      std::cout << "sum force (mag = " << obj[i].sum.magnitude3() << "):\n"; obj[i].sum.print();
      std::cout << "ground force (mag = " << obj[i].ground.magnitude3() << "):\n"; obj[i].ground.print();
      std::cout << "finger1 force (mag = " << obj[i].finger1.magnitude3() << "):\n"; obj[i].finger1.print();
      std::cout << "finger2 force (mag = " << obj[i].finger2.magnitude3() << "):\n"; obj[i].finger2.print();
      std::cout << "finger3 force (mag = " << obj[i].finger3.magnitude3() << "):\n"; obj[i].finger3.print();
      std::cout << "palm force (mag = " << obj[i].palm.magnitude3() << "):\n"; obj[i].palm.print();
    }
  }

  void print_obj_local() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    for (uint i = 0; i < obj.size(); i++) {
      std::cout << "Printing forces from object " << i << " in local frames:\n";
      std::cout << "finger1 local force (mag = " << obj[i].finger1_local.magnitude3() << "):\n"; obj[i].finger1_local.print();
      std::cout << "finger2 local force (mag = " << obj[i].finger2_local.magnitude3() << "):\n"; obj[i].finger2_local.print();
      std::cout << "finger3 local force (mag = " << obj[i].finger3_local.magnitude3() << "):\n"; obj[i].finger3_local.print();
      std::cout << "palm local force (mag = " << obj[i].palm.magnitude3() << "):\n"; obj[i].palm_local.print();
    }
  }

  void print_all_global() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces from all named geoms in global frame:\n";
    std::cout << "finger1 force (mag = " << all.finger1.magnitude3() << "):\n"; all.finger1.print();
    std::cout << "finger2 force (mag = " << all.finger2.magnitude3() << "):\n"; all.finger2.print();
    std::cout << "finger3 force (mag = " << all.finger3.magnitude3() << "):\n"; all.finger3.print();
    std::cout << "palm force (mag = " << all.palm.magnitude3() << "):\n"; all.palm.print();
  }

  void print_all_local() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces from all named geoms on the gripper fingers:\n";
    std::cout << "finger1 local force (mag = " << all.finger1_local.magnitude3() << "):\n"; all.finger1_local.print();
    std::cout << "finger2 local force (mag = " << all.finger2_local.magnitude3() << "):\n"; all.finger2_local.print();
    std::cout << "finger3 local force (mag = " << all.finger3_local.magnitude3() << "):\n"; all.finger3_local.print();
    std::cout << "palm local force (mag = " << all.palm.magnitude3() << "):\n"; all.palm_local.print();
  }

  void print_gnd_global() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces from gnd named geoms in global frame:\n";
    std::cout << "finger1 force (mag = " << gnd.finger1.magnitude3() << "):\n"; gnd.finger1.print();
    std::cout << "finger2 force (mag = " << gnd.finger2.magnitude3() << "):\n"; gnd.finger2.print();
    std::cout << "finger3 force (mag = " << gnd.finger3.magnitude3() << "):\n"; gnd.finger3.print();
  }

  void print_gnd_local() {
    if (empty) { std::cout << "Cannot print forces - it is empty\n"; return; }
    std::cout << "Printing forces from gnd named geoms in local frames:\n";
    std::cout << "finger1 local force (mag = " << gnd.finger1_local.magnitude3() << "):\n"; gnd.finger1_local.print();
    std::cout << "finger2 local force (mag = " << gnd.finger2_local.magnitude3() << "):\n"; gnd.finger2_local.print();
    std::cout << "finger3 local force (mag = " << gnd.finger3_local.magnitude3() << "):\n"; gnd.finger3_local.print();
 }

};

struct Gain {

  // gains for x,y,z motors
  double x = 0;
  double y = 0;
  double z = 0;

  // constructors
  Gain() {};
  Gain(double all) : x(all), y(all), z(all) {};
  Gain(double x, double y, double z) : x(x), y(y), z(z) {};

  // functions
  void set(double all) {
    x = all; y = all; z = all;
  }
  void set(double x_, double y_, double z_) {
    x = x_; y = y_; z = z_;
  }
};

struct QPos {

  // data
  double x, y, z, qx, qy, qz, qw;

  // constructor
  QPos() : x(0), y(0), z(0), qx(0), qy(0), qz(0), qw(1) {}
  QPos(double x, double y, double z, double qx, double qy, double qz, double qw)
    : x(x), y(y), z(z), qx(qx), qy(qy), qz(qz), qw(qw) {}
  void reset() { x = y = z = qx = qy = qz = 0; qw = 1; }

  // update using mjData and qpos address (idx)
  void update(const mjModel* m, const mjData* d, int idx) {
    if (idx == -1) {
      x = y = z = qx = qy = qz = 0; qw = 1;
      return;
    }
    if (idx + 6 > m->nq) {
      throw std::runtime_error("qpos.update() is out of range");
    }
    x = d->qpos[idx];
    y = d->qpos[idx + 1];
    z = d->qpos[idx + 2];
    qx = d->qpos[idx + 3];
    qy = d->qpos[idx + 4];
    qz = d->qpos[idx + 5];
    qw = d->qpos[idx + 6];
  }

  void print() {
    std::printf("QPos: xyz = (%.3f, %.3f, %.3f) quat = (%.3f, %.3f, %.3f, %.3f)\n",
      x, y, z, qx, qy, qz, qw);
  }
};

struct JointStates {

  // list of all possible joints and their state
  double gripper_x { 0.0 };
  double gripper_y { 0.0 };
  double gripper_z { 0.0 };
  double base_x { 0.0 };
  double base_y { 0.0 };
  double base_z { 0.0 };
  double base_roll { 0.0 };
  double base_pitch { 0.0 };
  double base_yaw { 0.0 };
};

struct RGBD
{
  std::vector<rgbint> rgb;
  std::vector<float> depth;
};

struct Base {

  /* describing the gripper base motions */
  
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;

  Base() : x(0), y(0), z(0), roll(0), pitch(0), yaw(0) {}

  Base(double x, double y, double z, double roll, double pitch, double yaw)
    : x(x), y(y), z(z), roll(roll), pitch(pitch), yaw(yaw)
    {}

  void reset() {
    x = y = z = roll = pitch = yaw = 0.0;
  }

  std::vector<double> to_vec() {
    std::vector<double> out { x, y, z, roll, pitch, yaw };
    return out;
  }

};

struct Target {

  Gripper end;                        // final target destination
  Gripper next;                       // target at the next step

  Base base;
  Base base_min;
  Base base_max;

  // std::array<double, 6> base {};      // target of base joints (only [0] used for z)
  std::array<double, 7> panda {};     // target for panda joints (never used)

  // for real life tracking what has just moved
  struct Robot {
    enum {
      none = 0,
      gripper,
      panda
    };
  };

  // what was the last robot motion achieved with
  int last_robot = Robot::none;

  /* ground is at -10mm, max finger tilt from start before tips touch lifts them another
  10mm, then we add 10mm of padding to get +-30mm */
  // static constexpr double base_z_min = -30e-3;
  // static constexpr double base_z_max = 30e-3;

  // calibration value since z height will not be perfect from controller stiffness
  double z_offset = 0;

  // static constexpr std::array<double, 6> base_lims_max {0.1, 1, 1, 1, 1, 1};
  // static constexpr std::array<double, 6> base_lims_min {-0.1, -1, -1, -1, -1, -1};

  // for testing, only used if flag 'log_test_data' is true in update_stepper(...) in myfunctions.cpp
  int datanum = 400;
  SlidingWindow<float> timedata {datanum};
  SlidingWindow<float> target_stepperx {datanum};
  SlidingWindow<float> target_steppery {datanum};
  SlidingWindow<float> target_stepperz {datanum};
  SlidingWindow<float> target_basex {datanum};
  SlidingWindow<float> target_basey {datanum};
  SlidingWindow<float> target_basez {datanum};
  SlidingWindow<float> target_baseroll {datanum};
  SlidingWindow<float> target_basepitch {datanum};
  SlidingWindow<float> target_baseyaw {datanum};
  SlidingWindow<float> actual_stepperx {datanum};
  SlidingWindow<float> actual_steppery {datanum};
  SlidingWindow<float> actual_stepperz {datanum};
  SlidingWindow<float> actual_basex {datanum};
  SlidingWindow<float> actual_basey {datanum};
  SlidingWindow<float> actual_basez {datanum};
  SlidingWindow<float> actual_baseroll {datanum};
  SlidingWindow<float> actual_basepitch {datanum};
  SlidingWindow<float> actual_baseyaw {datanum};

  void reset() {
    end.reset();
    next.reset();
    base.reset();
    panda.fill(0);
  }

  JointStates get_target_m()
  {
    /* returns in metres (or radians for rpy) the joint states */

    JointStates state;

    state.gripper_x = end.get_x_m();
    state.gripper_y = end.get_y_m();
    state.gripper_z = end.get_z_m();
    state.base_x = base.x;
    state.base_y = base.y;
    state.base_z = base.z;
    state.base_roll = base.roll;
    state.base_pitch = base.pitch;
    state.base_yaw = base.yaw;

    return state;
  }

  bool x_moving() {
    if (end.get_x_step() == next.get_x_step()) {
      return false;
    }
    return true;
  }

  bool y_moving() {
    if (end.get_y_step() == next.get_y_step()) {
      return false;
    }
    return true;
  }

  bool z_moving() {
    if (end.get_z_step() == next.get_z_step()) {
      return false;
    }
    return true;
  }

  bool prismatic_moving() {
    return x_moving();
  }

  bool revolute_moving() {

    // if angle changes by more than half a degree
    double tol = 5e-1;

    if (abs(end.get_th_deg() - next.get_th_deg()) < tol) {
      return false;
    }

    return true;
  }

};

} // namespace Luke

#endif