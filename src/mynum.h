#ifndef MYNUM_H_
#define MYNUM_H_

#include "mujoco.h"

namespace luke
{

class myNum {

  /* wrapper for mjtNum* to make it memory safe, the array is reallocated with
  a new pointer, then copied across.
  
  This class offers indexing, element-wise addition and subtraction, and matrix
  multiplication - all taking place on the internal mjtNum* array. There are
  also useful operations like print(), transpose(), magnitude(), and mean()
   */

private:
  mjtNum* data = NULL;      // data array pointer
  int nr = 0;               // number of rows
  int nc = 0;               // number of columns

  constexpr static bool debug = true;

public:
  // constructors and initialisers
  myNum() {}
  myNum(int nr_, int nc_) {
    nr = nr_;
    nc = nc_;
    data = new mjtNum[nr * nc];
    mju_zero(data, nr * nc);    // initialise to zero
  }
  myNum(mjtNum* data, int nr, int nc) {
    init(data, nr, nc);
  }
  void init(mjtNum* data_, int nr_, int nc_) {
    nr = nr_;
    nc = nc_;
    data = new mjtNum[nr * nc];
    for (int i = 0; i < nr * nc; i++) {
      data[i] = data_[i];
    }
  }

  // copy constructor
  myNum(const myNum& other) {
    nr = other.nr;
    nc = other.nc;
    if (other.size() > 0) {
      // create a new array on the heap, copy elements across
      data = new mjtNum[nr * nc];
      std::copy(other.data, other.data + (nr * nc), data);
    }
  }

  // destructor
  ~myNum() {
    if (data != NULL) delete data;
  }

  // swap: https://stackoverflow.com/questions/3279543/what-is-the-copy-and-swap-idiom
  friend void swap(myNum& first, myNum& second) {
    // mix namespaces to enable ADL (not needed I don't think, but good practice)
    using std::swap;
    swap(first.data, second.data);
    swap(first.nr, second.nr);
    swap(first.nc, second.nc);
  }

  // copy
  myNum& operator=(myNum other) {
    swap(*this, other);
    return *this;
  }

  // index
  mjtNum& operator[](int index) {
    return data[index];
  }

  // addition and subtraction
  void operator+=(const myNum& add) {
    additive_operator(add, +1);
  }
  void operator-=(const myNum& sub) {
    additive_operator(sub, -1);
  }
  myNum operator+(const myNum& add) {
    myNum out(*this);
    out += add;
    return out;
  }
  myNum operator-(const myNum& sub) {
    myNum out(*this);
    out -= sub;
    return out;
  }

  // multiplication
  friend myNum operator*(myNum lhs, myNum rhs);

  // print
  void print() {
    if (size() == 0)
      std::cout << "Cannot print myNum! nr or nc not set\n";
    mju_printMat(data, nr, nc);
  }

  // getters
  mjtNum get(int i) const { return data[i]; }
  int get_nr() const { return nr; }
  int get_nc() const { return nc; }
  int size() const { return nr * nc; }

  // utilities
  mjtNum sum() {
    int length = size();
    mjtNum result = 0.0;
    for (int i = 0; i < length; i++) {
      result += data[i];
    }
    return result;
  }

  mjtNum mean() {
    return (sum() / (mjtNum) size());
  }

  mjtNum magnitude() {
    int length = size();
    mjtNum result = 0.0;
    for (int i = 0; i < length; i++) {
      result += data[i] * data[i];
    }
    return sqrt(result);
  }

  mjtNum magnitude3(int n = 0) {
    // do magnitude on three indexes starting from n
    if (debug) {
      if (n + 3 > size()) throw std::runtime_error("magnitude3 got too large n");
    }
    mjtNum result = 0.0;
    for (int i = n; i < n + 3; i++) {
      result += data[i] * data[i];
    }
    return sqrt(result);
  }

  myNum rotate3_by(myNum rotation_matrix, int n = 0) {
    // return 3 elements of this rotated by a rotation matrix (must be 3x3)
    if (debug) {
      if (n + 3 > size()) throw std::runtime_error("rotate3_by got too large n");
    }
    // take the chosen 3 elements and form a vector
    mjtNum us[3];
    for (int i = n; i < n + 3; i++) {
      us[i] = data[i];
    }
    myNum vec3(us, 3, 1);
    myNum out = rotation_matrix * vec3;
    return out;
  }

  myNum transpose() {
    mjtNum t_arr[nr * nc];
    mju_transpose(t_arr, data, nr, nc);
    myNum out(t_arr, nc, nr);
    return out;
  }

private:
  // private helpers
  void additive_operator(const myNum& second, int sign) {
    /* all arrays in mujoco are in row-major format, hence:
          array = { a0, a1, a2, a3, a4, a5}, nr = 2, nc = 3
       corresponds to:
          a0 a1 a2
          a3 a4 a5
       however, since the arrays are the same size, we can simply add
       like for like for simplicity
    */
    if (debug) {
      if (second.get_nc() != nc or second.get_nr() != nr) {
        throw std::runtime_error("cannot add/subract matrices of different sizes");
      }
      if (size() == 0) {
        throw std::runtime_error("cannot add/subtract two empty matrices");
      }
    }
    int length = size();
    for (int i = 0; i < length; i++) {
      data[i] += (sign * second.get(i));
    }
  }
};

inline myNum operator*(myNum lhs, myNum rhs)
{
  /* perform matrix multiplication output = lhs * rhs */

  if (lhs.nc != rhs.nr) {
    throw std::runtime_error("matrix multiplication of wrong dimension");
  }

  // multiply into the result
  mjtNum result[lhs.nr * rhs.nc];
  mju_mulMatMat(result, lhs.data, rhs.data, lhs.nr, lhs.nc, rhs.nc);

  // convert to a mynum variable
  myNum out(result, lhs.nr, rhs.nc);
  return out;
}

} // namespace luke

#endif