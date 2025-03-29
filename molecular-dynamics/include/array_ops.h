#ifndef INCLUDE_INCLUDE_ARRAY_OPS_H_
#define INCLUDE_INCLUDE_ARRAY_OPS_H_

#include <array>
#include <cmath>

// Addition operator for vectors
inline std::array<double, 3> operator+(const std::array<double, 3> &lhs,
                                       const std::array<double, 3> &rhs) {
  return {lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]};
}

// Subtraction operator for vectors
inline std::array<double, 3> operator-(const std::array<double, 3> &lhs,
                                       const std::array<double, 3> &rhs) {
  return {lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]};
}

// Scalar multiplication (scalar * vector)
inline std::array<double, 3> operator*(double scalar,
                                       const std::array<double, 3> &vec) {
  return {scalar * vec[0], scalar * vec[1], scalar * vec[2]};
}

// Dot product
inline double operator*(const std::array<double, 3> &vec_lhs,
                        const std::array<double, 3> &vec_rhs) {
  return vec_lhs[0] * vec_rhs[0] + vec_lhs[1] * vec_rhs[1] +
         vec_lhs[2] * vec_rhs[2];
}

// Scalar multiplication (vector * scalar)
inline std::array<double, 3> operator*(const std::array<double, 3> &vec,
                                       double scalar) {
  return scalar * vec;
}

// Scalar division (vector / scalar)
inline std::array<double, 3> operator/(const std::array<double, 3> &vec,
                                       double scalar) {
  return {vec[0] / scalar, vec[1] / scalar, vec[2] / scalar};
}

// Compound addition
inline std::array<double, 3> &operator+=(std::array<double, 3> &lhs,
                                         const std::array<double, 3> &rhs) {
  lhs[0] += rhs[0];
  lhs[1] += rhs[1];
  lhs[2] += rhs[2];
  return lhs;
}

// Compound subtraction
inline std::array<double, 3> &operator-=(std::array<double, 3> &lhs,
                                         const std::array<double, 3> &rhs) {
  lhs[0] -= rhs[0];
  lhs[1] -= rhs[1];
  lhs[2] -= rhs[2];
  return lhs;
}

// Compound scalar multiplication
inline std::array<double, 3> &operator*=(std::array<double, 3> &vec,
                                         double scalar) {
  vec[0] *= scalar;
  vec[1] *= scalar;
  vec[2] *= scalar;
  return vec;
}

// Compound scalar division
inline std::array<double, 3> &operator/=(std::array<double, 3> &vec,
                                         double scalar) {
  vec[0] /= scalar;
  vec[1] /= scalar;
  vec[2] /= scalar;
  return vec;
}

// Dot product
inline double dot(const std::array<double, 3> &lhs,
                  const std::array<double, 3> &rhs) {
  return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
}

// Cross product
inline std::array<double, 3> cross(const std::array<double, 3> &lhs,
                                   const std::array<double, 3> &rhs) {
  return {lhs[1] * rhs[2] - lhs[2] * rhs[1], lhs[2] * rhs[0] - lhs[0] * rhs[2],
          lhs[0] * rhs[1] - lhs[1] * rhs[0]};
}

// Vector magnitude squared
inline double magnitudeSquared(const std::array<double, 3> &vec) {
  return dot(vec, vec);
}

// Vector magnitude
inline double magnitude(const std::array<double, 3> &vec) {
  return std::sqrt(magnitudeSquared(vec));
}

// Normalize a vector
inline std::array<double, 3> normalize(const std::array<double, 3> &vec) {
  double mag = magnitude(vec);
  if (mag > 0.0) {
    return vec / mag;
  }
  return {0.0, 0.0, 0.0};
}
#endif // INCLUDE_INCLUDE_ARRAY_OPS_H_
