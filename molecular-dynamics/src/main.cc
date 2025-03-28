#include <algorithm>
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <iomanip>
#include <iostream>
#include <vector>

// Define high precision type with 100 digits of precision
using mp_float = boost::multiprecision::cpp_dec_float_50;

// Constants
const mp_float PI = boost::math::constants::pi<mp_float>();

std::vector<std::vector<mp_float>> build_fourier_D(int N) {
  std::vector<std::vector<mp_float>> D(N + 1, std::vector<mp_float>(N + 1, 0));
  for (int j = 0; j < N + 1; j++) {
    for (int i = 0; i < N + 1; i++) {
      if (i != j) {
        D[j][i] = (pow(mp_float(-1), j + i)) /
                  (mp_float(2) * sin((j - i) * PI / (N + 1)));
      }
    }
  }
  return D;
}

mp_float analytical_u(mp_float x, int k) { return exp(mp_float(k) * sin(x)); }

mp_float analytical_du(mp_float x, int k) {
  return mp_float(k) * cos(x) * exp(mp_float(k) * sin(x));
}

mp_float calculate_relative_error(int k, int N) {
  std::vector<mp_float> x(N + 1);
  mp_float step = mp_float(2) * PI / (N + 1);
  for (int i = 0; i < N + 1; i++) {
    x[i] = i * step;
  }

  std::vector<mp_float> u(N + 1), analy_du(N + 1);
  for (int i = 0; i < N + 1; i++) {
    u[i] = analytical_u(x[i], k);
    analy_du[i] = analytical_du(x[i], k);
  }

  std::vector<std::vector<mp_float>> D = build_fourier_D(N);
  std::vector<mp_float> numerical_du(N + 1, 0);
  for (int j = 0; j < N + 1; j++) {
    for (int i = 0; i < N + 1; i++) {
      numerical_du[j] += D[j][i] * u[i];
    }
  }

  std::vector<mp_float> errors(N + 1);
  for (int i = 0; i < N + 1; i++) {
    mp_float denom = abs(analy_du[i]) + std::numeric_limits<double>::epsilon();
    errors[i] = abs(numerical_du[i] - analy_du[i]) / denom;
  }

  return *std::max_element(errors.begin(), errors.end());
}

std::vector<std::pair<int, mp_float>>
find_min_N(const std::vector<int> &k_values, mp_float error_threshold = 1e-5,
           int max_N = 100) {
  std::vector<std::pair<int, mp_float>> results;

  for (int k : k_values) {
    std::cout << "Finding N with max error < 10^-5 for k=" << k << std::endl;
    int best_N = -1;
    mp_float best_error = 1.0;

    for (int N = 5; N <= max_N; N++) {
      mp_float max_error = calculate_relative_error(k, N);
      if (max_error < best_error) {
        best_error = max_error;
        best_N = N;
        if (best_error < error_threshold) {
          std::cout << "Found best N=" << N << " for k=" << k
                    << " with error=" << best_error << std::endl
                    << std::endl;
          break;
        }
      }
    }
    results.emplace_back(best_N, best_error);
  }
  return results;
}

int main() {
  std::vector<int> k_values = {2, 4, 6, 8, 10, 12};
  mp_float error_threshold = 1e-5;
  int max_N = 100;

  std::cout << "Fourier differentiation matrix accuracy for u(x) = "
               "exp(k*sin(x))\n";
  std::cout << "Error threshold: " << error_threshold << std::endl << std::endl;

  std::vector<std::pair<int, mp_float>> results =
      find_min_N(k_values, error_threshold, max_N);

  std::cout << "------------------------------------------------------"
            << std::endl;
  std::cout << "k      | Minimum N | Max Relative Error" << std::endl;
  std::cout << "------------------------------------------------------"
            << std::endl;

  for (size_t i = 0; i < k_values.size(); i++) {
    std::cout << std::left << std::setw(7) << k_values[i] << "| "
              << std::setw(9) << results[i].first << " | " << std::scientific
              << std::setprecision(10) << results[i].second << std::endl;
  }
  std::cout << "------------------------------------------------------"
            << std::endl;

  return 0;
}
