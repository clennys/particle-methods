#include "../include/ising.hpp"
#include <iostream>
#include <matplot/matplot.h>
#include <vector>

int main() {
  std::vector<int> L = {5, 10, 15};
  // std::vector<int> L = {15};
  std::vector<double> temp = {0.0, 1.0, 1.5, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 3.0};
  // std::vector<double> temp = {1.0};
  int ntherm = 100000, n_sample = 5000;
  for (int l : L) {
    std::vector<double> temp_avg_en, temp_avg_mag;
    for (double t : temp) {
      std::cout << "Init Ising model with lattice size: [" << l << "x" << l
                << "] and Temperature: " << t << std::endl;
      Ising model(l, t);
      std::cout << "Start simulation" << std::endl;
      model.simulate(ntherm, n_sample, l * l);
      temp_avg_en.push_back(model.markov_avg_energy());
      temp_avg_mag.push_back(model.markov_avg_magnet());
    }
    matplot::plot(temp, temp_avg_en);
    matplot::save("img/avg_en_" + std::to_string(l) + ".png");
    matplot::hold(false);
    matplot::plot(temp, temp_avg_mag);
    matplot::save("img/avg_mag_" + std::to_string(l) + ".png");
  }
}
