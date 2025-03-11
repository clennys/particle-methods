#include "../include/ising.hpp"
#include "../include/util.hpp"
#include <iostream>
#include <matplot/matplot.h>
#include <string>
#include <vector>

int main() {
  std::vector<int> L = {5, 10, 15};
  std::vector<double> temp = {0.01, 1.0, 1.5, 2.0, 2.1,
                              2.2,  2.3, 2.4, 2.5, 3.0};
  int ntherm = 100000, n_sample = 5000;
  auto fig = matplot::figure(true);
  fig->quiet_mode(true);
  fig->backend()->run_command("unset warnings");

  for (int l : L) {
    std::vector<double> temp_avg_en, temp_avg_mag, temp_std_dev_en,
        temp_std_dev_mag;
    std::string lat_size = std::to_string(l);
    for (double t : temp) {
      Ising model(l, t);
      std::cout << "Start simulation with lattice size: [" << l << "x" << l << "] and Temperature: " << t << std::endl;
      model.simulate(ntherm, n_sample, l * l);
      double avg_en = model.markov_avg_energy();
      double avg_mag = model.markov_avg_magnet();
      temp_avg_en.push_back(avg_en);
      temp_avg_mag.push_back(avg_mag);
      temp_std_dev_en.push_back(
          model.std_dev(avg_en, model.energy_markov_chain));
      temp_std_dev_mag.push_back(
          model.std_dev(avg_mag, model.magnet_markov_chain, true));
      if (t < 2.3) {
        // Create a figure handle and set quiet mode to suppress warnings
        auto x = util::arange<double>(0, model.magnet_markov_chain.size());
        matplot::title("Magnetization at T=" + std::to_string(t) +
                       "for lettice [" + lat_size + "x" + lat_size + "]");
        matplot::xlabel("Time step");
        matplot::ylabel("Avg. Magnetization");
        matplot::grid(true);
        matplot::ylim({-1.1, 1.1});
        matplot::plot(x, model.magnet_markov_chain)->line_width(3);
        matplot::save("img/mag_time_" + std::to_string(t) + "_" + lat_size +
                      ".png");
      }
    }
    matplot::title("Energy at different Temperatures for a [" + lat_size + "x" +
                   lat_size + "] lattice");
    matplot::xlabel("Temperature");
    matplot::ylabel("Avg. Energy");
    matplot::ylim(matplot::automatic);
    matplot::plot(temp, temp_avg_en, "-o")->line_width(3);
    matplot::grid(true);
    matplot::save("img/avg_en_" + std::to_string(l) + ".png");
    matplot::hold(false);

    matplot::title("Magnetization at different Temperatures for a [" +
                   lat_size + "x" + lat_size + "] lattice");
    matplot::xlabel("Temperature");
    matplot::ylabel("Avg. absolute Magnetization");
    matplot::ylim({0.0, 1.1});
    matplot::grid(true);
    matplot::plot(temp, temp_avg_mag, "-o")->line_width(3);
    matplot::save("img/avg_mag_" + std::to_string(l) + ".png");
    std::cout << std::string(10, '=') + " OUTPUT FOR GRID SIZE [" << l << "X"
              << l << "]" + std::string(10, '=') << std::endl;
    for (unsigned long i = 0; i < temp.size(); i++) {
      std::cout << "T=" << temp[i] << " with avg_mag=" << temp_avg_mag[i]
                << ", mag std_dev=" << temp_std_dev_mag[i]
                << ", avg_energy=" << temp_avg_en[i]
                << ", energy std_dev=" << temp_std_dev_en[i] << std::endl;
    }
    std::cout << std::string(50, '#') << std::endl;
  }
}
