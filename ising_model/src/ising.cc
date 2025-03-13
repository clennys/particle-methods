#include "../include/ising.hpp"
#include <cmath>
#include <map>
#include <random>
#include <unistd.h>

Ising::Ising(int size, double T)
    : lat(size), temp(T), dist_lat(0, size - 1), dist_flip(0.0, 1.0) {
  std::random_device rd;
  this->gen = std::mt19937(rd());
  std::vector<int> pos_delta_en = {-8, -4, -2, 0, 2, 4, 8};
  for (int p : pos_delta_en) {
    this->exp_map[p] = std::exp(-p / this->temp);
  }
}

void Ising::simulate(int Ntherm, int Nsample, int Nsubsweep) {
  for (int i = 0; i < Ntherm; i++) {
    metropolis_step();
  }
  for (int i = 0; i < Nsample; i++) {
    metropolis_sweep(Nsubsweep);
    energy_markov_chain.push_back(overall_energy());
    magnet_markov_chain.push_back(avg_magnetization());
  }
}

void Ising::animate(int frames) {
  double dim = this->lat.get_dim();
  for (int i = 0; i < frames; i++) {
    metropolis_sweep(dim);
		this->lat.plot();
  }
}

void Ising::metropolis_sweep(int Nsubsweep) {
  for (int i = 0; i < Nsubsweep; i++) {
    metropolis_step();
  }
}
double Ising::markov_avg_energy() {
  double sum = std::accumulate(this->energy_markov_chain.begin(),
                               this->energy_markov_chain.end(), 0.0);
  return sum / static_cast<double>(this->energy_markov_chain.size());
}

double Ising::markov_avg_magnet() {
  double sum = std::transform_reduce(
      this->magnet_markov_chain.begin(), this->magnet_markov_chain.end(), 0.0,
      std::plus<>(), [](double value) { return std::abs(value); });
  return sum / static_cast<double>(this->magnet_markov_chain.size());
}

double Ising::avg_magnetization() {
  double sum = 0.;
  double dim = this->lat.get_dim();
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      sum += this->lat.L[i][j];
    }
  }
  return sum / (dim * dim);
}

void Ising::metropolis_step() {
  int rd_row = this->dist_lat(gen), rd_col = this->dist_lat(gen);
  int sum_neighbors = this->lat.sum_of_neighbors(rd_row, rd_col);
  int energy_change = 2 * this->lat.L[rd_row][rd_col] * sum_neighbors;
  if (energy_change <= 0 ||
      this->dist_flip(gen) < this->exp_map[energy_change]) {
    this->lat.L[rd_row][rd_col] *= -1;
  }
}

double Ising::overall_energy() {
  double energy = 0.;
  int dim = this->lat.get_dim();

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      energy += -this->lat.L[i][j] * this->lat.sum_of_neighbors(i, j);
    }
  }
  return energy / 2.;
}

double Ising::std_dev(const double &avg, const std::vector<double> &vec,
                      bool abs) {
  double sum = 0., diff = 0.;
  for (const double &el : vec) {
    if (abs)
      diff = std::abs(el) - avg;
    else
      diff = el - avg;

    sum += diff * diff;
  }
  return std::sqrt(sum / static_cast<double>(vec.size()));
}

Ising::~Ising() {}
