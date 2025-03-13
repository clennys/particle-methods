#ifndef ISING_HPP_
#define ISING_HPP_

#include "../include/lattice.hpp"
#include <random>
#include <map>

class Ising {
public:
  Ising(int size, double T);
  ~Ising();

  void simulate(int Ntherm, int Nsample, int Nsubsweep);
  double markov_avg_energy();
  double markov_avg_magnet();
  std::vector<double> magnet_markov_chain;
  std::vector<double> energy_markov_chain;
  double std_dev(const double &avg, const std::vector<double> &vec, bool abs=false);
	void animate(int frames);

private:
  Lattice lat;
  double temp;
	std::map<int, double> exp_map;
  std::mt19937 gen;
  std::uniform_int_distribution<int> dist_lat;      // Integer distribution
  std::uniform_real_distribution<double> dist_flip; // Real number distribution

  void metropolis_step();
  void metropolis_sweep(int Nsubsweep);
  double avg_magnetization();
  double overall_energy();
};

#endif // ISING_HPP_
