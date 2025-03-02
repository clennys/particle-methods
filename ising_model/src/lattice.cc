#include "../include/lattice.hpp"
#include <iomanip>
#include <iostream>
#include <matplot/matplot.h>
#include <matplot/backend/gnuplot.h>
#include <random>
#include <sstream>

std::string format_number(int num, int width = 3) {
  std::ostringstream oss;
  oss << std::setw(width) << std::setfill('0') << num;
  return oss.str();
}

Lattice::Lattice(int size) : frame(0) {
  this->L = std::vector<std::vector<int>>(size, std::vector<int>(size, 0));
  this->random_init();
}

void Lattice::random_init() {
  std::random_device rd;                         // Seed
  std::mt19937 gen(rd());                        // Mersenne Twister engine
  std::uniform_int_distribution<int> dist(0, 1); // Range [1, 10]
  for (int i = 0; i < this->get_dim(); ++i) {
    for (int j = 0; j < this->get_dim(); ++j) {
      this->L[i][j] = dist(gen) * 2 - 1;
    }
  }
}

void Lattice::plot() {
  using namespace matplot;
  colormap(palette::gray());
  std::string frame = format_number(this->frame++);
  title("Ising model at frame " + frame);
  image(this->L, true);
  save("img/lattice_" + frame + ".png");
  hold(false); // Ensure it updates properly
  // show();
}

int Lattice::get_dim() { return this->L.size(); }

int Lattice::sum_of_neighbors(int row, int col) {
  int dim = this->get_dim();
  return L[(row + 1) % dim][col] + L[(dim + row - 1) % dim][col] +
         L[row][(col + 1) % dim] + L[row][(dim + col - 1) % dim];
}

Lattice::~Lattice() {}
