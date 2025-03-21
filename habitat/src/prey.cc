#include "../include/prey.h"
#include <random>

Prey::Prey(double p_x, double p_y, int id, int lifespan)
    : Agent(p_x, p_y, id, lifespan) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<int> curr_age(1, lifespan - 1);

  this->lives = lifespan - curr_age(gen);
}

std::unique_ptr<Agent> Prey::clone() const {
  return std::make_unique<Prey>(*this);
}
