#include "../include/prey.h"
#include <random>

Prey::Prey(double p_x, double p_y, int id, int lifespan)
    : Agent(p_x, p_y, id, lifespan) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_int_distribution<int> curr_age(1, lifespan - 1);

  this->lives = lifespan - curr_age(gen);
}

void Prey::action(){
this->lives--;
}

std::unique_ptr<Agent> Prey::clone() const {
  return std::make_unique<Prey>(*this);
}

std::unique_ptr<Agent> Prey::replicate(double repl_prob) {
  // Check if replication should occur based on probability
  if (this->replication_success(repl_prob)) {
    // Create a new prey with the same properties as this one
    // but reset lives to max value
    return std::make_unique<Prey>(this->x, this->y, 0, this->max_ls);
  }
  return nullptr;
}

