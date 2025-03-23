#include "../include/predator.h"

Predator::Predator(double p_x, double p_y, int id, int lifespan) : Agent(p_x, p_y, id, lifespan)  {
}


std::unique_ptr<Agent> Predator::clone() const {
  return std::make_unique<Predator>(*this);
}

void Predator::action(){
	this->lives--;
}

std::unique_ptr<Agent> Predator::replicate(double repl_prob) {
  // Check if replication should occur based on probability
  if (this->replication_success(repl_prob)) {
    // Create a new prey with the same properties as this one
    // but reset lives to max value
    return std::make_unique<Predator>(this->x, this->y, 0, this->max_ls);
  }
  return nullptr;
}

