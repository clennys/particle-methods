#include "../include/predator.h"

Predator::Predator(double p_x, double p_y, int id, int lifespan) : Agent(p_x, p_y, id)  {
	this->lives = lifespan;
}


std::unique_ptr<Agent> Predator::clone() const {
  return std::make_unique<Predator>(*this);
}


