#include "../include/agents.h"
#include <random>
#include <iostream>

Agent::Agent(double p_x, double p_y, int i) : x(p_x), y(p_y), id(i) {}

void Agent::random_direction(double &dx, double &dy) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<double> dist(0.0, 2.0 * M_PI);

  double angle = dist(gen);
  dx = std::cos(angle);
  dy = std::sin(angle);
}

double Agent::random_step_size(double mean, double std_dev) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::normal_distribution<double> step_dist(mean, std_dev);

  return step_dist(gen);
}

void Agent::move(double mean, double std_dev, double pb_width, double pb_height) {
  double dx, dy;
  this->random_direction(dx, dy);
  double step_size = this->random_step_size(mean, std_dev);
  this->x += step_size * dx;
  this->y += step_size * dy;

  // Apply periodic boundary conditions
  this->x = std::fmod(this->x, pb_width);
  if (this->x < 0)
    this->x += pb_width;

  this->y = std::fmod(this->y, pb_height);
  if (this->y < 0)
    this->y += pb_height;
}

bool Agent::replication_success(double repl_prop) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  static std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

  if (prob_dist(gen) < repl_prop) {
    return true;
  }
  return false;
}

std::unique_ptr<Agent> Agent::replicate(double repl_prob) {
  // Check if replication should occur based on probability
  if (this->replication_success(repl_prob)) {
    // Create a new prey with the same properties as this one
    // but reset lives to max value
    return std::make_unique<Agent>(this->x, this->y, 0, this->max_ls);
  }
  return nullptr; 
}



bool Agent::exceeded_lifespan() { return this->lives < 0; }

bool Agent::death() {
  // agents.erase(std::remove_if(agents.begin(), agents.end(),
  //                             [this](const std::unique_ptr<Agent> &agent) {
  //                               return agent.get() == this;
  //                             }),
  //              agents.end());
  //
  return exceeded_lifespan();
}
