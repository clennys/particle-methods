#include "../include/sim.h"
#include <iostream>

Habitat::Habitat(int num_subg, double h, int num_pred, int num_prey,
                 int ls_pred, int ls_prey, double step_s)
    : domain(num_subg, h), step_size(step_s) {
  this->domain.generate_agents(num_pred, num_prey, ls_pred, ls_prey);
  std::cout << "Agents generated" << std::endl;
}

void Habitat::time_step() {
  this->domain.plot();
  for (int i = 0; i < 100; i++) {
    std::cout << "Timestep: " << i << std::endl;
    this->domain.move_agents(this->step_size);
    std::cout << "Move: " << i << std::endl;
    this->domain.perform_agent_action(0.2, 0.2, 0.5);
    std::cout << "Action: " << i << std::endl;
    this->domain.plot();
    std::cout << "Plot: " << i << std::endl;
  }
}
