#include "../include/sim.h"

Habitat::Habitat(int num_subg, double h, int num_pred, int num_prey,
                 int ls_pred, int ls_prey, double step_s)
    : domain(num_subg, h), step_size(step_s) {
  this->domain.generate_agents(num_pred, num_prey, ls_pred, ls_prey);
}

void Habitat::time_step() {
  this->domain.plot();
  for (int i = 0; i < 100; i++) {
    this->domain.move_agents(this->step_size);
    this->domain.plot();
  }
}
