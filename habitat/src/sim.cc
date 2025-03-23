#include "../include/sim.h"
#include "../include/util.h"
#include "matplot/matplot.h"
#include <iostream>

Habitat::Habitat(int num_subg, double h, int num_pred, int num_prey,
                 int ls_pred, int ls_prey, double step_s)
    : domain(num_subg, h), step_size(step_s) {
  this->domain.generate_agents(num_pred, num_prey, ls_pred, ls_prey);
  std::cout << "Agents generated" << std::endl;
}

void Habitat::time_step() {
  using namespace matplot;
  this->domain.plot();
  std::vector<int> history_pred, history_prey;
  int N = 1000;
  for (int i = 0; i < N; i++) {
    int pred, prey;
    std::cout << "Timestep: " << i << std::endl;
    this->domain.number_of_pred_prey(prey, pred);
    history_pred.push_back(pred);
    history_prey.push_back(prey);
    std::cout << "Pred: " << pred << " Prey:" << prey << std::endl;
    this->domain.move_agents(this->step_size);
    std::cout << "Move: " << i << std::endl;
    this->domain.perform_agent_action(0.02, 0.02, 0.5);
    std::cout << "Action: " << i << std::endl;
    // this->domain.plot();
    // std::cout << "Plot: " << i << std::endl;
  }
  auto fig = figure(true);
  // fig->quiet_mode(true);
  fig->backend()->run_command("unset warnings");
  std::vector<int> x = util::arange(0, N, 1);
  plot(x, history_prey);
  hold(on);
  plot(x, history_pred);
  show();
}
