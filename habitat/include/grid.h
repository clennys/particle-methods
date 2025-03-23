#ifndef INCLUDE_INCLUDE_GRID_H_
#define INCLUDE_INCLUDE_GRID_H_

#include "agents.h"
#include <array>
#include <memory>
#include <vector>

#define LEFT 0
#define TOP 1
#define RIGHT 2
#define BOTTOM 3

class Subgrid {
public:
  Subgrid(double len, std::array<double, 2> ap, std::array<int, 2> id);
  void move_to_neighbor(std::unique_ptr<Agent> &&agent);
  void add_agent(std::unique_ptr<Agent> &&agent);
  std::vector<std::unique_ptr<Agent>> agents;
  std::unique_ptr<Agent> remove_agent();
  double length_h;
  std::array<double, 2> anchor_point;
  std::array<std::weak_ptr<Subgrid>, 4> neighbors;
  std::array<int, 2> coords;
  std::vector<std::unique_ptr<Agent>>
  pred_eat(const std::unique_ptr<Agent> &predator_ptr, double rc);
};

class Grid {
public:
  std::vector<std::shared_ptr<Subgrid>> domain;

  int nr_subgrids;
  double h;
  double length;
  int frame = 0;

  Grid(int subgrids, double h_size);
  void generate_agents(int num_pred, int num_prey, int ls_pred, int ls_prey);
  void extract_agent_positions(std::vector<double> &pred_x,
                               std::vector<double> &pred_y,
                               std::vector<double> &prey_x,
                               std::vector<double> &prey_y);
  void perform_agent_action(double pred_repl_prob, double prey_repl_prob,
                            double eat_dist);
  void plot();
  void move_agents(double step_size);
  void perform_agent_action();
  ~Grid();
};

#endif // INCLUDE_INCLUDE_GRID_H_
