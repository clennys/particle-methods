#include "../include/grid.h"
#include "../include/predator.h"
#include "../include/prey.h"
#include "../include/util.h"
#include "matplot/core/figure_registry.h"
#include <matplot/matplot.h>
#include <random>

Subgrid::Subgrid(double len, std::array<double, 2> ap, std::array<int, 2> id)
    : length_h(len), anchor_point(ap), coords(id) {}

void Subgrid::add_agent(std::unique_ptr<Agent> &&agent) {
  agents.push_back(std::move(agent));
}

std::unique_ptr<Agent> Subgrid::remove_agent() {
  if (agents.empty())
    return nullptr;
  std::unique_ptr<Agent> agent = std::move(agents.back());
  agents.pop_back();
  return agent;
}

void Subgrid::move_to_neighbor(std::unique_ptr<Agent> &&agent) {
  // Compute the subgrid's boundaries
  double x_min = anchor_point[0];
  double x_max = x_min + length_h;
  double y_min = anchor_point[1];
  double y_max = y_min + length_h;

  // Determine which neighbor to move to based on boundaries
  int direction = -1; // -1 means no move
  if (agent->x < x_min)
    direction = LEFT;
  else if (agent->x >= x_max)
    direction = RIGHT;
  else if (agent->y < y_min)
    direction = BOTTOM;
  else if (agent->y >= y_max)
    direction = TOP;

  // If the agent has moved out, transfer it to the appropriate neighbor
  if (direction != -1 && !neighbors[direction].expired()) {
    if (auto neighbor = neighbors[direction].lock()) {
      std::cerr << "DEBUGPRINT: Number of agents before move: " << agents.size()
                << std::endl;
      std::cerr << "DEBUGPRINT: Number of neigbor agents bevor move: "
                << neighbor->agents.size() << std::endl;

      neighbor->add_agent(std::move(agent));
      std::cerr << "DEBUGPRINT[30]: grid.cc:50 (after "
                   "neighbor->add_agent(std::move(agent));)"
                << std::endl;
      auto it = std::find_if(
          agents.begin(), agents.end(),
          [&agent](const std::unique_ptr<Agent> &a) {
            return a.get() ==
                   agent.get(); // Check if they point to the same object
          });

      if (it != agents.end()) {
        agents.erase(it); // Remove the agent from the vector
      }
      std::cerr << "DEBUGPRINT: Number of agents after move: " << agents.size()
                << std::endl;
      std::cerr << "DEBUGPRINT: Number of neigbor agents after move: "
                << neighbor->agents.size() << std::endl;

    } else {
      std::cerr << "Error: Neighbor is invalid (expired)" << std::endl;
    }
  }
}

std::vector<std::unique_ptr<Agent>>
Subgrid::pred_eat(const std::unique_ptr<Agent> &predator_ptr, double rc) {
  Predator *predator = dynamic_cast<Predator *>(predator_ptr.get());
  if (!predator) {
    return {};
  }
  std::vector<std::unique_ptr<Agent>> eaten_prey;

  double pred_x = predator->x;
  double pred_y = predator->y;

  const double pe = 0.02; // Probability of eating

  auto check_prey_in_subgrid = [&](Subgrid *sg) {
    if (!sg)
      return;

    std::vector<size_t> indices_to_remove;

    for (size_t i = 0; i < sg->agents.size(); ++i) {
      auto &agent = sg->agents[i];

      if (!agent || agent.get() == predator) {
        continue;
      }

      auto prey = dynamic_cast<Prey *>(agent.get());
      if (!prey) {
        continue;
      }

      double prey_x = prey->x;
      double prey_y = prey->y;
      double distance = std::sqrt(std::pow(pred_x - prey_x, 2) +
                                  std::pow(pred_y - prey_y, 2));

      if (distance <= rc) {
				// TODO: (dhub) change random functions
        double random_value = static_cast<double>(std::rand()) / RAND_MAX;
        if (random_value <= pe) {
          indices_to_remove.push_back(i);
					predator->lives = predator->max_ls;
        }
      }
    }

    std::sort(indices_to_remove.rbegin(), indices_to_remove.rend());
    for (size_t idx : indices_to_remove) {
      if (idx < sg->agents.size()) {
        eaten_prey.push_back(std::move(sg->agents[idx]));
        sg->agents.erase(sg->agents.begin() + idx);
      }
    }
  };

  check_prey_in_subgrid(this);

  for (int i = 0; i < 4; i++) {
    if (auto neighbor = neighbors[i].lock()) {
      check_prey_in_subgrid(neighbor.get());
    }
  }

  return eaten_prey;
}

Grid::Grid(int subgrids, double h_size)
    : nr_subgrids(subgrids), h(h_size), length(subgrids * h_size) {

  for (int i = 0; i < nr_subgrids; i++) {
    for (int j = 0; j < nr_subgrids; j++) {
      std::array<double, 2> anchor_p = {i * h_size, j * h_size};
      std::array<int, 2> id = {i, j};
      this->domain.push_back(std::make_shared<Subgrid>(this->h, anchor_p, id));
    }
  }
  for (int i = 0; i < nr_subgrids; i++) {
    for (int j = 0; j < nr_subgrids; j++) {
      int current = i * nr_subgrids + j;

      // Left neighbor (wraps around to right side if on left edge)
      int left_j = (j > 0) ? j - 1 : nr_subgrids - 1;
      this->domain[current]->neighbors[LEFT] =
          this->domain[i * nr_subgrids + left_j];

      // Right neighbor (wraps around to left side if on right edge)
      int right_j = (j < nr_subgrids - 1) ? j + 1 : 0;
      this->domain[current]->neighbors[RIGHT] =
          this->domain[i * nr_subgrids + right_j];

      // Top neighbor (wraps around to bottom if on top edge)
      int top_i = (i > 0) ? i - 1 : nr_subgrids - 1;
      this->domain[current]->neighbors[TOP] =
          this->domain[top_i * nr_subgrids + j];

      // Bottom neighbor (wraps around to top if on bottom edge)
      int bottom_i = (i < nr_subgrids - 1) ? i + 1 : 0;
      this->domain[current]->neighbors[BOTTOM] =
          this->domain[bottom_i * nr_subgrids + j];
    }
  }
}
void Grid::generate_agents(int num_pred, int num_prey, int ls_pred,
                           int ls_prey) {
  std::random_device rd;
  std::mt19937 gen(rd());

  // Distribution for selecting a random subgrid
  std::uniform_int_distribution<> subgrid_dist(0,
                                               nr_subgrids * nr_subgrids - 1);

  // Distribution for position within a subgrid
  std::uniform_real_distribution<> position_dist(0.0, 1.0);

  for (int i = 0; i < num_pred + num_prey; i++) {
    int subgrid_idx = subgrid_dist(gen);

    double x_pos = position_dist(gen);
    double y_pos = position_dist(gen);

    int sg_i = subgrid_idx / nr_subgrids;
    int sg_j = subgrid_idx % nr_subgrids;

    // Set absolute position (grid coordinates)
    double abs_x = sg_i * h + x_pos * h; // x position
    double abs_y = sg_j * h + y_pos * h; // y position
    if (i < num_pred) {
      domain[subgrid_idx]->add_agent(
          std::make_unique<Predator>(abs_x, abs_y, i, ls_pred));
    } else {
      domain[subgrid_idx]->add_agent(
          std::make_unique<Prey>(abs_x, abs_y, i, ls_prey));
    }
  }
}

void Grid::extract_agent_positions(std::vector<double> &pred_x,
                                   std::vector<double> &pred_y,
                                   std::vector<double> &prey_x,
                                   std::vector<double> &prey_y) {
  for (const auto &subgrid : this->domain) { // Loop over all subgrids
    for (const auto &agent :
         subgrid->agents) { // Loop over agents in the subgrid
      if (util::instanceof<Predator>(agent.get())) {
        pred_x.push_back(agent->x);
        pred_y.push_back(agent->y);
      } else if (util::instanceof<Prey>(agent.get())) {
        prey_x.push_back(agent->x);
        prey_y.push_back(agent->y);
      }
    }
  }
}

void Grid::plot() {
  using namespace matplot;
  auto fig = figure(true);
  // fig->quiet_mode(true);
  fig->backend()->run_command("unset warnings");
  std::vector<double> predator_x, predator_y;
  std::vector<double> prey_x, prey_y;

  this->extract_agent_positions(predator_x, predator_y, prey_x, prey_y);
  // for (size_t i = 0; i < predator_x.size(); i++) {
  // 	std::cout << "("<< predator_x[i] << "," << predator_y[i] << ")" <<
  // std::endl;
  //
  // }

  scatter(prey_x, prey_y)->display_name("Prey");
  title("Predator Prey Model");
  xlabel("Y");
  ylabel("X");

  hold(on);
  scatter(predator_x, predator_y)->display_name("Predator");
  hold(off);
  ::matplot::legend(on);
  // show();
  std::string frame_str = util::format_number(++frame);
  save("anim/habitat_" + frame_str + ".png");
}

void Grid::move_agents(double step_size) {
  for (const auto &subgrid : this->domain) {
    std::vector<size_t> indices_to_remove;

    for (size_t i = 0; i < subgrid->agents.size(); ++i) {
      auto &agent = subgrid->agents[i];

      if (!agent) {
        std::cerr << "Warning: Encountered null agent at index " << i
                  << std::endl;
        indices_to_remove.push_back(i);
        continue;
      }

      // Move the agent physically
      agent->move(0, step_size, this->length, this->length);

      // Check if the agent needs to move to a different subgrid
      double x_min = subgrid->anchor_point[0];
      double x_max = x_min + subgrid->length_h;
      double y_min = subgrid->anchor_point[1];
      double y_max = y_min + subgrid->length_h;

      int direction = -1; // -1 means no move
      if (agent->x < x_min)
        direction = LEFT;
      else if (agent->x >= x_max)
        direction = RIGHT;
      else if (agent->y < y_min)
        direction = BOTTOM;
      else if (agent->y >= y_max)
        direction = TOP;

      // If the agent needs to move to a different subgrid
      if (direction != -1 && !subgrid->neighbors[direction].expired()) {
        if (auto neighbor = subgrid->neighbors[direction].lock()) {
          // Transfer the agent to the neighbor
          neighbor->add_agent(std::move(agent));

          // Mark this index for removal
          indices_to_remove.push_back(i);
        }
      }
    }

    std::sort(indices_to_remove.rbegin(), indices_to_remove.rend());
    for (size_t idx : indices_to_remove) {
      if (idx < subgrid->agents.size()) {
        subgrid->agents.erase(subgrid->agents.begin() + idx);
      }
    }
  }
}

void Grid::perform_agent_action(double pred_repl_prob, double prey_repl_prob,
                                double eat_dist) {
  for (const auto &subgrid : this->domain) {
    // Vector to collect new agents
    std::vector<std::unique_ptr<Agent>> new_agents;
    // Vector to track agents that need to be removed
    std::vector<size_t> indices_to_remove;

    // Process each agent in the subgrid
    for (size_t i = 0; i < subgrid->agents.size(); ++i) {
      auto &agent = subgrid->agents[i];

      // Skip null agents
      if (!agent)
        continue;

      // Update the agent's state
      agent->action();

      // Check if agent should be removed after action
      if (auto prey = dynamic_cast<Prey *>(agent.get())) {
        if (prey->exceeded_lifespan()) {
          indices_to_remove.push_back(i);
          continue; // Skip replication for dead agents
        }

        // Handle Prey replication
        std::unique_ptr<Agent> new_agent = prey->replicate(prey_repl_prob);
        if (new_agent) {
          new_agents.push_back(std::move(new_agent));
        }
      }
      // Note: Fixed the check for Predator - previous code had duplicate check
      // for Prey
      else if (auto predator = dynamic_cast<Predator *>(agent.get())) {
        if (predator->exceeded_lifespan()) {
          indices_to_remove.push_back(i);
          continue; // Skip replication for dead agents
        }

        // Handle predator eating prey and replication
        std::vector<std::unique_ptr<Agent>> eaten_prey =
            subgrid->pred_eat(agent, eat_dist);

        // Replicate for each prey eaten
        for (size_t j = 0; j < eaten_prey.size(); ++j) {
          std::unique_ptr<Agent> new_agent =
              predator->replicate(pred_repl_prob);
          if (new_agent) {
            new_agents.push_back(std::move(new_agent));
          }
        }
      }
    }

    // Remove dead agents in reverse order
    std::sort(indices_to_remove.rbegin(), indices_to_remove.rend());
    for (size_t idx : indices_to_remove) {
      if (idx < subgrid->agents.size()) {
        subgrid->agents.erase(subgrid->agents.begin() + idx);
      }
    }

    // Add all new agents to the subgrid
    for (auto &new_agent : new_agents) {
      subgrid->add_agent(std::move(new_agent));
    }
  }
}

Grid::~Grid() {}
