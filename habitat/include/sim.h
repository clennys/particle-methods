#ifndef INCLUDE_INCLUDE_SIM_H_
#define INCLUDE_INCLUDE_SIM_H_

#include "../include/grid.h"

class Habitat {
public:
  Habitat(int num_subg, double h, int num_pred, int num_prey, int ls_pred,
          int ls_prey, double step_s);
  void time_step();
  // Habitat(Habitat &&) = default;
  // Habitat(const Habitat &) = default;
  // Habitat &operator=(Habitat &&) = default;
  // Habitat &operator=(const Habitat &) = default;
  ~Habitat() = default;

private:
  Grid domain;
  double step_size;
};

#endif // INCLUDE_INCLUDE_SIM_H_
