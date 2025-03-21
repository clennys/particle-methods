#ifndef INCLUDE_INCLUDE_PREDATOR_H_
#define INCLUDE_INCLUDE_PREDATOR_H_

#include "agents.h"

class Predator : public Agent {
public:
  Predator(double p_x, double p_y, int id, int lifespan);
  ~Predator() = default;
  std::unique_ptr<Agent> clone() const override;
};

#endif  // INCLUDE_INCLUDE_PREDATOR_H_
