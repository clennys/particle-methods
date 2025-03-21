#ifndef INCLUDE_INCLUDE_PREY_H_
#define INCLUDE_INCLUDE_PREY_H_

#include "agents.h"

class Prey : public Agent {
public:
  Prey(double p_x, double p_y, int id, int lifespan);
  ~Prey() = default;

  std::unique_ptr<Agent> clone() const override;
  std::unique_ptr<Prey> replicate(double repl_prob);
	int max_ls;
};

#endif // INCLUDE_INCLUDE_PREY_H_
