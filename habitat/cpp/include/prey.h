#ifndef INCLUDE_INCLUDE_PREY_H_
#define INCLUDE_INCLUDE_PREY_H_

#include "agents.h"

class Prey : public Agent {
public:
  Prey(double p_x, double p_y, int id, int lifespan);
  ~Prey() = default;

  std::unique_ptr<Agent> clone() const override;
  std::unique_ptr<Agent> replicate(double repl_prob) override;
	void action() override;
};

#endif // INCLUDE_INCLUDE_PREY_H_
