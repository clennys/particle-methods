#ifndef INCLUDE_INCLUDE_AGENT_H_
#define INCLUDE_INCLUDE_AGENT_H_

#include <memory>

class Agent {
public:
  Agent(double p_x, double p_y, int id, int lifespan);
  virtual ~Agent() = default;
  double x, y, z; // x, y, z
  int id;
  int lives = 0;
  int max_ls = 0;
  virtual std::unique_ptr<Agent> clone() const = 0;
  virtual bool replication_success(double repl_prop);
  virtual std::unique_ptr<Agent> replicate(double repl_prob) = 0;
  // virtual bool death();
	virtual void action() = 0;

  virtual void move(double mean, double std_dev, double pb_width,
                    double pb_height);
  void random_direction(double &dx, double &dy);
  double random_step_size(double mean, double std_dev);
  virtual bool exceeded_lifespan();
};

#endif // INCLUDE_INCLUDE_AGENT_H_
