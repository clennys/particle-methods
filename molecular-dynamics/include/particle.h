#ifndef INCLUDE_INCLUDE_PARTICLE_H_
#define INCLUDE_INCLUDE_PARTICLE_H_
#include <array>
#include <vector>

class Particle {
public:
  Particle(double mass = 1.0);

  const std::array<double, 3> &position() const;
  const std::array<double, 3> &velocity() const;
  const std::array<double, 3> &force() const;
  double mass() const;

  void set_position(const std::array<double, 3> &position);
  void set_velocity(const std::array<double, 3> &velocity);
  void set_force(const std::array<double, 3> &force);

  // Reset force to zero
  void reset_force();

  // Compute kinetic energy
  double kinetic_energy() const;

private:
  std::array<double, 3> position_ = {0.0, 0.0, 0.0};
  std::array<double, 3> velocity_ = {0.0, 0.0, 0.0};
  // NOTE: We use forces here over acceleration due to it being a common design
  // choice in MD, because the performance impact of storing forces vs.
  // accelerations is negligible, while the benefits in terms of code
  // organization and physics accuracy are substantial.
  std::array<double, 3> force_ = {0.0, 0.0, 0.0};
  double mass_ = 1.0;
};

using Particles = std::vector<Particle>;

#endif // INCLUDE_INCLUDE_PARTICLE_H_
