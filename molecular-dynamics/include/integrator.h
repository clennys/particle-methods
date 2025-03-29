#ifndef INCLUDE_INCLUDE_INTEGRATOR_H_
#define INCLUDE_INCLUDE_INTEGRATOR_H_

#include "particle.h"

namespace integrator {

class Integrator {
public:
  virtual ~Integrator() = default;

  // Perform a full integration step
  virtual void integrate(Particles &particles, double dt) const = 0;

  // Apply periodic boundary conditions
  virtual void apply_PBC(Particles &particles,
                         const std::array<double, 3> &box_size) const;

  // Rescale velocities to match target temperature
  // virtual void rescale_velocities(Particles &particles,
                                  // double target_temperature) const;

  // Initialize velocities from Maxwell-Boltzmann distribution
  // virtual void initialize_velocities(Particles &particles,
                                     // double temperature) const;
};

class SemiImplicitEuler : public Integrator {
public:
  SemiImplicitEuler() = default;

  void integrate(Particles &particles, double dt) const override;
};

class VelocityVerlet : public Integrator {
public:
  VelocityVerlet() = default;

  // Perform a full velocity Verlet integration step
  void integrate(Particles &particles, double dt) const override;

  // First half of integration step (update positions and half-step velocities)
  void first_half_step(Particles &particles, double dt) const;

  // Second half of integration step (update velocities with new forces)
  void second_half_step(Particles &particles, double dt) const;
};

} // namespace integrator

#endif // INCLUDE_INCLUDE_INTEGRATOR_H_
