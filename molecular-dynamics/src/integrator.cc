#include "../include/integrator.h"
#include "../include/array_ops.h"

using namespace integrator;

void Integrator::apply_PBC(Particles &particles,
                           const std::array<double, 3> &box_size) const {}

void SemiImplicitEuler::integrate(Particles &particles, double dt) const {
  for (Particle &p : particles) {
    std::array<double, 3> position = p.position();
    std::array<double, 3> velocity = p.velocity();

    // NOTE: (dhub) Order of update is important
    velocity += p.force() * (1 / p.mass()) * dt;
    position += velocity * dt;

    p.set_position(position);
    p.set_velocity(velocity);
  }
}

void VelocityVerlet::first_half_step(Particles &particles, double dt) const {
  for (Particle &p : particles) {
    std::array<double, 3> position = p.position();
    std::array<double, 3> velocity = p.velocity();

    position += velocity * dt + 0.5 * p.force() * (1 / p.mass()) * dt * dt;
    velocity += 0.5 * p.force() * (1 / p.mass()) * dt;
    p.set_position(position);
    p.set_velocity(velocity);
  }
}
void VelocityVerlet::second_half_step(Particles &particles, double dt) const {
  for (Particle &p : particles) {
    std::array<double, 3> position = p.position();
    std::array<double, 3> velocity = p.velocity();

    velocity += 0.5 * p.force() * (1 / p.mass()) * dt;
    p.set_velocity(velocity);
  }
}
void VelocityVerlet::integrate(Particles &particles, double dt) const {
	first_half_step(particles, dt);

	// TODO: (dhub) Recalculate forces based on new positions
	

	second_half_step(particles, dt);
}
