#include "../include/particle.h"
#include "../include/array_ops.h"

Particle::Particle(double mass) : mass_(mass) {}

const std::array<double, 3> &Particle::position() const { return position_; }

const std::array<double, 3> &Particle::velocity() const { return velocity_; }

const std::array<double, 3> &Particle::force() const { return force_; }

double Particle::mass() const { return mass_; }

void Particle::set_position(const std::array<double, 3> &position) {
  position_ = position;
}

void Particle::set_velocity(const std::array<double, 3> &velocity) {
  velocity_ = velocity;
}

void Particle::set_force(const std::array<double, 3> &force) { force_ = force; }

void Particle::reset_force() { force_ *= 0; }

double Particle::kinetic_energy() const {
  return 0.5 * mass_ * velocity_ * velocity_;
}
