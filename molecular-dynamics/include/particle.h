#ifndef INCLUDE_INCLUDE_PARTICLE_H_
#define INCLUDE_INCLUDE_PARTICLE_H_
#include <vector>
#include <array>

class Particle {
public:
    Particle(double mass = 1.0);
    
    const std::array<double, 3>& position() const { return position_; }
    const std::array<double, 3>& velocity() const { return velocity_; }
    const std::array<double, 3>& force() const { return force_; }
    double mass() const { return mass_; }
    
    void set_position(const std::array<double, 3>& position) { position_ = position; }
    void set_velocity(const std::array<double, 3>& velocity) { velocity_ = velocity; }
    void set_force(const std::array<double, 3>& force) { force_ = force; }
    
    // Add force
    void add_force(const std::array<double, 3>& force);
    
    // Reset force to zero
    void reset_force();
    
    // Compute kinetic energy
    double kinetic_energy() const;
    
private:
    std::array<double, 3> position_ = {0.0, 0.0, 0.0};
    std::array<double, 3> velocity_ = {0.0, 0.0, 0.0};
    std::array<double, 3> force_ = {0.0, 0.0, 0.0};
    double mass_ = 1.0;
};

using Particles = std::vector<Particle>;

#endif  // INCLUDE_INCLUDE_PARTICLE_H_
