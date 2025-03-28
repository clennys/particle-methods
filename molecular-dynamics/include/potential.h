#ifndef INCLUDE_INCLUDE_POTENTIAL_H_
#define INCLUDE_INCLUDE_POTENTIAL_H_

#include "constants.h"
#include "particle.h"

class LennardJonesPotential {
public:
    LennardJonesPotential(
        double epsilon = constants::DEFAULT_EPSILON,
        double sigma = constants::DEFAULT_SIGMA,
        double cutoff = constants::DEFAULT_CUTOFF * constants::DEFAULT_SIGMA
    );
    
    // Calculate energy and force between two particles
    double pair_energy(const std::array<double, 3>& r_i, const std::array<double, 3>& r_j) const;
    std::array<double, 3> pair_force(const std::array<double, 3>& r_i, const std::array<double, 3>& r_j) const;
    
    // Calculate total potential energy for all particles
    double compute_energy(const Particles& particles, const std::array<double, 3>& box_size) const;
    
    // Calculate forces for all particles
    void compute_forces(Particles& particles, const std::array<double, 3>& box_size) const;
    
    // Getters
    double cutoff() const { return cutoff_; }
    double epsilon() const { return epsilon_; }
    double sigma() const { return sigma_; }
    
private:
    double epsilon_;  // energy parameter
    double sigma_;    // distance parameter
    double cutoff_;   // cutoff distance
    
    // Precomputed values for efficiency
    double cutoff_squared_;
    double e_shift_;  // energy shift at cutoff
};

#endif  // INCLUDE_INCLUDE_POTENTIAL_H_
