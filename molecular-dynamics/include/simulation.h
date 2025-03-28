#ifndef INCLUDE_INCLUDE_SIMULATION_H_
#define INCLUDE_INCLUDE_SIMULATION_H_

#include "../include/particle.h"
#include "../include/potential.h"
#include "../include/integrator.h"
#include "../include/cell_list.h"
#include <array>
#include <functional>

class Simulation {
public:
    enum class Ensemble {
        NVE,  // Microcanonical ensemble (constant energy)
        NVT,  // Canonical ensemble (constant temperature)
    };
    
    Simulation(const std::array<double, 3>& box_size, double dt = 0.005);
    
    // Setup methods
    void set_ensemble(Ensemble ensemble) { ensemble_ = ensemble; }
    void set_temperature(double temperature) { temperature_ = temperature; }
    void set_particles(const Particles& particles) { particles_ = particles; }
    
    // Initialize particle positions in a cubic lattice
    void initialize_cubic_lattice(int num_particles, double density);
    
    // Run for a specified number of steps
    void run(int num_steps);
    
    // Add observer to be called at specified interval
    using ObserverFunction = std::function<void(const Simulation&, int)>;
    void add_observer(ObserverFunction observer, int interval);
    
    // Getters
    const Particles& particles() const { return particles_; }
    double total_energy() const;
    double kinetic_energy() const;
    double potential_energy() const;
    double temperature() const;
    double time() const { return time_; }
    int step() const { return step_; }
    const std::array<double, 3>& box_size() const { return box_size_; }
    
private:
    // Perform a single time step
    void step();
    
    // Calculate system properties
    void calculateProperties();
    
    Particles particles_;
    std::array<double, 3> box_size_;
    
    // Simulation components
    LennardJonesPotential potential_;
		integrator::VelocityVerlet integrator_;
    std::unique_ptr<CellList> cell_list_;
    
    // Simulation parameters
    double dt_;
    double time_ = 0.0;
    int step_ = 0;
    Ensemble ensemble_ = Ensemble::NVE;
    double temperature_ = 1.0;
    
    // System properties
    double kinetic_energy_ = 0.0;
    double potential_energy_ = 0.0;
    
    // Observers
    std::vector<std::pair<ObserverFunction, int>> observers_;
};

#endif  // INCLUDE_INCLUDE_SIMULATION_H_
