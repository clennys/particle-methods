#ifndef INCLUDE_INCLUDE_CONSTANTS_H_
#define INCLUDE_INCLUDE_CONSTANTS_H_

namespace constants {
    // Physical constants
    constexpr double kB = 1.380649e-23;  // Boltzmann constant in J/K
    
    // Lennard-Jones parameters
    constexpr double DEFAULT_EPSILON = 1.0;  // energy parameter (typically in units of kJ/mol or kcal/mol)
    constexpr double DEFAULT_SIGMA = 1.0;    // distance parameter (typically in units of Angstrom)
    
    // Simulation parameters
    constexpr double DEFAULT_CUTOFF = 2.5;   // in units of sigma
    constexpr double DEFAULT_SKIN = 0.3;     // skin distance for cell list updates
}

#endif  // INCLUDE_INCLUDE_CONSTANTS_H_
