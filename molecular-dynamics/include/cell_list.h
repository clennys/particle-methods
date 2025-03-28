#ifndef INCLUDE_SRC_CELL_LIST_CC_
#define INCLUDE_SRC_CELL_LIST_CC_

#include "particle.h"
#include "potential.h"

class CellList {
public:
  CellList(double cutoff, double skin, const std::array<double, 3> &box_size);

  // Build cell lists from particle positions
  void build(const Particles &particles, const std::array<double, 3> &box_size);

  // Check if cell list needs rebuilding
  bool needs_rebuild(const Particles &particles) const;

  // Compute forces using cell list optimization
  void compute_forces(Particles &particles,
                      const LennardJonesPotential &potential,
                      const std::array<double, 3> &box_size);

  // Compute energy using cell list optimization
  double compute_energy(const Particles &particles,
                        const LennardJonesPotential &potential,
                        const std::array<double, 3> &box_size) const;

private:
  // Convert position to cell index
  std::array<int, 3>
  position_to_cell(const std::array<double, 3> &position) const;

  // Convert 3D cell indices to linear index
  int cell_indices_to_index(const std::array<int, 3> &indices) const;

  // Get neighboring cells
  std::vector<int> get_neighbor_cells(int cell_index) const;

  double cutoff_;
  double skin_;
  double cutoff_with_skin_;

  std::array<int, 3> num_cells_;
  std::array<double, 3> cell_size_;

  // Head-of-chain cell list
  std::vector<int> head_of_chain_; // First particle in each cell
  std::vector<int> linked_list_;   // Next particle in same cell

  // Particle reference positions (for rebuild check)
  std::vector<std::array<double, 3>> reference_positions_;
};

#endif // INCLUDE_SRC_CELL_LIST_CC_
