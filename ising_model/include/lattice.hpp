#ifndef LATTICE_H_
#define LATTICE_H_

#include <vector>

class Lattice {
public:
  Lattice(int size);
  // Lattice(Lattice &&) = default;
  // Lattice(const Lattice &) = default;
  // Lattice &operator=(Lattice &&) = default;
  // Lattice &operator=(const Lattice &) = default;
  ~Lattice();
	void plot();
	void random_init();
	int get_dim();
	int sum_of_neighbors(int row, int col);
  std::vector<std::vector<int> > L;

private:
	int frame;
};

#endif // LATTICE_H_
