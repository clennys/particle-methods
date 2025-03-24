#include "../include/sim.h"
#include <iostream>

int main(int argc, char *argv[]) {
  Habitat habitat(10, 1, 100, 900, 50, 100, 0.5);
  std::cout << "Start simulation" << std::endl;
  habitat.time_step();
  return 0;
}
