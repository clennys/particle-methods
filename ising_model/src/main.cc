#include "../include/ising.hpp"
#include <iostream>
int main() {
	std::cout << "Init Ising model" << std::endl;
	Ising model(100, 3);
	std::cout << "Start simulation" << std::endl;
	model.simulate(1);
	std::cout << "Simulation done" << std::endl;
}
