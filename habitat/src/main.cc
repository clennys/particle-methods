#include "../include/sim.h"

int main (int argc, char *argv[]) {
	Habitat habitat(10,1,100,900, 50, 50, 0.5);
	habitat.time_step();
	return 0;
}
