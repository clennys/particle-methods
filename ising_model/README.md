# Ising Model

## Dependencies
- cmake
- gnuplot
- Tested with g++14
- matplotlib++ (automatically installed using cmake)

## Compile
You can configure the project as shown below, you can specify the C++ compiler using the '-DCMAKE_CXX_COMPILER=' flag and there are two different builds using different compiler flags for optimization. 
```bash
cmake -B build -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_BUILD_TYPE=DEBUG
cmake -B build -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_BUILD_TYPE=RELEASE
```
The project can then be compiled using the following command:
```bash
cmake --build build
```
## Run
The project can then be simply run with following command
```bash
./build/Ising
```
The resulting plots can then be found in the './img/' folder.
