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
The project can then be simply run with following commands:
- For the plots from the exercises
    ```bash
    ./build/Ising -e
    ```
The resulting plots can then be found in the './img/' folder.
- For an animation (2000 frames on a 25x25 lattice) at a specific temperature
    ```bash
    ./build/Ising -a -T 0.2
    ```
    - The images of the grid can be found in the './anim/' folder.
    - Using the following ffmpeg command you can create the animation
    ```bash
    ffmpeg -framerate 60 -i anim/lattice_%03d.png -c:v libx264 -pix_fmt yuv420p -r 60 lattice_animation.mp4
    ```
