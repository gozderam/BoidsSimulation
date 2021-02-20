# Boids Simulation
Boids -  flocking behaviour simulation (C++, Nvidia CUDA, OpenGL). 

## Algoritm
This project is an implementation of **boids** model, which simulates flocking behaviour. It is based o Craig Reynolds' paper published in 1987. 
The program represents an example of artificial life algorithms - boids model aims to imitate the real behaviour of birds or fish (shaping shoals). 
The key aspect is the interaction between object (boids), which is handled by the algorithm with the following rules:

1. **separation** - set the direction of boid's movement to avoid crowding in one place.
2. **cohesion** - move towards the average position of another local boids.
3. **alignment** -  move in the direction which is the average heading of another local boids.

You can read more about the algorithm [here](http://www.red3d.com/cwr/boids/).

There are two algorithms implemented: one without CUDA (sequential version) and the other with CUDA (parallel - single thread for each boid).
One of the objectives of this program was to compare CUDA speedup capabilities with completely sequential version. You can see the differences while changing used algorithm in a runtime. 

## Sample

![](demo/boids_demo.gif)


## User guide
To change the algorithm used and the number of boids use appropriate key:
* 1 - CUDA algorithm, 16 boids
* 2 - CUDA algorithm, 256 boids 
* 3 - CUDA algorithm, 2048 boids 
* 4 - CUDA algorithm, 10240 boids 
* 5 - CUDA algorithm, 16348 boids 
* Q - no CUDA algorithm, 16 boids
* W - no CUDA algorithm, 256 boids 
* E - no CUDA algorithm, 2048 boids 
* R - no CUDA algorithm, 10240 boids 
* T - no CUDA algorithm, 16348 boids 

To pause/unpause program execution press P.

## Technology
C++ with CUDA and OpenGL (GLEW, GLFW, GLM). All dependecies included in the repo. 
