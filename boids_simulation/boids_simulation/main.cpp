#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "ProgramExecutionParameters.h"

// glfw
#include <GLFW/glfw3.h>
using namespace std;

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mode);

int main(void)
{
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(800, 800, "Boids model simulation. Micha³ Gozdera", NULL, NULL);

    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current 
    glfwMakeContextCurrent(window);

    // intin glew
    if (GLEW_OK != glewInit())
        std::cout << "Error" << std::endl;

	// key calbacks
	glfwSetKeyCallback(window, keyCallback);

	ProgramExecutionParameters::calculationModule = new CudaModule(64, 32, 8, 16, -1, 1, -1, 1, 0);

	double lastTime = glfwGetTime();
	int nbFrames = 0;

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
	{
		if (!ProgramExecutionParameters::isPaused)
		{
			double currentTime = glfwGetTime();
			nbFrames++;
			if (currentTime - lastTime >= 1.0) { // If last prinf() was more than 1 sec ago
				// printf and reset timer
				printf("%f ms/frame (%d fps)\n", 1000.0 / double(nbFrames), nbFrames);
				nbFrames = 0;
				lastTime += 1.0;
			}
			ProgramExecutionParameters::calculationModule->Draw();
		}

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

   
    glfwTerminate();
	delete ProgramExecutionParameters::calculationModule;
    return 0;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mode) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);
	if (key == GLFW_KEY_P && action == GLFW_PRESS)
	{
		ProgramExecutionParameters::isPaused = !ProgramExecutionParameters::isPaused;
	}
	else if (key == GLFW_KEY_1 && action == GLFW_PRESS) // 16 boids 
	{
		ProgramExecutionParameters::isPaused = true;
		delete ProgramExecutionParameters::calculationModule;
		ProgramExecutionParameters::calculationModule = new CudaModule(16, 1, 8, 16, -1, 1, -1, 1, 0);
		ProgramExecutionParameters::isPaused = false;
	} 
	else if (key == GLFW_KEY_2 && action == GLFW_PRESS)  // 256 boids
	{
		ProgramExecutionParameters::isPaused = true;
		delete ProgramExecutionParameters::calculationModule;
		ProgramExecutionParameters::calculationModule = new CudaModule(16, 16, 8, 16, -1, 1, -1, 1, 0);
		ProgramExecutionParameters::isPaused = false;
	}
	else if (key == GLFW_KEY_3 && action == GLFW_PRESS) // 2048 boids 
	{
		ProgramExecutionParameters::isPaused = true;
		delete ProgramExecutionParameters::calculationModule;
		ProgramExecutionParameters::calculationModule = new CudaModule(64, 32, 8, 16, -1, 1, -1, 1, 0);
		ProgramExecutionParameters::isPaused = false;
	}
	else if (key == GLFW_KEY_4 && action == GLFW_PRESS) // 10240 boids 
	{
		ProgramExecutionParameters::isPaused = true;
		delete ProgramExecutionParameters::calculationModule;
		ProgramExecutionParameters::calculationModule = new CudaModule(512, 20, 32, 64, -2, 2, -4, 4, 0);
		ProgramExecutionParameters::isPaused = false;
	}
	else if (key == GLFW_KEY_5 && action == GLFW_PRESS) // 16384 boids 
	{
		ProgramExecutionParameters::isPaused = true;
		delete ProgramExecutionParameters::calculationModule;
		ProgramExecutionParameters::calculationModule = new CudaModule(512, 32, 64, 64, -2, 2, -4, 4, 0);
		ProgramExecutionParameters::isPaused = false;
	}
	else if (key == GLFW_KEY_Q && action == GLFW_PRESS) // 16 boids without Cuda
	{
		ProgramExecutionParameters::isPaused = true;
		delete ProgramExecutionParameters::calculationModule;
		ProgramExecutionParameters::calculationModule = new NoCudaModule(16 *1, 8, 16, -1, 1, -1, 1, 0);
		ProgramExecutionParameters::isPaused = false;
	}
	else if (key == GLFW_KEY_W && action == GLFW_PRESS)  // 256 boids without Cuda
	{
		ProgramExecutionParameters::isPaused = true;
		delete ProgramExecutionParameters::calculationModule;
		ProgramExecutionParameters::calculationModule = new NoCudaModule(16 * 16, 8, 16, -1, 1, -1, 1, 0);
		ProgramExecutionParameters::isPaused = false;
	}
	else if (key == GLFW_KEY_E && action == GLFW_PRESS) // 2018 boids without Cuda
	{
		ProgramExecutionParameters::isPaused = true;
		delete ProgramExecutionParameters::calculationModule;
		ProgramExecutionParameters::calculationModule = new NoCudaModule(64 * 32, 8, 16, -1, 1, -1, 1, 0);
		ProgramExecutionParameters::isPaused = false;
	}
	else if (key == GLFW_KEY_R && action == GLFW_PRESS) // 10240 boids without Cuda
	{
		ProgramExecutionParameters::isPaused = true;
		delete ProgramExecutionParameters::calculationModule;
		ProgramExecutionParameters::calculationModule = new NoCudaModule(512 * 20, 32, 64, -2, 2, -4, 4, 0);
		ProgramExecutionParameters::isPaused = false;
	}
	else if (key == GLFW_KEY_T && action == GLFW_PRESS) // 16384 boids without Cuda
	{
		ProgramExecutionParameters::isPaused = true;
		delete ProgramExecutionParameters::calculationModule;
		ProgramExecutionParameters::calculationModule = new NoCudaModule(512* 32, 32, 64, -2, 2, -4, 4, 0);
		ProgramExecutionParameters::isPaused = false;
	}
}

