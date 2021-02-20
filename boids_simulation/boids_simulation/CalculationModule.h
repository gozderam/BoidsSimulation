#include "cudaModule.h"
#include "NoCudaModule.h"

#ifndef CALCUL_MODULE
#define CALCUL_MODULE
class CalculationModule
{
public:
	virtual void Draw() = 0;
	virtual ~CalculationModule() {}
};
#endif
