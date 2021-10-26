#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include "Neuron.h"
#include "Matrix.h"


int main(){
    Neuron n1 = Neuron(1.0, Func::SIGM);

    std::cout << "value:\t" << n1.get_value() << std::endl;
    std::cout << "activated value:\t" << n1.get_activate_value() << std::endl;
    std::cout << "derived value:\t" << n1.get_derived_value() << std::endl;

}
