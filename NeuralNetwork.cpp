#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include "Neuron.h"


int main(){
    Neuron n1 = Neuron(1.0, Func::SIGM);

    std::cout << "value:\t" << n1.getValue() << std::endl;
    std::cout << "activated value:\t" << n1.getActivateValue() << std::endl;
    std::cout << "derived value:\t" << n1.getDerivedValue() << std::endl;

    std::cout << "Ne Jopa clyanus'" << std::endl;

}