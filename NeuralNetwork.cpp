#include <iostream>
#include <vector>
#include <string>

#include "Neuron.h"



int main(){
    // Neuron n1 = Neuron(1.0, Func::SIGM);

    // std::cout << "value:\t" << n1.get_value() << std::endl;
    // std::cout << "activated value:\t" << n1.get_activate_value() << std::endl;
    // std::cout << "derived value:\t" << n1.get_derived_value() << std::endl;

    std::vector<int> Victor{1, 2, 3, 4, 5, 6};



    std::vector<int>::iterator joppa = Victor.begin(); 

    for(joppa; joppa < Victor.end(); joppa++){
        std::cout << *joppa << std::endl;
    }

}
