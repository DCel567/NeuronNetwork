#include "Neuron.h"
#include <math.h>

Neuron::Neuron(double value){
    this->set_value(value);
    this->activate();
    this->derive();
}

Neuron::Neuron(double value, Func func){
    this->set_value(value);
    this->funcType = func;
    this->activate();
    this->derive();
}

void Neuron::set_value(double value){
    this->value = value;
}

void Neuron::activate(){
    if(funcType == Func::RELU){
        this->activatedValue = 0;
        if(this->value > 0)
            this->activatedValue = this->value;
    }
    else if(funcType == Func::TAHN)
        this->activatedValue = tanh(this->value);
    else if(funcType == Func::SIGM)
        this->activatedValue = 1 / (1 + exp(-this->value));
    //else
    //    error!
}

void Neuron::derive(){
    if(funcType == Func::RELU){
        this->derivedValue = 0;
        if(this->value > 0)
            this->derivedValue = 1;
    }
    else if(funcType == Func::TAHN)
        this->derivedValue = 1 - this->activatedValue * this->activatedValue;
    else if(funcType == Func::SIGM)
        this->derivedValue = this->activatedValue * (1 - this->activatedValue);
    //else
    //    error!
}
