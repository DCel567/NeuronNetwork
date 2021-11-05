#include "LocalMath.h"


double sigmoid(double x){
    return 1/(1 + exp(x));
}

double d_sigmoid(double x){
    x = sigmoid(x);
    return x*(1 - x);
}

double relu(double x){
    return (x <= 0) ? 0 : x;
}

double d_relu(double x){
    return (x <= 0) ? 0 : 1;
}

double uniform_random(double a, double b){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(a, b);
    return dis(gen);
}