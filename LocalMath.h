#pragma once
#include <math.h>
#include <random>



double sigmoid(double x);
double d_sigmoid(double x);

double relu(double x);
double d_relu(double x);

double lrelu(double x);
double d_lrelu(double x);

double uniform_random(double a, double b);
double xavier_weight(double a, double b, int32_t neuron_num);
