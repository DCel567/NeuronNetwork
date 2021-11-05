#pragma once
#include <math.h>
#include <random>



double sigmoid(double x);
double d_sigmoid(double x);

double relu(double x);
double d_relu(double x);

double uniform_random(double a, double b);