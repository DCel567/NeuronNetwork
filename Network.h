#include <vector>
#include <string>
#include <math.h>

class NNLayer;
class NNWeight;
class NNNeuron;
class NNConnection;

typedef std::vector< NNLayer* >  VectorLayers;
typedef std::vector< NNWeight* >  VectorWeights;
typedef std::vector< NNNeuron* >  VectorNeurons;
typedef std::vector< NNConnection > VectorConnections;


// Neural Network class

class NeuralNetwork
{
public:
    NeuralNetwork();
    virtual ~NeuralNetwork();
    
    void Calculate(double* inputVector, uint32_t iCount, 
        double* outputVector = nullptr, uint32_t oCount = 0);

    void Backpropagate(double *actualOutput, 
         double *desiredOutput, uint32_t count);

    VectorLayers m_Layers;
};


// Layer class

class NNLayer
{
public:
    NNLayer(std::string str, NNLayer* pPrev = nullptr );
    virtual ~NNLayer();
    
    void Calculate();
    
    void Backpropagate( std::vector<double>& dErr_wrt_dXn /* in */, 
        std::vector<double>& dErr_wrt_dXnm1 /* out */, 
        double etaLearningRate );

    NNLayer* m_pPrevLayer;
    VectorNeurons m_Neurons;
    VectorWeights m_Weights;
};


// Neuron class

class NNNeuron{
public:
    NNNeuron( std::string str );
    virtual ~NNNeuron();

    void AddConnection( uint32_t iNeuron, uint32_t iWeight );
    void AddConnection( NNConnection const & conn );

    double output;

    VectorConnections m_Connections;
};


// Connection class

class NNConnection{
public: 
    NNConnection(uint32_t neuron = UINT32_MAX, uint32_t weight = UINT32_MAX);
    virtual ~NNConnection();

    uint32_t NeuronIndex;
    uint32_t WeightIndex;
};


// Weight class

class NNWeight{
public:
    NNWeight( std::string str, double val = 0.0 );
    virtual ~NNWeight();

    double value;
};


double SIGMOID(double x){
    return 1/(1 + exp(x));
}

double DSIGMOID(double x){
    x = SIGMOID(x);
    return x*(1 - x);
}