#include "Network.h"


/// Can't use this here for some reason. Check Network.h for definitions.

// NNLayer::NNLayer(std::string str, NNLayer* pPrev){
//     this->tag = str;
//     this->m_pPrevLayer = pPrev;
// }
//
// NNWeight::NNWeight( double v ){
//     this->value = v;
// }



void NeuralNetwork::Calculate(double* inputVector, uint32_t iCount, 
               double* outputVector /* =NULL */, 
               uint32_t oCount /* =0 */)
                              
{
    VectorLayers::iterator lit = m_Layers.begin();
    VectorNeurons::iterator nit;
    
    // first layer is input layer: directly
    // set outputs of all of its neurons
    // to the given input vector
    
    if ( lit < m_Layers.end() )  
    {
        nit = (*lit)->m_Neurons.begin();
        int count = 0;
        
        assert( iCount == (*lit)->m_Neurons.size() );
        // there should be exactly one neuron per input
        
        while( ( nit < (*lit)->m_Neurons.end() ) && ( count < iCount ) )
        {
            (*nit)->output = inputVector[ count ];
            nit++;
            count++;
        }
    }
    
    // iterate through remaining layers,
    // calling their Calculate() functions
    
    for( lit++; lit<m_Layers.end(); lit++ )
    {
        (*lit)->Calculate();
    }
    
    // load up output vector with results
    
    if ( outputVector != NULL )
    {
        lit = m_Layers.end();
        lit--;
        
        nit = (*lit)->m_Neurons.begin();
        
        for ( int ii=0; ii<oCount; ++ii )
        {
            outputVector[ ii ] = (*nit)->output;
            nit++;
        }
    }
}


void NNLayer::Calculate()
{
    assert( m_pPrevLayer != NULL );
    
    VectorNeurons::iterator nit;
    VectorConnections::iterator cit;
    
    double dSum;
    
    for( nit=m_Neurons.begin(); nit<m_Neurons.end(); nit++ )
    {
        NNNeuron& n = *(*nit);  // to ease the terminology
        
        cit = n.m_Connections.begin();
        
        // check if current weight of neuron is under m_Weights size

        assert( (*cit).WeightIndex < m_Weights.size() );
        
        // weight of the first connection is the bias;
        // its neuron-index is ignored

        dSum = m_Weights[ (*cit).WeightIndex ]->value;  
        
        for ( cit++ ; cit<n.m_Connections.end(); cit++ )
        {
            assert( (*cit).WeightIndex < m_Weights.size() );
            assert( (*cit).NeuronIndex < 
                     m_pPrevLayer->m_Neurons.size() );
            
            dSum += ( m_Weights[ (*cit).WeightIndex ]->value ) * 
                ( m_pPrevLayer->m_Neurons[ 
                   (*cit).NeuronIndex ]->output );
        }
        
        n.output = sigmoid(dSum);
    }
}

void NeuralNetwork::Backpropagate(double *actualOutput, 
     double *desiredOutput, uint32_t count)
{
    // Backpropagates through the neural net
    // Proceed from the last layer to the first, iteratively
    // We calculate the last layer separately, and first,
    // since it provides the needed derviative
    // (i.e., dErr_wrt_dXnm1) for the previous layers
    
    // nomenclature:
    //
    // Err is output error of the entire neural net
    // Xn is the output vector on the n-th layer
    // Xnm1 is the output vector of the previous layer
    // Wn is the vector of weights of the n-th layer
    // Yn is the activation value of the n-th layer,
    // i.e., the weighted sum of inputs BEFORE 
    //    the squashing function is applied
    // F is the squashing function: Xn = F(Yn)
    // F' is the derivative of the squashing function
    //   Conveniently, for F = tanh,
    //   then F'(Yn) = 1 - Xn^2, i.e., the derivative can be 
    //   calculated from the output, without knowledge of the input
    
    
    VectorLayers::iterator lit = m_Layers.end() - 1;
    
    std::vector<double> dErr_wrt_dXlast( (*lit)->m_Neurons.size() );
    std::vector<std::vector<double>> differentials;
    
    int iSize = m_Layers.size();
    
    differentials.resize(iSize);
    
    int ii;
    
    // start the process by calculating dErr_wrt_dXn for the last layer.
    // for the standard MSE Err function
    // (i.e., 0.5*sumof( (actual-target)^2 ), this differential is simply
    // the difference between the target and the actual
    
    for ( ii=0; ii<(*lit)->m_Neurons.size(); ++ii )
    {
        dErr_wrt_dXlast[ ii ] = 
            actualOutput[ ii ] - desiredOutput[ ii ];
    }
    


    differentials[ iSize-1 ] = dErr_wrt_dXlast;  // last one
    
    for ( ii=0; ii<iSize-1; ++ii )
    {
        differentials[ ii ].resize( 
             m_Layers[ii]->m_Neurons.size(), 0.0 );
    }
    
    // now iterate through all layers including
    // the last but excluding the first, and ask each of
    // them to backpropagate error and adjust
    // their weights, and to return the differential
    // dErr_wrt_dXnm1 for use as the input value
    // of dErr_wrt_dXn for the next iterated layer
    
    ii = iSize - 1;
    for ( lit; lit>m_Layers.begin(); lit--)
    {
        (*lit)->Backpropagate( differentials[ ii ], 
              differentials[ ii - 1 ], /*m_etaLearningRate*/ 0.05 );
        --ii;
    }
    
    differentials.clear();
}


void NNLayer::Backpropagate( std::vector< double >& dErr_wrt_dXn /* in */, 
                            std::vector< double >& dErr_wrt_dXnm1 /* out */, 
                            double etaLearningRate )
{
    double output;
    std::vector< double > dErr_wrt_dYn( dErr_wrt_dXn.size() );
    std::vector< double > dErr_wrt_dWn( dErr_wrt_dXn.size() );

    // calculate equation (3): dErr_wrt_dYn = F'(Yn) * dErr_wrt_Xn
    
    int ii;
    for ( ii=0; ii<m_Neurons.size(); ++ii )
    {
        output = m_Neurons[ ii ]->output;
        dErr_wrt_dYn[ ii ] = d_sigmoid( output ) * dErr_wrt_dXn[ ii ];
    }
    
    // calculate equation (4): dErr_wrt_Wn = Xnm1 * dErr_wrt_Yn
    // For each neuron in this layer, go through
    // the list of connections from the prior layer, and
    // update the differential for the corresponding weight
    
    ii = 0;
    for ( VectorNeurons::iterator nit=m_Neurons.begin(); nit<m_Neurons.end(); nit++ )
    {
        NNNeuron& n = *(*nit);  // for simplifying the terminology
        
        for ( VectorConnections::iterator cit=n.m_Connections.begin(); cit<n.m_Connections.end(); cit++ )
        {
            int kk = (*cit).NeuronIndex;
            if ( kk == UINT32_MAX )
            {
                output = 1.0;  // this is the bias weight
            }
            else
            {
                output = m_pPrevLayer->m_Neurons[ kk ]->output;
            }
            
            dErr_wrt_dWn[ (*cit).WeightIndex ] = dErr_wrt_dYn[ ii ] * output;
        }
        
        ii++;
    }
    
    
    // calculate equation (5): dErr_wrt_Xnm1 = Wn * dErr_wrt_dYn,
    // which is needed as the input value of
    // dErr_wrt_Xn for backpropagation of the next (i.e., previous) layer
    // For each neuron in this layer
    
    ii = 0;
    for ( VectorNeurons::iterator nit=m_Neurons.begin(); nit<m_Neurons.end(); nit++ )
    {
        NNNeuron& n = *(*nit);  // for simplifying the terminology
        
        for ( VectorConnections::iterator cit=n.m_Connections.begin(); 
              cit<n.m_Connections.end(); cit++ )
        {
            int kk=(*cit).NeuronIndex;
            if ( kk != UINT32_MAX )
            {
                // we exclude ULONG_MAX, which signifies
                // the phantom bias neuron with
                // constant output of "1",
                // since we cannot train the bias neuron
                
                int nIndex = kk;
                
                dErr_wrt_dXnm1[ nIndex ] += dErr_wrt_dYn[ ii ] * 
                       m_Weights[ (*cit).WeightIndex ]->value;
            }
            
        }
        
        ii++;  // ii tracks the neuron iterator
        
    }
    
    
    // calculate equation (6): update the weights
    // in this layer using dErr_wrt_dW (from 
    // equation (4)    and the learning rate eta

    for ( int jj=0; jj<m_Weights.size(); ++jj )
    {
        double oldValue = m_Weights[ jj ]->value;
        double newValue = oldValue - etaLearningRate * dErr_wrt_dWn[ jj ];
        m_Weights[ jj ]->value = newValue;
    }
}

void NNNeuron::AddConnection(uint32_t iNeuron, uint32_t iWeight){
    this->m_Connections.push_back(NNConnection(iNeuron, iWeight));
}

void NNNeuron::AddConnection(NNConnection const & con){
    this->m_Connections.push_back(con);
}

//TODO NNConnection constructor