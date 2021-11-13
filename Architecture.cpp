#include "Architecture.h"

// simplified code

bool Architecture::MNISTNet()
{    
    // initialize and build the neural net
    
    NeuralNetwork NN = NeuralNetwork();  // for easier nomenclature
    
    NNLayer* pLayer;
    
    int ii, jj, kk;
    double initWeight;
    
    // layer zero, the input layer.
    // Create neurons: exactly the same number of neurons as the input
    // vector of 29x29=841 pixels, and no weights/connections
    
    pLayer = new NNLayer( "Layer00" );
    NN.m_Layers.push_back( pLayer );
    
    for ( ii=0; ii<841; ++ii )
    {
        pLayer->m_Neurons.push_back( new NNNeuron() );
    }

    
    // layer one:
    // This layer is a convolutional layer that
    // has 6 feature maps.  Each feature 
    // map is 13x13, and each unit in the
    // feature maps is a 5x5 convolutional kernel
    // of the input layer.
    // So, there are 13x13x6 = 1014 neurons, (5x5+1)x6 = 156 weights
    
    pLayer = new NNLayer( "Layer01", pLayer );
    NN.m_Layers.push_back( pLayer );
    
    for ( ii=0; ii<1014; ++ii )
    {
        pLayer->m_Neurons.push_back( new NNNeuron() );
    }
    
    for ( ii=0; ii<156; ++ii )
    {
        initWeight = 0.05 * uniform_random(-1, 1);
        // My uniform random distribution

        pLayer->m_Weights.push_back( new NNWeight( initWeight ) );
    }
    
    // interconnections with previous layer: this is difficult
    // The previous layer is a top-down bitmap
    // image that has been padded to size 29x29
    // Each neuron in this layer is connected
    // to a 5x5 kernel in its feature map, which 
    // is also a top-down bitmap of size 13x13. 
    // We move the kernel by TWO pixels, i.e., we
    // skip every other pixel in the input image
    
    int kernelTemplate[25] = {
        0,  1,  2,  3,  4,
        29, 30, 31, 32, 33,
        58, 59, 60, 61, 62,
        87, 88, 89, 90, 91,
        116,117,118,119,120 };
        
    int iNumWeight;
        
    int fm;  // "fm" stands for "feature map"
        
    for ( fm=0; fm<6; ++fm)
    {
        for ( ii=0; ii<13; ++ii )
        {
            for ( jj=0; jj<13; ++jj )
            {
                // 26 is the number of weights per feature map
                iNumWeight = fm * 26;
                NNNeuron& n = 
                   *( pLayer->m_Neurons[ jj + ii*13 + fm*169 ] );
                
                n.AddConnection( UINT32_MAX, iNumWeight++ );  // bias weight
                
                for ( kk=0; kk<25; ++kk )
                {
                    // note: max val of index == 840, 
                    // corresponding to 841 neurons in prev layer
                    n.AddConnection( 2*jj + 58*ii + 
                        kernelTemplate[kk], iNumWeight++ );
                }
            }
        }
    }
    
    
    // layer two:
    // This layer is a convolutional layer
    // that has 50 feature maps.  Each feature 
    // map is 5x5, and each unit in the feature
    // maps is a 5x5 convolutional kernel
    // of corresponding areas of all 6 of the
    // previous layers, each of which is a 13x13 feature map
    // So, there are 5x5x50 = 1250 neurons, (5x5+1)x6x50 = 7800 weights
    
    pLayer = new NNLayer( "Layer02", pLayer );
    NN.m_Layers.push_back( pLayer );
    
    for ( ii=0; ii<1250; ++ii )
    {
        pLayer->m_Neurons.push_back( 
                // new NNNeuron( (LPCTSTR)label ) );  было так, не знаю, что такое label
                 new NNNeuron());
    }
    
    for ( ii=0; ii<7800; ++ii )
    {
        initWeight = 0.05 * uniform_random(-1, 1);
        pLayer->m_Weights.push_back( new NNWeight( initWeight ) );
    }
    
    // Interconnections with previous layer: this is difficult
    // Each feature map in the previous layer
    // is a top-down bitmap image whose size
    // is 13x13, and there are 6 such feature maps.
    // Each neuron in one 5x5 feature map of this 
    // layer is connected to a 5x5 kernel
    // positioned correspondingly in all 6 parent
    // feature maps, and there are individual
    // weights for the six different 5x5 kernels.  As
    // before, we move the kernel by TWO pixels, i.e., we
    // skip every other pixel in the input image.
    // The result is 50 different 5x5 top-down bitmap
    // feature maps
    
    int kernelTemplate2[25] = {
        0,  1,  2,  3,  4,
        13, 14, 15, 16, 17, 
        26, 27, 28, 29, 30,
        39, 40, 41, 42, 43, 
        52, 53, 54, 55, 56   };
        
        
    for ( fm=0; fm<50; ++fm)
    {
        for ( ii=0; ii<5; ++ii )
        {
            for ( jj=0; jj<5; ++jj )
            {
                // 26 is the number of weights per feature map
                iNumWeight = fm * 26;
                NNNeuron& n = *( pLayer->m_Neurons[ jj + ii*5 + fm*25 ] );
                
                n.AddConnection( UINT32_MAX, iNumWeight++ );  // bias weight
                
                for ( kk=0; kk<25; ++kk )
                {
                    // note: max val of index == 1013,
                    // corresponding to 1014 neurons in prev layer
                    n.AddConnection(       2*jj + 26*ii + 
                     kernelTemplate2[kk], iNumWeight++ );
                    n.AddConnection( 169 + 2*jj + 26*ii + 
                     kernelTemplate2[kk], iNumWeight++ );
                    n.AddConnection( 338 + 2*jj + 26*ii + 
                     kernelTemplate2[kk], iNumWeight++ );
                    n.AddConnection( 507 + 2*jj + 26*ii + 
                     kernelTemplate2[kk], iNumWeight++ );
                    n.AddConnection( 676 + 2*jj + 26*ii + 
                     kernelTemplate2[kk], iNumWeight++ );
                    n.AddConnection( 845 + 2*jj + 26*ii + 
                     kernelTemplate2[kk], iNumWeight++ );
                }
            }
        }
    }
            
    
    // layer three:
    // This layer is a fully-connected layer
    // with 100 units.  Since it is fully-connected,
    // each of the 100 neurons in the
    // layer is connected to all 1250 neurons in
    // the previous layer.
    // So, there are 100 neurons and 100*(1250+1)=125100 weights
    
    pLayer = new NNLayer( "Layer03", pLayer );
    NN.m_Layers.push_back( pLayer );
    
    for ( ii=0; ii<100; ++ii )
    {
        pLayer->m_Neurons.push_back( 
           //new NNNeuron( (LPCTSTR)label ) );  было так, не знаю, что за лэйбл
           new NNNeuron() );
    }
    
    for ( ii=0; ii<125100; ++ii )
    {
        initWeight = 0.05 * uniform_random(-1, 1);
    }
    
    // Interconnections with previous layer: fully-connected
    
    iNumWeight = 0;  // weights are not shared in this layer
    
    for ( fm=0; fm<100; ++fm )
    {
        NNNeuron& n = *( pLayer->m_Neurons[ fm ] );
        n.AddConnection( UINT32_MAX, iNumWeight++ );  // bias weight
        
        for ( ii=0; ii<1250; ++ii )
        {
            n.AddConnection( ii, iNumWeight++ );
        }
    }
    
    // layer four, the final (output) layer:
    // This layer is a fully-connected layer
    // with 10 units.  Since it is fully-connected,
    // each of the 10 neurons in the layer
    // is connected to all 100 neurons in
    // the previous layer.
    // So, there are 10 neurons and 10*(100+1)=1010 weights
    
    pLayer = new NNLayer( "Layer04", pLayer );
    NN.m_Layers.push_back( pLayer );
    
    for ( ii=0; ii<10; ++ii )
    {
        pLayer->m_Neurons.push_back( 
              //new NNNeuron( (LPCTSTR)label ) );  было так. не знаю, что за лэйбл
              new NNNeuron() );
    }
    
    for ( ii=0; ii<1010; ++ii )
    {
        initWeight = 0.05 * uniform_random(-1, 1);
    }
    
    // Interconnections with previous layer: fully-connected
    
    iNumWeight = 0;  // weights are not shared in this layer
    
    for ( fm=0; fm<10; ++fm )
    {
        NNNeuron& n = *( pLayer->m_Neurons[ fm ] );
        n.AddConnection( UINT32_MAX, iNumWeight++ );  // bias weight
        
        for ( ii=0; ii<100; ++ii )
        {
            n.AddConnection( ii, iNumWeight++ );
        }
    }
    
    
    //SetModifiedFlag( TRUE ); было так, не знаю, для чего это
    
    return TRUE;
}