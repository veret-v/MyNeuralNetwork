#ifndef CONVOLUTION_NN
#define CONVOLUTION_NN

#include <random>
#include <stddef.h> 
#include <vector>
#include <map>
#include <iostream> 

#include "../math/tensor.h"

class ConvolutionLayer
{
private:
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;

    std::vector<Tensor> kernels;
    std::vector<double> offsets;

    std::vector<Tensor> kernels_grad;
    std::vector<double> offsets_grad;

    TensorSize input_size; 
    TensorSize output_size; 
    TensorSize kernel_size;

    size_t step;
    size_t padding;
    size_t filters_num;
    
    void InitWeights();

public:
    ConvolutionLayer(
        TensorSize _size,
        TensorSize _kernel_size,
        size_t _filters_num,
        size_t _step,
        size_t _padding
    );

    Tensor Forward(const Tensor &X);
    Tensor Backward(const Tensor &grad, const Tensor &X);

    // функции для тестов 

    std::vector<Tensor> GetKernels();
    std::vector<Tensor> GetKernelsGrad();

    std::vector<double> GetOffsets();
    std::vector<double> GetOffsetsGrad();

    void SetKernels(Tensor _kernel);
    void SetOffset(double _offset);
};


#endif