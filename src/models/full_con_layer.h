#ifndef CONVOLUTION_NN
#define CONVOLUTION_NN

#include <random>
#include <stddef.h> 
#include <vector>
#include <map>
#include <iostream> 

#include "../math/tensor.h"

class FullyConnectedLayer
{
private:
    enum class ActivationType {
        None, // без активации
        Sigmoid, // сигмоидальная функция
        Tanh, // гиперболический тангенс
        ReLU, // выпрямитель
        LeakyReLU, // выпрямитель с утечкой
        ELU // экспоненциальный выпрямитель
    };

    std::default_random_engine generator;
    std::normal_distribution<double> distribution;

    int inputs; 
    int outputs;

    Tensor filter;
    Tensor filter_grad;

    std::vector<double> offset;
    std::vector<double> offset_grad;

    TensorSize input_size; 
    TensorSize output_size; 
    TensorSize filter_size;

    ActivationType activation_type; 
    Tensor activ_grad;
    
    ActivationType GetActivationType(const std::string& activation_type) const;

    void InitWeights();
    void Activate(Tensor &output);

public:
    FullyConnectedLayer(
        TensorSize _size,
        int outputs, 
        const std::string& activation_type = "none"
    );

    Tensor Forward(const Tensor &X);
    Tensor Backward(const Tensor &grad, const Tensor &X);

    void UpdateWeights(double learning_rate);
};


#endif