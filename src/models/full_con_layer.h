#ifndef FULL_CON_LAYER
#define FULL_CON_LAYER

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

    bool   soft_max;
    double soft_max_sum;

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
    Tensor softmax_grad;
    
    ActivationType GetActivationType(const std::string& activation_type) const;

    void InitWeights();
    void Activate(Tensor &output);
    void SoftMax(Tensor &output);

public:
    FullyConnectedLayer(
        TensorSize _size,
        int outputs, 
        const std::string& activation_type = "none",
        bool soft_max=0
    );

    Tensor Forward(const Tensor &X);
    Tensor Backward(const Tensor &grad, const Tensor &X);

    void UpdateWeights(double learning_rate);

    // функции для тестов 

    Tensor GetFilter();
    Tensor GetFilterGrad();

    std::vector<double> GetOffsets();
    std::vector<double> GetOffsetsGrad();

    void SetFilter(Tensor _filter);
    void SetOffset(std::vector<double> _offset);
};


#endif