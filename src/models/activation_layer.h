#ifndef ACTIV_LAYER
#define ACTIV_LAYER

#include <random>
#include <stddef.h> 
#include <vector>
#include <map>
#include <iostream> 

#include "../math/tensor.h"

class ActivationLayer
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

    TensorSize input_size; 

    ActivationType activation_type; 
    Tensor activ_grad;
    
    ActivationType GetActivationType(const std::string& activation_type) const;
    void Activate(Tensor &output);

public:
    ActivationLayer(
        TensorSize _size,
        const std::string& activation_type = "none"
    );

    Tensor Forward(const Tensor &X);
    Tensor Backward(const Tensor &grad);
};


#endif