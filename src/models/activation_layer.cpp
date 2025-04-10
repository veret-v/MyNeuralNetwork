#include "activation_layer.h"

// инициализация слоя
ActivationLayer::ActivationLayer(
        TensorSize _size,
        const std::string& activation_type
) : activ_grad(input_size)
{
    input_size.width  = _size.width;
    input_size.height = _size.height;
    input_size.depth  = _size.depth;

    this->activation_type = GetActivationType(activation_type);
}

// сопоставление строки и функции активации
ActivationLayer::ActivationType ActivationLayer::GetActivationType(const std::string& activation_type) const
{
    if (activation_type == "sigmoid")
        return ActivationType::Sigmoid;

    if (activation_type == "tanh")
        return ActivationType::Tanh;

    if (activation_type == "relu")
        return ActivationType::ReLU;

    if (activation_type == "leakyrelu")
        return ActivationType::LeakyReLU;

    if (activation_type == "elu")
        return ActivationType::ELU;

    if (activation_type == "none" || activation_type == "")
        return ActivationType::None;

    throw std::runtime_error("Invalid activation function");
}

// вычсление функции активации и ее производной
void ActivationLayer::Activate(Tensor &output)
{   
    if (activation_type == ActivationType::None)
    {
        for (int i = 0; i < input_size.height; i++) {
            for (int j = 0; j < input_size.width; j++) {
                for (int k = 0; k < input_size.depth; k++)
                    {   
                        activ_grad(k, i, j) = 1;
                    }
            }
        }
    } 
    else if (activation_type == ActivationType::Sigmoid)
    {
        for (int i = 0; i < input_size.height; i++) {
            for (int j = 0; j < input_size.width; j++) {
                for (int k = 0; k < input_size.depth; k++)
                    {   
                        output(i, 0, 0)     = 1 / (exp(-output(i, 0, 0)) + 1);
                        activ_grad(i, 0, 0) = output(i, 0, 0) * (1 - output(i, 0, 0));
                    }
            }
        } 
            
    }
    else if (activation_type == ActivationType::Tanh)
    {
        for (int i = 0; i < input_size.height; i++) {
            for (int j = 0; j < input_size.width; j++) {
                for (int k = 0; k < input_size.depth; k++)
                    {      
                        output(i, 0, 0)     = (exp(output(i, 0, 0)) - exp(-output(i, 0, 0)))/ (exp(-output(i, 0, 0)) + exp(output(i, 0, 0)));
                        activ_grad(i, 0, 0) = 1 - pow(output(i, 0, 0), 2);
                    }
            }
        }
    }
    else if (activation_type == ActivationType::ReLU)
    {
        for (int i = 0; i < input_size.height; i++) {
            for (int j = 0; j < input_size.width; j++) {
                for (int k = 0; k < input_size.depth; k++)
                {   
                    if (output(i, 0, 0) > 0)
                    {
                        activ_grad(i, 0, 0) = 1;
                    }
                    else
                    {
                        activ_grad(i, 0, 0) = 0;
                        output(i, 0, 0)     = 0; 
                    }
                }
            }
        }
    }
    else if (activation_type == ActivationType::LeakyReLU)
    {
        for (int i = 0; i < input_size.height; i++) {
            for (int j = 0; j < input_size.width; j++) {
                for (int k = 0; k < input_size.depth; k++)
                {   
                    if (output(i, 0, 0) > 0)
                    {
                        activ_grad(i, 0, 0) = 1;
                    }
                    else
                    {
                        activ_grad(i, 0, 0) = 0.01;
                        output(i, 0, 0)     *= 0.01; 
                    }
                }
            }
        }
    }
    else if (activation_type == ActivationType::ELU)
    {
        for (int i = 0; i < input_size.height; i++) {
            for (int j = 0; j < input_size.width; j++) {
                for (int k = 0; k < input_size.depth; k++)
                {   
                    if (output(i, 0, 0) > 0)
                    {
                        activ_grad(i, 0, 0) = 1;
                    }
                    else
                    {
                        activ_grad(i, 0, 0) = 0.01 * exp(output(i, 0, 0));
                        output(i, 0, 0)     = 0.01 * (exp(output(i, 0, 0)) - 1); 
                    }
                }
            }
        }
    }
}

// прямое прохождение через слой 
Tensor ActivationLayer::Forward(const Tensor &X) 
{   
    Tensor output(input_size);
    output = X;
    Activate(output);
    return output;
}

// обратное проождение через слой 
Tensor ActivationLayer::Backward(const Tensor &grad)
{
    Tensor X_grad(input_size);
    for (int i = 0; i < input_size.height; i++) {
        for (int j = 0; j < input_size.width; j++) {
            for (int k = 0; k < input_size.depth; k++)
            {
                X_grad(k, i, j) = activ_grad(k, i, j) * grad(k, i, j);
            }
        }
    }
    return X_grad;
}
