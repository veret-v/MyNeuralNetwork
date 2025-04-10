#include "full_con_layer.h"

// инициализация слоя
FullyConnectedLayer::FullyConnectedLayer(
        TensorSize _size,
        int outputs, 
        const std::string& activation_type,
        bool soft_max
) : filter(1, outputs, _size.height * _size.width * _size.depth), 
    filter_grad(1, outputs, _size.height * _size.width * _size.depth), 
    activ_grad(outputs, 1, 1),
    softmax_grad(outputs, 1, 1),
    distribution(0.0, sqrt(2.0 / (_size.width * _size.height * _size.depth)))
{
    input_size.width  = _size.width;
    input_size.height = _size.height;
    input_size.depth  = _size.depth;

    output_size.height = 1;
    output_size.width  = 1;
    output_size.depth  = outputs; 

    inputs                  = _size.height * _size.width * _size.depth;
    this -> outputs         = outputs;
    this -> activation_type = GetActivationType(activation_type);
    this -> soft_max        = soft_max;

    soft_max_sum = 0;

    offset      = std::vector<double>(outputs); 
    offset_grad = std::vector<double>(outputs); 

    InitWeights(); 
}

// инициализация весовых коэффициентов
void FullyConnectedLayer::InitWeights() {
        for (int i = 0; i < outputs; i++)
        {
            for (int j = 0; j < inputs; j++)
                filter(0, i, j) = distribution(generator);
            offset[i] = 0.01; 
        }
}

// сопоставление строки и функции активации
FullyConnectedLayer::ActivationType FullyConnectedLayer::GetActivationType(const std::string& activation_type) const
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
void FullyConnectedLayer::Activate(Tensor &output)
{   
    if (activation_type == ActivationType::None)
    {
        for (size_t i = 0; i < outputs; i++)
        {   
            activ_grad(i, 0, 0) = 1;
        }
    } 
    else if (activation_type == ActivationType::Sigmoid)
    {
        for (size_t i = 0; i < outputs; i++)
        {   
            output(i, 0, 0)     = 1 / (exp(-output(i, 0, 0)) + 1);
            activ_grad(i, 0, 0) = output(i, 0, 0) * (1 - output(i, 0, 0));
        }
    }
    else if (activation_type == ActivationType::Tanh)
    {
        for (size_t i = 0; i < outputs; i++)
        {   
            output(i, 0, 0)     = (exp(output(i, 0, 0)) - exp(-output(i, 0, 0)))/ (exp(-output(i, 0, 0)) + exp(output(i, 0, 0)));
            activ_grad(i, 0, 0) = 1 - pow(output(i, 0, 0), 2);
        }
    }
    else if (activation_type == ActivationType::ReLU)
    {
        for (size_t i = 0; i < outputs; i++)
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
    else if (activation_type == ActivationType::LeakyReLU)
    {
        for (size_t i = 0; i < outputs; i++)
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
    else if (activation_type == ActivationType::ELU)
    {
        for (size_t i = 0; i < outputs; i++)
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

// прямое прохождение через слой 
Tensor FullyConnectedLayer::Forward(const Tensor &X) 
{   
    Tensor output(output_size);
    for (size_t i = 0; i < outputs; i++)
    {
        output(i, 0, 0) = offset[i];
        for (size_t j = 0; j < inputs; j++)
            output(i, 0, 0) += filter(0, i, j) * X(j, 0 ,0);
    }
    Activate(output);
    if (soft_max)
        SoftMax(output);
    return output;
}

// обратное проождение через слой 
Tensor FullyConnectedLayer::Backward(const Tensor &grad, const Tensor &X)
{
    Tensor soft_max_grad(output_size);
    Tensor X_grad(input_size);

    if (soft_max)
    {
        for (size_t i = 0; i < outputs; i++) {
            for (size_t j = 0; j < outputs; j++) {
                if (i == j)
                {
                    soft_max_grad(i, 0, 0) += grad(j, 0, 0) * exp(X(j, 0, 0)) * (soft_max_sum - exp(X(j, 0, 0))) / (soft_max_sum * soft_max_sum);
                } else
                {
                    soft_max_grad(i, 0, 0) += grad(j, 0, 0) * (-exp(X(j, 0, 0)) * exp(X(i, 0, 0))) / (soft_max_sum * soft_max_sum);
                }       
           }
       }
    } else 
    {
        soft_max_grad = grad;
    }
    
    for (size_t i = 0; i < outputs; i++) {
        activ_grad(i, 0, 0) *= soft_max_grad(i, 0, 0);
    }
    
    for (size_t i = 0; i < inputs; i++) {
       for (size_t j = 0; j < outputs; j++)
            X_grad(i, 0, 0) +=  filter(0, j, i) * activ_grad(j, 0, 0);
    }

    for (size_t i = 0; i < outputs; i++) {
       for (size_t j = 0; j < inputs; j++)
            filter_grad(0, i, j) = activ_grad(j, 0, 0) * X(j, 0, 0);
        offset_grad[i] = activ_grad(i, 0, 0);
    }

    return X_grad;
}

// обновление весов 
void FullyConnectedLayer::UpdateWeights(double learning_rate) 
{
    for (int i = 0; i < outputs; i++) 
    {
        for (int j = 0; j < inputs; j++)
            filter(0, i, j) -= learning_rate * filter_grad(0, i, j);
        offset[i] -= learning_rate * offset_grad[i]; 
    }
}

// получение вероятностей 
void FullyConnectedLayer::SoftMax(Tensor &output)
{
    for (size_t i = 0; i < outputs; i++)
        soft_max_sum += exp(output(i, 0, 0));

    for (size_t i = 0; i < outputs; i++) {
        output(i, 0, 0) = exp(output(i, 0, 0)) / soft_max_sum;
    }
}

// получение ядер свертки
Tensor FullyConnectedLayer::GetFilter()
{
    return filter;
}

// получение коэффициентов смещения
std::vector<double> FullyConnectedLayer::GetOffsets()
{
    return offset;
}

// получение градиентов ядер свертки
Tensor FullyConnectedLayer::GetFilterGrad()
{
    return filter_grad;
}

// получение градиентов коэффициентов смещения
std::vector<double> FullyConnectedLayer::GetOffsetsGrad()
{
    return offset_grad;
}

// установка 1-го ядра свертки
void FullyConnectedLayer::SetFilter(Tensor _filter)
{
    filter = _filter;
}

// установка 1-го коэффициентов смещения
void FullyConnectedLayer::SetOffset(std::vector<double> _offset)
{
    for (size_t i = 0; i < offset.size(); i++)
    {
        offset[i] = _offset[i];
    }
}