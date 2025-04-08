#include "full_con_layer.h"

// инициализация слоя
FullyConnectedLayer::FullyConnectedLayer(
        TensorSize _size,
        int outputs, 
        const std::string& activation_type = "none"
) : filter(1, outputs, _size.height * _size.width * _size.depth), 
    filter_grad(1, outputs, _size.height * _size.width * _size.depth), 
    activ_grad(outputs, 1, 1),
    distribution(0.0, sqrt(2.0 / (_size.width * _size.height * _size.depth)))
{
    input_size.width  = _size.width;
    input_size.height = _size.height;
    input_size.depth  = _size.depth;

    output_size.height = 1;
    output_size.width  = 1;
    output_size.depth  = outputs; 

    inputs                = _size.height * _size.width * _size.depth;
    this->outputs         = outputs;
    this->activation_type = GetActivationType(activation_type);

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
    if (activation_type == ActivationType::Sigmoid)
    {
        for (size_t i = 0; i < outputs; i++)
        {   
            output(i, 0, 0)     = 1 / (exp(-output(i, 0, 0)) + 1);
            activ_grad(i, 0, 0) = output(i, 0, 0) * (1 - output(i, 0, 0));
        }
    }
    if (activation_type == ActivationType::Tanh)
    {
        for (size_t i = 0; i < outputs; i++)
        {   
            output(i, 0, 0)     = (exp(output(i, 0, 0)) - exp(-output(i, 0, 0)))/ (exp(-output(i, 0, 0)) + exp(output(i, 0, 0)));
            activ_grad(i, 0, 0) = 1 - pow(output(i, 0, 0), 2);
        }
    }
    if (activation_type == ActivationType::ReLU)
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
    if (activation_type == ActivationType::LeakyReLU)
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
    if (activation_type == ActivationType::ELU)
    {
        for (size_t i = 0; i < outputs; i++)
        {   
            output(i, 0, 0) = 1 / (exp(-output(i, 0, 0)) + 1);
            activ_grad(i, 0, 0) = exp(-output(i, 0, 0)) * pow(output(i, 0, 0), 2);
        }
    }
    

}

// прямое прохождение через слой 
Tensor FullyConnectedLayer::Forward(const Tensor &X) 
{
    Tensor output(output_size);

    for (size_t curr_depth = 0; curr_depth < output_size.depth; curr_depth++)
    {
        for (size_t x = 0; x < output_size.height; x++)
        {
            for (size_t y = 0; y < output_size.width; y++)
            {
                double sum = offsets[curr_depth];

                for (size_t i = 0; i < filter_size.height; i++)
                {
                    for (size_t j = 0; j < filter_size.width; j++)
                    {
                        size_t global_i = x * step + i - padding;
                        size_t global_j = y * step + j - padding;

                        if (global_i < 0 || global_i >= input_size.height || 
                            global_j < 0 || global_j >= input_size.width) 
                            continue;
                    
                        for (size_t c = 0; c < filter_size.depth; c++)
                            sum += X(c, global_i, global_j) * kernels[curr_depth](c, i, j); 
                    }
                }
                output(curr_depth, x, y) = sum;
            } 
        } 
    }
    return output;
}

// обратное проождение через слой 
Tensor FullyConnectedLayer::Backward(const Tensor &grad, const Tensor &X)
{
    Tensor X_grad(input_size);   

    for (size_t y = 0; y < input_size.height; y++)
    {
        for (size_t x = 0; x < input_size.width; x++)
        {
            for (size_t depth = 0; depth < input_size.depth; depth++)
            {
                double sum = 0;
                for (size_t i = 0; i < filter_size.height; i++)
                {
                    for (size_t j = 0; j < filter_size.width; j++)
                    {
                        int global_i = y * step - i - padding;
                        int global_j = x * step - j - padding;

                        if (global_i < 0 || global_i >= output_size.height || 
                            global_j < 0 || global_j >= output_size.width) 
                            continue;
                    
                        for (size_t filt_num = 0; filt_num < filters_num; filt_num++)
                            sum += kernels[filt_num](depth, i, j) * grad(filt_num, global_i, global_j);

                    }
                }

                X_grad(depth, x, y) = sum;
            } 
        } 
    }

    for (size_t filt_num = 0; filt_num < filters_num; filt_num++)
    {  

        for (size_t i = 0; i < filter_size.height; i++)
        {
            for (size_t j = 0; j < filter_size.width; j++)
            {   
                for(size_t depth = 0; depth < filter_size.depth; depth++)
                {
                    double sum_ker = 0;
                    double sum_off = 0;
                
                    for (size_t y = 0; y < output_size.height; y++)
                    {
                        for (size_t x = 0; x < output_size.width; x++)
                        {
                            size_t global_i = y * step + i - padding;
                            size_t global_j = x * step + j - padding;

                            if (global_i < 0 || global_i >= input_size.height || 
                                global_j < 0 || global_j >= input_size.width) 
                                continue;
                    
                            
                            sum_ker += X(depth, global_i, global_j) * grad(filt_num, y, x);
                            sum_off += grad(filt_num, y, x);                
                        }
                    }
                    kernels_grad[filt_num](depth, i, j) = sum_ker;
                    offsets_grad[filt_num] = sum_off;
                }

            } 
        } 
    }
    
    return X_grad;
}

