#include "conv_layer.h"

// инициализация слоя
ConvolutionLayer::ConvolutionLayer(
    TensorSize _size,
    TensorSize _kernel_size,
    size_t _filters_num,
    size_t _step,
    size_t _padding
) : distribution(0.0, sqrt(2.0 / (_kernel_size.width*_kernel_size.height*_kernel_size.depth)))
{
    filters_num    = _filters_num; 
    step           = _step; 
    padding        = _padding;

    kernel_size.width  = _kernel_size.width;
    kernel_size.height = _kernel_size.height;
    kernel_size.depth  = _kernel_size.depth;

    input_size.width  = _size.width;
    input_size.height = _size.height;
    input_size.depth  = _size.depth;

    output_size.height = (_size.height + padding * 2 - kernel_size.height) / step + 1;
    output_size.width  = (_size.width + padding * 2 - kernel_size.width) / step + 1;
    output_size.depth  = _filters_num; 

    if (_kernel_size.depth > input_size.depth)
    {
        std::cerr << "Incorrect kernel size" << std::endl;
    }

    kernels      = std::vector<Tensor>(filters_num, Tensor(kernel_size));
    kernels_grad = std::vector<Tensor>(filters_num, Tensor(kernel_size));
        
    offsets      = std::vector<double>(filters_num, 0);
    offsets_grad = std::vector<double>(filters_num, 0);

    InitWeights(); 
}

// инициализация весовых коэффициентов
void ConvolutionLayer::InitWeights() {
    for (int index = 0; index < filters_num; index++) {
        for (int i = 0; i < kernel_size.height; i++)
            for (int j = 0; j < kernel_size.width; j++)
                for (int k = 0; k < kernel_size.depth; k++)
                    kernels[index](k, i, j) = distribution(generator);

        offsets[index] = 0.01; 
    }
}

// прямое прохождение через слой 
Tensor ConvolutionLayer::Forward(const Tensor &X) 
{
    Tensor output(output_size);

    for (size_t curr_depth = 0; curr_depth < output_size.depth; curr_depth++)
    {
        for (size_t x = 0; x < output_size.height; x++)
        {
            for (size_t y = 0; y < output_size.width; y++)
            {
                double sum = offsets[curr_depth];

                for (size_t i = 0; i < kernel_size.height; i++)
                {
                    for (size_t j = 0; j < kernel_size.width; j++)
                    {
                        size_t global_i = x * step + i - padding;
                        size_t global_j = y * step + j - padding;

                        if (global_i < 0 || global_i >= input_size.height || 
                            global_j < 0 || global_j >= input_size.width) 
                            continue;
                    
                        for (size_t c = 0; c < kernel_size.depth; c++)
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
Tensor ConvolutionLayer::Backward(const Tensor &grad, const Tensor &X)
{
    Tensor X_grad(input_size);   

    for (size_t y = 0; y < input_size.height; y++)
    {
        for (size_t x = 0; x < input_size.width; x++)
        {
            for (size_t depth = 0; depth < input_size.depth; depth++)
            {
                double sum = 0;
                for (size_t i = 0; i < kernel_size.height; i++)
                {
                    for (size_t j = 0; j < kernel_size.width; j++)
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

        for (size_t i = 0; i < kernel_size.height; i++)
        {
            for (size_t j = 0; j < kernel_size.width; j++)
            {   
                for(size_t depth = 0; depth < kernel_size.depth; depth++)
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

// получение ядер свертки
std::vector<Tensor> ConvolutionLayer::GetKernels()
{
    return kernels;
}

// получение коэффициентов смещения
std::vector<double> ConvolutionLayer::GetOffsets()
{
    return offsets;
}

// получение градиентов ядер свертки
std::vector<Tensor> ConvolutionLayer::GetKernelsGrad()
{
    return kernels_grad;
}

// получение градиентов коэффициентов смещения
std::vector<double> ConvolutionLayer::GetOffsetsGrad()
{
    return offsets_grad;
}

// установка 1-го ядра свертки
void ConvolutionLayer::SetKernels(Tensor _kernel)
{
    kernels[0] = _kernel;
}

// установка 1-го коэффициентов смещения
void ConvolutionLayer::SetOffset(double _offset)
{
    offsets[0] = _offset;
}