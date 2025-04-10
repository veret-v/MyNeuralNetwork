#include "pooling_layer.h"

// инициализация слоя
PoolingLayer::PoolingLayer(
        TensorSize input_size,
        size_t scale,
        const std::string &pooling_type="max_pooling"
) : mask(input_size)
{
    this -> input_size.width  = input_size.width;
    this -> input_size.height = input_size.height;
    this -> input_size.depth  = input_size.depth;

    output_size.width  = input_size.width / scale;
    output_size.height = input_size.width / scale;
    output_size.depth  = input_size.depth;


    this -> scale        = scale;
    this -> pooling_type = GetPoolingType(pooling_type);
}

// сопоставление строки и функции активации
PoolingLayer::PoolingType PoolingLayer::GetPoolingType(const std::string& pooling_type) const
{
    if (pooling_type == "max_pooling")
        return PoolingType::MaxPooling;

    if (pooling_type == "average_pooling")
        return PoolingType::AveragePooling;

    if (pooling_type == "sum_pooling")
        return PoolingType::SumPooling;

    throw std::runtime_error("Invalid pooling");
}

// прямое прохождение через слой 
Tensor PoolingLayer::Forward(const Tensor &X) 
{   
    Tensor output(output_size);

    for (int d = 0; d < input_size.depth; d++) {
        for (int i = 0; i < input_size.height; i += scale) {
            for (int j = 0; j < input_size.width; j += scale) {
                if (pooling_type == PoolingType::MaxPooling)
                {       
                    int imax = i; 
                    int jmax = j; 
                    double max = X(d, i, j); 

                    for (int y = i; y < i + scale; y++) {
                        for (int x = j; x < j + scale; x++) {
                            double value = X(d, y, x); 
                            mask(d, y, x) = 0; 

                            if (value > max) {
                                max = value; 
                                imax = y; 
                                jmax = x; 
                            }
                        }
                    }

                    output(d, i / scale, j / scale) = max; 
                    mask(d, imax, jmax) = 1; 
                }
                else if (pooling_type == PoolingType::SumPooling)
                {       
                    double sum =0;
                    for (int y = i; y < i + scale; y++) {
                        for (int x = j; x < j + scale; x++) {
                            sum += X(d, y, x);
                        }
                    }

                    output(d, i / scale, j / scale) = sum;
                }
                else if (pooling_type == PoolingType::AveragePooling)
                {       
                    double sum =0;
                    for (int y = i; y < i + scale; y++) {
                        for (int x = j; x < j + scale; x++) {
                            sum += X(d, y, x);
                        }
                    }

                    output(d, i / scale, j / scale) = sum / (scale * scale);
                }
            }
        }
    } 
    return output;
}

// обратное проождение через слой 
Tensor PoolingLayer::Backward(const Tensor &grad)
{
    Tensor X_grad(output_size);

    if (pooling_type == PoolingType::MaxPooling)
    {       
        for (int d = 0; d < input_size.depth; d++) 
            for (int i = 0; i < input_size.height; i += scale) 
                for (int j = 0; j < input_size.width; j += scale) 
                    X_grad(d, i, j) = grad(d, i / scale, j / scale) * mask(d, i, j); 
    }
    else if (pooling_type == PoolingType::AveragePooling)
    {    
        for (int d = 0; d < input_size.depth; d++) {
            for (int i = 0; i < input_size.height; i += scale) {
                for (int j = 0; j < input_size.width; j += scale) {   
                    for (int y = i; y < i + scale; y++) {
                        for (int x = j; x < j + scale; x++) 
                            X_grad(d, y, x) = grad(d, i, j);
                    }
                }
            }
        }
    }
    else if (pooling_type == PoolingType::SumPooling)
    {       
        for (int d = 0; d < input_size.depth; d++) {
            for (int i = 0; i < input_size.height; i += scale) {
                for (int j = 0; j < input_size.width; j += scale) {   
                    for (int y = i; y < i + scale; y++) {
                        for (int x = j; x < j + scale; x++) 
                            X_grad(d, y, x) = grad(d, i, j) / (scale * scale);
                    }
                }
            }
        }
    }   

    return X_grad;
}
