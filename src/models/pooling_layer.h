#ifndef POOLING_LAYER
#define POOLING_LAYER

#include <random>
#include <stddef.h> 
#include <vector>
#include <map>
#include <iostream> 

#include "../math/tensor.h"

class PoolingLayer
{
private:
    enum class PoolingType {
        MaxPooling, // максимальное значение precision field
        AveragePooling, // среднее значение precision field
        SumPooling, // свертка
    };

    TensorSize input_size; 
    TensorSize output_size; 
    
    size_t scale;
    PoolingType pooling_type; 

    Tensor mask;

    PoolingType GetPoolingType(const std::string& pooling_type) const;

public:
    PoolingLayer(
        TensorSize input_size,
        size_t scale,
        const std::string &pooling_type="max_pooling"
    );
        
    Tensor Forward(const Tensor &X);
    Tensor Backward(const Tensor &grad);

};


#endif