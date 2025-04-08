#ifndef MATRIX
#define MATRIX

#include <stddef.h> 
#include <vector>
#include <iostream>

struct TensorSize {
    size_t depth; 
    size_t height; 
    size_t width; 

    TensorSize(size_t _depth,  size_t _height, size_t _width) : depth(_depth), height(_height), width(_width) {};
    TensorSize(size_t _height, size_t _width) : TensorSize(0, _height, width) {};
    TensorSize() : TensorSize(1, 3, 3) {};

};

class Tensor {
    TensorSize size; 
    std::vector<double> values; 

    size_t dw; 

    void Init(size_t depth, size_t height, size_t width);

public:
    Tensor(size_t depth, size_t height, size_t width); 
    Tensor(size_t height, size_t width) : Tensor(0, height, width) {};
    Tensor(const TensorSize &size); 

    double& operator()(size_t d, size_t i, size_t j); 
    double operator()(size_t d, size_t i, size_t j) const;
    Tensor operator=(const Tensor &_tensor);

    TensorSize GetSize() const; 

    friend std::ostream& operator<<(std::ostream& os, const Tensor &tensor); 
};

#endif