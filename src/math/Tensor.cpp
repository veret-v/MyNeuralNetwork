#include "tensor.h"

// инициализация по размерам
void Tensor::Init(size_t depth, size_t height, size_t width) 
{
    size.width = width;
    size.height = height; 
    size.depth = depth; 

    dw = depth * width; 

    values = std::vector<double>(width * height * depth, 0); 
}

// создание из размеров
Tensor::Tensor(size_t depth, size_t height, size_t width) 
{
    Init(depth, height, width);  
}

// создание из размера
Tensor::Tensor(const TensorSize &size) 
{
    Init(size.depth, size.height, size.width);
}

// индексация
double& Tensor::operator()(size_t d, size_t i, size_t j) 
{
    return values[i * dw + j * size.depth + d];
}

// индексация
double Tensor::operator()(size_t d, size_t i, size_t j) const 
{
    return values[i * dw + j * size.depth + d];
}

// получение размера
TensorSize Tensor::GetSize() const 
{
    return size;
}

// вывод тензора
std::ostream& operator<<(std::ostream& os, const Tensor &tensor) 
{
    for (size_t d = 0; d < tensor.size.depth; d++) {
        for (size_t i = 0; i < tensor.size.height; i++) {
            for (size_t j = 0; j < tensor.size.width; j++) {
                os << tensor(d, i, j) << " ";
            }
            os << std::endl;
        }
        os << std::endl;
    }
    return os;
}

// оператор приравнивания тензора
Tensor Tensor::operator=(const Tensor &_tensor)
{
    size = _tensor.GetSize();
    for (size_t d = 0; d < size.depth; d++) {
        for (size_t i = 0; i < size.height; i++) {
            for (size_t j = 0; j < size.width; j++)
                this -> operator()(d, i, j) = _tensor(d, i, j);
        }
    }
    return *this;
}