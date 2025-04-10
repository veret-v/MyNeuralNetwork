#include <iostream>
#include <stddef.h>

#include "../src/models/conv_layer.h"
#include "../src/math/tensor.h"

void test_conv_1 () {
  Tensor input(1, 3, 3);
  Tensor output(1, 2, 2);
  Tensor grad(1, 2, 2);

  size_t p = 0;

  for (size_t i = 0; i < input.GetSize().height; i++) {
      for (size_t j = 0; j < input.GetSize().width; j++)
          input(0, i, j) = p;
          p++;
  }
  
  ConvolutionLayer Layer1(TensorSize(1, 3, 3), TensorSize(1, 2, 2), 1, 1, 0);
  Tensor kernel(1, 2, 2);
  double offset = 0;

  kernel((size_t)0, (size_t)0, (size_t)0) = 1;
  kernel((size_t)0, (size_t)1, (size_t)0) = 1;
  kernel((size_t)0, (size_t)0, (size_t)1) = 1;
  kernel((size_t)0, (size_t)1, (size_t)1) = 1;

  grad((size_t)0, (size_t)0, (size_t)0) = 1;
  grad((size_t)0, (size_t)1, (size_t)0) = 1;
  grad((size_t)0, (size_t)0, (size_t)1) = 1;
  grad((size_t)0, (size_t)1, (size_t)1) = 1;

  Layer1.SetKernels(kernel);
  Layer1.SetOffset(offset);

  std::cout << "input" << std::endl;
  std::cout << input << std::endl;

  std::cout << "kernel" << std::endl;
  std::cout << kernel << std::endl;

  std::cout << "offset" << std::endl;
  std::cout << offset << std::endl;

  std::cout << "forward" << std::endl;
  std::cout << Layer1.Forward(input) << std::endl;

  std::cout << "backward" << std::endl;
  std::cout << Layer1.Backward(grad, input) << std::endl;

  std::cout << "w_grad" << std::endl;
  std::cout << Layer1.GetKernelsGrad()[0] << std::endl;

  std::cout << "b_grad" << std::endl;
  std::cout << Layer1.GetOffsetsGrad()[0] << std::endl;
}


void test_conv_2 () 
{
  Tensor input(2, 4, 1);
  Tensor output(1, 3, 1);
  Tensor grad(1, 3, 1);

  size_t p = 0;

  for (size_t i = 0; i < input.GetSize().height; i++) {
      for (size_t d = 0; d < input.GetSize().depth; d++)
      {
          input(d, i, 0) = p;
          p++;
      }
  }
  
  ConvolutionLayer Layer1(TensorSize(2, 4, 1), TensorSize(2, 2, 1), 1, 1, 0);
  Tensor kernel(2, 2, 1);
  double offset = 0;

  kernel((size_t)0, (size_t)0, (size_t)0) = 1;
  kernel((size_t)0, (size_t)1, (size_t)0) = 1;
  kernel((size_t)1, (size_t)0, (size_t)0) = 1;
  kernel((size_t)1, (size_t)1, (size_t)0) = 1;

  grad((size_t)0, (size_t)0, (size_t)0) = 1;
  grad((size_t)0, (size_t)1, (size_t)0) = 1;
  grad((size_t)0, (size_t)2, (size_t)0) = 1;

  Layer1.SetKernels(kernel);
  Layer1.SetOffset(offset);

  std::cout << "input" << std::endl;
  std::cout << input << std::endl;

  std::cout << "kernel" << std::endl;
  std::cout << kernel << std::endl;

  std::cout << "offset" << std::endl;
  std::cout << offset << std::endl;

  std::cout << "forward" << std::endl;
  std::cout << Layer1.Forward(input) << std::endl;

  std::cout << "backward" << std::endl;
  std::cout << Layer1.Backward(grad, input) << std::endl;

  std::cout << "w_grad" << std::endl;
  std::cout << Layer1.GetKernelsGrad()[0] << std::endl;

  std::cout << "b_grad" << std::endl;
  std::cout << Layer1.GetOffsetsGrad()[0] << std::endl;
}

int main()
{
  test_conv_1();
  test_conv_2();
}