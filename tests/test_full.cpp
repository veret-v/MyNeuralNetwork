#include <iostream>
#include <stddef.h>

#include "../src/models/full_con_layer.h"
#include "../src/math/tensor.h"

void test_full_1 () {
  Tensor input(3, 1, 1);
  Tensor grad(2, 1, 1);

  size_t p = 0;

  for (size_t i = 0; i < input.GetSize().depth; i++) {
      input(i, 0, 0) = p;
      p++;
  }
  
  FullyConnectedLayer Layer1(TensorSize(3, 1, 1), 2);
  Tensor kernel(1, 2, 3);

  kernel((size_t)0, (size_t)0, (size_t)0) = 1;
  kernel((size_t)0, (size_t)0, (size_t)1) = 1;
  kernel((size_t)0, (size_t)0, (size_t)2) = 1;
  kernel((size_t)0, (size_t)1, (size_t)0) = 1;
  kernel((size_t)0, (size_t)1, (size_t)1) = 1;
  kernel((size_t)0, (size_t)1, (size_t)2) = 1;

  grad((size_t)0, (size_t)0, (size_t)0) = 1;
  grad((size_t)1, (size_t)0, (size_t)0) = 1;
  
  Layer1.SetFilter(kernel);

  std::cout << "input" << std::endl;
  std::cout << input << std::endl;

  std::cout << "kernel" << std::endl;
  std::cout << kernel << std::endl;

  std::cout << "forward" << std::endl;
  std::cout << Layer1.Forward(input) << std::endl;

  std::cout << "backward" << std::endl;
  std::cout << Layer1.Backward(grad, input) << std::endl;

  std::cout << "w_grad" << std::endl;
  std::cout << Layer1.GetFilterGrad() << std::endl;

  std::cout << "b_grad" << std::endl;
  std::cout << Layer1.GetOffsetsGrad()[0] << std::endl;
}

int main()
{
  test_full_1();
}