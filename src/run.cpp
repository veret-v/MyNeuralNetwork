#include "models/activation_layer.h"
#include "models/conv_layer.h"
#include "models/pooling_layer.h"
#include "models/full_con_layer.h"

#include "data_processing/texts_corpus.h"

#include <iostream>
#include <tuple>

typedef std::tuple<Tensor, int, bool> info;

// TextCorpus corpus("../data/IMDB_dataset-1.csv");

// ConvolutionLayer    conv();
// ActivationLayer     activ();
// PoolingLayer        pool();
// FullyConnectedLayer full();

// double CrossEntropyLoss(const Tensor &out)
// {
//   double loss = 0;
//   for (size_t i = 0; i < out.GetSize().depth; i++)
//   {
//     loss += i * log(out(i, 0, 0));
//   }
//   return loss;
// }

// Tensor CrossEntropyGradient(const Tensor &out)
// {
//   Tensor grad(out.GetSize());
//   for (size_t i = 0; i < out.GetSize().depth; i++)
//   {
//     grad(i, 0, 0) = - 1 / out(i, 0, 0);
//   }
//   return grad;
// }

// info ForwardProp(const Tensor &sentence, size_t label)
// { 
  

// }


int main()
{
  TextCorpus corpus("../data/texts_imdb.txt", "../data/target_imdb.txt");
  corpus.OneHotEncoding();
  std::cout << corpus.GetEmbeddingSize() << std::endl;
  std::cout << corpus.GetMaxTextLen() << std::endl;

}