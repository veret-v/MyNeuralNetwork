#ifndef DATA_FRAME
#define DATA_FRAME

#include <string>
#include <vector>
#include <stddef.h> 
#include <iostream>
#include <fstream>
#include <regex>

#include "../math/tensor.h"


enum class CSVState {
    UnquotedField,
    QuotedField,
    QuotedQuote
};

class TextCorpus
{
private:
    size_t texts_num;
    size_t embedding_size;
    size_t max_text_len;

    std::vector<Tensor>      coded_texts;
    std::vector<std::string> texts;
    std::vector<size_t>      target;

    std::vector<std::string> readCSVRow(const std::string &row);
    std::vector<std::string> Tokenizer(std::string text);
    
public:
    TextCorpus(std::string file_name);
    TextCorpus(std::string texts_file, std::string target_file);

    void OneHotEncoding();
    void TfIdf();
    void Word2Vec();

    size_t GetMaxTextLen();
    size_t GetEmbeddingSize();

};


#endif