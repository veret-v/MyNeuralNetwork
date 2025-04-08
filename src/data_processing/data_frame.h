#ifndef DATA_FRAME
#define DATA_FRAME

#include <string>
#include <vector>
#include <stddef.h> 
#include <iostream>
#include <fstream>
#include <regex>


enum class CSVState {
    UnquotedField,
    QuotedField,
    QuotedQuote
};

class DataFrame
{
private:
    size_t texts_num;
    std::vector<std::vector<std::string>> data;

    std::vector<std::string> readCSVRow(const std::string &row);
    
public:
    DataFrame(std::string file_name);

    void OneHotEncoding(size_t column);
    void TfIdf(size_t column);
    void Word2Vec(size_t column);

};


#endif