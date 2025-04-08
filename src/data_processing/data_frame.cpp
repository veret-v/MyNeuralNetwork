#include "data_frame.h"

// конструктор класса DataFrame
DataFrame::DataFrame(std::string corpus_name)
{
    std::ifstream in;
    in.open(corpus_name);
    std::string row;
    while (!in.eof()) {
        std::getline(in, row);
        if (in.bad() || in.fail()) {
            break;
        }
        auto fields = readCSVRow(row);
        data.push_back(fields);
        texts_num += 1;
    }
    in.close();
}

// парсер строки csv файла
std::vector<std::string> DataFrame::readCSVRow(const std::string &row)
{
    CSVState state = CSVState::UnquotedField;
    std::vector<std::string> fields {""};
    size_t i = 0; 
    for (char c : row) {
        switch (state) {
            case CSVState::UnquotedField:
                switch (c) {
                    case ',': 
                        fields.push_back(""); i++;
                        break;
                    case '"': 
                        state = CSVState::QuotedField;
                        break;
                    default:  
                        fields[i].push_back(c);
                        break; 
                }
                break;
            case CSVState::QuotedField:
                switch (c) {
                    case '"': 
                        state = CSVState::QuotedQuote;
                        break;
                    default:  
                        fields[i].push_back(c);
                            break;
                }
                break;
            case CSVState::QuotedQuote:
                switch (c) {
                    case ',': 
                        fields.push_back(""); i++;
                        state = CSVState::UnquotedField;
                        break;
                    case '"': 
                        fields[i].push_back('"');
                        state = CSVState::QuotedField;
                        break;
                    default:  
                        state = CSVState::UnquotedField;
                        break; 
                    }
                break;
        }
    }
    return fields;
}

// one hot кодировка
void DataFrame::OneHotEncoding(size_t column)
{   
    std::vector<std::string> vocabulary;
    for (auto row = 0; row < texts_num; row++) 
    {
        std::string sentence = data[row][column];
        std::regex re("[\\|,:]");
        std::sregex_token_iterator first{sentence.begin(), sentence.end(), re, -1}, last;
        std::vector<std::string> words{first, last};
        for (auto word = words.begin(); word != words.end(); ++word)
        {
            for (size_t voc_index = 0; voc_index < vocabulary.size(); voc_index++)
            {
                if (*word != vocabulary[voc_index])
                {
                    vocabulary.push_back(*word);
                    break;
                }
            }
        }
    }
    for (auto row = 0; row < texts_num; row++) 
    {
        std::string sentence = data[row][column];
        std::regex re("[\\|,:]");
        std::sregex_token_iterator first{sentence.begin(), sentence.end(), re, -1}, last;
        std::vector<std::string> words{first, last};
        for (size_t voc_index = 0; voc_index < vocabulary.size(); voc_index++)
        {
            bool word_detected = 0;
            for (auto word = words.begin(); word != words.end(); ++word) 
            {
                if (*word == vocabulary[voc_index])
                {
                    data[row].push_back("1");
                    word_detected = 1;
                    break;
                }
            }
            if (!word_detected)
                data[row].push_back("0");
        }
    }
}