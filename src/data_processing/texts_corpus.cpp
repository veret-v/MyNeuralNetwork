#include "texts_corpus.h"

// конструктор класса TextCorpus
TextCorpus::TextCorpus(std::string corpus_name)
{
    max_text_len = 0;

    std::ifstream in;
    in.open(corpus_name);
    std::string row;

    while (!in.eof()) {
        std::getline(in, row);

        if (in.bad() || in.fail()) {
            break;
        }

        auto fields = readCSVRow(row);
        texts.push_back(fields[0]);
        target.push_back(stoi(fields[1]));
        texts_num += 1;
    }
    in.close();
}

// конструктор класса TextCorpus из двух файлов
TextCorpus::TextCorpus(std::string texts_file, std::string target_file)
{
    max_text_len = 0;

    std::ifstream target_in;
    std::ifstream texts_in;

    texts_in.open(texts_file);
    target_in.open(target_file);

    std::string text;
    std::string label;

    while (!texts_in.eof()) {
        std::getline(texts_in, text);
        std::getline(target_in, label);

        if (texts_in.bad() || texts_in.fail() || 
            target_in.bad() || target_in.fail()) 
            break;

        texts.push_back(text);
        target.push_back(stoi(label));
        texts_num += 1;
    }

    texts_in.close();
    target_in.close();
}

// парсер строки csv файла
std::vector<std::string> TextCorpus::readCSVRow(const std::string &row)
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
void TextCorpus::OneHotEncoding()
{   
    std::vector<std::string> vocabulary;

    for (auto row = 0; row < texts_num; row++) {
        Tensor coded_text(embedding_size, max_text_len, 1);
        std::vector<std::string> words = Tokenizer(texts[row]);

        for (size_t word_id = 0; word_id < words.size(); word_id++) {
            if (row == 0 && word_id == 0)
                vocabulary.push_back(words[word_id]);
    
            for (size_t voc_index = 0; voc_index < vocabulary.size(); voc_index++) {
                if (words[word_id] == vocabulary[voc_index]) 
                    break;
                vocabulary.push_back(words[word_id]);
            }
        }

        if (words.size() > max_text_len)
            max_text_len = words.size();
        
    }

    embedding_size = vocabulary.size();

    for (auto row = 0; row < texts_num; row++) {
        Tensor coded_text(embedding_size, max_text_len, 1);
        std::vector<std::string> words = Tokenizer(texts[row]);

        for (size_t word_id = 0; word_id < words.size(); word_id++) {
            for (size_t voc_index = 0; voc_index < embedding_size; voc_index++) {
                if (words[word_id] == vocabulary[voc_index]) {
                    coded_text(voc_index, word_id, 0) = 1;
                } else
                {
                    coded_text(voc_index, word_id, 0) = 0;
                }
            }
        }
        coded_texts.push_back(coded_text);
    }
}

// токкенизация
std::vector<std::string> TextCorpus::Tokenizer(std::string text)
{
    const std::regex re(R"([\s|,]+)");
    std::sregex_token_iterator it{ text.begin(), text.end(), re, -1 };
    std::vector<std::string> tokenized{ it, {} };

    tokenized.erase(
        std::remove_if(
            tokenized.begin(), 
            tokenized.end(),
            [](std::string const& s) {
                           return s.size() == 0;
            }),
        tokenized.end()
    );

    return tokenized;
}


// получение максимальной длины текста
size_t TextCorpus::GetMaxTextLen()
{
    return max_text_len;
}

// получение максимальной длины текста
size_t TextCorpus::GetEmbeddingSize()
{
    return embedding_size;
}
