#pragma once
#include "byte_pair_encoding.h"
#include "modelparams.h"
#include "encoding.h"
#include "emdedded_resource_reader.h"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <fstream>
class TFilePathResourceReader : public IResourceReader {
public:
    TFilePathResourceReader(const std::string& path) 
        : path_(path)
    {
    }

    std::vector<std::string> readLines() override {
        std::ifstream file(path_);
        if (!file.is_open()) {
            throw std::runtime_error("Embedded resource '" + path_ + "' not found.");
        }

        std::string line;
        std::vector<std::string> lines;
        while (std::getline(file, line)) {
            lines.push_back(line);
        }

        return lines;
    }
private:
    std::string path_;
};

class Llama3Tokenizer {
    int bos_id, eos_id, pad_id, n_words;
    std::unordered_map<std::string, int> special_tokens;
    GptEncoding model;
    std::unordered_set<int> stop_tokens;

public:
    Llama3Tokenizer(const std::string model_path);
    std::vector<int> encode(const std::string &line_to_encode, bool bos, bool eos, const std::unordered_set<std::string> &allowed_special = {},
        const std::unordered_set<std::string> &disallowed_special = { "all" });
    std::string decode(const std::vector<int> &input_tokens_to_decode);

};