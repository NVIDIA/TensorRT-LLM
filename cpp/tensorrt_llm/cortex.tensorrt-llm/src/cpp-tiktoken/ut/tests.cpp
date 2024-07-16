#include "encoding.h"
#include "emdedded_resource_reader.h"

#include "gtest/gtest.h"

#include <fstream>
#include <iostream>
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

TEST(TestGetEncoding, TestDefaultEncod)
{
    auto encoder = GptEncoding::get_encoding(LanguageModel::CL100K_BASE);
    std::vector<int> tokens = encoder->encode("This is a test sentence.");
    ASSERT_EQ(tokens.size(), 6);
    ASSERT_EQ(tokens[0], 2028);
    ASSERT_EQ(tokens[1], 374);
    ASSERT_EQ(tokens[2], 264);
    ASSERT_EQ(tokens[3], 1296);
    ASSERT_EQ(tokens[4], 11914);
    ASSERT_EQ(tokens[5], 13);
}

TEST(TestGetEncoding, TestLlama3Encod)
{
    auto encoder = GptEncoding::get_encoding(LanguageModel::CL100K_BASE);
    std::vector<int> tokens = encoder->encode("This is a test sentence.");
    ASSERT_EQ(tokens.size(), 6);
    ASSERT_EQ(tokens[0], 2028);
    ASSERT_EQ(tokens[1], 374);
    ASSERT_EQ(tokens[2], 264);
    ASSERT_EQ(tokens[3], 1296);
    ASSERT_EQ(tokens[4], 11914);
    ASSERT_EQ(tokens[5], 13);

    std::vector<int> role_user = encoder->encode("user");
    std::vector<int> role_system = encoder->encode("system");
    std::vector<int> paragraph = encoder->encode("\n\n");
    ASSERT_EQ(role_user[0], 882);
    ASSERT_EQ(role_system[0], 9125);
    ASSERT_EQ(paragraph[0], 271);


}

TEST(TestGetEncoding, TestEncode_O200K_BASE)
{
    auto encoder = GptEncoding::get_encoding(LanguageModel::O200K_BASE);
    std::vector<int> tokens = encoder->encode("hello world");
    ASSERT_EQ(tokens.size(), 2);
    ASSERT_EQ(tokens[0], 24912);
    ASSERT_EQ(tokens[1], 2375);
}

// Test cases below are inspired by meta-llama3 https://github.com/meta-llama/llama3/blob/main/llama/test_tokenizer.py

TEST(TestGetEncoding, TestCustomResourceReader)
{
    TFilePathResourceReader reader("../tokenizers/tokenizer.model");
    auto encoder = GptEncoding::get_encoding_llama3(LanguageModel::CL100K_BASE, &reader);
    std::vector<int> tokens = encoder->encode("This is a test sentence.");
    for(int i = 0;i<tokens.size();i++){
        std::cout<< encoder->decode({tokens[i]})<<" ";
    }
    std::cout<<"\n";
    ASSERT_EQ(tokens.size(), 6);
    ASSERT_EQ(tokens[0], 2028);
    ASSERT_EQ(tokens[1], 374);
    ASSERT_EQ(tokens[2], 264);
    ASSERT_EQ(tokens[3], 1296);
    ASSERT_EQ(tokens[4], 11914);
    ASSERT_EQ(tokens[5], 13);

    std::vector<int> role_user = encoder->encode("user");
    std::vector<int> role_system = encoder->encode("system");
    std::vector<int> paragraph = encoder->encode("\n\n");

    std::string decode_str = encoder->decode({128000, 2028, 374, 264, 1296, 11914, 13, 128001});

    ASSERT_EQ(role_user[0], 882);
    ASSERT_EQ(role_system[0], 9125);
    ASSERT_EQ(paragraph[0], 271);
    ASSERT_EQ(decode_str, "<|begin_of_text|>This is a test sentence.<|end_of_text|>");
}
