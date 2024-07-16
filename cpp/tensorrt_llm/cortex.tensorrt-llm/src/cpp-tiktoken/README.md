# Cpp-Tiktoken

This is a C++ implementation of a tiktoken tokenizer library for C++. It was heavily inspired
by https://github.com/dmitry-brazhenko/SharpToken

To use, first somewhere have a lines in your project that reads something like:

    #include "tiktoken/enconding.h"

    ....

    auto encoder = GptEncoding::get_encoding(<model name>);
The value returned from this function is an `std::shared_ptr` and you will not have to manage its memory.

Supported language models that you can pass as a parameter to this function are:

    LanguageModel::O200K_BASE
    LanguageModel::CL100K_BASE 
    LanguageModel::R50K_BASE
    LanguageModel::P50K_BASE
    LanguageModel::P50K_EDIT
After obtaining an encoder, you can then call

    auto tokens = encoder->encode(string_to_encode);
This returns a vector of the tokens for that language model.

You can decode a vector of tokens back into its original string with

    auto string_value = encoder->decode(tokens)
If you like this project, and find it useful, you are invited to make a donation of whatever amount you believe
is appropriate via paypal to markt AT nerdflat.com.  There is absolutely no obligation to donate.

Install pcre2-devel
