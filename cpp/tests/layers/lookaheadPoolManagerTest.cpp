
#include <gtest/gtest.h>

#include "tensorrt_llm/layers/lookaheadDecodingUtils.h"
#include "tensorrt_llm/layers/lookaheadPoolManager.h"

namespace tensorrt_llm::tests::layers
{
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::layers;
using TensorPtr = runtime::ITensor::SharedPtr;

void printMap(char const* name, std::unordered_map<LookaheadPoolManager::Key, std::list<TensorPtr>> const& tokenMap)
{
    std::ostringstream buf;
    buf << name << std::endl;
    for (auto const& [key, value] : tokenMap)
    {
        buf << static_cast<char>(key) << ": ";
        for (auto const& tup : value)
        {
            buf << "(";
            for (auto const& token : BufferRange<LookaheadPoolManager::Key>(*tup))
            {
                buf << static_cast<char>(token) << ",";
            }
            buf << "),";
        }
        buf << std::endl;
    }
    TLLM_LOG_DEBUG(buf.str());
}

TensorPtr initTensor(
    std::shared_ptr<BufferManager>& mBufferManager, std::string str, std::optional<ITensor::Shape> shape = std::nullopt)
{
    std::vector<TokenIdType> data(str.begin(), str.end());
    auto shape1d = ITensor::makeShape({static_cast<SizeType32>(data.size())});
    if (shape)
    {
        TLLM_CHECK(ITensor::volume(shape1d) == ITensor::volume(shape.value()));
    }
    return ITensor::view(mBufferManager->copyFrom(data, MemoryType::kCPU), shape.value_or(shape1d));
}

bool isTensorEqString(TensorPtr const& a, std::string b)
{
    TLLM_CHECK(ITensor::volume(a->getShape()) == static_cast<SizeType32>(b.size()));
    auto ar = BufferRange<TokenIdType>(*a);
    return std::equal(ar.begin(), ar.end(), b.begin());
}

TEST(LookaheadPoolManagerTest, fillAndUpdate)
{
    std::shared_ptr<CudaStream> mStream = std::make_shared<CudaStream>();
    std::shared_ptr<BufferManager> mBufferManager = std::make_shared<BufferManager>(mStream);

    SizeType32 constexpr W{5};
    SizeType32 constexpr N{4};
    SizeType32 constexpr G{5};
    auto prompt = initTensor(mBufferManager, "hello world; hello world. live is life.");
    LookaheadPoolManager pm(G, mBufferManager);
    pm.fillWithPrompt(prompt, N);
    printMap("Token map after fill with prompt", pm.getMap());
    /***
     s: ( ,l,i,),
     v: (e, ,i,),
     i: (v,e, ,),(s, ,l,),(f,e,.,),
     d: (;, ,h,),(., ,l,),
     w: (o,r,l,),
      : (h,e,l,),(w,o,r,),(l,i,v,),(i,s, ,),(l,i,f,),
     .: ( ,l,i,),
     ;: ( ,h,e,),
     o: ( ,w,o,),(r,l,d,),
     l: (l,o, ,),(o, ,w,),(d,., ,),(i,v,e,),(i,f,e,),
     r: (l,d,;,),(l,d,.,),
     e: (l,l,o,),( ,i,s,),
     h: (e,l,l,),
     **/

    LookaheadPoolManager::Key lastToken = 'l';
    auto list = pm.guess(lastToken, G);
    for (auto const& ngram : list)
    {
        PRINT_TOKENS(ngram);
    }
    /***
     l: (l,o, ,),(o, ,w,),(d,., ,),(i,v,e,),(i,f,e,),
     **/

    auto pastTokens = initTensor(mBufferManager, std::string("abcde12345hijkm"), ITensor::makeShape({5, 3}));
    auto keyTokens = initTensor(mBufferManager, std::string("lvwxy"));
    pm.update(keyTokens, pastTokens);
    printMap("Token map after update", pm.getMap());
    /** Noted, we update the map with N=4, so the map has different sizes of ngrams.
     y: (j,k,m,),
     x: (5,h,i,),
     e: (l,l,o,),( ,i,s,),
     r: (l,d,;,),(l,d,.,),
     l: (o, ,w,),(d,., ,),(i,v,e,),(i,f,e,),(a,b,c,),
     o: ( ,w,o,),(r,l,d,),
     ;: ( ,h,e,),
     h: (e,l,l,),
     .: ( ,l,i,),
      : (h,e,l,),(w,o,r,),(l,i,v,),(i,s, ,),(l,i,f,),
     w: (o,r,l,),(2,3,4,),
     d: (;, ,h,),(., ,l,),
     i: (v,e, ,),(s, ,l,),(f,e,.,),
     v: (e, ,i,),(d,e,1,),
     s: ( ,l,i,),
    */

    lastToken = 'w';
    list = pm.guess(lastToken, G);
    for (auto const& ngram : list)
    {
        PRINT_TOKENS(ngram);
    }
    /**
     w: (o,r,l,),(2,3,4,),
    */

    ASSERT_EQ(list.size(), 2);
    auto it = list.begin();
    EXPECT_TRUE(isTensorEqString(*it, "orl"));
    it++;
    EXPECT_TRUE(isTensorEqString(*it, "234"));

    pastTokens = initTensor(mBufferManager, std::string("dogde12345hijkm"), ITensor::makeShape({5, 3}));
    pm.update(keyTokens, pastTokens);

    pastTokens = initTensor(mBufferManager, std::string("catde12345hijkm"), ITensor::makeShape({5, 3}));
    pm.update(keyTokens, pastTokens);

    pastTokens = initTensor(mBufferManager, std::string("abcde12345hijkm"), ITensor::makeShape({5, 3}));
    pm.update(keyTokens, pastTokens);

    printMap("Token map after update more for key 'l'", pm.getMap());
    /**
     y: (j,k,m,),
     x: (5,h,i,),
     e: (l,l,o,),( ,i,s,),
     r: (l,d,;,),(l,d,.,),
     l: (i,v,e,),(i,f,e,),(d,o,g,),(c,a,t,),(a,b,c,),
     o: ( ,w,o,),(r,l,d,),
     ;: ( ,h,e,),
     h: (e,l,l,),
     .: ( ,l,i,),
      : (h,e,l,),(w,o,r,),(l,i,v,),(i,s, ,),(l,i,f,),
     w: (o,r,l,),(2,3,4,),
     d: (;, ,h,),(., ,l,),
     i: (v,e, ,),(s, ,l,),(f,e,.,),
     v: (e, ,i,),(d,e,1,),
     s: ( ,l,i,),
    */
    lastToken = 'l';
    list = pm.guess(lastToken, G);
    ASSERT_EQ(list.size(), G);
    it = list.begin();
    EXPECT_TRUE(isTensorEqString(*it, "ive"));
    it++;
    EXPECT_TRUE(isTensorEqString(*it, "ife"));
    it++;
    EXPECT_TRUE(isTensorEqString(*it, "dog"));
    it++;
    EXPECT_TRUE(isTensorEqString(*it, "cat"));
    it++;
    EXPECT_TRUE(isTensorEqString(*it, "abc"));
}

} // namespace tensorrt_llm::tests::layers
