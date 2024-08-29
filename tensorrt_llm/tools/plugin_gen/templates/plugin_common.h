// Below are some utilities for plugins, we paste it here to decouple the PluginGen from TRT-LLM libraries
#ifndef PLUGIN_UTILITY
#define PLUGIN_UTILITY

#include "NvInferPlugin.h"

#include <cstring>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define PLUGIN_ASSERT(assertion)                                                                                       \
    {                                                                                                                  \
        if (!(assertion))                                                                                              \
        {                                                                                                              \
            nvinfer1::plugin::reportAssertion(#assertion, __FILE__, __LINE__);                                         \
        }                                                                                                              \
    }

#define PLUGIN_CUBLASASSERT(status_)                                                                                   \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != CUBLAS_STATUS_SUCCESS)                                                                               \
        {                                                                                                              \
            nvinfer1::plugin::throwCublasError(__FILE__, FN_NAME, __LINE__, s_);                                       \
        }                                                                                                              \
    }

#define PLUGIN_CUASSERT(status_)                                                                                       \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != cudaSuccess)                                                                                         \
        {                                                                                                              \
            const char* msg = cudaGetErrorString(s_);                                                                  \
            nvinfer1::plugin::throwCudaError(__FILE__, FN_NAME, __LINE__, s_, msg);                                    \
        }                                                                                                              \
    }

#include <mutex>
#include <sstream>

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

namespace nvinfer1
{
namespace plugin
{

namespace
{
template <ILogger::Severity kSeverity>
class LogStream : public std::ostream
{
    class Buf : public std::stringbuf
    {
    public:
        int sync() override;
    };

    Buf buffer;
    std::mutex mLogStreamMutex;

public:
    std::mutex& getMutex()
    {
        return mLogStreamMutex;
    }

    LogStream()
        : std::ostream(&buffer){};
};

// Use mutex to protect multi-stream write to buffer
template <ILogger::Severity kSeverity, typename T>
LogStream<kSeverity>& operator<<(LogStream<kSeverity>& stream, T const& msg)
{
    std::lock_guard<std::mutex> guard(stream.getMutex());
    auto& os = static_cast<std::ostream&>(stream);
    os << msg;
    return stream;
}

// Special handling static numbers
template <ILogger::Severity kSeverity>
inline LogStream<kSeverity>& operator<<(LogStream<kSeverity>& stream, int32_t num)
{
    std::lock_guard<std::mutex> guard(stream.getMutex());
    auto& os = static_cast<std::ostream&>(stream);
    os << num;
    return stream;
}

// Special handling std::endl
template <ILogger::Severity kSeverity>
inline LogStream<kSeverity>& operator<<(LogStream<kSeverity>& stream, std::ostream& (*f)(std::ostream&) )
{
    std::lock_guard<std::mutex> guard(stream.getMutex());
    auto& os = static_cast<std::ostream&>(stream);
    os << f;
    return stream;
}

} // namespace

class TRTException : public std::exception
{
public:
    TRTException(char const* fl, char const* fn, int ln, int st, char const* msg, char const* nm)
        : file(fl)
        , function(fn)
        , line(ln)
        , status(st)
        , message(msg)
        , name(nm)
    {
    }

    virtual void log(std::ostream& logStream) const
    {
        logStream << file << " (" << line << ") - " << name << " Error in " << function << ": " << status;
        if (message != nullptr)
        {
            logStream << " (" << message << ")";
        }
        logStream << std::endl;
    }

    void setMessage(char const* msg)
    {
        message = msg;
    }

protected:
    char const* file{};
    char const* function{};
    int line{};
    int status{};
    char const* message{};
    char const* name{};
};

class TritonError : public TRTException
{
public:
    TritonError(char const* fl, char const* fn, int ln, int stat, char const* msg = nullptr)
        : TRTException(fl, fn, ln, stat, msg, "cuBLAS")
    {
    }
};

class PluginError : public TRTException
{
public:
    PluginError(char const* fl, char const* fn, int ln, int stat, char const* msg = nullptr)
        : TRTException(fl, fn, ln, stat, msg, "Plugin")
    {
    }
};

class CudaError : public TRTException
{
public:
    CudaError(char const* fl, char const* fn, int ln, int stat, char const* msg = nullptr)
        : TRTException(fl, fn, ln, stat, msg, "Cuda")
    {
    }
};

class CublasError : public TRTException
{
public:
    CublasError(char const* fl, char const* fn, int ln, int stat, char const* msg = nullptr)
        : TRTException(fl, fn, ln, stat, msg, "cuBLAS")
    {
    }
};

// Write values into buffer
template <typename T>
void write(char*& buffer, T const& val)
{
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
void read(char const*& buffer, T& val)
{
    std::memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
}

void reportAssertion(char const* msg, char const* file, int line);
void caughtError(std::exception const& e);
void reportValidationFailure(char const* msg, char const* file, int line);
void throwCublasError(char const* file, char const* function, int line, int status, char const* msg = nullptr);
void throwCudaError(char const* file, char const* function, int line, int status, char const* msg = nullptr);

} // namespace plugin

} // namespace nvinfer1

#endif // end ifndef PLUGIN_UTILITY
