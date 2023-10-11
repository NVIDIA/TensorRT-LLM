// Below are some utilities for plugins, we paste it here to decouple the PluginGen from TRT-LLM libraries
#ifndef PLUGIN_UTILITY
#define PLUGIN_UTILITY

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
    TRTException(const char* fl, const char* fn, int ln, int st, const char* msg, const char* nm)
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

    void setMessage(const char* msg)
    {
        message = msg;
    }

protected:
    const char* file{};
    const char* function{};
    int line{};
    int status{};
    const char* message{};
    const char* name{};
};

class TritonError : public TRTException
{
public:
    TritonError(const char* fl, const char* fn, int ln, int stat, const char* msg = nullptr)
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
    CudaError(const char* fl, const char* fn, int ln, int stat, const char* msg = nullptr)
        : TRTException(fl, fn, ln, stat, msg, "Cuda")
    {
    }
};

class CublasError : public TRTException
{
public:
    CublasError(const char* fl, const char* fn, int ln, int stat, const char* msg = nullptr)
        : TRTException(fl, fn, ln, stat, msg, "cuBLAS")
    {
    }
};

// Write values into buffer
template <typename T>
void write(char*& buffer, const T& val)
{
    std::memcpy(buffer, &val, sizeof(T));
    buffer += sizeof(T);
}

// Read values from buffer
template <typename T>
void read(const char*& buffer, T& val)
{
    std::memcpy(&val, buffer, sizeof(T));
    buffer += sizeof(T);
}

void reportAssertion(const char* msg, const char* file, int line);
void caughtError(const std::exception& e);
void reportValidationFailure(char const* msg, char const* file, int line);
void throwCublasError(const char* file, const char* function, int line, int status, const char* msg = nullptr);
void throwCudaError(const char* file, const char* function, int line, int status, const char* msg = nullptr);

} // namespace plugin

} // namespace nvinfer1

#endif // end ifndef PLUGIN_UTILITY
