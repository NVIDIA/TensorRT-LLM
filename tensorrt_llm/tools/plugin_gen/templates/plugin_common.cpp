#include "plugin_common.h"

using namespace nvinfer1::plugin;

namespace nvinfer1
{
namespace plugin
{

LogStream<ILogger::Severity::kERROR> gLogError;
LogStream<ILogger::Severity::kWARNING> gLogWarning;
LogStream<ILogger::Severity::kINFO> gLogInfo;
LogStream<ILogger::Severity::kVERBOSE> gLogVerbose;

ILogger* gLogger{};

void throwPluginError(char const* file, char const* function, int line, int status, char const* msg)
{
    PluginError error(file, function, line, status, msg);
    reportValidationFailure(msg, file, line);
    throw error;
}

void reportValidationFailure(char const* msg, char const* file, int line)
{
    std::ostringstream stream;
    stream << "Validation failed: " << msg << std::endl << file << ':' << line << std::endl;
    getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
}

void caughtError(std::exception const& e)
{
    gLogError << e.what() << std::endl;
}

// break-pointable
void reportAssertion(char const* msg, char const* file, int line)
{
    std::ostringstream stream;
    stream << "Assertion failed: " << msg << std::endl
           << file << ':' << line << std::endl
           << "Aborting..." << std::endl;
    getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
    PLUGIN_CUASSERT(cudaDeviceReset());
    abort();
}

void throwCublasError(char const* file, char const* function, int line, int status, char const* msg)
{
    if (msg == nullptr)
    {
        auto s_ = static_cast<cublasStatus_t>(status);
        switch (s_)
        {
        case CUBLAS_STATUS_SUCCESS: msg = "CUBLAS_STATUS_SUCCESS"; break;
        case CUBLAS_STATUS_NOT_INITIALIZED: msg = "CUBLAS_STATUS_NOT_INITIALIZED"; break;
        case CUBLAS_STATUS_ALLOC_FAILED: msg = "CUBLAS_STATUS_ALLOC_FAILED"; break;
        case CUBLAS_STATUS_INVALID_VALUE: msg = "CUBLAS_STATUS_INVALID_VALUE"; break;
        case CUBLAS_STATUS_ARCH_MISMATCH: msg = "CUBLAS_STATUS_ARCH_MISMATCH"; break;
        case CUBLAS_STATUS_MAPPING_ERROR: msg = "CUBLAS_STATUS_MAPPING_ERROR"; break;
        case CUBLAS_STATUS_EXECUTION_FAILED: msg = "CUBLAS_STATUS_EXECUTION_FAILED"; break;
        case CUBLAS_STATUS_INTERNAL_ERROR: msg = "CUBLAS_STATUS_INTERNAL_ERROR"; break;
        case CUBLAS_STATUS_NOT_SUPPORTED: msg = "CUBLAS_STATUS_NOT_SUPPORTED"; break;
        case CUBLAS_STATUS_LICENSE_ERROR: msg = "CUBLAS_STATUS_LICENSE_ERROR"; break;
        }
    }
    CublasError error(file, function, line, status, msg);
    error.log(gLogError);
    throw error;
}

template <ILogger::Severity kSeverity>
int LogStream<kSeverity>::Buf::sync()
{
    std::string s = str();
    while (!s.empty() && s.back() == '\n')
    {
        s.pop_back();
    }
    if (gLogger != nullptr)
    {
        gLogger->log(kSeverity, s.c_str());
    }
    str("");
    return 0;
}

void throwCudaError(char const* file, char const* function, int line, int status, char const* msg)
{
    CudaError error(file, function, line, status, msg);
    error.log(gLogError);
    throw error;
}

} // namespace plugin
} // namespace nvinfer1
