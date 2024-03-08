#include "utils/nitro_utils.h"
#include <climits> // for PATH_MAX
#include <drogon/HttpAppFramework.h>
#include <drogon/drogon.h>
#include <iostream>

#if defined(__APPLE__) && defined(__MACH__)
#include <libgen.h> // for dirname()
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <libgen.h> // for dirname()
#include <unistd.h> // for readlink()
#elif defined(_WIN32)
#include <windows.h>
#undef max
#else
#error "Unsupported platform!"
#endif

int main(int argc, char* argv[])
{
    int thread_num = 1;
    std::string host = "127.0.0.1";
    int port = 3928;
    std::string uploads_folder_path;

    // Number of nitro threads
    if (argc > 1)
    {
        thread_num = std::atoi(argv[1]);
    }

    // Check for host argument
    if (argc > 2)
    {
        host = argv[2];
    }

    // Check for port argument
    if (argc > 3)
    {
        port = std::atoi(argv[3]); // Convert string argument to int
    }

    // Uploads folder path
    if (argc > 4)
    {
        uploads_folder_path = argv[4];
    }

    int logical_cores = std::thread::hardware_concurrency();
    int drogon_thread_num = 1; // temporarily set thread num to 1
    nitro_utils::nitro_logo();
#ifdef NITRO_VERSION
    LOG_INFO << "Nitro version: " << NITRO_VERSION;
#else
    LOG_INFO << "Nitro version: undefined";
#endif
    LOG_INFO << "Server started, listening at: " << host << ":" << port;
    LOG_INFO << "Please load your model";
    drogon::app().addListener(host, port);
    drogon::app().setThreadNum(drogon_thread_num);
    if (!uploads_folder_path.empty())
    {
        LOG_INFO << "Drogon uploads folder is at: " << uploads_folder_path;
        drogon::app().setUploadPath(uploads_folder_path);
    }
    LOG_INFO << "Number of thread is:" << drogon::app().getThreadNum();

    drogon::app().run();

    return 0;
}
