/**
 * @file dylib.hpp
 * @version 2.2.1
 * @brief C++ cross-platform wrapper around dynamic loading of shared libraries
 * @link https://github.com/martin-olivier/dylib
 * 
 * @author Martin Olivier <martin.olivier@live.fr>
 * @copyright (c) 2023 Martin Olivier
 *
 * This library is released under MIT license
 */

#pragma once

#include <string>
#include <stdexcept>
#include <utility>

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
#define DYLIB_CPP17
#include <filesystem>
#endif

#if (defined(_WIN32) || defined(_WIN64))
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#define DYLIB_UNDEFINE_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#define DYLIB_UNDEFINE_NOMINMAX
#endif
#include <windows.h>
#ifdef DYLIB_UNDEFINE_LEAN_AND_MEAN
#undef WIN32_LEAN_AND_MEAN
#undef DYLIB_UNDEFINE_LEAN_AND_MEAN
#endif
#ifdef DYLIB_UNDEFINE_NOMINMAX
#undef NOMINMAX
#undef DYLIB_UNDEFINE_NOMINMAX
#endif
#else
#include <dlfcn.h>
#endif

#if (defined(_WIN32) || defined(_WIN64))
#define DYLIB_WIN_MAC_OTHER(win_def, mac_def, other_def) win_def
#define DYLIB_WIN_OTHER(win_def, other_def) win_def
#elif defined(__APPLE__)
#define DYLIB_WIN_MAC_OTHER(win_def, mac_def, other_def) mac_def
#define DYLIB_WIN_OTHER(win_def, other_def) other_def
#else
#define DYLIB_WIN_MAC_OTHER(win_def, mac_def, other_def) other_def
#define DYLIB_WIN_OTHER(win_def, other_def) other_def
#endif

/**
 *  The `dylib` class represents a single dynamic library instance,
 *  allowing the access of symbols like functions or global variables
 */
class dylib {
public:
    struct filename_components {
        static constexpr const char *prefix = DYLIB_WIN_OTHER("", "lib");
        static constexpr const char *suffix = DYLIB_WIN_MAC_OTHER(".dll", ".dylib", ".so");
    };
    using native_handle_type = DYLIB_WIN_OTHER(HINSTANCE, void *);
    using native_symbol_type = DYLIB_WIN_OTHER(FARPROC, void *);

    static_assert(std::is_pointer<native_handle_type>::value, "Expecting HINSTANCE to be a pointer");
    static_assert(std::is_pointer<native_symbol_type>::value, "Expecting FARPROC to be a pointer");

    static constexpr bool add_filename_decorations = true;
    static constexpr bool no_filename_decorations = false;

    /**
     *  This exception is raised when a library fails to load or a symbol fails to resolve
     */
    class exception : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    /**
     *  This exception is raised when a library fails to load
     */
    class load_error : public exception {
        using exception::exception;
    };

    /**
     *  This exception is raised when a symbol fails to resolve
     */
    class symbol_error : public exception {
        using exception::exception;
    };

    dylib(const dylib&) = delete;
    dylib& operator=(const dylib&) = delete;

    dylib(dylib &&other) noexcept : m_handle(other.m_handle) {
        other.m_handle = nullptr;
    }

    dylib& operator=(dylib &&other) noexcept {
        if (this != &other)
            std::swap(m_handle, other.m_handle);
        return *this;
    }

    /**
     *  Loads a dynamic library
     *
     *  @throws `dylib::load_error` if the library could not be opened (including
     *  the case of the library file not being found)
     *  @throws `std::invalid_argument` if the arguments are null
     *
     *  @param dir_path the directory path where the dynamic library is located
     *  @param name the name of the dynamic library to load
     *  @param decorations adds OS-specific decorations to the library name
     */
    ///@{
    dylib(const char *dir_path, const char *lib_name, bool decorations = add_filename_decorations) {
        if (!dir_path)
            throw std::invalid_argument("The directory path is null");
        if (!lib_name)
            throw std::invalid_argument("The library name is null");

        std::string final_name = lib_name;
        std::string final_path = dir_path;

        if (decorations)
            final_name = filename_components::prefix + final_name + filename_components::suffix;

        if (!final_path.empty() && final_path.find_last_of('/') != final_path.size() - 1)
            final_path += '/';

        m_handle = open((final_path + final_name).c_str());

        if (!m_handle)
            throw load_error("Could not load library \"" + final_path + final_name + "\"\n" + get_error_description());
    }

    dylib(const std::string &dir_path, const std::string &lib_name, bool decorations = add_filename_decorations)
        : dylib(dir_path.c_str(), lib_name.c_str(), decorations) {}

    dylib(const std::string &dir_path, const char *lib_name, bool decorations = add_filename_decorations)
        : dylib(dir_path.c_str(), lib_name, decorations) {}

    dylib(const char *dir_path, const std::string &lib_name, bool decorations = add_filename_decorations)
        : dylib(dir_path, lib_name.c_str(), decorations) {}

    explicit dylib(const std::string &lib_name, bool decorations = add_filename_decorations)
        : dylib("", lib_name.c_str(), decorations) {}

    explicit dylib(const char *lib_name, bool decorations = add_filename_decorations)
        : dylib("", lib_name, decorations) {}

#ifdef DYLIB_CPP17
    explicit dylib(const std::filesystem::path &lib_path)
        : dylib("", lib_path.string().c_str(), no_filename_decorations) {}

    dylib(const std::filesystem::path &dir_path, const std::string &lib_name, bool decorations = add_filename_decorations)
        : dylib(dir_path.string().c_str(), lib_name.c_str(), decorations) {}

    dylib(const std::filesystem::path &dir_path, const char *lib_name, bool decorations = add_filename_decorations)
        : dylib(dir_path.string().c_str(), lib_name, decorations) {}
#endif
    ///@}

    ~dylib() {
        if (m_handle)
            close(m_handle);
    }

    /**
     *  Get a symbol from the currently loaded dynamic library
     * 
     *  @throws `dylib::symbol_error` if the symbol could not be found
     *  @throws `std::invalid_argument` if the argument or library handle is null
     *
     *  @param symbol_name the symbol name to lookup
     *
     *  @return a pointer to the requested symbol
     */
    native_symbol_type get_symbol(const char *symbol_name) const {
        if (!symbol_name)
            throw std::invalid_argument("The symbol name to lookup is null");
        if (!m_handle)
            throw std::logic_error("The dynamic library handle is null. This object may have been moved from.");

        auto symbol = locate_symbol(m_handle, symbol_name);

        if (symbol == nullptr)
            throw symbol_error("Could not get symbol \"" + std::string(symbol_name) + "\"\n" + get_error_description());
        return symbol;
    }

    native_symbol_type get_symbol(const std::string &symbol_name) const {
        return get_symbol(symbol_name.c_str());
    }

    /**
     *  Get a function from the currently loaded dynamic library
     * 
     *  @throws `dylib::symbol_error` if the function could not be found
     *  @throws `std::invalid_argument` if the argument is null
     *
     *  @tparam T the function type, e.g., `double(int, int)`
     *  @param symbol_name the function name to lookup
     *
     *  @return a pointer to the requested function
     */
    template<typename T>
    T *get_function(const char *symbol_name) const {
#if (defined(__GNUC__) && __GNUC__ >= 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
        return reinterpret_cast<T *>(get_symbol(symbol_name));
#if (defined(__GNUC__) && __GNUC__ >= 8)
#pragma GCC diagnostic pop
#endif
    }

    template<typename T>
    T *get_function(const std::string &symbol_name) const {
        return get_function<T>(symbol_name.c_str());
    }

    /**
     *  Get a variable from the currently loaded dynamic library
     * 
     *  @throws `dylib::symbol_error` if the variable could not be found
     *  @throws `std::invalid_argument` if the argument is null
     *
     *  @tparam T the variable type
     *  @param symbol_name the variable name to lookup
     *
     *  @return a reference to the requested variable
     */
    template<typename T>
    T &get_variable(const char *symbol_name) const {
        return *reinterpret_cast<T *>(get_symbol(symbol_name));
    }

    template<typename T>
    T &get_variable(const std::string &symbol_name) const {
        return get_variable<T>(symbol_name.c_str());
    }

    /**
     *  Check if a symbol exists in the currently loaded dynamic library. 
     *  This method will return false if no dynamic library is currently loaded 
     *  or if the symbol name is nullptr
     *
     *  @param symbol_name the symbol name to look for
     *
     *  @return true if the symbol exists in the dynamic library, false otherwise
     */
    bool has_symbol(const char *symbol_name) const noexcept {
        if (!m_handle || !symbol_name)
            return false;
        return locate_symbol(m_handle, symbol_name) != nullptr;
    }

    bool has_symbol(const std::string &symbol) const noexcept {
        return has_symbol(symbol.c_str());
    }

    /**
     *  @return the dynamic library handle
     */
    native_handle_type native_handle() noexcept {
        return m_handle;
    }

protected:
    native_handle_type m_handle{nullptr};

    static native_handle_type open(const char *path) noexcept {
#if (defined(_WIN32) || defined(_WIN64))
        return LoadLibraryA(path);
#else
        return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
    }

    static native_symbol_type locate_symbol(native_handle_type lib, const char *name) noexcept {
        return DYLIB_WIN_OTHER(GetProcAddress, dlsym)(lib, name);
    }

    static void close(native_handle_type lib) noexcept {
        DYLIB_WIN_OTHER(FreeLibrary, dlclose)(lib);
    }

    static std::string get_error_description() noexcept {
#if (defined(_WIN32) || defined(_WIN64))
        constexpr const size_t BUF_SIZE = 512;
        const auto error_code = GetLastError();
        if (!error_code)
            return "No error reported by GetLastError";
        char description[BUF_SIZE];
        const auto lang = MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US);
        const DWORD length =
            FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, nullptr, error_code, lang, description, BUF_SIZE, nullptr);
        return (length == 0) ? "Unknown error (FormatMessage failed)" : description;
#else
        const auto description = dlerror();
        return (description == nullptr) ? "No error reported by dlerror" : description;
#endif
    }
};

#undef DYLIB_WIN_MAC_OTHER
#undef DYLIB_WIN_OTHER
#undef DYLIB_CPP17
