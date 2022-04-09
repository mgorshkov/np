/*
C++ numpy-like template-based array implementation

Copyright (c) 2022 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include <cstddef>
#include <exception>
#include <string>
#include <cerrno>
#include <cstring>

#ifdef WIN32
#include <windows.h>
#endif

#define NP_THROW_UNLESS(cond, message) \
if (!(cond)) throw np::Exception(message);

#define NP_THROW_UNLESS_WITH_ARG(cond, message, arg) \
if (!(cond)) throw np::Exception(message, arg);

namespace np {
class Exception: public std::runtime_error {
public:
    inline explicit Exception(const std::string& message)
            : std::runtime_error(message)
    {
    }

    inline Exception(const std::string& message, const std::string& arg)
        : std::runtime_error(message + arg + ", Error: " + getLastError())
    {
    }

private:
    inline static std::string getLastError()
    {
#ifdef WIN32
        DWORD lastError = ::GetLastError();
        if (lastError == 0)
            return std::string();

        LPSTR messageBuffer = nullptr;
        size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
            NULL, lastError, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);

        std::string message(messageBuffer, size);

        LocalFree(messageBuffer);

        return message;
#else
        return std::strerror(errno);
#endif
    }
};
}
