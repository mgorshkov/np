/*
⚡ NumPy-style arrays in C++ | CUDA GPU + SIMD (AVX2/AVX512/AMX) CPU

Copyright (c) 2022-2026 Mikhail Gorshkov (mikhail.gorshkov@gmail.com)

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

#include <string>

#ifdef WIN32
#include <dbghelp.h>
#include <windows.h>
#pragma comment(lib, "dbghelp.lib")
#else
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <execinfo.h>
#endif

namespace np {
    inline std::string getStackTrace() {
        std::string trace;
#ifdef WIN32
        // Windows stack trace using CaptureStackBackTrace and SymFromAddr
        const int kMaxFrames = 20;
        void *stack[kMaxFrames];
        USHORT frames = CaptureStackBackTrace(0, kMaxFrames, stack, nullptr);
        HANDLE process = GetCurrentProcess();
        SymInitialize(process, nullptr, TRUE);
        SYMBOL_INFO *symbol = (SYMBOL_INFO *) calloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char), 1);
        symbol->MaxNameLen = 255;
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        for (USHORT i = 0; i < frames; ++i) {
            DWORD64 address = (DWORD64) (stack[i]);
            SymFromAddr(process, address, nullptr, symbol);
            trace += "  #" + std::to_string(i) + " " + std::string(symbol->Name) + " (0x" + std::to_string(address) + ")\n";
        }
        free(symbol);
        SymCleanup(process);
#else
        // Linux/macOS stack trace using backtrace
        const int kMaxFrames = 20;
        void *stack[kMaxFrames];
        int frames = backtrace(stack, kMaxFrames);
        char **symbols = backtrace_symbols(stack, frames);
        if (symbols) {
            for (int i = 0; i < frames; ++i) {
                // Demangle C++ symbols if possible
                char *begin_name = nullptr;
                char *begin_offset = nullptr;
                char *end_offset = nullptr;
                // Iterate over the string to find parentheses and +offset
                for (char *p = symbols[i]; *p; ++p) {
                    if (*p == '(') begin_name = p;
                    else if (*p == '+')
                        begin_offset = p;
                    else if (*p == ')' && begin_offset) {
                        end_offset = p;
                        break;
                    }
                }
                if (begin_name && begin_offset && end_offset && begin_name < begin_offset) {
                    *begin_name++ = '\0';
                    *begin_offset++ = '\0';
                    *end_offset = '\0';
                    int status;
                    char *demangled = abi::__cxa_demangle(begin_name, nullptr, nullptr, &status);
                    if (status == 0) {
                        trace += "  #" + std::to_string(i) + " " + std::string(symbols[i]) + " : " + std::string(demangled) + "+" + std::string(begin_offset) + "\n";
                        free(demangled);
                    } else {
                        trace += "  #" + std::to_string(i) + " " + std::string(symbols[i]) + " : " + std::string(begin_name) + "+" + std::string(begin_offset) + "\n";
                    }
                } else {
                    trace += "  #" + std::to_string(i) + " " + std::string(symbols[i]) + "\n";
                }
            }
            free(symbols);
        }
#endif
        return trace;
    }
}// namespace np
