#!/bin/bash

find include samples unit_tests -type f -iname *.cpp -o -iname *.hpp | xargs clang-format -i
