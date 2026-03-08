#!/bin/bash

find include samples unit_tests -type f -iname *.cpp -o -iname *.hpp -o -iname *.cu -o -iname *.cuh | xargs clang-format -i
