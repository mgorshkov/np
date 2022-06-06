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

#include <random>
#include <algorithm>

#include <np/Shape.hpp>

#include <np/ndarray/static/NDArrayStaticDecl.hpp>

namespace np {
	namespace ndarray {
		namespace array_static {
			// Array creators
			template<typename DType, Size SizeT, Size... SizeTs>
			inline NDArrayStatic<DType, SizeT, SizeTs...>::NDArrayStatic() noexcept
				: m_ArrayImpl{} {
			}

			template<typename DType, Size SizeT, Size... SizeTs>
			inline NDArrayStatic<DType, SizeT, SizeTs...>::NDArrayStatic(const DType &value) noexcept
				: m_ArrayImpl{ value } {
			}

			template<typename DType, Size SizeT, Size... SizeTs>
			inline NDArrayStatic<DType, SizeT, SizeTs...>::NDArrayStatic(CArrayType data) noexcept
				: m_ArrayImpl{ data } {
			}

			template<typename DType, Size SizeT, Size... SizeTs>
			inline NDArrayStatic<DType, SizeT, SizeTs...>::NDArrayStatic(
				const NDArrayStatic<DType, SizeT, SizeTs...> &another) noexcept
				: m_ArrayImpl{ another.m_ArrayImpl } {
			}

			template<typename DType, Size SizeT, Size... SizeTs>
			inline NDArrayStatic<DType, SizeT, SizeTs...>::NDArrayStatic(
				NDArrayStatic<DType, SizeT, SizeTs...> &&another) noexcept
				: m_ArrayImpl{ std::move(another.m_ArrayImpl) } {
			}

			template<typename DType, Size SizeT, Size... SizeTs>
			inline NDArrayStatic<DType, SizeT, SizeTs...>::NDArrayStatic(
				const internal::NDArrayStaticInternal<DType, SizeT, SizeTs...> &array) noexcept
				: m_ArrayImpl{ array } {
			}

			template<typename DType, Size SizeT, Size... SizeTs>
			inline NDArrayStatic<DType, SizeT, SizeTs...>::NDArrayStatic(
				internal::NDArrayStaticInternal<DType, SizeT, SizeTs...> &&array) noexcept
				: m_ArrayImpl{ std::move(array) } {
			}

			template<typename DType, Size SizeT, Size... SizeTs>
			inline NDArrayStatic<DType, SizeT, SizeTs...>::NDArrayStatic(const StdArrayType &array) noexcept
				: m_ArrayImpl{ array } {
			}

			template<typename DType, Size SizeT, Size... SizeTs>
			inline NDArrayStatic<DType, SizeT, SizeTs...>::NDArrayStatic(StdArrayType &&array) noexcept
				: m_ArrayImpl{ std::move(array) } {
			}

			template<typename DType, Size SizeT, Size... SizeTs>
			inline NDArrayStatic<DType, SizeT, SizeTs...>::NDArrayStatic(const StdVectorType &vector) noexcept
				: m_ArrayImpl{ vector } {
			}

			template<typename DType, Size SizeT, Size... SizeTs>
			inline NDArrayStatic<DType, SizeT, SizeTs...>::NDArrayStatic(StdVectorType &&vector) noexcept
				: m_ArrayImpl{ std::move(vector) } {
			}

			template<typename DType,
				Size SizeT, Size... SizeTs>
				inline NDArrayStatic<DType, SizeT, SizeTs...>::NDArrayStatic(std::initializer_list<DType> init_list) noexcept
				: m_ArrayImpl{ init_list } {
			}

			template<typename DType, Size SizeT, Size... SizeTs>
			inline NDArrayStatic<DType, SizeT, SizeTs...>::~NDArrayStatic() noexcept {
			}

			template<typename DType, Size SizeT, Size... SizeTs>
			inline NDArrayStatic<DType, SizeT, SizeTs...> &NDArrayStatic<DType, SizeT, SizeTs...>::operator=(
				const NDArrayStatic<DType, SizeT, SizeTs...> &another) noexcept {
				m_ArrayImpl = another.m_ArrayImpl;
				return *this;
			}

			template<typename DType, Size SizeT, Size... SizeTs>
			inline NDArrayStatic<DType, SizeT, SizeTs...> &
				NDArrayStatic<DType, SizeT, SizeTs...>::operator=(NDArrayStatic<DType, SizeT, SizeTs...> &&another) noexcept {
				m_ArrayImpl = std::move(another.m_ArrayImpl);
				return *this;
			}

			template<typename DType, Size SizeT, Size... SizeTs>
			inline NDArrayStatic<DType, SizeT, SizeTs...> &
				NDArrayStatic<DType, SizeT, SizeTs...>::operator=(const NDArrayStatic<DType, SizeT, SizeTs...>::StdVectorType &vector) noexcept {
				m_ArrayImpl = vector;
				return *this;
			}
		}
	}
}
