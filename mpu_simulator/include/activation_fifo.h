/* Copyright 2019, 2020 Kevin Stehle
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * @file        activation_fifo.h
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   MIT License
 */

#ifndef ACTIVATION_FIFO_H
#define ACTIVATION_FIFO_H

#include <vector>
#include <iostream>
#include <cstdint>
#include <cassert>

/**
 * @class           ActivationFifo
 * @brief           The activation FIFOs are used to buffer
 *                  activation data between the SDSU and the
 *                  systolic array. They are implemented as
 *                  a ring buffer. Data memory is emulated
 *                  using a std::vector of the respective
 *                  datatype. Attempting to push to a full
 *                  FIFO does not result in an error, the
 *                  push operation is simply ignored in
 *                  that case. Popping from an empty FIFO
 *                  causes an assertion failure in debug
 *                  builds.
 * @tparam DataType The datatype of the stored activations
 */

template<typename DataType> class ActivationFifo
{

public:

    ActivationFifo(const size_t size): m_size{size},
                                        m_dataVector(m_size)
    {
    }

    void push(const DataType value)
    {
        if(!isFull())
        {
            m_dataVector.at(m_writePtr) = value;
            m_writePtr = (m_writePtr + 1) % m_size;
            ++m_contentSize;
        }
    }

    DataType pop()
    {
        DataType returnValue{m_dataVector.at(m_readPtr)};

        assert(!isEmpty());

        if(!isEmpty())
        {
            m_readPtr = (m_readPtr + 1) % m_size;
            --m_contentSize;
        }

        return returnValue;
    }

    size_t getContentSize() const
    {
        return m_contentSize;
    }
    
    /**
     * @deprecated
     */

    void setContent(const std::vector<DataType>& vector)
    {
        assert(vector.size() == m_size);
        m_dataVector.assign(vector.begin(), vector.end());
        m_readPtr = 0;
        m_writePtr = m_size - 1;
        m_contentSize = m_size - 1;
    }

    bool isEmpty() const
    {
        return m_readPtr == m_writePtr;
    }

    bool isEmptyNextIteration() const
    {
        return ((m_readPtr + 1) % m_size) == m_writePtr;
    }

    bool isEmptyInTwoIterations() const
    {
        return ((m_readPtr + 2) % m_size) == m_writePtr;
    }

    bool isFull() const
    {
        return ((m_writePtr + 1) % m_size) == m_readPtr;
    }

private:

    const size_t m_size;

    std::vector<DataType> m_dataVector;
    size_t m_readPtr{0};
    size_t m_writePtr{0};

    size_t m_contentSize{0};

};

#endif
