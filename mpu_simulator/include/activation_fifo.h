/* This file is part of mpusim.
 *
 * mpusim is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * mpusim is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with mpusim.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @file        activation_fifo.h
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   GNU Public License version 3 (GPLv3)
 */


#ifndef ACTIVATION_FIFO_H
#define ACTIVATION_FIFO_H

#include <vector>
#include <iostream>
#include <cstdint>
#include <cassert>

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
