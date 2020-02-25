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

    size_t getSize() const
    {
        return m_size;
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
