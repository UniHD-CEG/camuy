/* Copyright (c) 2019, 2020 Kevin Stehle
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
 * @file        processing_element.h
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   MIT License
 */

#ifndef PROCESSING_ELEMENT_H
#define PROCESSING_ELEMENT_H

#include <mutex>
#include <iostream>
#include <cstdio>
#include <cmath>

/**
 * @struct  PEPosition
 * @brief   
 */

struct PEPosition
{
    PEPosition(size_t x, size_t y): x{x},
                                        y{y}
    {
    }

    size_t x;
    size_t y;
};

/**
 * @class ProcessingElement
 * @brief
 * @tparam WeightDatatype
 * @tparam ActivationDatatype
 * @tparam SumDatatype
 */

template<typename WeightDatatype,
            typename ActivationDatatype,
            typename SumDatatype> class ProcessingElement
{

public:
    
    /**
     * @brief
     * @param position
     */

    ProcessingElement(const PEPosition position): m_position{position}
    {
    }

    WeightDatatype loadWeight() const
    {
        return (*m_weightRegisterReadPtrCurrent);
    }

    void storeWeight(const WeightDatatype weight)
    {
        *m_weightRegisterWritePtrCurrent = weight;
    }

    void updateWeight()
    {
        m_weightRegisterReadPtrNext =
                            m_weightRegisterWritePtrCurrent;

        m_weightRegisterWritePtrNext =
                            m_weightRegisterReadPtrCurrent;

    }

    SumDatatype getSum() const
    {
        return m_sumCurrent;
    }

    ActivationDatatype getActivation() const
    {
        return m_activationCurrent;
    }

    PEPosition getPosition() const
    {
         return m_position;
    }

    bool hasValidSignal() const
    {
        return m_validCurrent;
    }

    bool hasUpdateWeightSignal() const
    {
        return m_updateWeightCurrent;
    }

    /**
     * @brief
     */
    
    virtual void readUpdateWeightSignals() = 0;
    
    /**
     * @brief
     * @param intraPeDataMovements
     * @param interPeDataMovements
     * @param weightZeroCount
     */

    virtual void computeSum(size_t& intraPeDataMovements,
                                size_t& interPeDataMovements,
                                size_t& weightZeroCount) = 0;

    /**
     * @brief
     */                            
    
    virtual void updateState()
    {
        if(m_updateWeightNext)
        {
            updateWeight();
        }

        m_sumCurrent = m_sumNext;

        m_activationCurrent = m_activationNext;

        m_weightRegisterReadPtrCurrent =
                        m_weightRegisterReadPtrNext;

        m_weightRegisterWritePtrCurrent =
                        m_weightRegisterWritePtrNext;

        m_validCurrent = m_validNext;
        m_updateWeightCurrent = m_updateWeightNext;

        m_validNext = false;
        m_updateWeightNext = false;
    }

    virtual ~ProcessingElement() = default;

protected:

    const PEPosition m_position;

    SumDatatype m_sumCurrent{0};
    SumDatatype m_sumNext{0};
    ActivationDatatype m_activationCurrent{0};
    ActivationDatatype m_activationNext{0};

    bool m_validCurrent{false};
    bool m_validNext{false};
    bool m_updateWeightCurrent{false};
    bool m_updateWeightNext{false};

private:

    WeightDatatype m_weightRegister0{0};
    WeightDatatype m_weightRegister1{0};

    WeightDatatype* m_weightRegisterReadPtrCurrent{&m_weightRegister0};
    WeightDatatype* m_weightRegisterReadPtrNext{&m_weightRegister0};
    WeightDatatype* m_weightRegisterWritePtrCurrent{&m_weightRegister1};
    WeightDatatype* m_weightRegisterWritePtrNext{&m_weightRegister1};



};

#endif
