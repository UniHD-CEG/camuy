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
 * @file        processing_element.h
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   GNU Public License version 3 (GPLv3)
 */


#ifndef PROCESSING_ELEMENT_H
#define PROCESSING_ELEMENT_H

#include <mutex>
#include <iostream>
#include <cstdio>
#include <cmath>

struct PEPosition
{
    PEPosition(size_t x, size_t y): x{x},
                                        y{y}
    {
    }

    size_t x;
    size_t y;
};

template<typename WeightDatatype,
            typename ActivationDatatype,
            typename SumDatatype> class ProcessingElement
{

public:

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

//        std::cout << "PE (" << m_position.x << ", "
//                    << m_position.y << ") switched weight" << std::endl;
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

    virtual void readUpdateWeightSignals() = 0;

    virtual void computeSum(size_t& intraPeDataMovements,
                                size_t& interPeDataMovements,
                                size_t& weightZeroCount) = 0;

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
