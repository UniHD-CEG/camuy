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
 * @file        processing_element_left_border.h
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   GNU Public License version 3 (GPLv3)
 */


#ifndef PROCESSING_ELEMENT_LEFT_BORDER_H
#define PROCESSING_ELEMENT_LEFT_BORDER_H

#include <sstream>
#include <cassert>

#include "processing_element.h"
#include "activation_fifo.h"

template<typename WeightDatatype,
            typename ActivationDatatype,
            typename SumDatatype> class ProcessingElementLeftBorder: public ProcessingElement<WeightDatatype,
                                                                                                ActivationDatatype,
                                                                                                SumDatatype>
{

public:

    ProcessingElementLeftBorder(const PEPosition position,
                                const ProcessingElement<WeightDatatype,
                                                            ActivationDatatype,
                                                            SumDatatype>* const neighborUpperPtr,
                                ActivationFifo<ActivationDatatype>* const activationFifoPtr):
                                                                ProcessingElement<WeightDatatype,
                                                                                    ActivationDatatype,
                                                                                    SumDatatype>::ProcessingElement(position),
                                                                m_neighborUpperPtr{neighborUpperPtr},
                                                                m_activationFifoPtr{activationFifoPtr}
    {

        assert((ProcessingElement<WeightDatatype,
                                    ActivationDatatype,
                                    SumDatatype>::m_position.x) == 0);
    }

    void enableFifoInput(bool enabled)
    {
        m_fifoInputEnabledNext = enabled;
    }

    bool fifoInputEnabled() const
    {
        return m_fifoInputEnabledCurrent;
    }

    void setUpdateWeightSignal(const bool updateWeight)
    {
        ProcessingElement<WeightDatatype,
                            ActivationDatatype,
                            SumDatatype>::m_updateWeightNext = updateWeight;
    }

    void updateState() final
    {
        ProcessingElement<WeightDatatype,
                            ActivationDatatype,
                            SumDatatype>::updateState();

        m_fifoInputEnabledCurrent = m_fifoInputEnabledNext;
    }

    void readUpdateWeightSignals() final
    {
        if(m_neighborUpperPtr)
        {
            ProcessingElement<WeightDatatype,
                                ActivationDatatype,
                                SumDatatype>::m_updateWeightNext =
                                                        m_neighborUpperPtr->hasUpdateWeightSignal();
        }
    }

    void computeSum(size_t& intraPeDataMovements,
                        size_t& interPeDataMovements,
                        size_t& weightZeroCount) final
    {

        if(m_fifoInputEnabledCurrent)
        {
            bool validSignalUpperNeighbor{true};

            if(m_neighborUpperPtr)
            {
                validSignalUpperNeighbor =  m_neighborUpperPtr->hasValidSignal();
            }

            if(validSignalUpperNeighbor)
            {

                ProcessingElement<WeightDatatype,
                                    ActivationDatatype,
                                    SumDatatype>::m_activationNext =
                                                            m_activationFifoPtr->pop();

                ProcessingElement<WeightDatatype,
                                    ActivationDatatype,
                                    SumDatatype>::m_sumNext =
                                        ProcessingElement<WeightDatatype,
                                                            ActivationDatatype,
                                                            SumDatatype>::m_activationNext*
                                        ProcessingElement<WeightDatatype,
                                                            ActivationDatatype,
                                                            SumDatatype>::loadWeight();

                intraPeDataMovements += 3UL;
                interPeDataMovements += 1UL;

                ProcessingElement<WeightDatatype,
                                    ActivationDatatype,
                                    SumDatatype>::m_validNext = true;

                if(m_neighborUpperPtr)
                {
                    ProcessingElement<WeightDatatype,
                                        ActivationDatatype,
                                        SumDatatype>::m_sumNext += m_neighborUpperPtr->getSum();
                    ProcessingElement<WeightDatatype,
                                        ActivationDatatype,
                                        SumDatatype>::m_validNext &= validSignalUpperNeighbor;

                    interPeDataMovements += 1UL;
                }
                
                if(!(ProcessingElement<WeightDatatype,
                                        ActivationDatatype,
                                        SumDatatype>::loadWeight()))
                {
                    ++weightZeroCount;
                }
            }
        }
    }

private:

    const ProcessingElement<WeightDatatype,
                                ActivationDatatype,
                                SumDatatype>* const m_neighborUpperPtr{nullptr};

    ActivationFifo<ActivationDatatype>* const m_activationFifoPtr{nullptr};

    bool m_fifoInputEnabledCurrent{false};
    bool m_fifoInputEnabledNext{false};

};

#endif
