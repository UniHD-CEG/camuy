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
 * @file        processing_element_center.h
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   GNU Public License version 3 (GPLv3)
 */


#ifndef PROCESSING_ELEMENT_CENTER_H
#define PROCESSING_ELEMENT_CENTER_H

#include <sstream>

#include "processing_element.h"

template<typename WeightDatatype,
            typename ActivationDatatype,
            typename SumDatatype> class ProcessingElementCenter: public ProcessingElement<WeightDatatype,
                                                                                            ActivationDatatype,
                                                                                            SumDatatype>
{

public:

    ProcessingElementCenter(const PEPosition position,
                            const ProcessingElement<WeightDatatype,
                                                    ActivationDatatype,
                                                    SumDatatype>* const neighborLeftPtr,
                            const ProcessingElement<WeightDatatype,
                                                    ActivationDatatype,
                                                    SumDatatype>* const neighborUpperPtr):
                                                                            ProcessingElement<WeightDatatype,
                                                                                                ActivationDatatype,
                                                                                                SumDatatype>::ProcessingElement(position),
                                                                            m_neighborLeftPtr{neighborLeftPtr},
                                                                            m_neighborUpperPtr{neighborUpperPtr}
    {
    }

    void readUpdateWeightSignals() final
    {
        ProcessingElement<WeightDatatype,
                            ActivationDatatype,
                            SumDatatype>::m_updateWeightNext =
                                                        m_neighborLeftPtr->hasUpdateWeightSignal() &&
                                                        m_neighborUpperPtr->hasUpdateWeightSignal();
    }

    void computeSum(size_t& intraPeDataMovements,
                        size_t& interPeDataMovements,
                        size_t& weightZeroCount) final
    {

        const bool validSignalNeighbors{
                            m_neighborLeftPtr->hasValidSignal() &&
                            m_neighborUpperPtr->hasValidSignal()};

        if(validSignalNeighbors)
        {
            ProcessingElement<WeightDatatype,
                                ActivationDatatype,
                                SumDatatype>::m_activationNext =
                                                m_neighborLeftPtr->getActivation();

            ProcessingElement<WeightDatatype,
                                ActivationDatatype,
                                SumDatatype>::m_sumNext =
                                            ProcessingElement<WeightDatatype,
                                                                ActivationDatatype,
                                                                SumDatatype>::m_activationNext*
                                            ProcessingElement<WeightDatatype,
                                                                ActivationDatatype,
                                                                SumDatatype>::loadWeight() +
                                            m_neighborUpperPtr->getSum();

            intraPeDataMovements += 3UL;
            interPeDataMovements += 2UL;
            
            if(!(ProcessingElement<WeightDatatype,
                                    ActivationDatatype,
                                    SumDatatype>::loadWeight()))
            {
                ++weightZeroCount;
            }

            ProcessingElement<WeightDatatype,
                                ActivationDatatype,
                                SumDatatype>::m_validNext = validSignalNeighbors;

        }
    }

private:

    const ProcessingElement<WeightDatatype,
                                ActivationDatatype,
                                SumDatatype>* const m_neighborLeftPtr{nullptr};

    const ProcessingElement<WeightDatatype,
                            ActivationDatatype,
                            SumDatatype>* const m_neighborUpperPtr{nullptr};

};

#endif
