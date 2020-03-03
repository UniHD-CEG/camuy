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
 * @file        processing_element_top_border.h
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   MIT License
 */

#ifndef PROCESSING_ELEMENT_TOP_BORDER_H
#define PROCESSING_ELEMENT_TOP_BORDER_H

#include <sstream>
#include <cassert>

#include "processing_element.h"

template<typename WeightDatatype,
            typename ActivationDatatype,
            typename SumDatatype> class ProcessingElementTopBorder: public ProcessingElement<WeightDatatype,
                                                                                                ActivationDatatype,
                                                                                                SumDatatype>
{

public:

    ProcessingElementTopBorder(const PEPosition position,
                                const ProcessingElement<WeightDatatype,
                                                        ActivationDatatype,
                                                        SumDatatype>* const neighborLeftPtr):
                                                                                ProcessingElement<WeightDatatype,
                                                                                                    ActivationDatatype,
                                                                                                    SumDatatype>::ProcessingElement(position),
                                                                                m_neighborLeftPtr{neighborLeftPtr}
    {

        assert((ProcessingElement<WeightDatatype,
                                    ActivationDatatype,
                                    SumDatatype>::m_position.y) == 0);

//        std::cout << "PE (" << ProcessingElement<Datatype>::m_position.x << ", "
//                    << ProcessingElement<Datatype>::m_position.y << ") constructed, left neighbor position ("
//                    << m_neighborLeftPtr->getPosition().x
//                    << ", " << m_neighborLeftPtr->getPosition().y << ')' << std::endl;
    }

    void readUpdateWeightSignals() final
    {
        ProcessingElement<WeightDatatype,
                            ActivationDatatype,
                            SumDatatype>::m_updateWeightNext =
                                        m_neighborLeftPtr->hasUpdateWeightSignal();
    }

    void computeSum() final
    {
        const bool validSignalLeftNeighbor{
                            m_neighborLeftPtr->hasValidSignal()};

        if(validSignalLeftNeighbor)
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
                                                        SumDatatype>::loadWeight();

            ProcessingElement<WeightDatatype,
                                ActivationDatatype,
                                SumDatatype>::m_validNext = validSignalLeftNeighbor;

//            std::ostringstream outputStringStream;

//            outputStringStream << "PE (" << ProcessingElement<Datatype>::m_position.x << ", "
//                                << ProcessingElement<Datatype>::m_position.y << ") weight: "
//                                << ProcessingElement<Datatype>::loadWeight() << " input act: "
//                                << ProcessingElement<Datatype>::m_activationNext << " result : "
//                                << ProcessingElement<Datatype>::m_sumNext << " valid signal: "
//                                << ProcessingElement<Datatype>::m_validNext << " update weight signal: "
//                                << ProcessingElement<Datatype>::m_updateWeightNext << std::endl;

//            std::cout << outputStringStream.str();
        }
    }

private:

    const ProcessingElement<WeightDatatype,
                                ActivationDatatype,
                                SumDatatype>* const m_neighborLeftPtr{nullptr};

};

#endif
