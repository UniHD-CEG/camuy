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
    }

    void readUpdateWeightSignals() final
    {
        ProcessingElement<WeightDatatype,
                            ActivationDatatype,
                            SumDatatype>::m_updateWeightNext =
                                        m_neighborLeftPtr->hasUpdateWeightSignal();
    }

    void computeSum(size_t& intraPeDataMovements,
                        size_t& interPeDataMovements,
                        size_t& weightZeroCount) final
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

            intraPeDataMovements += 3UL;
            interPeDataMovements += 1UL;
            
            if(!(ProcessingElement<WeightDatatype,
                                    ActivationDatatype,
                                    SumDatatype>::loadWeight()))
            {
                ++weightZeroCount;
            }

            ProcessingElement<WeightDatatype,
                                ActivationDatatype,
                                SumDatatype>::m_validNext = validSignalLeftNeighbor;
        }
    }

private:

    const ProcessingElement<WeightDatatype,
                                ActivationDatatype,
                                SumDatatype>* const m_neighborLeftPtr{nullptr};

};

#endif
