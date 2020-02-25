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
