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
