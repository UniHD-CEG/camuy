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
