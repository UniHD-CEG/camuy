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
 * @file        systolic_array.h
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   MIT License
 */

#ifndef SYSTOLIC_ARRAY_H
#define SYSTOLIC_ARRAY_H

#include <vector>
#include <memory>
#include <climits>

#include <omp.h>

#include <opencv2/opencv.hpp>

#include "processing_element.h"
#include "processing_element_top_border.h"
#include "processing_element_left_border.h"
#include "processing_element_center.h"
#include "activation_fifo.h"
#include "accumulator_array.h"

//#define SYSTOLIC_ARRAY_DEBUG SYSTOLIC_ARRAY_DEBUG

#define ACTIVATION_FIFO_ADDRESS_DOES_NOT_STORE_VALUE cv::Scalar(0xE0, 0xE0 , 0xE0)
//#define ACTIVATION_FIFO_ADDRESS_STORES_VALUE cv::Scalar(0x00, 0x00, 0x99)
#define ACTIVATION_FIFO_ADDRESS_STORES_VALUE cv::Scalar(0x25, 0xA8 , 0x00)

constexpr size_t activationFifoElementWidth{10UL};
constexpr size_t activationFifoElementHeight{40UL};

template<typename WeightDatatype,
            typename ActivationDatatype,
            typename SumDatatype> class SystolicArray
{

public:

    SystolicArray(const size_t width,
                    const size_t height,
                    const size_t activationFifoDepth): m_width{width},
                                                        m_height{height},
                                                        m_activationFifoDepth{activationFifoDepth},
                                                        m_pePtrArray(m_height),
                                                        m_peDiagonalsArray(m_width + m_height - 1)
    {

        for(size_t heightCount{0}; heightCount < m_height; ++heightCount)
        {
            m_activationFifoArray.emplace_back(ActivationFifo<ActivationDatatype>{m_activationFifoDepth});
        }

        m_pePtrArray.at(0).emplace_back(std::unique_ptr<ProcessingElement<WeightDatatype,
                                                                            ActivationDatatype,
                                                                            SumDatatype>>{
                                                         new ProcessingElementLeftBorder<WeightDatatype,
                                                                                            ActivationDatatype,
                                                                                            SumDatatype>(
                                                                                        PEPosition(0, 0), nullptr,
                                                                                        &(m_activationFifoArray.at(0)))});

        for(size_t heightCount{1}; heightCount < m_height; ++heightCount)
        {
            m_pePtrArray.at(heightCount).emplace_back(std::unique_ptr<ProcessingElement<WeightDatatype,
                                                                                        ActivationDatatype,
                                                                                        SumDatatype>>{
                                                                       new ProcessingElementLeftBorder<WeightDatatype,
                                                                                                        ActivationDatatype,
                                                                                                        SumDatatype>(
                                                                                                        PEPosition(0, heightCount),
                                                                                                        m_pePtrArray.at(heightCount - 1).at(0).get(),
                                                                                                        &(m_activationFifoArray.at(heightCount)))});
        }

        for(size_t widthCount = 1; widthCount < m_width; widthCount++)
        {
            m_pePtrArray.at(0).emplace_back(std::unique_ptr<ProcessingElement<WeightDatatype,
                                                                                ActivationDatatype,
                                                                                SumDatatype>>{
                                                             new ProcessingElementTopBorder<WeightDatatype,
                                                                                            ActivationDatatype,
                                                                                            SumDatatype>(
                                                                                        PEPosition(widthCount, 0),
                                                                                        m_pePtrArray.at(0).at(widthCount - 1).get())});
        }

        for(size_t heightCount{1}; heightCount < m_height; ++heightCount)
        {
            for(size_t widthCount{1}; widthCount < m_width; ++widthCount)
            {
                m_pePtrArray.at(heightCount).emplace_back(std::unique_ptr<ProcessingElement<WeightDatatype,
                                                                                            ActivationDatatype,
                                                                                            SumDatatype>>{
                                                                        new ProcessingElementCenter<WeightDatatype,
                                                                                                    ActivationDatatype,
                                                                                                    SumDatatype>(
                                                                                        PEPosition(widthCount, heightCount),
                                                                                        m_pePtrArray.at(heightCount).at(widthCount - 1).get(),
                                                                                        m_pePtrArray.at(heightCount - 1).at(widthCount).get())});
            }
        }

        for(size_t columnCount{0}; columnCount < m_height; ++columnCount)
        {
            for(size_t rowCount{0}; rowCount < m_width; ++rowCount)
            {
                m_peDiagonalsArray.at(rowCount + columnCount).push_back(
                                                                m_pePtrArray.at(columnCount).at(rowCount).get());
            }
        }
    }

    size_t getWidth() const
    {
        return m_width;
    }

    size_t getHeight() const
    {
        return m_height;
    }

    size_t getActivationFifoDepth() const
    {
        return m_activationFifoDepth;
    }

    size_t getIterationCount() const
    {
        return m_iterationCount;
    }

    size_t getDataRegisterBytesSystolicArray() const
    {
        return m_width*m_height*(2UL*sizeof(WeightDatatype) +
                                    sizeof(ActivationDatatype) +
                                    sizeof(SumDatatype));
    }

    size_t getDataRegisterBitsSystolicArray() const
    {
        return getDataRegisterBytesSystolicArray()*CHAR_BIT;
    }

    size_t getControlRegisterBitsSystolicArray() const
    {
        return m_height*(3UL*m_width + 1UL) + 1UL;
    }

    size_t getDataRegisterCountActivationFifos() const
    {
        return m_height*m_activationFifoDepth;
    }

    size_t getDataRegisterBytesActivationFifos() const
    {
        return getDataRegisterCountActivationFifos()*
                                sizeof(ActivationDatatype);
    }

    size_t getDataRegisterBitsActivationFifos() const
    {
        return getDataRegisterBytesActivationFifos()*CHAR_BIT;
    }

    size_t getAddressRegisterCountActivationFifos() const
    {
        return 2UL*m_height;
    }

    size_t getActivationFifoAddressBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(static_cast<double>(
                                        m_activationFifoDepth)));
    }

    size_t getControlRegisterBitsActivationFifos() const
    {
        return getActivationFifoAddressBitwidthRequiredMin()*
                        getAddressRegisterCountActivationFifos();
    }

    std::vector<ProcessingElement<WeightDatatype,
                                    ActivationDatatype,
                                    SumDatatype>*> getDiagonal(const size_t diagonal)
    {
        assert(diagonal < (m_width + m_height - 1));

        return m_peDiagonalsArray.at(diagonal);
    }

    std::vector<std::unique_ptr<ProcessingElement<WeightDatatype,
                                                    ActivationDatatype,
                                                    SumDatatype>>>* getBottomPePtrRowPtr()
    {
        return &(*(m_pePtrArray.rbegin()));
    }

    std::vector<ActivationFifo<ActivationDatatype>>* getActivationFifoArrayPtr()
    {
        return &m_activationFifoArray;
    }

    void storeWeight(const PEPosition& position,
                            const WeightDatatype value)
    {
        m_pePtrArray.at(position.y).at(position.x)->storeWeight(value);
    }

    void resetIterationCount()
    {
        m_iterationCount = 0;
    }

    void setUpdateWeightsSignal(const bool updateWeights)
    {
        dynamic_cast<ProcessingElementLeftBorder<WeightDatatype,
                                                    ActivationDatatype,
                                                    SumDatatype>*>(
                                                        m_pePtrArray.at(0).at(0).get())->setUpdateWeightSignal(updateWeights);
    }

    void readUpdateWeightSignals()
    {
        for(std::vector<std::unique_ptr<ProcessingElement<WeightDatatype,
                                                            ActivationDatatype,
                                                            SumDatatype>>>& pePtrRow : m_pePtrArray)
        {
            for(std::unique_ptr<ProcessingElement<WeightDatatype,
                                                    ActivationDatatype,
                                                    SumDatatype>>& pePtr : pePtrRow)
            {
                pePtr->readUpdateWeightSignals();
            }
        }
    }

    void runIteration()
    {
        if(m_iterationCount < m_height)
        {
            dynamic_cast<ProcessingElementLeftBorder<WeightDatatype,
                                                        ActivationDatatype,
                                                        SumDatatype>*>(
                                                            m_pePtrArray.at(m_iterationCount).at(0).get())->enableFifoInput(true);
        }

        for(size_t activationFifoCount{0}; activationFifoCount < m_activationFifoArray.size();
                                                                                activationFifoCount++)
        {
            if(m_activationFifoArray.at(activationFifoCount).isEmptyNextIteration())
            {
                dynamic_cast<ProcessingElementLeftBorder<WeightDatatype, ActivationDatatype, SumDatatype>*>(
                                             m_pePtrArray.at(activationFifoCount).at(0).get())->enableFifoInput(false);

#ifdef SYSTOLIC_ARRAY_DEBUG
                std::cout << "FIFO " << activationFifoCount
                            << " empty in next iteration" << std::endl;
#endif
            }
        }

        readUpdateWeightSignals();

        for(size_t rowCount = 0; rowCount < m_height; ++rowCount)
        {
            for(std::unique_ptr<ProcessingElement<WeightDatatype,
                                                    ActivationDatatype,
                                                    SumDatatype>>& pePtr : m_pePtrArray.at(rowCount))
            {
                pePtr->computeSum();
            }
        }
    }

    void updateState(cv::Mat& outputFrameMat,
                        const cv::Point& outputFrameMatOffset)
    {
        for(std::vector<std::unique_ptr<ProcessingElement<WeightDatatype,
                                                            ActivationDatatype,
                                                            SumDatatype>>>& pePtrRow : m_pePtrArray)
        {
            for(std::unique_ptr<ProcessingElement<WeightDatatype,
                                                    ActivationDatatype,
                                                    SumDatatype>>& pePtr : pePtrRow)
            {
                pePtr->updateState();
            }
        }

        paintState(outputFrameMat, outputFrameMatOffset);

        ++m_iterationCount;
    }

    void paintState(cv::Mat& outputFrameMat,
                        const cv::Point& outputFrameOffset) const
    {

        for(const std::vector<std::unique_ptr<ProcessingElement<WeightDatatype,
                                                            ActivationDatatype,
                                                                    SumDatatype>>>& pePtrRow : m_pePtrArray)
        {
            for(const std::unique_ptr<ProcessingElement<WeightDatatype,
                                                    ActivationDatatype,
                                                            SumDatatype>>& pePtr : pePtrRow)
            {
                pePtr->paintState(outputFrameMat,
                                    cv::Point(outputFrameOffset.x + 90,
                                                    outputFrameOffset.y));
            }
        }

        for(size_t activationFifoCount{0UL};
                activationFifoCount < m_activationFifoArray.size();
                                                    ++activationFifoCount)
        {
            for(size_t elementCount{1UL}; elementCount < m_activationFifoDepth;
                                                                    ++elementCount)
            {
                const cv::Point rectanglePointTopLeft(outputFrameOffset.x +
                                                        elementCount*
                                                        activationFifoElementWidth,
                                                        outputFrameOffset.y +
                                                        activationFifoCount*
                                                        peSize +
                                                        3);

                const cv::Point rectanglePointBottomRight(outputFrameOffset.x +
                                                            (elementCount + 1)*
                                                            activationFifoElementWidth,
                                                            outputFrameOffset.y +
                                                            activationFifoCount*
                                                            peSize +
                                                            activationFifoElementHeight +
                                                            3);

                if((m_activationFifoDepth - elementCount - 1) <
                        m_activationFifoArray.at(activationFifoCount).getContentSize())
                {
                    cv::rectangle(outputFrameMat,
                                    rectanglePointTopLeft,
                                    rectanglePointBottomRight,
                                    ACTIVATION_FIFO_ADDRESS_STORES_VALUE,
                                    cv::FILLED);
                }

                else
                {
                    cv::rectangle(outputFrameMat,
                                    rectanglePointTopLeft,
                                    rectanglePointBottomRight,
                                    ACTIVATION_FIFO_ADDRESS_DOES_NOT_STORE_VALUE,
                                    cv::FILLED);
                }

                cv::rectangle(outputFrameMat,
                                rectanglePointTopLeft,
                                rectanglePointBottomRight,
                                cv::Scalar(0, 0, 0));
            }
        }
    }

private:

    const size_t m_width;
    const size_t m_height;
    const size_t m_activationFifoDepth;

    std::vector<std::vector<std::unique_ptr<ProcessingElement<WeightDatatype,
                                                                ActivationDatatype,
                                                                SumDatatype>>>> m_pePtrArray;
    std::vector<std::vector<ProcessingElement<WeightDatatype,
                                                ActivationDatatype,
                                                SumDatatype>*>> m_peDiagonalsArray;

    std::vector<ActivationFifo<ActivationDatatype>> m_activationFifoArray;

    size_t m_iterationCount{0UL};

};

#endif
