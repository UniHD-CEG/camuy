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
 * @file        accumulator_array.h
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   MIT License
 */

#ifndef ACCUMULATOR_ARRAY_H
#define ACCUMULATOR_ARRAY_H

#include <vector>
#include <memory>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <climits>

#include <opencv2/opencv.hpp>

#include "mpu_visualizer_global_constants.h"
#include "processing_element.h"

//#define ACCUMULATOR_ARRAY_DEBUG ACCUMULATOR_ARRAY_DEBUG

constexpr size_t accumulatorElementWidth{47UL};
constexpr size_t accumulatorElementHeight{10UL};

#define ACCUMULATOR_ARRAY_ELEMENT_COLOR_INACTIVE cv::Scalar(0xE0, 0xE0 , 0xE0)
#define ACCUMULATOR_ARRAY_ELEMENT_COLOR_READ_ACCUMULATOR_ARRAY cv::Scalar(0xCC, 0x40, 0x99)
#define ACCUMULATOR_ARRAY_ELEMENT_COLOR_READ_UNIFIED_BUFFER cv::Scalar(0xFF, 0xFF, 0xFF)
#define ACCUMULATOR_ARRAY_ELEMENT_COLOR_READ_UNIFIED_BUFFER_PERSISTENT cv::Scalar(0xCC, 0x40, 0x99)
#define ACCUMULATOR_ARRAY_ELEMENT_COLOR_WRITE cv::Scalar(0x25, 0xA8 , 0x00)

namespace
{
constexpr bool low{false};
constexpr bool high{true};
}

enum class SystolicArrayStartupMode
{
    WeightsPreloaded,
    WeightsNotPreloaded
};

template<typename WeightDatatype,
            typename ActivationDatatype,
            typename AccumulatorDatatype> class AccumulatorArray
{

public:

    AccumulatorArray(std::vector<std::unique_ptr<
                        ProcessingElement<WeightDatatype,
                                            ActivationDatatype,
                                            AccumulatorDatatype>>>* pePtrVector,
                                                    const size_t systolicArrayWidth,
                                                    const size_t accumulatorArrayHeight): m_pePtrVector{pePtrVector},
                                                                                            m_width{systolicArrayWidth},
                                                                                            m_height{accumulatorArrayHeight},
                                                                                            m_bufferHeight{m_height/2UL},
                                                                                            m_dataArray(m_width*m_height),
                                                                                            m_buffer0Address(m_dataArray.begin()),
                                                                                            m_buffer1Address(m_dataArray.begin() +
                                                                                                                            m_width*
                                                                                                                            m_bufferHeight),
                                                                                            m_rowPtrArrayCurrent(m_width),
                                                                                            m_rowPtrArrayNext(m_width),
                                                                                            m_rowAdditionCountArrayCurrent(m_width),
                                                                                            m_rowAdditionCountArrayNext(m_width),
                                                                                            m_buffer0RowReadArray(m_width,
                                                                                                                    std::numeric_limits<size_t>::max()),
                                                                                            m_buffer1RowReadArray(m_width,
                                                                                                                    std::numeric_limits<size_t>::max()),
                                                                                            m_buffer0RowWriteArray(m_width,
                                                                                                                    std::numeric_limits<size_t>::max()),
                                                                                            m_buffer1RowWriteArray(m_width,
                                                                                                                    std::numeric_limits<size_t>::max()),
                                                                                            m_writeAddressSelectBitArrayCurrent(m_width),
                                                                                            m_writeAddressSelectBitArrayNext(m_width),
                                                                                            m_firstWeightUpdateDoneArrayCurrent(m_width),
                                                                                            m_firstWeightUpdateDoneArrayNext(m_width),
                                                                                            m_validSignalArray(m_width)
    {
    }

    size_t getRowPtrBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(static_cast<double>(
                                            m_bufferHeight)));
    }

    size_t getAdditionCounterBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(static_cast<double>(
                                        m_additionCountMax)));
    }

    size_t getDataRegisterCount() const
    {
        return m_width*m_height;
    }

    size_t getDataRegisterBytes() const
    {
        return getDataRegisterCount()*sizeof(AccumulatorDatatype);
    }

    size_t getDataRegisterBits() const
    {
        return getDataRegisterBytes()*CHAR_BIT;
    }

    size_t getControlRegisterBits() const
    {      
        /* All control data arrays have a size of m_width.
         * To calculate the bits in the accumulator array
         * required for a MPU model to perform all matrix
         * multiplications performed since the addition
         * count maximum values were last reset using
         * resetAdditionCountMaxValue(), we calculate the
         * number of bits required per accumulator array
         * column.
         * The required bits are the minimum bitwidth of the
         * row pointers (modelled by m_rowPtrArrayCurrent
         * and m_rowPtrArrayNext), the minimum bitwidth of
         * the addition counters (modelled by
         * m_rowAdditionCountArrayCurrent and
         * m_rowAdditionCountArrayNext) plus the write
         * address select bit (modelled by
         * m_writeAddressSelectBitArrayCurrent and
         * m_writeAddressSelectBitArrayNext) and the first
         * weight update done flag bit (modelled by
         * m_firstWeightUpdateDoneArrayCurrent and
         * m_firstWeightUpdateDoneArrayNext).
         * Additionally, the addition count needs to be
         * stored.
         * The additional four flag bits not dependent on
         * the width of the systolic array are the
         * weight preload mode select flag, the
         * "got first input" flag bit, the the data ready
         * flag bit, and the buffer write done flag bit. */


        return m_width*(getRowPtrBitwidthRequiredMin() +
                        getAdditionCounterBitwidthRequiredMin() + 2UL) +
                        getAdditionCounterBitwidthRequiredMin() + 4UL;
    }

    size_t getWidth() const
    {
        return m_width;
    }

    void resetAdditionCountMaxValue()
    {
        m_additionCountMax = 0;
    }

    void readRow(AccumulatorDatatype* dest,
                    const bool bufferSelectBit,
                    const size_t accumulatorArrayRow,
                    const size_t width)
    {
        auto readAddress{getBufferAddress(bufferSelectBit) +
                                            accumulatorArrayRow*m_width};

        std::copy(readAddress, readAddress + width, dest);
    }

    void paintDiagonalRead(const AccumulatorDatatype* const dest,
                            const size_t destMatrixWidth,
                            const bool bufferSelectBit,
                            const size_t accumulatorArrayBufferDiagonal,
                            const size_t blockHeight,
                            const size_t blockWidth,
                            const AccumulatorDatatype* const unifiedBufferPtr,
                            cv::Mat& memoryAccessMapMat,
                            cv::Mat& memoryAccessMapMatPersistent)
    {
        const auto blockDimensionsOrdered{
                        std::minmax(blockWidth, blockHeight)};

        const size_t diagonalCount{blockHeight + blockWidth - 1};

        const size_t diagonalElements{(accumulatorArrayBufferDiagonal <
                                                blockDimensionsOrdered.second) ?
                                                    std::min(blockDimensionsOrdered.first,
                                                            (accumulatorArrayBufferDiagonal + 1)) :
                                                    (diagonalCount -
                                                            accumulatorArrayBufferDiagonal)};

        const size_t rowStart = std::max(0L, static_cast<ssize_t>(
                                                accumulatorArrayBufferDiagonal - blockWidth + 1));

        const size_t columnStartConst{std::min(accumulatorArrayBufferDiagonal,
                                                                blockWidth - 1UL)};

        const size_t matrixOffset = dest - unifiedBufferPtr;

        for(size_t elementCount{0}; elementCount < diagonalElements;
                                                            ++elementCount)
        {
            if(bufferSelectBit)
            {
                m_buffer1RowReadArray.at(columnStartConst - elementCount) =
                                                    (rowStart + elementCount);
            }

            else
            {
                m_buffer0RowReadArray.at(columnStartConst - elementCount) =
                                                    (rowStart + elementCount);
            }

            const size_t rectangleOriginCoordinateX{
                                                memoryAccessMapElementSize*
                                                ((matrixOffset +
                                                (rowStart +
                                                elementCount)*
                                                destMatrixWidth +
                                                (columnStartConst -
                                                elementCount)) %
                                                memoryAccessMapWidth)};

            const size_t rectangleOriginCoordinateY{
                                                memoryAccessMapElementSize*
                                                ((matrixOffset +
                                                (rowStart +
                                                elementCount)*
                                                destMatrixWidth +
                                                (columnStartConst -
                                                elementCount))/
                                                memoryAccessMapWidth)};

            const cv::Point rectanglePointTopLeft(rectangleOriginCoordinateX,
                                                  rectangleOriginCoordinateY);

            const cv::Point rectanglePointBottomRight(rectangleOriginCoordinateX +
                                                        memoryAccessMapElementSize - 1,
                                                      rectangleOriginCoordinateY +
                                                        memoryAccessMapElementSize - 1);

            cv::rectangle(memoryAccessMapMat,
                          rectanglePointTopLeft,
                          rectanglePointBottomRight,
                          ACCUMULATOR_ARRAY_ELEMENT_COLOR_READ_UNIFIED_BUFFER,
                          cv::FILLED);

            cv::rectangle(memoryAccessMapMatPersistent,
                          rectanglePointTopLeft,
                          rectanglePointBottomRight,
                          ACCUMULATOR_ARRAY_ELEMENT_COLOR_READ_UNIFIED_BUFFER_PERSISTENT,
                          cv::FILLED);
        }
    }

    void readDiagonal(AccumulatorDatatype* dest,
                            const size_t destMatrixWidth,
                            const bool bufferSelectBit,
                            const size_t accumulatorArrayBufferDiagonal,
                            const size_t blockHeight,
                            const size_t blockWidth,
                            size_t& loadCount,
                            size_t& columnStart,
                            size_t& columnEnd)
    {
        const auto blockDimensionsOrdered{
                        std::minmax(blockWidth, blockHeight)};

        const size_t diagonalCount{blockHeight + blockWidth - 1};

        const size_t diagonalElements{(accumulatorArrayBufferDiagonal <
                                                blockDimensionsOrdered.second) ?
                                                    std::min(blockDimensionsOrdered.first,
                                                            (accumulatorArrayBufferDiagonal + 1)) :
                                                    (diagonalCount -
                                                            accumulatorArrayBufferDiagonal)};

        auto readAddress{getBufferAddress(bufferSelectBit)};

        const size_t rowStart = std::max(0L, static_cast<ssize_t>(
                                                accumulatorArrayBufferDiagonal - blockWidth + 1));

        columnStart = std::min(accumulatorArrayBufferDiagonal,
                                                    blockWidth - 1UL);

        const size_t columnStartConst{columnStart};

        columnEnd = columnStart - diagonalElements + 1;

        for(size_t elementCount{0}; elementCount < diagonalElements;
                                                            ++elementCount)
        {

            dest[(rowStart + elementCount)*destMatrixWidth +
                                    (columnStartConst - elementCount)] =
                                                            readAddress[(rowStart + elementCount)*m_width +
                                                                                    (columnStartConst - elementCount)];

            ++loadCount;

#ifdef ACCUMULATOR_ARRAY_DEBUG
            std::cout << "Read element ("
                        << rowStart + elementCount
                        << ", " << columnStart - elementCount
                        << ") from accumulator buffer "
                        << bufferSelectBit << std::endl;
#endif

        }
    }

    bool hasDataReadySignal() const
    {
        return m_dataReadyCurrent;
    }

    bool hasBufferWriteDoneSignal() const
    {
        return m_bufferWriteDoneCurrent;
    }

    void setSystolicArrayStartupMode(
                        const SystolicArrayStartupMode systolicArrayStartupMode)
    {
        m_systolicArrayStartupModeNext = systolicArrayStartupMode;
    }

    void setAdditionCount(const size_t additionCount)
    {
        m_additionCountNext = additionCount;

        if(m_additionCountNext > m_additionCountMax)
        {
            m_additionCountMax = m_additionCountNext;
        }
    }

    void clearGotFirstInputBit()
    {
        m_gotFirstInputNext = false;
    }

    void clearDataReadyBit()
    {
        m_dataReadyNext = false;
    }

    void clearBufferWriteDoneBit()
    {
        m_bufferWriteDoneNext = false;
    }

    void clearFirstUpdateDoneBits()
    {
        for(size_t elementCount{0}; elementCount < m_width;
                                                    ++elementCount)
        {
            m_firstWeightUpdateDoneArrayNext.at(elementCount) = false;
        }
    }

    void resetCounters()
    {
        for(size_t elementCount{0}; elementCount < m_width;
                                                    ++elementCount)
        {
            m_rowPtrArrayNext.at(elementCount) = 0UL;
            m_rowAdditionCountArrayNext.at(elementCount) = 0UL;
            m_writeAddressSelectBitArrayNext.at(elementCount) = false;
        }
    }

    void runIteration()
    {
        for(std::unique_ptr<ProcessingElement<WeightDatatype,
                                                ActivationDatatype,
                                                AccumulatorDatatype>>& pePtr : *m_pePtrVector)
        {
            const size_t column{pePtr->getPosition().x};

            if(pePtr->hasValidSignal())
            {
                m_gotFirstInputNext = true;

                auto writeAddress{getBufferAddress(
                                    m_writeAddressSelectBitArrayCurrent.at(column)) +
                                        m_width*m_rowPtrArrayCurrent.at(column) + column};

                if(m_rowAdditionCountArrayCurrent.at(column) != 0)
                {
                   *writeAddress += pePtr->getSum();
                }

                else
                {
                    *writeAddress = pePtr->getSum();
                }

                m_rowPtrArrayNext.at(column) =
                            m_rowPtrArrayCurrent.at(column) + 1;

                m_validSignalArray.at(column) = true;

            }

            else
            {
                m_validSignalArray.at(column) = false;
            }

            if(pePtr->hasUpdateWeightSignal())
            {
                if(m_systolicArrayStartupModeCurrent ==
                            SystolicArrayStartupMode::WeightsNotPreloaded)
                {
                    if(m_firstWeightUpdateDoneArrayCurrent.at(column) == true)
                    {
                        m_rowPtrArrayNext.at(column) = 0;
                        m_rowAdditionCountArrayNext.at(column) =
                                        m_rowAdditionCountArrayCurrent.at(column) + 1;
                    }

                    else
                    {
                        m_firstWeightUpdateDoneArrayNext.at(column) = true;
                    }
                }

                else
                {
                    m_rowPtrArrayNext.at(column) = 0;
                    m_rowAdditionCountArrayNext.at(column) =
                                    m_rowAdditionCountArrayCurrent.at(column) + 1;
                }

            }

            if((pePtr->hasValidSignal() || m_gotFirstInputCurrent) &&
                                                        (column == 0) &&
                        (m_rowAdditionCountArrayNext.at(column) ==
                                            (m_additionCountCurrent - 1)) &&
                                            (m_rowPtrArrayCurrent.at(column) == 0))
            {
                m_dataReadyNext = true;

#ifdef ACCUMULATOR_ARRAY_DEBUG
                std::cout << "Accumulator array: Buffer: "
                            << m_writeAddressSelectBitArrayCurrent.at(column)
                            << " data ready" << std::endl;
#endif
            }

            if((column == (m_width - 1)) &&
                    (m_rowAdditionCountArrayNext.at(column) ==
                                             m_additionCountCurrent))
            {
                m_bufferWriteDoneNext = true;

#ifdef ACCUMULATOR_ARRAY_DEBUG
                std::cout << "Accumulator array: Buffer "
                            << m_writeAddressSelectBitArrayCurrent.at(column)
                            << " write done" << std::endl;
#endif
            }

            if(m_rowAdditionCountArrayNext.at(column) ==
                                            m_additionCountCurrent)
            {
                m_rowAdditionCountArrayNext.at(column) = 0;

                m_writeAddressSelectBitArrayNext.at(column) =
                        !m_writeAddressSelectBitArrayCurrent.at(column);
            }
        }
    }

    void updateState(cv::Mat& outputFrameMat,
                        const cv::Point& outputFrameOffset)
    {
        for(size_t column{0}; column < m_width; ++column)
        {
            m_rowPtrArrayCurrent.at(column) =
                            m_rowPtrArrayNext.at(column);

            m_rowAdditionCountArrayCurrent.at(column) =
                            m_rowAdditionCountArrayNext.at(column);

            m_writeAddressSelectBitArrayCurrent.at(column) =
                            m_writeAddressSelectBitArrayNext.at(column);

            m_firstWeightUpdateDoneArrayCurrent.at(column) =
                            m_firstWeightUpdateDoneArrayNext.at(column);

            if(m_writeAddressSelectBitArrayCurrent.at(column))
            {
                m_buffer1RowWriteArray.at(column) =
                                m_rowPtrArrayCurrent.at(column);
            }

            else
            {
                m_buffer0RowWriteArray.at(column) =
                                m_rowPtrArrayCurrent.at(column);
            }

        }

        m_additionCountCurrent =
                        m_additionCountNext;

        m_systolicArrayStartupModeCurrent =
                        m_systolicArrayStartupModeNext;

        m_gotFirstInputCurrent =
                        m_gotFirstInputNext;

        m_dataReadyCurrent =
                        m_dataReadyNext;

        m_bufferWriteDoneCurrent =
                        m_bufferWriteDoneNext;

        paintState(outputFrameMat, outputFrameOffset);

    }

    void paintState(cv::Mat& outputFrameMat,
                        const cv::Point& outputFrameOffset)
    {
        for(size_t columnCount{0UL}; columnCount < m_width;
                                                    ++columnCount)
        {
            for(size_t rowCount{0UL}; rowCount < m_bufferHeight; ++rowCount)
            {
                const cv::Point rectanglePointTopLeft(outputFrameOffset.x +
                                                        columnCount*
                                                        accumulatorElementWidth,
                                                        outputFrameOffset.y +
                                                        rowCount*
                                                        accumulatorElementHeight);

                const cv::Point rectanglePointBottomRight(outputFrameOffset.x +
                                                            (columnCount + 1)*
                                                            accumulatorElementWidth,
                                                            outputFrameOffset.y +
                                                            (rowCount + 1)*
                                                            accumulatorElementHeight);

                if(rowCount == m_buffer0RowReadArray.at(columnCount))
                {
                    cv::rectangle(outputFrameMat,
                                    rectanglePointTopLeft,
                                    rectanglePointBottomRight,
                                    ACCUMULATOR_ARRAY_ELEMENT_COLOR_READ_ACCUMULATOR_ARRAY,
                                    cv::FILLED);

                    m_buffer0RowReadArray.at(columnCount) =
                                            std::numeric_limits<size_t>::max();
                }

                else if(rowCount == m_buffer0RowWriteArray.at(columnCount))
                {
                    cv::rectangle(outputFrameMat,
                                    rectanglePointTopLeft,
                                    rectanglePointBottomRight,
                                    ACCUMULATOR_ARRAY_ELEMENT_COLOR_WRITE,
                                    cv::FILLED);

                    m_buffer0RowWriteArray.at(columnCount) =
                                            std::numeric_limits<size_t>::max();
                }

                else
                {
                    cv::rectangle(outputFrameMat,
                                    rectanglePointTopLeft,
                                    rectanglePointBottomRight,
                                    ACCUMULATOR_ARRAY_ELEMENT_COLOR_INACTIVE,
                                    cv::FILLED);
                }

                cv::rectangle(outputFrameMat,
                                rectanglePointTopLeft,
                                rectanglePointBottomRight,
                                cv::Scalar(0, 0, 0));
            }

            const size_t buffer1OffsetY{accumulatorElementHeight*m_bufferHeight};

            for(size_t rowCount{0UL}; rowCount < m_bufferHeight; ++rowCount)
            {
                const cv::Point rectanglePointTopLeft(outputFrameOffset.x +
                                                        columnCount*
                                                        accumulatorElementWidth,
                                                        outputFrameOffset.y +
                                                        buffer1OffsetY +
                                                        rowCount*
                                                        accumulatorElementHeight);

                const cv::Point rectanglePointBottomRight(outputFrameOffset.x +
                                                            (columnCount + 1)*
                                                            accumulatorElementWidth,
                                                            outputFrameOffset.y +
                                                            buffer1OffsetY +
                                                            (rowCount + 1)*
                                                            accumulatorElementHeight);

                if(rowCount == m_buffer1RowReadArray.at(columnCount))
                {
                    cv::rectangle(outputFrameMat,
                                    rectanglePointTopLeft,
                                    rectanglePointBottomRight,
                                    ACCUMULATOR_ARRAY_ELEMENT_COLOR_READ_ACCUMULATOR_ARRAY,
                                    cv::FILLED);

                    m_buffer1RowReadArray.at(columnCount) =
                                            std::numeric_limits<size_t>::max();
                }

                else if(rowCount == m_buffer1RowWriteArray.at(columnCount))
                {
                    cv::rectangle(outputFrameMat,
                                    rectanglePointTopLeft,
                                    rectanglePointBottomRight,
                                    ACCUMULATOR_ARRAY_ELEMENT_COLOR_WRITE,
                                    cv::FILLED);

                    m_buffer1RowWriteArray.at(columnCount) =
                                            std::numeric_limits<size_t>::max();
                }

                else
                {
                    cv::rectangle(outputFrameMat,
                                    rectanglePointTopLeft,
                                    rectanglePointBottomRight,
                                    ACCUMULATOR_ARRAY_ELEMENT_COLOR_INACTIVE,
                                    cv::FILLED);
                }

                cv::rectangle(outputFrameMat,
                                rectanglePointTopLeft,
                                rectanglePointBottomRight,
                                cv::Scalar(0, 0, 0));
            }
        }
    }

    void printElements()
    {
        for(size_t rowCount = 0; rowCount < m_height; rowCount++)
        {
            for(size_t columnCount = 0; columnCount < m_width; columnCount++)
            {
                std::cout << m_dataArray.at(rowCount*m_width + columnCount) << '\t';
            }

            std::cout << '\n';
        }

        std::cout << std::endl;
    }

private:

    typename std::vector<AccumulatorDatatype>::iterator& getBufferAddress(const bool bufferSelectBit)
    {
        return (bufferSelectBit == ::low) ? m_buffer0Address :
                                                m_buffer1Address;
    }

    std::vector<std::unique_ptr<ProcessingElement<WeightDatatype,
                                                    ActivationDatatype,
                                                    AccumulatorDatatype>>>* const m_pePtrVector;
    const size_t m_width;
    const size_t m_height;
    const size_t m_bufferHeight;

    std::vector<AccumulatorDatatype> m_dataArray;

    typename std::vector<AccumulatorDatatype>::iterator m_buffer0Address;
    typename std::vector<AccumulatorDatatype>::iterator m_buffer1Address;

    std::vector<size_t> m_rowPtrArrayCurrent;
    std::vector<size_t> m_rowPtrArrayNext;

    std::vector<size_t> m_rowAdditionCountArrayCurrent;
    std::vector<size_t> m_rowAdditionCountArrayNext;

    std::vector<size_t> m_buffer0RowReadArray;
    std::vector<size_t> m_buffer1RowReadArray;

    std::vector<size_t> m_buffer0RowWriteArray;
    std::vector<size_t> m_buffer1RowWriteArray;

    std::vector<bool> m_writeAddressSelectBitArrayCurrent;
    std::vector<bool> m_writeAddressSelectBitArrayNext;

    std::vector<bool> m_firstWeightUpdateDoneArrayCurrent;
    std::vector<bool> m_firstWeightUpdateDoneArrayNext;

    std::vector<bool> m_validSignalArray;

    size_t m_additionCountCurrent{0UL};
    size_t m_additionCountNext{0UL};

    size_t m_additionCountMax{0UL};

    SystolicArrayStartupMode m_systolicArrayStartupModeCurrent{
                                    SystolicArrayStartupMode::WeightsNotPreloaded};
    SystolicArrayStartupMode m_systolicArrayStartupModeNext{
                                    SystolicArrayStartupMode::WeightsNotPreloaded};

    bool m_gotFirstInputCurrent{false};
    bool m_gotFirstInputNext{false};

    bool m_dataReadyCurrent{false};
    bool m_dataReadyNext{false};

    bool m_bufferWriteDoneCurrent{false};
    bool m_bufferWriteDoneNext{false};

};


#endif
