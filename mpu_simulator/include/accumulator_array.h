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
 * @file        accumulator_array.h
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   GNU Public License version 3 (GPLv3)
 */

#ifndef ACCUMULATOR_ARRAY_H
#define ACCUMULATOR_ARRAY_H

#include <vector>
#include <memory>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <climits>

#include "processing_element.h"

//#define ACCUMULATOR_ARRAY_DEBUG ACCUMULATOR_ARRAY_DEBUG

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

/**
 * @class                       AccumulatorArray
 * @brief                       Class template emulating the accumulator array of the MPU
 * @details                     Class template emulating the accumulator array of the MPU.
 *                              This functional unit is used to accumulate the partial sums
 *                              produced by the systolic array. The partial sums are
 *                              stored in dedicated memory, the size of which is defined
 *                              by the width of the systolic array systolicArrayWidth and
 *                              the height of the accumulator array accumulatorArrayHeight.
 *                              The total accumulator array memory size is given by
 *                              systolicArrayWidth*accumulatorArrayHeight*sizeof(AccumulatorDatatype).
 *                              The accumulator array uses double buffering, and produces
 *                              output matrix tiles with a maximum height of accumulatorArrayHeight/2
 *                              and a width of systolicArrayWidth. These tiles are stored in
 *                              the memory segment in row major order. When a tile is finished,
 *                              the accumulator array sets the data ready, represented by the
 *                              members dataReadyCurrent and dataReadyNext, to high, which signals
 *                              the MCU that a tile can be read from accumulator array memory.
 *                              This is currently done using the readDiagonal method, to ensure
 *                              that tiles of all heights, down to a height of 1, can be retrieved
 *                              from memory without stalling the systolic array.
 *                              The module has two startup modes: In WeightsPreloaded mode, the
 *                              first weight tile is assumed to already be stored into the
 *                              PE weight registers before the start of the active systolic array
 *                              and accumulator array operation. In WeightsNotPreloaded mode, the
 *                              first wave of updates is expected to occur during active systolic array
 *                              and accumulator array operation. The two operation modes have to be
 *                              accounted for, as they require different partial sum addressing behavior.
 *                              This is because the order of update weight and valid signals arriving
 *                              at the accumulator array differs between them.
 * @todo                        Remove no longer required WeightsPreloaded operation mode
 * @tparam WeightDatatype       The weight datatype used by the MPU
 * @tparam ActivationDatatype   The activation datatype used by the MPU
 * @tparam AccumulatorDatatype  The accumulator datatype used by the MPU
 */

template<typename WeightDatatype,
            typename ActivationDatatype,
            typename AccumulatorDatatype> class AccumulatorArray
{

public:

    /**
     * @brief                           AccumulatorArray constructor. Allocates memory representing
     *                                  the accumulator array memory segment and control registers.
     * @tparam WeightDatatype           The weight datatype used by the MPU
     * @tparam ActivationDatatype       The activation datatype used by the MPU
     * @tparam AccumulatorDatatype      The accumulator datatype used by the MPU
     * @param pePtrVector               A pointer to a std::vector of std::unique_ptr objects pointing
     *                                  to ProcessingElement subclass objects representing the PEs
     *                                  at the lower edge of the systolic array
     * @param systolicArrayWidth        The width of the systolic array
     * @param accumulatorArrayHeight    The desired height of the accumulator array
     */
    
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
                                                                                            m_writeAddressSelectBitArrayCurrent(m_width),
                                                                                            m_writeAddressSelectBitArrayNext(m_width),
                                                                                            m_firstWeightUpdateDoneArrayCurrent(m_width),
                                                                                            m_firstWeightUpdateDoneArrayNext(m_width)
    {
    }

    size_t getRowPtrBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(m_bufferHeight));
    }

    size_t getAdditionCounterBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(m_additionCountMax));
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

    /**
     * @brief
     * @param dest
     * @param destMatrixWidth
     * @param bufferSelectBit
     * @param accumulatorArrayBufferDiagonal
     * @param blockHeight
     * @param blockWidth
     * @param loadCount
     * @param columnStart
     * @param columnEnd
     */
    
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

    void updateState()
    {
        for(size_t elementCount{0}; elementCount < m_width;
                                                    ++elementCount)
        {
            m_rowPtrArrayCurrent.at(elementCount) =
                            m_rowPtrArrayNext.at(elementCount);

            m_rowAdditionCountArrayCurrent.at(elementCount) =
                            m_rowAdditionCountArrayNext.at(elementCount);

            m_writeAddressSelectBitArrayCurrent.at(elementCount) =
                            m_writeAddressSelectBitArrayNext.at(elementCount);

            m_firstWeightUpdateDoneArrayCurrent.at(elementCount) =
                            m_firstWeightUpdateDoneArrayNext.at(elementCount);
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

    std::vector<bool> m_writeAddressSelectBitArrayCurrent;
    std::vector<bool> m_writeAddressSelectBitArrayNext;

    std::vector<bool> m_firstWeightUpdateDoneArrayCurrent;
    std::vector<bool> m_firstWeightUpdateDoneArrayNext;

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
