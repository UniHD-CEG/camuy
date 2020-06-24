/* Copyright (c) 2020 Computing Systems Group
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
 * @file        weight_fetcher.h
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   MIT License
 */

#ifndef WEIGHT_FIFO_H
#define WEIGHT_FIFO_H

#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstdint>

#include "systolic_array.h"

/**
 * @struct  WeightUpdateRequest
 * @brief   Struct representing the data required for a
 *          weight update request: The x and y coordinate
 *          of the block to be stored to the weight registers
 *          of the systolic array PEs, as well as the
 *          systolic array PE diagonals already updates
 */

struct WeightUpdateRequest
{

    WeightUpdateRequest(const size_t blockCoordinateX,
                            const size_t blockCoordinateY): blockCoordinateX{blockCoordinateX},
                                                                blockCoordinateY{blockCoordinateY}
    {
    }

    size_t blockCoordinateX;
    size_t blockCoordinateY;

    size_t diagonalsUpdated{0UL};
};

/**
 * @class                       WeightFetcher
 * @brief                       Functional unit responsible for storing weight matrix tiles
 *                              to the weight registers of the systolic array PEs
 * @tparam WeightDatatype       The weight datatype of the MPU
 * @tparam ActivationDatatype   The activation datatype of the MPU
 * @tparam AccumulatorDatatype  The accumulator datatype of the MPU
 */

template<typename WeightDatatype,
            typename ActivationDatatype,
            typename AccumulatorDatatype> class WeightFetcher
{

public:

    /**
     * @brief
     * @param systolicArrayPtr
     */
    
    WeightFetcher(SystolicArray<WeightDatatype,
                                    ActivationDatatype,
                                    AccumulatorDatatype>* const systolicArrayPtr):
                                                            m_systolicArrayPtr{systolicArrayPtr},
                                                            m_systolicArrayWidth{m_systolicArrayPtr->getWidth()},
                                                            m_systolicArrayHeight{m_systolicArrayPtr->getHeight()},
                                                            m_systolicArrayDiagonals{m_systolicArrayWidth +
                                                                                            m_systolicArrayHeight - 1}
    {
    } 

    size_t getDiagonalCountBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(m_systolicArrayDiagonals));
    }

    size_t getWeightUpdateRequestQueueAddressBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(m_weightUpdateRequestQueueLengthMax));
    }

    size_t getMatrixAddressBitwidthRequiredMin(const size_t unifiedBufferSize) const
    {
        return std::ceil(std::log2(unifiedBufferSize));
    }

    size_t getMatrixWidthBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(m_matrixWidthMax));
    }

    size_t getMatrixHeightBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(m_matrixHeightMax));
    }

    size_t getBlocksXBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(m_blocksXMax));
    }

    size_t getBlocksYBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(m_blocksYMax));
    }

    size_t getActiveColumnsBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(m_activeColumnsMax));
    }

    size_t getIdleRowsBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(m_idleRowsLastBlockMax));
    }

    size_t getLoadCount() const
    {
        return m_loadCount;
    }

    size_t getConcurrentLoadsMax() const
    {
        return m_concurrentLoadCountMax;
    }

    size_t getConcurrentLoadsPerColumnMax() const
    {
        return m_concurrentLoadCountPerColumnMax;
    }

    size_t getControlRegisterBits(const size_t unifiedBufferSize) const
    {
        /* To calculate the number of control bits
         * dependent on the maximum update request
         * queue length required  for the weight fetcher
         * to perform all the  updates necessary for all
         * matrix multiplication  operations performed
         * since the maximum register values were last
         * reset using  resetMaxRegisterValues(), we need
         * to  multiply the maximum queue length with the
         * bits required for the registers used to
         * store each queue element. These registers are
         * the block  count x register (modelled by
         * blockX),  the block count y register (modelled
         * by blockY), and the updated diagonals counter
         * (modelled by  diagonalsUpdated).
         * Additional to these registers required for each
         * element in the update request queue, there exist
         * a number of registers not dependent on the maximum
         * update request queue length. These are the matrix
         * address register (modelled by m_matrixPtrCurrent
         * and m_matrixPtrNext), the matrix width register
         * (modelled by m_matrixWidthCurrent and
         * m_matrixWidthNext), the matrix height register
         * (modelled by m_matrixHeightCurrent and
         * m_matrixHeightNext), the block count x register
         * (modelled by m_blockCountXCurrent and
         * m_blocksXNext), the block count y register
         * (modelled by m_blockCountYCurrent and
         * m_blocksYNext), the active columns register
         * (modelled by m_activeColumnsCurrent and
         * m_activeColumnsNext), the idle row count
         * register (modelled by m_idleRowsCurrent and
         * m_idleRowsNext), and the busy flag bit
         * (modelled by m_busyCurrent). As the state of the
         * busy flag bit only changes in the updateState()
         * function, no corresponding next signal register
         * was used to model it, as its next state does not
         * need to be stored in between state updates.
         */

        return m_weightUpdateRequestQueueLengthMax*(
                        getBlocksXBitwidthRequiredMin() +
                        getBlocksYBitwidthRequiredMin() +
                        getDiagonalCountBitwidthRequiredMin()) +
                        getMatrixAddressBitwidthRequiredMin(unifiedBufferSize) +
                        getMatrixWidthBitwidthRequiredMin() +
                        getMatrixHeightBitwidthRequiredMin() +
                        getBlocksXBitwidthRequiredMin() +
                        getBlocksYBitwidthRequiredMin() +
                        getActiveColumnsBitwidthRequiredMin() +
                        getIdleRowsBitwidthRequiredMin() + 1UL;

    }

    void resetDataMovementCounters()
    {
        m_loadCount = 0UL;
        m_concurrentLoadCountMax = 0UL;
        m_concurrentLoadCountPerColumnMax = 0UL;
    }

    void resetMaxRegisterValues()
    {
        m_weightUpdateRequestQueueLengthMax = 0UL;
        m_matrixWidthMax = 0UL;
        m_matrixHeightMax = 0UL;
        m_blocksXMax = 0UL;
        m_blocksYMax = 0UL;
        m_activeColumnsMax = 0UL;
        m_idleRowsLastBlockMax = 0UL;
    }

    bool hasBusySignal() const
    {
        return m_busyCurrent;
    }

    size_t getBlockCountX() const
    {
        return m_blocksXCurrent;
    }

    size_t getBlockCountY() const
    {
        return m_blocksYCurrent;
    }

    size_t getActiveColumnsLastBlock() const
    {
        return m_activeColumnsLastBlockCurrent;
    }
    
    /**
     * @brief
     * @param weightArrayPtr
     * @param width
     * @param height
     */
    
    void setInput(const WeightDatatype* const weightArrayPtr,
                                          const size_t width,
                                          const size_t height)
    {
        m_matrixPtrNext = weightArrayPtr;

        m_matrixWidthNext = width;

        if(m_matrixWidthMax < m_matrixWidthNext)
        {
            m_matrixWidthMax = m_matrixWidthNext;
        }

        m_matrixHeightNext = height;

        if(m_matrixHeightMax < m_matrixHeightNext)
        {
            m_matrixHeightMax = m_matrixHeightNext;
        }

        m_blocksXNext = std::ceil(static_cast<float>(m_matrixWidthNext)/
                                        static_cast<float>(m_systolicArrayWidth));

        if(m_blocksXMax < m_blocksXNext)
        {
            m_blocksXMax = m_blocksXNext;
        }

        m_blocksYNext = std::ceil(static_cast<float>(m_matrixHeightNext)/
                                        static_cast<float>(m_systolicArrayHeight));

        if(m_blocksYMax < m_blocksYNext)
        {
            m_blocksYMax = m_blocksYNext;
        }

        m_activeColumnsLastBlockNext = m_systolicArrayWidth*(1L - m_blocksXNext) +
                                                                        m_matrixWidthNext;

        if(m_activeColumnsMax < m_activeColumnsLastBlockNext)
        {
            m_activeColumnsMax = m_activeColumnsLastBlockNext;
        }

        m_idleRowsLastBlockNext = m_blocksYNext*m_systolicArrayHeight -
                                                                    m_matrixHeightNext;

        if(m_idleRowsLastBlockMax < m_idleRowsLastBlockNext)
        {
            m_idleRowsLastBlockMax = m_idleRowsLastBlockNext;
        }
    }
    
    /**
     * @brief           
     * @param blockX    
     * @param blockY    
     */

    void updateWeights(const size_t blockX,
                        const size_t blockY)
    {
        assert(blockX < m_blocksXCurrent);
        assert(blockY < m_blocksYCurrent);

        m_weightUpdateRequestQueue.emplace_back(
                                    WeightUpdateRequest(blockX,
                                                            blockY));

        if(m_weightUpdateRequestQueueLengthMax <
                            m_weightUpdateRequestQueue.size())
        {
            m_weightUpdateRequestQueueLengthMax =
                            m_weightUpdateRequestQueue.size();
        }
    }

    void clearWeightUpdateRequestQueue()
    {
        m_clearWeightUpdateRequestQueueNext = true;
    }
    
    /**
     * @brief
     */

    void runIteration()
    {
        size_t concurrentLoadCount{0UL};

        std::vector<size_t> concurrentLoadsPerColumn(m_systolicArrayWidth);

        for(WeightUpdateRequest& weightUpdateRequest :
                                                m_weightUpdateRequestQueue)
        {   

            const size_t activeColumns{(weightUpdateRequest.blockCoordinateX !=
                                                            (m_blocksXCurrent - 1)) ? m_systolicArrayWidth :
                                                                                m_activeColumnsLastBlockCurrent};

           const size_t idleRows{(weightUpdateRequest.blockCoordinateY !=
                                                            (m_blocksYCurrent - 1)) ? 0UL :
                                                                           m_idleRowsLastBlockNext};

            for(ProcessingElement<WeightDatatype,
                                    ActivationDatatype,
                                    AccumulatorDatatype>* pePtr : m_systolicArrayPtr->getDiagonal(
                                                                            weightUpdateRequest.diagonalsUpdated))
            {
                if((pePtr->getPosition().x < activeColumns) &&
                                (pePtr->getPosition().y >= idleRows))
                {
                    pePtr->storeWeight(m_matrixPtrCurrent[(weightUpdateRequest.blockCoordinateY*
                                                                m_systolicArrayHeight +
                                                                pePtr->getPosition().y -
                                                                idleRows)*
                                                                m_matrixWidthCurrent +
                                                                weightUpdateRequest.blockCoordinateX*
                                                                m_systolicArrayWidth +
                                                                pePtr->getPosition().x]);

                    ++m_loadCount;
                    ++concurrentLoadCount;
                    ++concurrentLoadsPerColumn.at(pePtr->getPosition().x);

                }

                else
                {
                    pePtr->storeWeight(WeightDatatype(0));
                }
            }


            weightUpdateRequest.diagonalsUpdated++;

        }

        if(m_concurrentLoadCountMax < concurrentLoadCount)
        {
            m_concurrentLoadCountMax = concurrentLoadCount;
        }

        for(const size_t& columnConcurrentLoadCount :
                                    concurrentLoadsPerColumn)
        {
            if(m_concurrentLoadCountPerColumnMax <
                                    columnConcurrentLoadCount)
            {
                m_concurrentLoadCountPerColumnMax = columnConcurrentLoadCount;
            }
        }

        size_t count{0UL};

        for(auto it{m_weightUpdateRequestQueue.begin()}; it < m_weightUpdateRequestQueue.end(); ++it)
        {
            if(it->diagonalsUpdated == m_systolicArrayDiagonals)
            {
                m_weightUpdateRequestQueue.erase(it);
            }

            ++count;
        }
    }

    /**
     * @brief
     */
    
    void updateState()
    {
        m_matrixPtrCurrent = m_matrixPtrNext;

        m_matrixWidthCurrent = m_matrixWidthNext;
        m_matrixHeightCurrent = m_matrixHeightNext;
        m_blocksXCurrent = m_blocksXNext;
        m_blocksYCurrent = m_blocksYNext;
        m_activeColumnsLastBlockCurrent = m_activeColumnsLastBlockNext;
        m_idleRowsLastBlockCurrent = m_idleRowsLastBlockNext;

        m_busyCurrent =
            (m_weightUpdateRequestQueue.size() > 0UL) ? true :
                                                        false;

        if(m_clearWeightUpdateRequestQueueNext)
        {
            m_weightUpdateRequestQueue.clear();
        }

        m_clearWeightUpdateRequestQueueNext = false;

    }

private:

    SystolicArray<WeightDatatype,
                        ActivationDatatype,
                        AccumulatorDatatype>* const m_systolicArrayPtr;

    const size_t m_systolicArrayWidth;
    const size_t m_systolicArrayHeight;
    const size_t m_systolicArrayDiagonals;

    std::vector<WeightUpdateRequest> m_weightUpdateRequestQueue;

    size_t m_weightUpdateRequestQueueLengthMax{0UL};

    const WeightDatatype* m_matrixPtrCurrent{nullptr};
    const WeightDatatype* m_matrixPtrNext{nullptr};

    size_t m_matrixWidthCurrent{0UL};
    size_t m_matrixWidthNext{0UL};

    size_t m_matrixWidthMax{0UL};

    size_t m_matrixHeightCurrent{0UL};
    size_t m_matrixHeightNext{0UL};

    size_t m_matrixHeightMax{0UL};

    size_t m_blocksXCurrent{0UL};
    size_t m_blocksXNext{0UL};

    size_t m_blocksXMax{0UL};

    size_t m_blocksYCurrent{0UL};
    size_t m_blocksYNext{0UL};

    size_t m_blocksYMax{0UL};

    size_t m_activeColumnsLastBlockCurrent{0UL};
    size_t m_activeColumnsLastBlockNext{0UL};

    size_t m_activeColumnsMax{0UL};

    size_t m_idleRowsLastBlockCurrent{0UL};
    size_t m_idleRowsLastBlockNext{0UL};

    size_t m_idleRowsLastBlockMax{0UL};

    size_t m_loadCount{0UL};
    size_t m_concurrentLoadCountMax{0UL};
    size_t m_concurrentLoadCountPerColumnMax{0UL};

    bool m_busyCurrent{false};

    bool m_clearWeightUpdateRequestQueueNext{false};

};

#endif
