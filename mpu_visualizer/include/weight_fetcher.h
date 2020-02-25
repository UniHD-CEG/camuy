#ifndef WEIGHT_FIFO_H
#define WEIGHT_FIFO_H

#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstdint>

#include "mpu_visualizer_global_constants.h"
#include "systolic_array.h"

#define WEIGHT_FETCHER_MEMORY_ACCESS_COLOR cv::Scalar(0xFF, 0xFF , 0xFF)
#define WEIGHT_FETCHER_MEMORY_ACCESS_COLOR_PERSISTENT cv::Scalar(0xCC, 0x40, 0x00)


struct WeightUpdateRequest
{

    WeightUpdateRequest(const size_t blockX,
                            const size_t blockY): blockX{blockX},
                                                    blockY{blockY}
    {
    }

    size_t blockX;
    size_t blockY;

    size_t diagonalsUpdated{0UL};
};

template<typename WeightDatatype,
            typename ActivationDatatype,
            typename AccumulatorDatatype> class WeightFetcher
{

public:

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
        return std::ceil(std::log2(static_cast<double>(
                                    m_systolicArrayDiagonals)));
    }

    size_t getWeightUpdateRequestQueueAddressBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(static_cast<double>(
                            m_weightUpdateRequestQueueLengthMax)));
    }

    size_t getMatrixAddressBitwidthRequiredMin(const size_t unifiedBufferSize) const
    {
        return std::ceil(std::log2(static_cast<double>(
                                        unifiedBufferSize)));
    }

    size_t getMatrixWidthBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(static_cast<double>(
                                        m_matrixWidthMax)));
    }

    size_t getMatrixHeightBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(static_cast<double>(
                                        m_matrixHeightMax)));
    }

    size_t getBlockCountXBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(static_cast<double>(
                                        m_blockCountXMax)));
    }

    size_t getBlockCountYBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(static_cast<double>(
                                        m_blockCountYMax)));
    }

    size_t getActiveColumnsBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(static_cast<double>(
                                        m_activeColumnsMax)));
    }

    size_t getIdleRowsBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(static_cast<double>(
                                        m_idleRowsLastBlockMax)));
    }

    size_t getLoadCount() const
    {
        return m_loadCount;
    }

    size_t getMaxConcurrentLoads() const
    {
        return m_concurrentLoadCountMax;
    }

    size_t getMaxConcurrentLoadsPerColumn() const
    {
        return m_concurrentLoadCountPerColumnMax;
    }

    size_t getControlBits(const size_t unifiedBufferSize) const
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
         * m_blockCountXNext), the block count y register
         * (modelled by m_blockCountYCurrent and
         * m_blockCountYNext), the active columns register
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
                        getBlockCountXBitwidthRequiredMin() +
                        getBlockCountYBitwidthRequiredMin() +
                        getDiagonalCountBitwidthRequiredMin()) +
                        getMatrixAddressBitwidthRequiredMin(unifiedBufferSize) +
                        getMatrixWidthBitwidthRequiredMin() +
                        getMatrixHeightBitwidthRequiredMin() +
                        getBlockCountXBitwidthRequiredMin() +
                        getBlockCountYBitwidthRequiredMin() +
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
        m_blockCountXMax = 0UL;
        m_blockCountYMax = 0UL;
        m_activeColumnsMax = 0UL;
        m_idleRowsLastBlockMax = 0UL;
    }

    bool hasBusySignal() const
    {
        return m_busyCurrent;
    }

    size_t getBlockCountX() const
    {
        return m_blockCountXCurrent;
    }

    size_t getBlockCountY() const
    {
        return m_blockCountYCurrent;
    }

    size_t getActiveColumnsLastBlock() const
    {
        return m_activeColumnsLastBlockCurrent;
    }

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

        m_blockCountXNext = std::ceil(static_cast<float>(m_matrixWidthNext)/
                                        static_cast<float>(m_systolicArrayWidth));

        if(m_blockCountXMax < m_blockCountXNext)
        {
            m_blockCountXMax = m_blockCountXNext;
        }

        m_blockCountYNext = std::ceil(static_cast<float>(m_matrixHeightNext)/
                                        static_cast<float>(m_systolicArrayHeight));

        if(m_blockCountYMax < m_blockCountYNext)
        {
            m_blockCountYMax = m_blockCountYNext;
        }

        m_activeColumnsLastBlockNext = m_systolicArrayWidth*(1L - m_blockCountXNext) +
                                                                        m_matrixWidthNext;

        if(m_activeColumnsMax < m_activeColumnsLastBlockNext)
        {
            m_activeColumnsMax = m_activeColumnsLastBlockNext;
        }

        m_idleRowsLastBlockNext = m_blockCountYNext*m_systolicArrayHeight -
                                                                    m_matrixHeightNext;

        if(m_idleRowsLastBlockMax < m_idleRowsLastBlockNext)
        {
            m_idleRowsLastBlockMax = m_idleRowsLastBlockNext;
        }
    }

    void updateWeights(const size_t blockX,
                        const size_t blockY)
    {
        assert(blockX < m_blockCountXCurrent);
        assert(blockY < m_blockCountYCurrent);

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

    void paintMemoryAccesses(cv::Mat& memoryAccessMapMat,
                                cv::Mat& memoryAccessMapMatPersistent,
                                const WeightDatatype* const unifiedBufferPtr) const
    {

        for(const WeightUpdateRequest& weightUpdateRequest :
                                                m_weightUpdateRequestQueue)
        {

            const size_t activeColumns{(weightUpdateRequest.blockX !=
                                                (m_blockCountXCurrent - 1)) ? m_systolicArrayWidth :
                                                                                m_activeColumnsLastBlockCurrent};

           const size_t idleRows{(weightUpdateRequest.blockY !=
                                           (m_blockCountYCurrent - 1)) ? 0UL :
                                                                           m_idleRowsLastBlockNext};

            for(ProcessingElement<WeightDatatype,
                                    ActivationDatatype,
                                    AccumulatorDatatype>* pePtr : m_systolicArrayPtr->getDiagonal(
                                                                            weightUpdateRequest.diagonalsUpdated))
            {
                if((pePtr->getPosition().x < activeColumns) &&
                                (pePtr->getPosition().y >= idleRows))
                {

                    const size_t matrixOffset = m_matrixPtrCurrent -
                                                        unifiedBufferPtr;

                    const size_t rectangleOriginCoordinateX{
                                                    memoryAccessMapElementSize*
                                                    ((matrixOffset +
                                                    (weightUpdateRequest.blockY*
                                                    m_systolicArrayHeight +
                                                    pePtr->getPosition().y -
                                                    idleRows)*
                                                    m_matrixWidthCurrent +
                                                    weightUpdateRequest.blockX*
                                                    m_systolicArrayWidth +
                                                    pePtr->getPosition().x) %
                                                    memoryAccessMapWidth)};

                    const size_t rectangleOriginCoordinateY{
                                                    memoryAccessMapElementSize*
                                                    ((matrixOffset +
                                                    (weightUpdateRequest.blockY*
                                                    m_systolicArrayHeight +
                                                    pePtr->getPosition().y -
                                                    idleRows)*
                                                    m_matrixWidthCurrent +
                                                    weightUpdateRequest.blockX*
                                                    m_systolicArrayWidth +
                                                    pePtr->getPosition().x)/
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
                                    WEIGHT_FETCHER_MEMORY_ACCESS_COLOR,
                                    cv::FILLED);

                    cv::rectangle(memoryAccessMapMatPersistent,
                                    rectanglePointTopLeft,
                                    rectanglePointBottomRight,
                                    WEIGHT_FETCHER_MEMORY_ACCESS_COLOR_PERSISTENT,
                                    cv::FILLED);

                }
            }
        }
    }

    void runIteration()
    {
        size_t concurrentLoadCount{0UL};

        std::vector<size_t> concurrentLoadsPerColumn(m_systolicArrayWidth);

        for(WeightUpdateRequest& weightUpdateRequest :
                                                m_weightUpdateRequestQueue)
        {   

            const size_t activeColumns{(weightUpdateRequest.blockX !=
                                                (m_blockCountXCurrent - 1)) ? m_systolicArrayWidth :
                                                                                m_activeColumnsLastBlockCurrent};

           const size_t idleRows{(weightUpdateRequest.blockY !=
                                           (m_blockCountYCurrent - 1)) ? 0UL :
                                                                           m_idleRowsLastBlockNext};

            for(ProcessingElement<WeightDatatype,
                                    ActivationDatatype,
                                    AccumulatorDatatype>* pePtr : m_systolicArrayPtr->getDiagonal(
                                                                            weightUpdateRequest.diagonalsUpdated))
            {
                if((pePtr->getPosition().x < activeColumns) &&
                                (pePtr->getPosition().y >= idleRows))
                {
                    pePtr->storeWeight(m_matrixPtrCurrent[(weightUpdateRequest.blockY*
                                                                m_systolicArrayHeight +
                                                                pePtr->getPosition().y -
                                                                idleRows)*
                                                                m_matrixWidthCurrent +
                                                                weightUpdateRequest.blockX*
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

    void updateState(cv::Mat& memoryAccessMapMat,
                         cv::Mat& memoryAccessMapMatPersistent,
                         const WeightDatatype* const unifiedBufferPtr)
    {
        m_matrixPtrCurrent = m_matrixPtrNext;

        m_matrixWidthCurrent = m_matrixWidthNext;
        m_matrixHeightCurrent = m_matrixHeightNext;
        m_blockCountXCurrent = m_blockCountXNext;
        m_blockCountYCurrent = m_blockCountYNext;
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

        paintMemoryAccesses(memoryAccessMapMat,
                                memoryAccessMapMatPersistent,
                                unifiedBufferPtr);

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

    size_t m_blockCountXCurrent{0UL};
    size_t m_blockCountXNext{0UL};

    size_t m_blockCountXMax{0UL};

    size_t m_blockCountYCurrent{0UL};
    size_t m_blockCountYNext{0UL};

    size_t m_blockCountYMax{0UL};

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
