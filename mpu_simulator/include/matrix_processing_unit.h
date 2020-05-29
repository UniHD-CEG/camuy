/* Copyright 2019, 2020 Kevin Stehle
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
 * @file        matrix_processing_unit.h
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   MIT License
 */

#ifndef MATRIX_PROCESSING_UNIT_H
#define MATRIX_PROCESSING_UNIT_H

#include <vector>
#include <utility>
#include <exception>
#include <cstring>
#include <cmath>
#include <cstddef>
#include <cassert>

#include <eigen3/Eigen/Dense>

#include "mpu_exception.h"
#include "systolic_data_setup_unit.h"
#include "systolic_array.h"
#include "weight_fetcher.h"
#include "memory_management_unit.h"
#include "mpu_statistics_log_entry.h"

template<typename T> using RMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/**
 * @struct  AccumulatorArrayReadOperation
 * @brief   Struct containing the data required for performing
 *          a read operation from the accumulator array to the
 *          unified buffer: The row and column of the tile
 *          within the result matrix, the accumulator array
 *          double buffer containing the result matrix tile,
 *          and the width and height of the tile.
 */

struct AccumulatorArrayReadOperation
{

    /**
     * @brief                                   AccumulatorArrayReadOperation constructor
     * @param destMatrixRowStart                The row of the result matrix tile within the result matrix
     * @param destMatrixColumnStart             The column of the result matrix tile within the result matrix
     * @param accumulatorArrayBufferSelectBit   The accumulator array double buffer select bit value of the
     *                                          of the buffer containing the result matrix tile
     * @param blockHeight                       The height of the result matrix tile
     * @param blockWidth                        The width of the result matrix tile
     */
    
    AccumulatorArrayReadOperation(const size_t destMatrixRowStart,
                                    const size_t destMatrixColumnStart,
                                    const bool accumulatorArrayBufferSelectBit,
                                    const size_t blockHeight,
                                    const size_t blockWidth): destMatrixRowStart{destMatrixRowStart},
                                                                destMatrixColumnStart{destMatrixColumnStart},
                                                                blockHeight{blockHeight},
                                                                blockWidth{blockWidth},
                                                                blockDiagonals{blockHeight + blockWidth - 1},
                                                                accumulatorArrayBufferSelectBit{accumulatorArrayBufferSelectBit}
    {
    }

    size_t destMatrixRowStart;
    size_t destMatrixColumnStart;
    size_t blockHeight;
    size_t blockWidth;
    size_t blockDiagonals;
    size_t diagonalCoordinate{0UL};

    bool accumulatorArrayBufferSelectBit;
};

/**
 * @class                       MatrixProcessingUnit
 * @brief                       Class containing all of the required MPU submodules and the main control unit (MCU) logic
 * @tparam WeightDatatype       The weight datatype used by the MPU
 * @tparam ActivationDatatype   The activation datatype used by the MPU
 * @tparam AccumulatorDatatype  The partial sum/result datatype used by the MPU
 */

template<typename WeightDatatype,
            typename ActivationDatatype,
            typename AccumulatorDatatype> class MatrixProcessingUnit
{

public:

    
    /**
     * @brief                           MatrixProcessingUnit constructor initializing all MPU submodules
     * @param systolicArrayWidth        The width of the systolic array
     * @param systolicArrayHeight       The height of the systolic array
     * @param activationFifoDepth       The depth of the activation FIFOs connecting the SDSU and the systolic array
     * @param accumulatorArrayHeight    The height of the accumulator array
     * @param unifiedBufferSizeByteMax  The maximum size of the unified buffer
     */
    
    MatrixProcessingUnit(const size_t systolicArrayWidth,
                            const size_t systolicArrayHeight,
                            const size_t activationFifoDepth,
                            const size_t accumulatorArrayHeight,
                            const size_t unifiedBufferSizeByteMax): m_systolicArrayWidth{systolicArrayWidth},
                                                                    m_systolicArrayHeight{systolicArrayHeight},
                                                                    m_systolicArrayDiagonals{m_systolicArrayWidth +
                                                                                            m_systolicArrayHeight - 1UL},
                                                                    m_activationFifoDepth{activationFifoDepth},
                                                                    m_accumulatorArrayHeight{accumulatorArrayHeight},
                                                                    m_accumulatorArrayBufferHeight{m_accumulatorArrayHeight/2UL},
                                                                    m_unifiedBufferSizeByteMax{unifiedBufferSizeByteMax},
                                                                    m_systolicArray(m_systolicArrayWidth,
                                                                                        m_systolicArrayHeight,
                                                                                        m_activationFifoDepth),
                                                                    m_systolicDataSetupUnit(
                                                                                m_systolicArray.getActivationFifoArrayPtr()),
                                                                    m_weightFetcher(&m_systolicArray),
                                                                    m_accumulatorArray(m_systolicArray.getBottomPePtrRowPtr(),
                                                                                                            m_systolicArrayWidth,
                                                                                                            accumulatorArrayHeight),
                                                                    m_memoryManagementUnit(&m_unifiedBuffer,
                                                                                                m_unifiedBufferSizeByteMax)
    {
        std::cout << "Constructed MPU object\n\tWeights size: "
                    << sizeof(WeightDatatype)
                    << " byte\n\tActivations datatype size: "
                    << sizeof(ActivationDatatype)
                    << " byte\n\tResults datatype size: "
                    << sizeof(AccumulatorDatatype)
                    << " byte\n\tSystolic array height: "
                    << m_systolicArrayHeight 
                    << "\n\tSystolic array width: "
                    << m_systolicArrayWidth
                    << "\n\tActivation FIFO depth: "
                    << m_activationFifoDepth
                    << "\n\tAccumulator array height: "
                    << m_accumulatorArrayHeight
                    << "\n\tMax unified buffer size: "
                    << m_unifiedBufferSizeByteMax
                    << " byte" << std::endl;
    }

    size_t getActivationMatrixBlocksYBitwidthMin() const
    {
        return std::ceil(std::log2(m_activationMatrixBlocksYMax));
    }

    size_t getActivationMatrixRowsLastBlockBitwidthMin() const
    {
        return std::ceil(std::log2(m_activationMatrixRowsLastBlockMax));
    }

    size_t getWeightMatrixBlocksXBitwidthMin() const
    {
        return std::ceil(std::log2(m_weightMatrixBlocksXMax));
    }

    size_t getWeightMatrixBlocksYBitwidthMin() const
    {
        return std::ceil(std::log2(m_weightMatrixBlocksYMax));
    }

    size_t getWeightMatrixColumnsLastBlockBitwidthMin() const
    {
        return std::ceil(std::log2(m_weightMatrixColumnsLastBlockMax));
    }

    size_t getSystolicArrayInputCountBitwidthMin() const
    {
        return std::ceil(std::log2(m_systolicArrayInputCountMax));
    }

    size_t getAccumulatorArrayReadOperationBitwidthMin() const
    {

        return std::ceil(std::log2(m_accumulatorArrayBufferHeight*
                                        m_activationMatrixBlocksY)) +
                std::ceil(std::log2(m_systolicArrayWidth*
                                        m_weightMatrixBlocksX)) +
                std::ceil(std::log2(m_accumulatorArrayBufferHeight)) +
                std::ceil(std::log2(m_systolicArrayWidth)) +
                std::ceil(std::log2(getSystolicArrayDiagonals())) + 1UL;
    }

    size_t getControlRegisterBitsMpu() const
    {

        return m_accumulatorArrayReadOperationQueueLengthMax*
                    getAccumulatorArrayReadOperationBitwidthMin() +
                    5UL*getActivationMatrixBlocksYBitwidthMin() +
                    getActivationMatrixRowsLastBlockBitwidthMin() +
                    3UL*getWeightMatrixBlocksXBitwidthMin() +
                    getWeightMatrixBlocksYBitwidthMin() +
                    getWeightMatrixColumnsLastBlockBitwidthMin() +
                    getSystolicArrayInputCountBitwidthMin() + 1UL;
    }

    size_t getControlRegisterBitsTotal() const
    {
        return m_systolicDataSetupUnit.getControlRegisterBits(
                    m_memoryManagementUnit.getMemoryUsageMaxByte()) +
                m_systolicArray.getControlRegisterBitsSystolicArray() +
                m_systolicArray.getControlRegisterBitsActivationFifos() +
                m_weightFetcher.getControlRegisterBits(
                    m_memoryManagementUnit.getMemoryUsageMaxByte()) +
                m_accumulatorArray.getControlRegisterBits() +
                getControlRegisterBitsMpu();
    }

    size_t getDataRegisterBits() const
    {
        return m_systolicArray.getDataRegisterBitsSystolicArray() +
                m_systolicArray.getDataRegisterBitsActivationFifos() +
                m_accumulatorArray.getDataRegisterBits();
    }

    size_t getUnifiedBufferSizeMinByte() const
    {
        return m_memoryManagementUnit.getMemoryUsageMaxByte();
    }

    size_t getUnifiedBufferSizeMinBit() const
    {
        return m_memoryManagementUnit.getMemoryUsageMaxBit();
    }

    void registerLogEntryAvailableCallback(const std::function<void(MpuStatisticsLogEntry&&)>& statisticsLogEntryAvailableCallback)
    {
        m_statisticsLogEntryAvailableCallback =
                        statisticsLogEntryAvailableCallback;
    }

    void resetDataMovementAndFootprintMetrics()
    {
        m_accumulatorArrayReadOperationQueueLengthMax = 0UL;
        m_activationMatrixBlocksYMax = 0UL;
        m_activationMatrixRowsLastBlockMax = 0UL;
        m_weightMatrixBlocksXMax = 0UL;
        m_weightMatrixBlocksYMax = 0UL;
        m_weightMatrixColumnsLastBlockMax = 0UL;
        m_systolicArrayInputCountMax = 0UL;
        m_accumulatorArrayLoadCount = 0UL;
        m_concurrentAccumulatorLoadCountMax = 0UL;
        m_concurrentAccumulatorArrayLoadCountPerColumnMax = 0UL;

        m_systolicDataSetupUnit.resetLoadCount();
        m_systolicDataSetupUnit.resetMaxRegisterValues();
        m_systolicArray.resetExecutionMetrics();
        m_weightFetcher.resetDataMovementCounters();
        m_accumulatorArray.resetAdditionCountMaxValue();
    }

    void resetIterationCounts()
    {
        m_iterationCountTotal = 0UL;
        m_iterationCountStalled = 0UL;
    }

    size_t getSystolicArrayWidth() const
    {
        return m_systolicArrayWidth;
    }

    size_t getSystolicArrayHeight() const
    {
        return m_systolicArrayHeight;
    }

    size_t getSystolicArrayDiagonals() const
    {
        return m_systolicArrayDiagonals;
    }

    size_t getActivationFifoDepth() const
    {
        return m_activationFifoDepth;
    }

    size_t getAccumulatorBufferHeight() const
    {
        return m_accumulatorArrayBufferHeight;
    }

    size_t getUnifiedBufferSizeBytes() const
    {
        return m_unifiedBufferSizeByteMax;
    }

    mpusim::byte* getUnifiedBufferAddress()
    {
        return m_unifiedBuffer.data();
    }

    void setDebugFlag(const bool debugFlag)
    {
        m_debugFlag = debugFlag;
    }

    bool getDebugFlag() const
    {
        return m_debugFlag;
    }

    void setDebugOutputVerboseFlag(const bool debugOutputVerboseFlag)
    {
        m_verboseDebugOutputFlag = debugOutputVerboseFlag;
    }

    bool getDebugOutputVerboseFlag() const
    {
        return m_verboseDebugOutputFlag;
    }

    void setUnifiedBufferDynamicResize(const bool unifiedBufferDynamicResize)
    {
        m_memoryManagementUnit.setUnifiedBufferDynamicResize(
                                        unifiedBufferDynamicResize);
    }

    void resetMemoryManagementUnit()
    {
        m_memoryManagementUnit.reset();
    }

    void loadFromUnifiedBuffer(mpusim::byte* const dest,
                                const mpusim::byte* const src,
                                const size_t size)
    {
        m_memoryManagementUnit.loadFromUnifiedBuffer(
                                        dest, src, size);
    }

    void storeToUnifiedBuffer(mpusim::byte* const dest,
                                const mpusim::byte* const src,
                                const size_t size)
    {
        m_memoryManagementUnit.storeToUnifiedBuffer(
                                        dest, src, size);
    }

    /**
     * @brief                   Function to store weight matrices to the unified buffer
     * @param operationName     The string identifier of the weight matrix to be stored
     * @param weightMatrixPtr   A pointer to the weight matrix to be stored
     * @param rows              The rows of the weight matrix
     * @param columns           The columns of the weight matrix
     */
    
    void storeWeightMatrix(const std::string& operationName,
                                            const WeightDatatype* const weightMatrixPtr,
                                            const size_t rows,
                                            const size_t columns)
    {
        m_memoryManagementUnit.storeWeightMatrixManaged(
                                                operationName,
                                                weightMatrixPtr,
                                                rows,
                                                columns);
    }
    
    /**
     * @brief                       Function to store activation matrices to the unified buffer
     * @param activationMatrixPtr   A pointer to the activation matrix to be stored
     * @param rows                  The rows of the activation matrix
     * @param columns               The columns of the activation matrix
     */

    void storeActivationMatrix(const ActivationDatatype* const activationMatrixPtr,
                                                                    const size_t rows,
                                                                    const size_t columns)
    {
        m_memoryManagementUnit.storeActivationMatrixManaged(
                                                activationMatrixPtr,
                                                rows,
                                                columns);
    }
    
    /**
     * @brief       Function to load result matrices from the unified buffer
     * @param dest  A pointer to which the result matrix will be stored
     * @param size  The size of the result matrix
     */

    void loadResultMatrix(AccumulatorDatatype* const dest,
                                            const size_t size) const
    {
        m_memoryManagementUnit.loadResultMatrixManaged(dest, size);
    }

    void printUnifiedBufferLayout() const
    {
        m_memoryManagementUnit.printMemoryLayout();
    }

    
    /**
     * @brief
     * @param sizeM
     * @param sizeN
     * @param sizeK
     * @param matrixAPtr
     * @param matrixBPtr
     * @param matrixCPtr
     */
    
    void runMultiplication(const size_t sizeM,
                            const size_t sizeN,
                            const size_t sizeK,
                            const ActivationDatatype* const __restrict__ matrixAPtr,
                            const WeightDatatype* const __restrict__ matrixBPtr,
                            AccumulatorDatatype* const __restrict__ matrixCPtr)
    {

        assert((reinterpret_cast<const mpusim::byte* const>(matrixAPtr) >=
                                            &(*m_unifiedBuffer.begin())) &&
                (reinterpret_cast<const mpusim::byte* const>(matrixAPtr  + sizeM*sizeK) <=
                                                            &(*m_unifiedBuffer.end())));

        if((reinterpret_cast<const mpusim::byte* const>(matrixAPtr) <
                                    &(*m_unifiedBuffer.begin())) ||
            (reinterpret_cast<const mpusim::byte* const>(matrixAPtr + sizeM*sizeK) >
                                                            &(*m_unifiedBuffer.end())))
        {
            throw MpuException("MPU matrix multiplication "
                                "matrix A outside MPU address space");
        }

        assert((reinterpret_cast<const mpusim::byte* const>(matrixBPtr) >=
                                            &(*m_unifiedBuffer.begin())) &&
                (reinterpret_cast<const mpusim::byte* const>(matrixBPtr + sizeK*sizeN) <=
                                                            &(*m_unifiedBuffer.end())));

        if((reinterpret_cast<const mpusim::byte* const>(matrixBPtr) <
                                        &(*m_unifiedBuffer.begin())) ||
            (reinterpret_cast<const mpusim::byte* const>(matrixBPtr + sizeK*sizeN) >
                                                            &(*m_unifiedBuffer.end())))
        {
            throw MpuException("MPU matrix multiplication "
                                "matrix B outside MPU address space");
        }

        assert((reinterpret_cast<mpusim::byte* const>(matrixCPtr) >=
                                        &(*m_unifiedBuffer.begin())) &&
                (reinterpret_cast<mpusim::byte* const>(matrixCPtr + sizeM*sizeN) <=
                                                        &(*m_unifiedBuffer.end())));

        if((reinterpret_cast<mpusim::byte* const>(matrixCPtr) <
                                        &(*m_unifiedBuffer.begin())) ||
            (reinterpret_cast<mpusim::byte* const>(matrixCPtr + sizeM*sizeN) >
                                                    &(*m_unifiedBuffer.end())))
        {
            throw MpuException("MPU matrix multiplication result "
                                    "matrix outside MPU address space");
        }

        /* Startup */

        if(m_debugFlag)
        {
            std::cout << "Matrix Processing Unit: Matrix multiplication: "
                        << "Startup\nInput matrix dimensions:\tM: " << sizeM
                        << "\tN: " << sizeN << "\tK: " << sizeK << std::endl;
        }

        m_weightMatrixBlockCoordinateX = 0UL;
        m_weightMatrixBlockCoordinateY = 0UL;

        m_weightFetcherActivationMatrixRowBlockCoordinate = 0UL;
        m_systolicArrayActivationMatrixRowBlockCoordinate = 0UL;

        m_resultMatrixReadInProgressBlockCoordinateX = 0UL;
        m_resultMatrixReadInProgressBlockCoordinateY = 0UL;

        m_resultMatrixReadDoneBlockCoordinateX = 0UL;
        m_resultMatrixReadDoneBlockCoordinateY = 0UL;

        m_accumulatorArrayBufferSelectBit = false;

        m_weightFetcher.setInput(matrixBPtr, sizeN, sizeK);
        m_weightFetcher.clearWeightUpdateRequestQueue();
        m_weightFetcher.updateState();

        m_weightMatrixBlocksX = m_weightFetcher.getBlockCountX();

        if(m_weightMatrixBlocksXMax < m_weightMatrixBlocksX)
        {
            m_weightMatrixBlocksXMax = m_weightMatrixBlocksX;
        }

        m_weightMatrixBlocksY = m_weightFetcher.getBlockCountY();

        if(m_weightMatrixBlocksYMax < m_weightMatrixBlocksY)
        {
            m_weightMatrixBlocksYMax = m_weightMatrixBlocksY;
        }

        m_weightMatrixColumnsLastBlock =
                        m_weightFetcher.getActiveColumnsLastBlock();

        if(m_weightMatrixColumnsLastBlockMax < m_weightMatrixColumnsLastBlock)
        {
            m_weightMatrixColumnsLastBlockMax = m_weightMatrixColumnsLastBlock;
        }

        m_systolicDataSetupUnit.addInputMatrix(matrixAPtr, sizeK,
                                    (m_accumulatorArrayBufferHeight < sizeM) ?
                                                        m_accumulatorArrayBufferHeight : sizeM,
                                                                            m_weightMatrixBlocksX);

        m_activationMatrixBlocksY =
                            std::ceil(static_cast<float>(sizeM)/
                                        static_cast<float>(m_accumulatorArrayBufferHeight));

        if(m_activationMatrixBlocksYMax < m_activationMatrixBlocksY)
        {
            m_activationMatrixBlocksYMax = m_activationMatrixBlocksY;
        }

        m_activationMatrixRowsLastBlock =
                            m_accumulatorArrayBufferHeight*(1L - m_activationMatrixBlocksY) + sizeM;

        if(m_activationMatrixRowsLastBlockMax <
                                m_activationMatrixRowsLastBlock)
        {
            m_activationMatrixRowsLastBlockMax =
                            m_activationMatrixRowsLastBlock;
        }

        m_activationMatrixBlockCoordinateY = 1UL;

        if(m_debugFlag)
        {
            std::cout << "Weight matrix:\nBlock count x: "
                                << m_weightMatrixBlocksX
                                << "\nBlock count y: "
                                << m_weightMatrixBlocksY
                                << "\nActive columns last block column: "
                                << m_weightMatrixColumnsLastBlock << std::endl;

            std::cout << "Activation matrix:\nBlock count y: "
                            << m_activationMatrixBlocksY
                            << "\nActivation matrix rows last block: "
                            << m_activationMatrixRowsLastBlock << std::endl;
        }

        m_weightFetcher.updateWeights(0UL, 0UL);
        m_weightFetcher.updateState();

        m_weightFetcher.runIteration();
        m_weightFetcher.updateState();
        m_weightFetcher.runIteration();
        m_weightFetcher.updateState();

        m_systolicArray.setUpdateWeightsSignal(true);
        m_systolicArray.updateState();

        m_systolicArray.readUpdateWeightSignals();
        m_systolicArray.updateState();

        m_systolicDataSetupUnit.runIteration();
        m_systolicDataSetupUnit.updateState();
        m_systolicDataSetupUnit.runIteration();
        m_systolicDataSetupUnit.updateState();
        m_systolicDataSetupUnit.runIteration();
        m_systolicDataSetupUnit.updateState();
        m_systolicDataSetupUnit.runIteration();
        m_systolicDataSetupUnit.updateState();

        m_accumulatorArray.resetCounters();
        m_accumulatorArray.clearGotFirstInputBit();
        m_accumulatorArray.clearFirstUpdateDoneBits();
        m_accumulatorArray.clearBufferWriteDoneBit();
        m_accumulatorArray.setSystolicArrayStartupMode(
                                SystolicArrayStartupMode::WeightsNotPreloaded);
        m_accumulatorArray.setAdditionCount(m_weightMatrixBlocksY);
        m_accumulatorArray.updateState();

        m_systolicArray.resetIterationCount();

        m_iterationCountTotal += 4UL;
        m_iterationCountStalled += 4UL;

        m_systolicArrayInputCount = 0UL;

        /* Matrix multiplication */
        
        if(!((m_weightMatrixBlocksX == 1UL) &&
                (m_weightMatrixBlocksY == 1UL) &&
                (m_activationMatrixBlocksY == 1UL)))
        {

            do
            {
                if(m_debugFlag && m_verboseDebugOutputFlag)
                {
                    std::cout << "------------------------------------------- "
                                << "Iteration " << m_iterationCountTotal
                                << " ------------------------------------------"
                                << std::endl;
                }

                m_systolicDataSetupUnit.runIteration();
                m_weightFetcher.runIteration();
                m_systolicArray.runIteration();
                m_accumulatorArray.runIteration();

                const size_t weightMatrixOutputRowsWeightUpdateSignal{
                                    (m_systolicArrayActivationMatrixRowBlockCoordinate !=
                                                                        (m_activationMatrixBlocksY - 1)) ?
                                                                        m_accumulatorArrayBufferHeight :
                                                                        m_activationMatrixRowsLastBlock};

                if((m_systolicArrayInputCount ==
                            weightMatrixOutputRowsWeightUpdateSignal) &&
                        (m_systolicArrayActivationMatrixRowBlockCoordinate !=
                                                                m_activationMatrixBlocksY))
                {
                    m_systolicArrayActivationMatrixRowBlockCoordinate =
                                    m_weightFetcherActivationMatrixRowBlockCoordinate;

                    if(m_systolicArrayActivationMatrixRowBlockCoordinate !=
                                                        m_activationMatrixBlocksY)
                    {
                        m_systolicArray.setUpdateWeightsSignal(true);

                        if(m_debugFlag && m_verboseDebugOutputFlag)
                        {
                            std::cout << "Set update weights signal" << std::endl;
                        }

                        m_systolicArrayInputCount = 0;
                    }
                }

                if(!m_systolicDataSetupUnit.hasBusySignal() &&
                                (m_activationMatrixBlockCoordinateY != m_activationMatrixBlocksY))
                {

                    const size_t activationMatrixInputRows{
                                        (m_activationMatrixBlockCoordinateY !=
                                                    (m_activationMatrixBlocksY - 1)) ?
                                                                m_accumulatorArrayBufferHeight :
                                                                m_activationMatrixRowsLastBlock};

                    m_systolicDataSetupUnit.addInputMatrix(matrixAPtr +
                                                            m_activationMatrixBlockCoordinateY*
                                                            m_accumulatorArrayBufferHeight*sizeK,
                                                            sizeK, activationMatrixInputRows,
                                                            m_weightMatrixBlocksX);

                    if(m_debugFlag && m_verboseDebugOutputFlag)
                    {
                        std::cout << "Systolic data setup unit: Added input block "
                                                << m_activationMatrixBlockCoordinateY
                                                << ", rows: " << activationMatrixInputRows
                                                << ", columns: " << sizeK
                                                << ", repetition count: "  << m_weightMatrixBlocksX
                                                << std::endl;
                    }

                    ++m_activationMatrixBlockCoordinateY;

                }

                const size_t weightMatrixOutputRowsWeightUpdate{
                                    (m_weightFetcherActivationMatrixRowBlockCoordinate !=
                                                                    (m_activationMatrixBlocksY - 1)) ?
                                                                    m_accumulatorArrayBufferHeight :
                                                                    m_activationMatrixRowsLastBlock};

                if((m_systolicArrayInputCount ==
                        (weightMatrixOutputRowsWeightUpdate - 1)) &&
                        (m_weightFetcherActivationMatrixRowBlockCoordinate !=
                                                        m_activationMatrixBlocksY))
                {

                    if(m_weightMatrixBlockCoordinateY != (m_weightMatrixBlocksY - 1))
                    {
                        ++m_weightMatrixBlockCoordinateY;
                    }

                    else
                    {
                        m_weightMatrixBlockCoordinateY = 0;

                        if(m_weightMatrixBlockCoordinateX != (m_weightMatrixBlocksX - 1))
                        {

                            ++m_weightMatrixBlockCoordinateX;
                        }

                        else
                        {
                            m_weightMatrixBlockCoordinateX = 0;
                            ++m_weightFetcherActivationMatrixRowBlockCoordinate;
                        }
                    }

                    if(m_weightFetcherActivationMatrixRowBlockCoordinate !=
                                                            m_activationMatrixBlocksY)
                    {
                        m_weightFetcher.updateWeights(m_weightMatrixBlockCoordinateX,
                                                        m_weightMatrixBlockCoordinateY);

                        if(m_debugFlag && m_verboseDebugOutputFlag)
                        {
                            std::cout << "Weight fetcher: Updating to block ("
                                        << m_weightMatrixBlockCoordinateX
                                        << ", "
                                        << m_weightMatrixBlockCoordinateY
                                        << ") of {"
                                        << m_weightMatrixBlocksX - 1
                                        << ", "
                                        << m_weightMatrixBlocksY - 1
                                        << '}' << std::endl;
                        }
                    }
                }

                std::vector<size_t> accumulatorArrayColumnAccessCountVector(m_systolicArrayWidth);
                size_t concurrentAccumulatorArrayLoadCount{0UL};

                for(auto readOperationQueueIterator{m_accumulatorArrayReadOperationQueue.begin()};
                            readOperationQueueIterator < m_accumulatorArrayReadOperationQueue.end();)
                {

                    size_t accumulatorArrayColumnAccessStart{0UL};
                    size_t accumulatorArrayColumnAccessEnd{0UL};

                    loadAccumulatorData(matrixCPtr,
                                            sizeN,
                                            readOperationQueueIterator->destMatrixRowStart,
                                            readOperationQueueIterator->destMatrixColumnStart,
                                            readOperationQueueIterator->accumulatorArrayBufferSelectBit,
                                            readOperationQueueIterator->diagonalCoordinate,
                                            readOperationQueueIterator->blockHeight,
                                            readOperationQueueIterator->blockWidth,
                                            concurrentAccumulatorArrayLoadCount,
                                            accumulatorArrayColumnAccessStart,
                                            accumulatorArrayColumnAccessEnd);

                    for(size_t columnCount{accumulatorArrayColumnAccessEnd};
                                        columnCount <= accumulatorArrayColumnAccessStart; ++columnCount)
                    {
                        ++accumulatorArrayColumnAccessCountVector.at(columnCount);
                    }

                    ++(readOperationQueueIterator->diagonalCoordinate);

                    if(readOperationQueueIterator->diagonalCoordinate ==
                                            readOperationQueueIterator->blockDiagonals)
                    {
                        if(m_debugFlag)
                        {
                            if(m_verboseDebugOutputFlag)
                            {
                                std::cout << "Result matrix read at queue position "
                                            << readOperationQueueIterator -
                                                        m_accumulatorArrayReadOperationQueue.begin()
                                            << " done, coordinate ("
                                            << m_resultMatrixReadDoneBlockCoordinateX
                                            << ", "
                                            << m_resultMatrixReadDoneBlockCoordinateY
                                            << ") of {"
                                            << m_weightMatrixBlocksX - 1
                                            << ", "
                                            << m_activationMatrixBlocksY - 1
                                            << "}, columns: "
                                            << readOperationQueueIterator->blockWidth
                                            << ", rows: "
                                            << readOperationQueueIterator->blockHeight
                                            << std::endl;
                            }

                            else
                            {
                                std::cout << m_resultMatrixReadDoneBlockCoordinateY*
                                                m_weightMatrixBlocksX +
                                                m_resultMatrixReadDoneBlockCoordinateX + 1
                                            << " of "
                                            << m_weightMatrixBlocksX*m_activationMatrixBlocksY
                                            << " output blocks done" << std::endl;
                            }
                        }

                        if(m_resultMatrixReadDoneBlockCoordinateX <
                                                    (m_weightMatrixBlocksX - 1))
                        {
                            ++m_resultMatrixReadDoneBlockCoordinateX;
                        }

                        else
                        {
                            m_resultMatrixReadDoneBlockCoordinateX = 0;
                            ++m_resultMatrixReadDoneBlockCoordinateY;
                        }

                        m_accumulatorArrayReadOperationQueue.erase(readOperationQueueIterator);

                        if(m_debugFlag && m_verboseDebugOutputFlag)
                        {
                            std::cout << "Read operations currently in progress: "
                                        << m_accumulatorArrayReadOperationQueue.size() << std::endl;
                        }
                    }

                    else
                    {
                        ++readOperationQueueIterator;
                    }
                }

                for(const size_t& element : accumulatorArrayColumnAccessCountVector)
                {
                    if(m_concurrentAccumulatorArrayLoadCountPerColumnMax <
                                                                    element)
                    {
                        m_concurrentAccumulatorArrayLoadCountPerColumnMax = element;
                    }
                }

                if(m_concurrentAccumulatorLoadCountMax <
                                    concurrentAccumulatorArrayLoadCount)
                {
                    m_concurrentAccumulatorLoadCountMax =
                                concurrentAccumulatorArrayLoadCount;
                }

                if(m_accumulatorArray.hasDataReadySignal()  &&
                                (m_resultMatrixReadInProgressBlockCoordinateY != m_activationMatrixBlocksY))
                {
                    m_accumulatorArray.clearDataReadyBit();

                    const size_t outputRows{(m_resultMatrixReadInProgressBlockCoordinateY !=
                                                                (m_activationMatrixBlocksY - 1)) ?
                                                                        m_accumulatorArrayBufferHeight :
                                                                        m_activationMatrixRowsLastBlock};

                    const size_t outputColumns{(m_resultMatrixReadInProgressBlockCoordinateX !=
                                                                        (m_weightMatrixBlocksX - 1)) ?
                                                                                    m_systolicArrayWidth :
                                                                                    m_weightMatrixColumnsLastBlock};

                    m_accumulatorArrayReadOperationQueue.emplace_back(
                                                    AccumulatorArrayReadOperation(
                                                                        m_resultMatrixReadInProgressBlockCoordinateY*
                                                                        m_accumulatorArrayBufferHeight,
                                                                        m_systolicArrayWidth*
                                                                        m_resultMatrixReadInProgressBlockCoordinateX,
                                                                        m_accumulatorArrayBufferSelectBit,
                                                                        outputRows,
                                                                        outputColumns));

                    if(m_accumulatorArrayReadOperationQueueLengthMax <
                                        m_accumulatorArrayReadOperationQueue.size())
                    {
                        m_accumulatorArrayReadOperationQueueLengthMax =
                                        m_accumulatorArrayReadOperationQueue.size();
                    }

                    if(m_debugFlag && m_verboseDebugOutputFlag)
                    {
                        std::cout << "Added accumulator array read operation, "
                                                                "queue position: "
                                    << m_accumulatorArrayReadOperationQueue.size() - 1
                                    << ", accumulator array buffer: "
                                    << m_accumulatorArrayBufferSelectBit
                                    << ", block coordinate: ("
                                    << m_resultMatrixReadInProgressBlockCoordinateX
                                    << ", "
                                    << m_resultMatrixReadInProgressBlockCoordinateY
                                    << "), columns: "
                                    << outputColumns
                                    << ", rows: "
                                    << outputRows
                                    << std::endl;
                    }

                    m_accumulatorArrayBufferSelectBit =
                                !m_accumulatorArrayBufferSelectBit;

                    if(m_resultMatrixReadInProgressBlockCoordinateX <
                                                        (m_weightMatrixBlocksX - 1))
                    {
                        ++m_resultMatrixReadInProgressBlockCoordinateX;
                    }

                    else
                    {
                        m_resultMatrixReadInProgressBlockCoordinateX = 0;
                        ++m_resultMatrixReadInProgressBlockCoordinateY;
                    }
                }

                m_systolicDataSetupUnit.updateState();
                m_weightFetcher.updateState();
                m_systolicArray.updateState();
                m_accumulatorArray.updateState();

                ++m_systolicArrayInputCount;
                ++m_iterationCountTotal;

                if(m_systolicArrayInputCountMax <
                                m_systolicArrayInputCount)
                {
                    m_systolicArrayInputCountMax =
                                m_systolicArrayInputCount;
                }
            }

            while(m_resultMatrixReadDoneBlockCoordinateY !=
                                                m_activationMatrixBlocksY);
        
        }
        
        else
        {
            /* TODO: Add simplified execution path for single activation/weight/output block */
        }

        Eigen::Map<const RMatrix<ActivationDatatype>> matrixAEigen(matrixAPtr, sizeM, sizeK);
        Eigen::Map<const RMatrix<WeightDatatype>> matrixBEigen(matrixBPtr, sizeK, sizeN);
        const RMatrix<AccumulatorDatatype> matrixCEigen{matrixAEigen.template cast<AccumulatorDatatype>()*
                                                            matrixBEigen.template cast<AccumulatorDatatype>()};

        bool sanityCheckPassed{true};

        for(size_t rowCount{0}; rowCount < sizeM; ++rowCount)
        {
            for(size_t columnCount{0}; columnCount < sizeN; ++columnCount)
            {
                if(matrixCPtr[rowCount*sizeN + columnCount] !=
                                    matrixCEigen(rowCount, columnCount))
                {
                    if(m_debugFlag && m_verboseDebugOutputFlag)
                    {
                        std::cout << "Systolic array output incorrect at ("
                                    << columnCount << ", " << rowCount
                                    << "): Expected value: "
                                    << matrixCEigen(rowCount, columnCount)
                                    << " actual value: "
                                    << matrixCPtr[rowCount*sizeN + columnCount] << std::endl;
                    }

                    sanityCheckPassed = false;
                }
            }
        }

        if(m_debugFlag)
        {
            std::cout << "Sanity check "
                        << (sanityCheckPassed ? "passed" : "failed") << std::endl;
        }

        assert(sanityCheckPassed);

        if(!sanityCheckPassed)
        {
            throw MpuException("MPU: Matrix multiplication "
                                    "result failed sanity check");
        }

        if(m_debugFlag)
        {
            std::cout << "Matrix processing unit: "
                            "Matrix multiplication: Done\nRequired iterations: "
                        << m_iterationCountTotal
                        << "\nStalled iterations: "
                        << m_iterationCountStalled
                        << "\nUnified buffer I/O:\nSystolic data setup unit:"
                                                        "\n\tLoad operations: "
                        << m_systolicDataSetupUnit.getLoadCount()
                        << "\nWeight fetcher:\n\tLoad operations: "
                        << m_weightFetcher.getLoadCount()
                        << "\n\tMax. concurrent loads per column: "
                        << m_weightFetcher.getConcurrentLoadsPerColumnMax()
                        << "\n\tMax. concurrent loads total: "
                        << m_weightFetcher.getConcurrentLoadsMax()
                        << "\n\tMax. concurrent load operations: "
                        << m_concurrentAccumulatorLoadCountMax
                        << "\n\tMax. concurrent load operations per column: "
                        << m_concurrentAccumulatorArrayLoadCountPerColumnMax
                        << "\nStore operations to unified buffer: "
                        << m_accumulatorArrayLoadCount << std::endl;
        }
    }

    
    /**
     * @brief
     * @param operationName
     */
    
    void runMultiplication(const std::string& operationName)
    {

        const auto weightMatrixDimensions{
                        m_memoryManagementUnit.getWeightMatrixDimensionsManaged(operationName)};

        const auto activationMatrixDimensions{
                        m_memoryManagementUnit.getActivationMatrixDimensionsManaged()};

        assert(weightMatrixDimensions.first ==
                        activationMatrixDimensions.second);

        if(weightMatrixDimensions.first !=
                        activationMatrixDimensions.second)
        {
            throw MpuException("Stored activation matrix "
                                "column count not equal to "
                                "requested weight matrix row count");
        }

        m_memoryManagementUnit.setResultMatrixSizeManaged(
                                        activationMatrixDimensions.first,
                                        weightMatrixDimensions.second);

        runMultiplication(activationMatrixDimensions.first,
                                weightMatrixDimensions.second,
                                activationMatrixDimensions.second,
                                m_memoryManagementUnit.getActivationMatrixPtrManaged(),
                                m_memoryManagementUnit.getWeightMatrixPtrManaged(operationName),
                                m_memoryManagementUnit.getResultMatrixPtrManaged());

        m_statisticsLogEntryAvailableCallback(
                                MpuStatisticsLogEntry{operationName,
                                                        activationMatrixDimensions.first,
                                                        weightMatrixDimensions.second,
                                                        activationMatrixDimensions.second,
                                                        m_systolicArrayHeight,
                                                        m_systolicArrayWidth,
                                                        m_activationFifoDepth,
                                                        m_accumulatorArrayHeight,
                                                        getControlRegisterBitsMpu(),
                                                        m_systolicDataSetupUnit.getControlRegisterBits(
                                                            m_memoryManagementUnit.getMemoryUsageMaxByte()),
                                                        m_systolicArray.getControlRegisterBitsActivationFifos(),
                                                        m_weightFetcher.getControlRegisterBits(
                                                            m_memoryManagementUnit.getMemoryUsageMaxByte()),
                                                        m_systolicArray.getControlRegisterBitsSystolicArray(),
                                                        m_accumulatorArray.getControlRegisterBits(),
                                                        m_systolicArray.getDataRegisterBitsActivationFifos(),
                                                        m_systolicArray.getDataRegisterBitsSystolicArray(),
                                                        m_accumulatorArray.getDataRegisterBits(),
                                                        m_memoryManagementUnit.getMemoryUsageMaxBit(),
                                                        m_systolicArray.getIntraPeDataMovements(),
                                                        m_systolicArray.getInterPeDataMovements(),
                                                        m_systolicDataSetupUnit.getLoadCount(),
                                                        m_weightFetcher.getLoadCount(),
                                                        m_weightFetcher.getConcurrentLoadsMax(),
                                                        m_weightFetcher.getConcurrentLoadsPerColumnMax(),
                                                        m_accumulatorArrayLoadCount,
                                                        m_concurrentAccumulatorLoadCountMax,
                                                        m_concurrentAccumulatorArrayLoadCountPerColumnMax,
                                                        m_iterationCountTotal,
                                                        m_iterationCountStalled,
                                                        m_systolicArray.getMuliplicationsWithWeightZeroCountTotal()});

    }
    
    ~MatrixProcessingUnit()
    {
        std::cout << "MPU object destroyed" << std::endl;
    }

private:

    void loadAccumulatorData(AccumulatorDatatype* const destMatrixPtr,
                                const size_t matrixWidth,
                                const size_t matrixRowStart,
                                const size_t matrixColumnStart,
                                const size_t accumulatorArrayBufferSelectBit,
                                const size_t accumulatorArrayDiagonal,
                                const size_t blockHeight,
                                const size_t blockWidth,
                                size_t& loadCount,
                                size_t& columnStart,
                                size_t& columnEnd)
    {

        assert(blockWidth <= m_systolicArrayWidth);

        if(blockWidth > m_systolicArrayWidth)
        {
            throw MpuException("MPU accumulator array read operation width "
                                "larger than accumulator array buffer width");
        }

        AccumulatorDatatype* const dest{destMatrixPtr +
                                            matrixRowStart*matrixWidth +
                                            matrixColumnStart};

        m_accumulatorArray.readDiagonal(dest,
                                        matrixWidth,
                                        accumulatorArrayBufferSelectBit,
                                        accumulatorArrayDiagonal,
                                        blockHeight,
                                        blockWidth,
                                        loadCount,
                                        columnStart,
                                        columnEnd);

        m_accumulatorArrayLoadCount += loadCount;

    }

    const size_t m_systolicArrayWidth;
    const size_t m_systolicArrayHeight;
    const size_t m_systolicArrayDiagonals;
    const size_t m_activationFifoDepth;
    const size_t m_accumulatorArrayHeight;
    const size_t m_accumulatorArrayBufferHeight;

    const size_t m_unifiedBufferSizeByteMax;

    SystolicArray<WeightDatatype, ActivationDatatype, AccumulatorDatatype> m_systolicArray;
    SystolicDataSetupUnit<ActivationDatatype> m_systolicDataSetupUnit;
    WeightFetcher<WeightDatatype, ActivationDatatype, AccumulatorDatatype> m_weightFetcher;
    AccumulatorArray<WeightDatatype, ActivationDatatype, AccumulatorDatatype> m_accumulatorArray;

    std::vector<mpusim::byte> m_unifiedBuffer;

    MemoryManagementUnit<WeightDatatype, ActivationDatatype, AccumulatorDatatype> m_memoryManagementUnit;

    std::vector<AccumulatorArrayReadOperation> m_accumulatorArrayReadOperationQueue;

    std::function<void(MpuStatisticsLogEntry&&)> m_statisticsLogEntryAvailableCallback;

    size_t m_accumulatorArrayReadOperationQueueLengthMax{0UL};

    size_t m_activationMatrixBlocksY{0UL};

    size_t m_activationMatrixBlocksYMax{0UL};

    size_t m_activationMatrixBlockCoordinateY{0UL};

    size_t m_activationMatrixRowsLastBlock{0UL};

    size_t m_activationMatrixRowsLastBlockMax{0UL};

    size_t m_weightMatrixBlocksX{0UL};
    size_t m_weightMatrixBlocksY{0UL};

    size_t m_weightMatrixBlocksXMax{0UL};
    size_t m_weightMatrixBlocksYMax{0UL};

    size_t m_weightMatrixBlockCoordinateX{0UL};
    size_t m_weightMatrixBlockCoordinateY{0UL};

    size_t m_weightMatrixColumnsLastBlock{0UL};

    size_t m_weightMatrixColumnsLastBlockMax{0UL};

    size_t m_weightFetcherActivationMatrixRowBlockCoordinate{0UL};

    size_t m_systolicArrayActivationMatrixRowBlockCoordinate{0UL};

    size_t m_resultMatrixReadInProgressBlockCoordinateX{0UL};
    size_t m_resultMatrixReadInProgressBlockCoordinateY{0UL};

    size_t m_resultMatrixReadDoneBlockCoordinateX{0UL};
    size_t m_resultMatrixReadDoneBlockCoordinateY{0UL};

    size_t m_systolicArrayInputCount{0UL};

    size_t m_systolicArrayInputCountMax{0UL};

    size_t m_iterationCountTotal{0UL};
    size_t m_iterationCountStalled{0UL};

    size_t m_accumulatorArrayLoadCount{0UL};

    size_t m_concurrentAccumulatorLoadCountMax{0UL};
    size_t m_concurrentAccumulatorArrayLoadCountPerColumnMax{0UL};

    bool m_accumulatorArrayBufferSelectBit{false};

    bool m_debugFlag{false};
    bool m_verboseDebugOutputFlag{false};

};


#endif
