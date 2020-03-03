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
 * @file        matrix_processing_unit.h
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   MIT License
 */

#ifndef MATRIX_PROCESSING_UNIT_H
#define MATRIX_PROCESSING_UNIT_H

#include <exception>
#include <vector>
#include <functional>
#include <cstring>
#include <cstddef>
#include <cassert>

#include <eigen3/Eigen/Dense>

#include <opencv2/opencv.hpp>

#include "mpu_visualizer_global_constants.h"
#include "mpu_exception.h"
#include "systolic_data_setup_unit.h"
#include "systolic_array.h"
#include "weight_fetcher.h"
#include "memory_management_unit.h"

#define MATRIX_BLOCK_ACCESS_SCHEMATIC_COLOR_MATRIX_A cv::Scalar(0xD3, 0x00, 0x94)
#define MATRIX_BLOCK_ACCESS_SCHEMATIC_COLOR_MATRIX_B cv::Scalar(0x73, 0x9E, 0x00)
#define MATRIX_BLOCK_ACCESS_SCHEMATIC_COLOR_MATRIX_C cv::Scalar(0x00, 0x9F, 0xE6)

#define MATRIX_BLOCK_ACCESS_SCHEMATIC_COLOR_BLOCK_INACTIVE cv::Scalar(0xFF, 0xFF, 0xFF)
#define MATRIX_BLOCK_ACCESS_SCHEMATIC_COLOR_BLOCK_ACTIVE cv::Scalar(0x00, 0x00, 0x00)

#define MATRIX_BLOCK_ACCESS_SCHEMATIC_COLOR_INACTIVE_ROWS_COLUMNS cv::Scalar(0xA9, 0xA9, 0xA9)

constexpr int clockPeriodSlowMs{1000};
constexpr int clockPeriodFastMs{1};

constexpr int videoWriterFramerate{60};

constexpr size_t visualizationMatrixSizeM{200UL};
constexpr size_t visualizationMatrixSizeN{180UL};
constexpr size_t visualizationMatrixSizeK{240UL};

constexpr size_t outputFrameMatWidth{1920UL};
constexpr size_t outputFrameMatHeight{1080UL};

constexpr size_t mpuSchematicOutputFrameMatOffsetXMemoryAccessMap{140UL};
constexpr size_t mpuSchematicOutputFrameMatOffsetYMemoryAccessMap{200UL};

constexpr size_t mpuSchematicOutputFrameMatOffsetXSystolicArray{1000UL};
constexpr size_t mpuSchematicOutputFrameMatOffsetYSystolicArray{270UL};

constexpr size_t mpuSchematicOutputFrameMatOffsetXAccumulatorArray{1090UL};
constexpr size_t mpuSchematicOutputFrameMatOffsetYAccumulatorArray{702UL};

constexpr size_t matrixBlockAccessSchematicElementSize{2UL};

constexpr size_t matrixBlockAccessSchematicOutputFrameMatOffsetXMatrixA{500UL};
constexpr size_t matrixBlockAccessSchematicOutputFrameMatOffsetYMatrixA{990UL -
                                                                        visualizationMatrixSizeM*
                                                                        matrixBlockAccessSchematicElementSize};

constexpr size_t matrixBlockAccessSchematicOutputFrameMatOffsetXMatrixB{
                                                    matrixBlockAccessSchematicOutputFrameMatOffsetXMatrixA +
                                                    visualizationMatrixSizeK*
                                                    matrixBlockAccessSchematicElementSize + 50UL};
constexpr size_t matrixBlockAccessSchematicOutputFrameMatOffsetYMatrixB{
                                                    matrixBlockAccessSchematicOutputFrameMatOffsetYMatrixA -
                                                    (visualizationMatrixSizeN*
                                                    matrixBlockAccessSchematicElementSize + 165UL)};
constexpr size_t matrixBlockAccessSchematicOutputFrameMatOffsetXMatrixC{
                                            matrixBlockAccessSchematicOutputFrameMatOffsetXMatrixB};
constexpr size_t matrixBlockAccessSchematicOutputFrameMatOffsetYMatrixC{
                                            matrixBlockAccessSchematicOutputFrameMatOffsetYMatrixA};


template<typename T> using RMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct AccumulatorArrayReadOperation
{

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
    size_t diagonalsRead{0UL};

    bool accumulatorArrayBufferSelectBit;
};

void copyMemoryAccessMapMatToOutputFrameMatAndPaintBoarder(cv::Mat& memoryAccessMapMat,
                                                                cv::Mat& outputFrameMat)
{
    memoryAccessMapMat.copyTo(outputFrameMat(cv::Rect(
                                    mpuSchematicOutputFrameMatOffsetXMemoryAccessMap,
                                    mpuSchematicOutputFrameMatOffsetYMemoryAccessMap,
                                    memoryAccessMapWidth*
                                    memoryAccessMapElementSize,
                                    memoryAccessMapHeight*
                                    memoryAccessMapElementSize)));

    cv::rectangle(outputFrameMat,
                    cv::Point(mpuSchematicOutputFrameMatOffsetXMemoryAccessMap - 1,
                                mpuSchematicOutputFrameMatOffsetYMemoryAccessMap - 1),
                    cv::Point(mpuSchematicOutputFrameMatOffsetXMemoryAccessMap +
                                memoryAccessMapWidth*
                                memoryAccessMapElementSize,
                                mpuSchematicOutputFrameMatOffsetYMemoryAccessMap +
                                memoryAccessMapHeight*
                                memoryAccessMapElementSize),
                    cv::Scalar(0x00, 0x00 , 0x00));
}

void paintMatrixBlockAccessSchematic(const size_t systolicArrayWidth,
                                            const size_t systolicArrayHeight,
                                            const size_t accumulatorArrayHeight,
                                            const size_t activationMatrixBlocksY,
                                            const size_t activationMatrixBlockCoordinateY,
                                            const size_t weightMatrixBlocksX,
                                            const size_t weightMatrixBlocksY,
                                            const size_t weightMatrixBlockCoordinateX,
                                            const size_t weightMatrixBlockCoordinateY,
                                            const size_t resultMatrixBlockCoordinateX,
                                            const size_t resultMatrixBlockCoordinateY,
                                            cv::Mat& matrixABlockAccessSchematicMat,
                                            cv::Mat& matrixBBlockAccessSchematicMat,
                                            cv::Mat& matrixCBlockAccessSchematicMat)
{

    for(size_t activationMatrixBlockCount{0UL};
                activationMatrixBlockCount < activationMatrixBlocksY;
                                                ++activationMatrixBlockCount)
    {
        const cv::Point activationMatrixBlockPointTopLeft(0,
                                        matrixBlockAccessSchematicElementSize*
                                        accumulatorArrayHeight*
                                        activationMatrixBlockCount);

        const cv::Point activationMatrixBlockPointBottomRight(
                                                matrixBlockAccessSchematicElementSize*
                                                weightMatrixBlocksY*
                                                systolicArrayHeight,
                                                matrixBlockAccessSchematicElementSize*
                                                accumulatorArrayHeight*
                                                (activationMatrixBlockCount + 1));

        if(activationMatrixBlockCount != activationMatrixBlockCoordinateY)
        {
            cv::rectangle(matrixABlockAccessSchematicMat,
                                activationMatrixBlockPointTopLeft,
                                activationMatrixBlockPointBottomRight,
                                MATRIX_BLOCK_ACCESS_SCHEMATIC_COLOR_BLOCK_INACTIVE);
        }

    }

    const cv::Point activationMatrixBlockPointTopLeft(0,
                                    matrixBlockAccessSchematicElementSize*
                                    accumulatorArrayHeight*
                                    activationMatrixBlockCoordinateY);

    const cv::Point activationMatrixBlockPointBottomRight(
                                            matrixBlockAccessSchematicElementSize*
                                            systolicArrayHeight*
                                            weightMatrixBlocksY,
                                            matrixBlockAccessSchematicElementSize*
                                            accumulatorArrayHeight*
                                            (activationMatrixBlockCoordinateY + 1));

    cv::rectangle(matrixABlockAccessSchematicMat,
                        activationMatrixBlockPointTopLeft,
                        activationMatrixBlockPointBottomRight,
                        MATRIX_BLOCK_ACCESS_SCHEMATIC_COLOR_BLOCK_ACTIVE);

    for(size_t weightMatrixBlockCountX{0UL};
                weightMatrixBlockCountX < weightMatrixBlocksX;
                                                ++weightMatrixBlockCountX)
    {

        for(size_t weightMatrixBlockCountY{0UL};
                    weightMatrixBlockCountY < weightMatrixBlocksY;
                                                    ++weightMatrixBlockCountY)
        {

            const cv::Point weightMatrixBlockPointTopLeft(
                                            matrixBlockAccessSchematicElementSize*
                                            systolicArrayWidth*
                                            weightMatrixBlockCountX,
                                            matrixBlockAccessSchematicElementSize*
                                            systolicArrayHeight*
                                            weightMatrixBlockCountY);

            const cv::Point weightMatrixBlockPointBottomRight(
                                                matrixBlockAccessSchematicElementSize*
                                                systolicArrayWidth*
                                                (weightMatrixBlockCountX + 1),
                                                matrixBlockAccessSchematicElementSize*
                                                systolicArrayHeight*
                                                (weightMatrixBlockCountY + 1));

            if((weightMatrixBlockCountX != weightMatrixBlockCoordinateX) ||
                        (weightMatrixBlockCountY != weightMatrixBlockCoordinateY))
            {
                cv::rectangle(matrixBBlockAccessSchematicMat,
                                    weightMatrixBlockPointTopLeft,
                                    weightMatrixBlockPointBottomRight,
                                    MATRIX_BLOCK_ACCESS_SCHEMATIC_COLOR_BLOCK_INACTIVE);
            }
        }
    }

    const cv::Point weightMatrixBlockPointTopLeft(
                                    matrixBlockAccessSchematicElementSize*
                                    systolicArrayWidth*
                                    weightMatrixBlockCoordinateX,
                                    matrixBlockAccessSchematicElementSize*
                                    systolicArrayHeight*
                                    weightMatrixBlockCoordinateY);

    const cv::Point weightMatrixBlockPointBottomRight(
                                        matrixBlockAccessSchematicElementSize*
                                        systolicArrayWidth*
                                        (weightMatrixBlockCoordinateX + 1),
                                        matrixBlockAccessSchematicElementSize*
                                        systolicArrayHeight*
                                        (weightMatrixBlockCoordinateY + 1));

    cv::rectangle(matrixBBlockAccessSchematicMat,
                        weightMatrixBlockPointTopLeft,
                        weightMatrixBlockPointBottomRight,
                        MATRIX_BLOCK_ACCESS_SCHEMATIC_COLOR_BLOCK_ACTIVE);

    for(size_t resultMatrixBlockCountX{0UL};
                resultMatrixBlockCountX < weightMatrixBlocksX;
                                                ++resultMatrixBlockCountX)
    {

        for(size_t resultMatrixBlockCountY{0UL};
                    resultMatrixBlockCountY < activationMatrixBlocksY;
                                                    ++resultMatrixBlockCountY)
        {

            const cv::Point resultMatrixBlockPointTopLeft(
                                            matrixBlockAccessSchematicElementSize*
                                            resultMatrixBlockCountX*
                                            systolicArrayWidth,
                                            matrixBlockAccessSchematicElementSize*
                                            resultMatrixBlockCountY*
                                            accumulatorArrayHeight);

            const cv::Point resultMatrixBlockPointBottomRight(
                                                matrixBlockAccessSchematicElementSize*
                                                (resultMatrixBlockCountX + 1)*
                                                systolicArrayWidth,
                                                matrixBlockAccessSchematicElementSize*
                                                (resultMatrixBlockCountY + 1)*
                                                accumulatorArrayHeight);

            if((resultMatrixBlockCountX != resultMatrixBlockCoordinateX) ||
                        (resultMatrixBlockCountY != resultMatrixBlockCoordinateY))
            {
                cv::rectangle(matrixCBlockAccessSchematicMat,
                                    resultMatrixBlockPointTopLeft,
                                    resultMatrixBlockPointBottomRight,
                                    MATRIX_BLOCK_ACCESS_SCHEMATIC_COLOR_BLOCK_INACTIVE);
            }
        }
    }


    const cv::Point resultMatrixBlockPointTopLeft(
                                    matrixBlockAccessSchematicElementSize*
                                    resultMatrixBlockCoordinateX*
                                    systolicArrayWidth,
                                    matrixBlockAccessSchematicElementSize*
                                    resultMatrixBlockCoordinateY*
                                    accumulatorArrayHeight);

    const cv::Point resultMatrixBlockPointBottomRight(
                                        matrixBlockAccessSchematicElementSize*
                                        (resultMatrixBlockCoordinateX + 1)*
                                        systolicArrayWidth,
                                        matrixBlockAccessSchematicElementSize*
                                        (resultMatrixBlockCoordinateY + 1)*
                                        accumulatorArrayHeight);

        cv::rectangle(matrixCBlockAccessSchematicMat,
                            resultMatrixBlockPointTopLeft,
                            resultMatrixBlockPointBottomRight,
                            MATRIX_BLOCK_ACCESS_SCHEMATIC_COLOR_BLOCK_ACTIVE);

}

void copyMatrixBlockAccessSchematicMatsToOutputFrameMatAndPaintBoarders(
                                                    cv::Mat& matrixABlockAccessSchematicMat,
                                                    cv::Mat& matrixBBlockAccessSchematicMat,
                                                    cv::Mat& matrixCBlockAccessSchematicMat,
                                                    cv::Mat& outputFrameMat)
{
    matrixABlockAccessSchematicMat.copyTo(outputFrameMat(cv::Rect(
                                            matrixBlockAccessSchematicOutputFrameMatOffsetXMatrixA,
                                            matrixBlockAccessSchematicOutputFrameMatOffsetYMatrixA,
                                            matrixABlockAccessSchematicMat.cols,
                                            matrixABlockAccessSchematicMat.rows)));

    cv::rectangle(outputFrameMat,
                    cv::Point(matrixBlockAccessSchematicOutputFrameMatOffsetXMatrixA - 1,
                                matrixBlockAccessSchematicOutputFrameMatOffsetYMatrixA - 1),
                    cv::Point(matrixBlockAccessSchematicOutputFrameMatOffsetXMatrixA +
                                matrixABlockAccessSchematicMat.cols,
                                matrixBlockAccessSchematicOutputFrameMatOffsetYMatrixA +
                                matrixABlockAccessSchematicMat.rows),
                    cv::Scalar(0x00, 0x00 , 0x00));

    matrixBBlockAccessSchematicMat.copyTo(outputFrameMat(cv::Rect(
                                            matrixBlockAccessSchematicOutputFrameMatOffsetXMatrixB,
                                            matrixBlockAccessSchematicOutputFrameMatOffsetYMatrixB,
                                            matrixBBlockAccessSchematicMat.cols,
                                            matrixBBlockAccessSchematicMat.rows)));

    cv::rectangle(outputFrameMat,
                    cv::Point(matrixBlockAccessSchematicOutputFrameMatOffsetXMatrixB - 1,
                                matrixBlockAccessSchematicOutputFrameMatOffsetYMatrixB - 1),
                    cv::Point(matrixBlockAccessSchematicOutputFrameMatOffsetXMatrixB +
                                matrixBBlockAccessSchematicMat.cols,
                                matrixBlockAccessSchematicOutputFrameMatOffsetYMatrixB +
                                matrixBBlockAccessSchematicMat.rows),
                    cv::Scalar(0x00, 0x00 , 0x00));

    matrixCBlockAccessSchematicMat.copyTo(outputFrameMat(cv::Rect(
                                            matrixBlockAccessSchematicOutputFrameMatOffsetXMatrixC,
                                            matrixBlockAccessSchematicOutputFrameMatOffsetYMatrixC,
                                            matrixCBlockAccessSchematicMat.cols,
                                            matrixCBlockAccessSchematicMat.rows)));

    cv::rectangle(outputFrameMat,
                    cv::Point(matrixBlockAccessSchematicOutputFrameMatOffsetXMatrixC - 1,
                                matrixBlockAccessSchematicOutputFrameMatOffsetYMatrixC - 1),
                    cv::Point(matrixBlockAccessSchematicOutputFrameMatOffsetXMatrixC +
                                matrixCBlockAccessSchematicMat.cols,
                                matrixBlockAccessSchematicOutputFrameMatOffsetYMatrixC +
                                matrixCBlockAccessSchematicMat.rows),
                    cv::Scalar(0x00, 0x00 , 0x00));
}

template<typename WeightDatatype,
            typename ActivationDatatype,
            typename AccumulatorDatatype> class MatrixProcessingUnit
{

public:

    MatrixProcessingUnit(const size_t systolicArrayWidth,
                            const size_t systolicArrayHeight,
                            const size_t activationFifoDepth,
                            const size_t accumulatorArrayHeight,
                            const size_t unifiedBufferSizeByteMax): m_systolicArrayWidth{systolicArrayWidth},
                                                                    m_systolicArrayHeight{systolicArrayHeight},
                                                                    m_activationFifoDepth{activationFifoDepth},
                                                                    m_accumulatorArrayBufferHeight{accumulatorArrayHeight/2UL},
                                                                    m_systolicArrayLatency{m_systolicArrayWidth +
                                                                                            m_systolicArrayHeight - 1UL},
                                                                    m_unifiedBufferSizeByteMax{unifiedBufferSizeByteMax},
                                                                    m_systolicArray(m_systolicArrayWidth,
                                                                                        m_systolicArrayHeight,
                                                                                        m_activationFifoDepth),
                                                                    m_systolicDataSetupUnit(m_systolicArray.getActivationFifoArrayPtr()),
                                                                    m_weightFetcher(&m_systolicArray),
                                                                    m_accumulatorArray(m_systolicArray.getBottomPePtrRowPtr(),
                                                                                                            m_systolicArrayWidth,
                                                                                                            accumulatorArrayHeight),
                                                                    m_memoryManagementUnit(&m_unifiedBuffer,
                                                                                                m_unifiedBufferSizeByteMax)
    {
    }

    void resetDataMovementAndFootprintStatistics()
    {
        m_storeCount = 0UL;
        m_concurrentAccumulatorLoadCountMax = 0UL;
        m_concurrentAccumulatorArrayLoadCountPerColumnMax = 0UL;

        m_systolicDataSetupUnit.resetLoadCount();
        m_systolicDataSetupUnit.resetMaxRegisterValues();
        m_weightFetcher.resetDataMovementCounters();
        m_accumulatorArray.resetAdditionCountMaxValue();
    }

    size_t getSystolicArrayWidth() const
    {
        return m_systolicArrayWidth;
    }

    size_t getSystolicArrayHeight() const
    {
        return m_systolicArrayHeight;
    }

    size_t getActivationFifoDepth() const
    {
        return m_activationFifoDepth;
    }

    size_t getAccumulatorBufferHeight() const
    {
        return m_accumulatorArrayBufferHeight;
    }

    size_t getSystolicArrayLatency() const
    {
        return m_systolicArrayLatency;
    }

    size_t getUnifiedBufferSizeBytes() const
    {
        return m_unifiedBufferSizeByteMax;
    }

    std::byte* getUnifiedBufferAddress()
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

    void loadFromUnifiedBuffer(std::byte* const dest,
                                const std::byte* const src,
                                const size_t size)
    {
        m_memoryManagementUnit.loadFromUnifiedBuffer(
                                        dest, src, size);
    }

    void storeToUnifiedBuffer(std::byte* const dest,
                                const std::byte* const src,
                                const size_t size)
    {
        m_memoryManagementUnit.storeToUnifiedBuffer(
                                        dest, src, size);
    }

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

    void storeActivationMatrix(const ActivationDatatype* const activationMatrixPtr,
                                                                    const size_t rows,
                                                                    const size_t columns)
    {
        m_memoryManagementUnit.storeActivationMatrixManaged(
                                                activationMatrixPtr,
                                                rows,
                                                columns);
    }

    void loadResultMatrix(AccumulatorDatatype* const dest,
                                            const size_t size) const
    {
        m_memoryManagementUnit.loadResultMatrixManaged(dest, size);
    }

    size_t getUnifiedBufferMemoryUsageMax() const
    {
        return m_memoryManagementUnit.getMemoryUsageMaxByte();
    }

    void printUnifiedBufferLayout() const
    {
        m_memoryManagementUnit.printMemoryLayout();
    }

    void runMultiplication(const size_t sizeM,
                            const size_t sizeN,
                            const size_t sizeK,
                            const ActivationDatatype* const __restrict__ matrixAPtr,
                            const WeightDatatype* const __restrict__ matrixBPtr,
                            AccumulatorDatatype* const __restrict__ matrixCPtr)
    {

        assert((reinterpret_cast<const std::byte* const>(matrixAPtr) >=
                                            &(*m_unifiedBuffer.begin())) &&
                (reinterpret_cast<const std::byte* const>(matrixAPtr  + sizeM*sizeK) <=
                                                            &(*m_unifiedBuffer.end())));

        if((reinterpret_cast<const std::byte* const>(matrixAPtr) <
                                    &(*m_unifiedBuffer.begin())) ||
            (reinterpret_cast<const std::byte* const>(matrixAPtr + sizeM*sizeK) >
                                                            &(*m_unifiedBuffer.end())))
        {
            throw MpuException("TPU matrix multiplication "
                                "matrix A outside TPU address space");
        }

        assert((reinterpret_cast<const std::byte* const>(matrixBPtr) >=
                                            &(*m_unifiedBuffer.begin())) &&
                (reinterpret_cast<const std::byte* const>(matrixBPtr + sizeK*sizeN) <=
                                                            &(*m_unifiedBuffer.end())));

        if((reinterpret_cast<const std::byte* const>(matrixBPtr) <
                                        &(*m_unifiedBuffer.begin())) ||
            (reinterpret_cast<const std::byte* const>(matrixBPtr + sizeK*sizeN) >
                                                            &(*m_unifiedBuffer.end())))
        {
            throw MpuException("TPU matrix multiplication "
                                "matrix B outside TPU address space");
        }

        assert((reinterpret_cast<std::byte* const>(matrixCPtr) >=
                                        &(*m_unifiedBuffer.begin())) &&
                (reinterpret_cast<std::byte* const>(matrixCPtr + sizeM*sizeN) <=
                                                        &(*m_unifiedBuffer.end())));

        if((reinterpret_cast<std::byte* const>(matrixCPtr) <
                                        &(*m_unifiedBuffer.begin())) ||
            (reinterpret_cast<std::byte* const>(matrixCPtr + sizeM*sizeN) >
                                                    &(*m_unifiedBuffer.end())))
        {
            throw MpuException("TPU matrix multiplication result "
                                    "matrix outside TPU address space");
        }

        /* Startup */

        if(m_debugFlag)
        {
            std::cout << "Tensor Processing Unit: Matrix multiplication: "
                        << "Startup\nInput matrix dimensions:\tM: " << sizeM
                        << "\tN: " << sizeN << "\tK: " << sizeK << std::endl;
        }

        m_weightMatrixBlockCoordinateX = 0UL;
        m_weightMatrixBlockCoordinateY = 0UL;

        m_weightFetcherActivationMatrixRowBlockCount = 0UL;
        m_systolicArrayActivationMatrixColumnBlockCount = 0UL;

        m_resultMatrixWriteBlockCoordinateX = 0UL;
        m_resultMatrixWriteBlockCoordinateY = 0UL;

        m_resultMatrixReadInProgressBlockCoordinateX = 0UL;
        m_resultMatrixReadInProgressBlockCoordinateY = 0UL;

        m_resultMatrixReadDoneBlockCoordinateX = 0UL;
        m_resultMatrixReadDoneBlockCoordinateY = 0UL;

        m_accumulatorArrayBufferSelectBit = false;

//        cv::namedWindow("MPU state");
//        cv::resizeWindow("MPU state", 1920, 1080);

        cv::namedWindow("Block accesses");
        cv::resizeWindow("Block accesses", 1920, 1080);

        cv::VideoWriter videoWriter("matrix_block_access_schematic.mp4",
                                        cv::VideoWriter::fourcc(
                                                    'F', 'M', 'P', '4'),
                                        videoWriterFramerate,
                                        cv::Size(outputFrameMatWidth,
                                                    outputFrameMatHeight),
                                        true);

        cv::Mat memoryAccessMapMat(memoryAccessMapHeight*
                                    memoryAccessMapElementSize,
                                    memoryAccessMapWidth*
                                    memoryAccessMapElementSize,
                                    CV_8UC3);

        memoryAccessMapMat.setTo(cv::viz::Color(
                                    cv::Scalar(0x00, 0x00 , 0x00)));

        cv::Mat memoryAccessMapMatPersistent(memoryAccessMapHeight*
                                                memoryAccessMapElementSize,
                                                memoryAccessMapWidth*
                                                memoryAccessMapElementSize,
                                                CV_8UC3);

        memoryAccessMapMatPersistent.setTo(cv::viz::Color(
                                            cv::Scalar(0x00, 0x00 , 0x00)));

        cv::Mat memoryAccessMapMatOverlayed(memoryAccessMapHeight*
                                                memoryAccessMapElementSize,
                                                memoryAccessMapWidth*
                                                memoryAccessMapElementSize,
                                                CV_8UC3);

        const cv::Mat mpuSchematicOutputFrameBackgroundMat{cv::imread(
                                                            "mpu_visualizer_background.png")};

        cv::Mat mpuSchematicOutputFrameMat(outputFrameMatHeight,
                                            outputFrameMatWidth,
                                            CV_8UC3);

        mpuSchematicOutputFrameMat =
                    mpuSchematicOutputFrameBackgroundMat;

        m_weightFetcher.setInput(matrixBPtr, sizeN, sizeK);
        m_weightFetcher.clearWeightUpdateRequestQueue();
        m_weightFetcher.updateState(memoryAccessMapMat,
                                        memoryAccessMapMatPersistent,
                                        reinterpret_cast<WeightDatatype* const>(
                                                                    m_unifiedBuffer.data()));

        m_systolicDataSetupUnit.paintMemoryAccesses(memoryAccessMapMat,
                                                    memoryAccessMapMatPersistent,
                                                    reinterpret_cast<ActivationDatatype* const>(
                                                                                    m_unifiedBuffer.data()));

        m_systolicArray.paintState(mpuSchematicOutputFrameMat,
                                    cv::Point(mpuSchematicOutputFrameMatOffsetXSystolicArray,
                                                mpuSchematicOutputFrameMatOffsetYSystolicArray));

        m_accumulatorArray.paintState(mpuSchematicOutputFrameMat,
                                        cv::Point(mpuSchematicOutputFrameMatOffsetXAccumulatorArray,
                                                    mpuSchematicOutputFrameMatOffsetYAccumulatorArray));

//        auto overlayMemoryAccessMapMats{std::bind(cv::add,
//                                                    memoryAccessMapMat,
//                                                    memoryAccessMapMatPersistent,
//                                                    memoryAccessMapMatOverlayed,
//                                                    cv::noArray(),  -1)};
//        overlayMemoryAccessMapMats();

//        copyMemoryAccessMapMatToOutputFrameMatAndPaintBoarder(
//                                                    memoryAccessMapMatOverlayed,
//                                                    outputFrameMat);


//        cv::imshow("MPU state", mpuSchematicOutputFrameMat);

//        for(size_t frameCount{0}; frameCount < videoWriterFramerate;
//                                                            frameCount++)
//        {
//            videoWriter << mpuSchematicOutputFrameMat;
//            cv::waitKey(clockPeriodSlowMs/
//                            videoWriterFramerate);
//        }

        mpuSchematicOutputFrameMat = mpuSchematicOutputFrameBackgroundMat;

        m_weightMatrixBlocksX = m_weightFetcher.getBlockCountX();
        m_weightMatrixBlocksY = m_weightFetcher.getBlockCountY();

        m_weightMatrixColumnsLastBlock =
                        m_weightFetcher.getActiveColumnsLastBlock();

        m_systolicDataSetupUnit.addInputMatrix(matrixAPtr, sizeK,
                                    (m_accumulatorArrayBufferHeight < sizeM) ?
                                                        m_accumulatorArrayBufferHeight : sizeM,
                                                                            m_weightMatrixBlocksX);

        m_activationMatrixBlocksY =
                            std::ceil(static_cast<float>(sizeM)/
                                        static_cast<float>(m_accumulatorArrayBufferHeight));

        m_activationMatrixRowsLastBlock =
                            m_accumulatorArrayBufferHeight*(1L - m_activationMatrixBlocksY) + sizeM;

        m_activationMatrixBlockCoordinateY = 1UL;

        cv::Mat matrixBlockAccessSchematicOutputFrameMat(cv::imread(
                                                            "matrix_block_access_schematic_background.png"));

        cv::Mat matrixABlockAccessSchematicMat(m_activationMatrixBlocksY*
                                                m_accumulatorArrayBufferHeight*
                                                matrixBlockAccessSchematicElementSize + 1,
                                                m_weightMatrixBlocksY*
                                                m_systolicArrayHeight*
                                                matrixBlockAccessSchematicElementSize + 1,
                                                CV_8UC3);

        matrixABlockAccessSchematicMat.setTo(
                           MATRIX_BLOCK_ACCESS_SCHEMATIC_COLOR_INACTIVE_ROWS_COLUMNS);

        const cv::Rect matrixASizeRectSchematic(0, 0, visualizationMatrixSizeK*
                                                        matrixBlockAccessSchematicElementSize + 1,
                                                        visualizationMatrixSizeM*
                                                        matrixBlockAccessSchematicElementSize + 1);

        matrixABlockAccessSchematicMat(matrixASizeRectSchematic).setTo(cv::viz::Color(
                                                MATRIX_BLOCK_ACCESS_SCHEMATIC_COLOR_MATRIX_A));

        cv::Mat matrixBBlockAccessSchematicMat(m_weightMatrixBlocksY*
                                                m_systolicArrayHeight*
                                                matrixBlockAccessSchematicElementSize + 1,
                                                m_weightMatrixBlocksX*
                                                m_systolicArrayWidth*
                                                matrixBlockAccessSchematicElementSize + 1,
                                                CV_8UC3);

        matrixBBlockAccessSchematicMat.setTo(
                           MATRIX_BLOCK_ACCESS_SCHEMATIC_COLOR_INACTIVE_ROWS_COLUMNS);

        const cv::Rect matrixBSizeRectSchematic(0, 0, visualizationMatrixSizeN*
                                                        matrixBlockAccessSchematicElementSize + 1,
                                                        visualizationMatrixSizeK*
                                                        matrixBlockAccessSchematicElementSize + 1);

        matrixBBlockAccessSchematicMat(matrixBSizeRectSchematic).setTo(cv::viz::Color(
                                                MATRIX_BLOCK_ACCESS_SCHEMATIC_COLOR_MATRIX_B));

        cv::Mat matrixCBlockAccessSchematicMat(m_activationMatrixBlocksY*
                                                m_accumulatorArrayBufferHeight*
                                                matrixBlockAccessSchematicElementSize + 1,
                                                m_weightMatrixBlocksX*
                                                m_systolicArrayWidth*
                                                matrixBlockAccessSchematicElementSize + 1,
                                                CV_8UC3);

        matrixCBlockAccessSchematicMat.setTo(
                           MATRIX_BLOCK_ACCESS_SCHEMATIC_COLOR_INACTIVE_ROWS_COLUMNS);

        const cv::Rect matrixCSizeRectSchematic(0, 0, visualizationMatrixSizeN*
                                                        matrixBlockAccessSchematicElementSize + 1,
                                                        visualizationMatrixSizeM*
                                                        matrixBlockAccessSchematicElementSize + 1);

        matrixCBlockAccessSchematicMat(matrixCSizeRectSchematic).setTo(cv::viz::Color(
                                                MATRIX_BLOCK_ACCESS_SCHEMATIC_COLOR_MATRIX_C));

//        paintMatrixBlockAccessSchematicGrids(m_systolicArrayWidth,
//                                                m_systolicArrayHeight,
//                                                m_accumulatorArrayBufferHeight,
//                                                m_activationMatrixBlocksY,
//                                                m_activationMatrixBlockCoordinateY,
//                                                m_weightMatrixBlocksX,
//                                                m_weightMatrixBlocksY,
//                                                m_weightMatrixBlockCoordinateX,
//                                                m_weightMatrixBlockCoordinateY,
//                                                m_resultMatrixWriteBlockCoordinateX,
//                                                m_resultMatrixWriteBlockCoordinateY,
//                                                matrixABlockAccessSchematicMat,
//                                                matrixBBlockAccessSchematicMat,
//                                                matrixCBlockAccessSchematicMat);

//        cv::imshow("Activation matrix block access",
//                            matrixABlockAccessSchematicMat);
//        cv::imshow("Weight matrix block access",
//                            matrixBBlockAccessSchematicMat);
//        cv::imshow("Result matrix block access",
//                            matrixCBlockAccessSchematicMat);

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

        m_systolicDataSetupUnit.runIteration();
        m_systolicDataSetupUnit.updateState(memoryAccessMapMat,
                                                memoryAccessMapMatPersistent,
                                                reinterpret_cast<ActivationDatatype* const>(
                                                                                m_unifiedBuffer.data()));


        m_systolicArray.paintState(mpuSchematicOutputFrameMat,
                                    cv::Point(mpuSchematicOutputFrameMatOffsetXSystolicArray,
                                                mpuSchematicOutputFrameMatOffsetYSystolicArray));

        m_accumulatorArray.paintState(mpuSchematicOutputFrameMat,
                                        cv::Point(mpuSchematicOutputFrameMatOffsetXAccumulatorArray,
                                                    mpuSchematicOutputFrameMatOffsetYAccumulatorArray));

//        overlayMemoryAccessMapMats();

//        copyMemoryAccessMapMatToOutputFrameMatAndPaintBoarder(
//                                                    memoryAccessMapMatOverlayed,
//                                                    outputFrameMat);

//        cv::imshow("MPU state", mpuSchematicOutputFrameMat);

//        for(size_t frameCount{0}; frameCount < videoWriterFramerate;
//                                                            frameCount++)
//        {
//            videoWriter << mpuSchematicOutputFrameMat;
//            cv::waitKey(clockPeriodSlowMs/
//                            videoWriterFramerate);
//        }

//        mpuSchematicOutputFrameMat = mpuSchematicOutputFrameBackgroundMat;

        m_systolicDataSetupUnit.runIteration();
        m_weightFetcher.updateWeights(0UL, 0UL);

        m_systolicDataSetupUnit.updateState(memoryAccessMapMat,
                                                memoryAccessMapMatPersistent,
                                                reinterpret_cast<ActivationDatatype* const>(
                                                                                m_unifiedBuffer.data()));
        m_weightFetcher.updateState(memoryAccessMapMat,
                                    memoryAccessMapMatPersistent,
                                    reinterpret_cast<WeightDatatype* const>(
                                                                m_unifiedBuffer.data()));

        m_systolicArray.paintState(mpuSchematicOutputFrameMat,
                                    cv::Point(mpuSchematicOutputFrameMatOffsetXSystolicArray,
                                                mpuSchematicOutputFrameMatOffsetYSystolicArray));

        m_accumulatorArray.paintState(mpuSchematicOutputFrameMat,
                                        cv::Point(mpuSchematicOutputFrameMatOffsetXAccumulatorArray,
                                                    mpuSchematicOutputFrameMatOffsetYAccumulatorArray));

//        overlayMemoryAccessMapMats();

//        copyMemoryAccessMapMatToOutputFrameMatAndPaintBoarder(
//                                                    memoryAccessMapMatOverlayed,
//                                                    outputFrameMat);

//        cv::imshow("MPU state", mpuSchematicOutputFrameMat);

//        for(size_t frameCount{0}; frameCount < videoWriterFramerate;
//                                                            frameCount++)
//        {
//            videoWriter << mpuSchematicOutputFrameMat;
//            cv::waitKey(clockPeriodSlowMs/
//                            videoWriterFramerate);
//        }

        mpuSchematicOutputFrameMat = mpuSchematicOutputFrameBackgroundMat;

        m_systolicDataSetupUnit.runIteration();
        m_systolicArray.setUpdateWeightsSignal(true);
        m_weightFetcher.runIteration();

        m_systolicDataSetupUnit.updateState(memoryAccessMapMat,
                                                memoryAccessMapMatPersistent,
                                                reinterpret_cast<ActivationDatatype* const>(
                                                                                m_unifiedBuffer.data()));

        m_systolicArray.updateState(mpuSchematicOutputFrameMat,
                                    cv::Point(mpuSchematicOutputFrameMatOffsetXSystolicArray,
                                                mpuSchematicOutputFrameMatOffsetYSystolicArray));

        m_accumulatorArray.paintState(mpuSchematicOutputFrameMat,
                                        cv::Point(mpuSchematicOutputFrameMatOffsetXAccumulatorArray,
                                                    mpuSchematicOutputFrameMatOffsetYAccumulatorArray));

        m_weightFetcher.updateState(memoryAccessMapMat,
                                        memoryAccessMapMatPersistent,
                                        reinterpret_cast<WeightDatatype* const>(
                                                                    m_unifiedBuffer.data()));

//        overlayMemoryAccessMapMats();

//        copyMemoryAccessMapMatToOutputFrameMatAndPaintBoarder(
//                                                    memoryAccessMapMatOverlayed,
//                                                    outputFrameMat);


//        for(size_t frameCount{0}; frameCount < videoWriterFramerate;
//                                                            frameCount++)
//        {
//            videoWriter << mpuSchematicOutputFrameMat;
//            cv::waitKey(clockPeriodSlowMs/
//                            videoWriterFramerate);
//        }

        mpuSchematicOutputFrameMat = mpuSchematicOutputFrameBackgroundMat;

        m_systolicDataSetupUnit.runIteration();

        m_accumulatorArray.resetCounters();
        m_accumulatorArray.clearGotFirstInputBit();
        m_accumulatorArray.clearFirstUpdateDoneBits();
        m_accumulatorArray.clearBufferWriteDoneBit();
        m_accumulatorArray.setSystolicArrayStartupMode(
                                SystolicArrayStartupMode::WeightsNotPreloaded);
        m_accumulatorArray.setAdditionCount(m_weightMatrixBlocksY);

        m_systolicArray.readUpdateWeightSignals();

        m_weightFetcher.runIteration();

        m_systolicDataSetupUnit.updateState(memoryAccessMapMat,
                                                memoryAccessMapMatPersistent,
                                                reinterpret_cast<ActivationDatatype* const>(
                                                                                m_unifiedBuffer.data()));

        m_systolicArray.updateState(mpuSchematicOutputFrameMat,
                                    cv::Point(mpuSchematicOutputFrameMatOffsetXSystolicArray,
                                                mpuSchematicOutputFrameMatOffsetYSystolicArray));

        m_accumulatorArray.updateState(mpuSchematicOutputFrameMat,
                                        cv::Point(mpuSchematicOutputFrameMatOffsetXAccumulatorArray,
                                                   mpuSchematicOutputFrameMatOffsetYAccumulatorArray));

        m_weightFetcher.updateState(memoryAccessMapMat,
                                        memoryAccessMapMatPersistent,
                                        reinterpret_cast<WeightDatatype* const>(
                                                                    m_unifiedBuffer.data()));

//        overlayMemoryAccessMapMats();

//        copyMemoryAccessMapMatToOutputFrameMatAndPaintBoarder(
//                                                    memoryAccessMapMatOverlayed,
//                                                    outputFrameMat);

//        cv::imshow("MPU state", mpuSchematicOutputFrameMat);

//        for(size_t frameCount{0}; frameCount < videoWriterFramerate;
//                                                            frameCount++)
//        {
//            videoWriter << mpuSchematicOutputFrameMat;
//            cv::waitKey(clockPeriodSlowMs/
//                            videoWriterFramerate);
//        }

        mpuSchematicOutputFrameMat = mpuSchematicOutputFrameBackgroundMat;

        m_systolicArray.resetIterationCount();

        m_iterationCountTotal += 4UL;
        m_iterationCountStalled += 4UL;

        size_t systolicArrayInputCount{0UL};

        /* Matrix multiplication */

        do
        {
            if(m_debugFlag && m_verboseDebugOutputFlag)
            {
                std::cout << "------------------------------------------- "
                            << "Iteration " << m_iterationCountTotal
                            << " ------------------------------------------"
                            << std::endl;
            }

            if(m_resultMatrixWriteBlockCoordinateY != m_activationMatrixBlocksY)
            {
                m_systolicDataSetupUnit.runIteration();
                m_systolicArray.runIteration();
                m_accumulatorArray.runIteration();
                m_weightFetcher.runIteration();

                const size_t weightMatrixOutputRowsWeightUpdateSignal{
                                    (m_systolicArrayActivationMatrixColumnBlockCount !=
                                                                        (m_activationMatrixBlocksY - 1)) ?
                                                                        m_accumulatorArrayBufferHeight :
                                                                        m_activationMatrixRowsLastBlock};

                if((systolicArrayInputCount ==
                                        weightMatrixOutputRowsWeightUpdateSignal) &&
                        (m_systolicArrayActivationMatrixColumnBlockCount !=
                                                                m_activationMatrixBlocksY))
                {
                    m_systolicArrayActivationMatrixColumnBlockCount =
                                    m_weightFetcherActivationMatrixRowBlockCount;

                    if(m_systolicArrayActivationMatrixColumnBlockCount !=
                                                        m_activationMatrixBlocksY)
                    {
                        m_systolicArray.setUpdateWeightsSignal(true);

                        if(m_debugFlag && m_verboseDebugOutputFlag)
                        {
                            std::cout << "Set update weights signal" << std::endl;
                        }

                        systolicArrayInputCount = 0;
                    }
                }

                if(!m_systolicDataSetupUnit.hasBusySignal() &&
                        (m_activationMatrixBlockCoordinateY < m_activationMatrixBlocksY))
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
                                    (m_weightFetcherActivationMatrixRowBlockCount !=
                                                                    (m_activationMatrixBlocksY - 1)) ?
                                                                    m_accumulatorArrayBufferHeight :
                                                                    m_activationMatrixRowsLastBlock};

                if((systolicArrayInputCount ==
                        (weightMatrixOutputRowsWeightUpdate - 1)) &&
                        (m_weightFetcherActivationMatrixRowBlockCount !=
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
                            ++m_weightFetcherActivationMatrixRowBlockCount;
                        }
                    }

                    if(m_weightFetcherActivationMatrixRowBlockCount !=
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
            }

            else
            {
                assert(0);
                throw MpuException("TPU entered unexpected state");
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
                                        readOperationQueueIterator->diagonalsRead,
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

                ++(readOperationQueueIterator->diagonalsRead);

                if(readOperationQueueIterator->diagonalsRead ==
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

                    if(m_resultMatrixReadDoneBlockCoordinateX < (m_weightMatrixBlocksX - 1))
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
                                                                    m_resultMatrixReadInProgressBlockCoordinateX*
                                                                    m_systolicArrayWidth,
                                                                    m_accumulatorArrayBufferSelectBit,
                                                                    outputRows,
                                                                    outputColumns));

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

                if(m_resultMatrixReadInProgressBlockCoordinateX < (m_weightMatrixBlocksX - 1))
                {
                   ++m_resultMatrixReadInProgressBlockCoordinateX;
                }

                else
                {
                   m_resultMatrixReadInProgressBlockCoordinateX = 0;
                   ++m_resultMatrixReadInProgressBlockCoordinateY;
                }
            }

            if(m_accumulatorArray.hasBufferWriteDoneSignal())
            {
                if(m_resultMatrixWriteBlockCoordinateX < (m_weightMatrixBlocksX - 1))
                {
                    ++m_resultMatrixWriteBlockCoordinateX;
                }

                else
                {
                    m_resultMatrixWriteBlockCoordinateX = 0;
                    ++m_resultMatrixWriteBlockCoordinateY;
                }

                m_accumulatorArray.clearBufferWriteDoneBit();
            }

            for(auto readOperationQueueIterator{m_accumulatorArrayReadOperationQueue.begin()};
                        readOperationQueueIterator < m_accumulatorArrayReadOperationQueue.end();
                                                                        ++readOperationQueueIterator)
            {
                paintAccumulatorLoadOperation(matrixCPtr,
                                                sizeN,
                                                readOperationQueueIterator->destMatrixRowStart,
                                                readOperationQueueIterator->destMatrixColumnStart,
                                                readOperationQueueIterator->accumulatorArrayBufferSelectBit,
                                                readOperationQueueIterator->diagonalsRead,
                                                readOperationQueueIterator->blockHeight,
                                                readOperationQueueIterator->blockWidth,
                                                memoryAccessMapMat,
                                                memoryAccessMapMatPersistent);
            }

            m_systolicDataSetupUnit.updateState(memoryAccessMapMat,
                                                    memoryAccessMapMatPersistent,
                                                    reinterpret_cast<ActivationDatatype* const>(
                                                                                    m_unifiedBuffer.data()));

            m_systolicArray.updateState(mpuSchematicOutputFrameMat,
                                        cv::Point(mpuSchematicOutputFrameMatOffsetXSystolicArray,
                                                    mpuSchematicOutputFrameMatOffsetYSystolicArray));

            m_weightFetcher.updateState(memoryAccessMapMat,
                                            memoryAccessMapMatPersistent,
                                            reinterpret_cast<WeightDatatype* const>(
                                                                        m_unifiedBuffer.data()));

            m_accumulatorArray.updateState(mpuSchematicOutputFrameMat,
                                           cv::Point(mpuSchematicOutputFrameMatOffsetXAccumulatorArray,
                                                       mpuSchematicOutputFrameMatOffsetYAccumulatorArray));

            ++systolicArrayInputCount;
            ++m_iterationCountTotal;

//            overlayMemoryAccessMapMats();

//            copyMemoryAccessMapMatToOutputFrameMatAndPaintBoarder(
//                                                        memoryAccessMapMatOverlayed,
//                                                        outputFrameMat);

//            cv::imshow("MPU state", mpuSchematicOutputFrameMat);

//            const int clockPeriodMs{(((m_resultMatrixReadDoneBlockCoordinateY*
//                                            m_weightMatrixBlocksX +
//                                            m_resultMatrixReadDoneBlockCoordinateX) ==
//                                            m_weightMatrixBlocksX*m_activationMatrixBlocksY - 1) ||
//                                        ((m_resultMatrixReadDoneBlockCoordinateY*
//                                            m_weightMatrixBlocksX +
//                                            m_resultMatrixReadDoneBlockCoordinateX) == 0UL)) ?
//                                                                                    clockPeriodSlowMs :
//                                                                                    clockPeriodFastMs};

//            cv::waitKey(clockPeriodMs);

//            for(size_t frameCount{0}; frameCount < videoWriterFramerate;
//                                                                frameCount++)
//            {
//                videoWriter << mpuSchematicOutputFrameMat;
//                cv::waitKey(clockPeriodSlowMs/
//                                videoWriterFramerate);
//            }

//            memoryAccessMapMat.setTo(cv::viz::Color(cv::Scalar(0x00, 0x00 , 0x00)));
//            mpuSchematicOutputFrameMat = mpuSchematicOutputFrameBackgroundMat;

            paintMatrixBlockAccessSchematic(m_systolicArrayWidth,
                                                    m_systolicArrayHeight,
                                                    m_accumulatorArrayBufferHeight,
                                                    m_activationMatrixBlocksY,
                                                    m_activationMatrixBlockCoordinateY - 2UL,
                                                    m_weightMatrixBlocksX,
                                                    m_weightMatrixBlocksY,
                                                    m_weightMatrixBlockCoordinateX,
                                                    m_weightMatrixBlockCoordinateY,
                                                    m_resultMatrixWriteBlockCoordinateX,
                                                    m_resultMatrixWriteBlockCoordinateY,
                                                    matrixABlockAccessSchematicMat,
                                                    matrixBBlockAccessSchematicMat,
                                                    matrixCBlockAccessSchematicMat);

            copyMatrixBlockAccessSchematicMatsToOutputFrameMatAndPaintBoarders(
                                                    matrixABlockAccessSchematicMat,
                                                    matrixBBlockAccessSchematicMat,
                                                    matrixCBlockAccessSchematicMat,
                                                    matrixBlockAccessSchematicOutputFrameMat);


            cv::imshow("Block accesses",
                            matrixBlockAccessSchematicOutputFrameMat);

            videoWriter << matrixBlockAccessSchematicOutputFrameMat;

            cv::waitKey(1);

        }

        while(m_resultMatrixReadDoneBlockCoordinateY !=
                                            m_activationMatrixBlocksY);

        Eigen::Map<const RMatrix<ActivationDatatype>> matrixAEigen(matrixAPtr, sizeM, sizeK);
        Eigen::Map<const RMatrix<WeightDatatype>> matrixBEigen(matrixBPtr, sizeK, sizeN);
        RMatrix<AccumulatorDatatype> matrixCEigen = matrixAEigen*matrixBEigen;

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
            throw MpuException("TPU: Matrix multiplication "
                                    "result failed sanity check");
        }

        if(m_debugFlag)
        {
            std::cout << "Tensor processing unit: "
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
                        << m_weightFetcher.getMaxConcurrentLoadsPerColumn()
                        << "\n\tMax. concurrent loads total: "
                        << m_weightFetcher.getMaxConcurrentLoads()
                        << "\nAccumulator array:\n\tBits required for row pointers: "
                        << m_accumulatorArray.getRowPtrBitwidthRequiredMin()
                        << "\n\tBits required for addition counters: "
                        << m_accumulatorArray.getAdditionCounterBitwidthRequiredMin()
                        << "\n\tMax. concurrent load operations: "
                        << m_concurrentAccumulatorLoadCountMax
                        << "\n\tMax. concurrent load operations per column: "
                        << m_concurrentAccumulatorArrayLoadCountPerColumnMax
                        << "\nStore operations to unified buffer: "
                        << m_storeCount << std::endl;
        }
    }

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
            throw MpuException("TPU accumulator array read operation width "
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
        m_storeCount += loadCount;

    }

    void paintAccumulatorLoadOperation(const AccumulatorDatatype* const destMatrixPtr,
                                                            const size_t matrixWidth,
                                                            const size_t matrixRowStart,
                                                            const size_t matrixColumnStart,
                                                            const size_t accumulatorArrayBufferSelectBit,
                                                            const size_t accumulatorArrayDiagonal,
                                                            const size_t blockHeight,
                                                            const size_t blockWidth,
                                                            cv::Mat& memoryAccessMapMat,
                                                            cv::Mat& memoryAccessMapMatPersistent)
    {

        assert(blockWidth <= m_systolicArrayWidth);

        if(blockWidth > m_systolicArrayWidth)
        {
            throw MpuException("TPU accumulator array read operation width "
                                "larger than accumulator array buffer width");
        }

        const AccumulatorDatatype* const dest{destMatrixPtr +
                                                matrixRowStart*matrixWidth +
                                                matrixColumnStart};

        m_accumulatorArray.paintDiagonalRead(dest,
                                                matrixWidth,
                                                accumulatorArrayBufferSelectBit,
                                                accumulatorArrayDiagonal,
                                                blockHeight,
                                                blockWidth,
                                                reinterpret_cast<const AccumulatorDatatype* const>(
                                                                                m_unifiedBuffer.data()),
                                                memoryAccessMapMat,
                                                memoryAccessMapMatPersistent);

    }


    const size_t m_systolicArrayWidth;
    const size_t m_systolicArrayHeight;
    const size_t m_activationFifoDepth;
    const size_t m_accumulatorArrayBufferHeight;
    const size_t m_systolicArrayLatency;

    const size_t m_unifiedBufferSizeByteMax;

    SystolicArray<WeightDatatype, ActivationDatatype, AccumulatorDatatype> m_systolicArray;
    SystolicDataSetupUnit<ActivationDatatype> m_systolicDataSetupUnit;
    WeightFetcher<WeightDatatype, ActivationDatatype, AccumulatorDatatype> m_weightFetcher;
    AccumulatorArray<WeightDatatype, ActivationDatatype, AccumulatorDatatype> m_accumulatorArray;

    std::vector<std::byte> m_unifiedBuffer;

    MemoryManagementUnit<WeightDatatype, ActivationDatatype, AccumulatorDatatype> m_memoryManagementUnit;

    std::vector<AccumulatorArrayReadOperation> m_accumulatorArrayReadOperationQueue;

    size_t m_activationMatrixBlocksY{0UL};

    size_t m_activationMatrixBlockCoordinateY{0UL};

    size_t m_activationMatrixRowsLastBlock{0UL};

    size_t m_weightMatrixBlocksX{0UL};
    size_t m_weightMatrixBlocksY{0UL};

    size_t m_weightMatrixBlockCoordinateX{0UL};
    size_t m_weightMatrixBlockCoordinateY{0UL};

    size_t m_weightMatrixColumnsLastBlock{0UL};

    size_t m_weightFetcherActivationMatrixRowBlockCount{0UL};
    size_t m_systolicArrayActivationMatrixColumnBlockCount{0UL};

    size_t m_resultMatrixReadInProgressBlockCoordinateX{0UL};
    size_t m_resultMatrixReadInProgressBlockCoordinateY{0UL};

    size_t m_resultMatrixReadDoneBlockCoordinateX{0UL};
    size_t m_resultMatrixReadDoneBlockCoordinateY{0UL};

    size_t m_resultMatrixWriteBlockCoordinateX{0UL};
    size_t m_resultMatrixWriteBlockCoordinateY{0UL};

    size_t m_iterationCountTotal{0UL};
    size_t m_iterationCountStalled{0UL};

    size_t m_storeCount{0UL};

    size_t m_concurrentAccumulatorLoadCountMax{0UL};
    size_t m_concurrentAccumulatorArrayLoadCountPerColumnMax{0UL};

    bool m_accumulatorArrayBufferSelectBit{false};

    bool m_debugFlag{false};
    bool m_verboseDebugOutputFlag{false};

};


#endif
