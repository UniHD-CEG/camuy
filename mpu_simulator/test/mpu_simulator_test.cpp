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
 * @file        mpu_simulator_test.cpp
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   MIT License
 */

#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <cstddef>
#include <cmath>

#include "matrix_processing_unit.h"
#include "mpu_statistics_logger.h"

int main(int argc, char** argv)
{

    using WeightDatatype = int8_t;
    using ActivationDatatype = int8_t;
    using AccumulatorDatatype = int32_t;

    constexpr size_t systolicArrayWidth{64UL};
    constexpr size_t systolicArrayHeight{64UL};
    constexpr size_t accumulatorArrayHeight{256UL};

    constexpr size_t activationFifoDepth{8UL};
    constexpr size_t unifiedBufferSizeByte{3UL*1024UL*1024UL*1024UL};

    MatrixProcessingUnit<WeightDatatype, ActivationDatatype, AccumulatorDatatype> matrixProcessingUnit(
                                                                                            systolicArrayWidth,
                                                                                            systolicArrayHeight,
                                                                                            activationFifoDepth,
                                                                                            accumulatorArrayHeight,
                                                                                            unifiedBufferSizeByte);


    matrixProcessingUnit.setDebugFlag(true);

    MpuStatisticsLogger mpuStatisticsLogger("test", sizeof(WeightDatatype),
                                                sizeof(ActivationDatatype),
                                                sizeof(AccumulatorDatatype));

    matrixProcessingUnit.registerLogEntryAvailableCallback(
                            [&mpuStatisticsLogger](MpuStatisticsLogEntry&& mpuStatisticsLogEntry){
        mpuStatisticsLogger.addMpuStatisticsLogEntry(std::move(mpuStatisticsLogEntry));
    });

    std::default_random_engine rng(static_cast<unsigned long>(
                                    std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::time_point_cast<std::chrono::milliseconds>(
                                    std::chrono::high_resolution_clock::now()).time_since_epoch()).count()));

    std::uniform_int_distribution<size_t> matrixDimensionDistribution(1UL, 8192UL);
    std::normal_distribution<float> matrixValueDistribution(0.0F, 8.0F);

    std::vector<ActivationDatatype> activationMatrix;
    std::vector<WeightDatatype> weightMatrix;
    std::vector<AccumulatorDatatype> resultMatrix;

    size_t multiplicationCount{0UL};
    
    bool sanityCheckPassedDynamic{true};
    bool sanityCheckPassedStatic{true};

    std::cout << "MPU test 0: Dynamic unified buffer resize" << std::endl;

    for(size_t memoryManagementUnitTestCount{0UL}; memoryManagementUnitTestCount < 16UL;
                                                                    ++memoryManagementUnitTestCount)
    {
        ++multiplicationCount;

        std::cout << "Multiplication " << multiplicationCount << std::endl;

        size_t sizeM;
        size_t sizeN;
        size_t sizeK;

        do
        {
            sizeM = matrixDimensionDistribution(rng);
            sizeN = matrixDimensionDistribution(rng);
            sizeK = matrixDimensionDistribution(rng);
        }

        while((sizeM*sizeN*sizeK > (1UL << 24)) ||
                            ((sizeN > systolicArrayWidth) &&
                                (sizeK <= systolicArrayHeight)));

        const size_t sizeMConst{sizeM};
        const size_t sizeNConst{sizeN};
        const size_t sizeKConst{sizeK};

        activationMatrix.clear();

        for(size_t rowCount{0}; rowCount < sizeMConst; ++rowCount)
        {
            for(size_t columnCount{0}; columnCount < sizeKConst; ++columnCount)
            {
                activationMatrix.emplace_back(static_cast<ActivationDatatype>(
                                                            matrixValueDistribution(rng)));
            }
        }

        weightMatrix.clear();

        for(size_t rowCount{0}; rowCount < sizeKConst; ++rowCount)
        {
            for(size_t columnCount{0}; columnCount < sizeNConst; ++columnCount)
            {
                weightMatrix.emplace_back(static_cast<WeightDatatype>(
                                                    matrixValueDistribution(rng)));
            }
        }

        matrixProcessingUnit.storeActivationMatrix(activationMatrix.data(),
                                                        sizeMConst, sizeKConst);

        const std::string weightMatrixNameString{"test" + std::to_string(multiplicationCount)};

        matrixProcessingUnit.storeWeightMatrix(weightMatrixNameString,
                                                    weightMatrix.data(),
                                                    sizeKConst, sizeNConst);

        matrixProcessingUnit.runMultiplication(weightMatrixNameString);

        resultMatrix.clear();
        resultMatrix.resize(sizeMConst*sizeNConst);

        matrixProcessingUnit.loadResultMatrix(resultMatrix.data(),
                                                resultMatrix.size());

        Eigen::Map<const RMatrix<ActivationDatatype>> matrixAEigen(
                                                            activationMatrix.data(),
                                                                    sizeMConst, sizeKConst);

        Eigen::Map<const RMatrix<WeightDatatype>> matrixBEigen(
                                                        weightMatrix.data(),
                                                                sizeKConst, sizeNConst);

        const RMatrix<AccumulatorDatatype> matrixCEigen{matrixAEigen.template cast<AccumulatorDatatype>()*
                                                        matrixBEigen.template cast<AccumulatorDatatype>()};

        for(size_t rowCount{0}; rowCount < sizeMConst; ++rowCount)
        {
            for(size_t columnCount{0}; columnCount < sizeNConst; ++columnCount)
            {
                if(resultMatrix[rowCount*sizeNConst + columnCount] !=
                                    matrixCEigen(rowCount, columnCount))
                {
                    std::cout << "Systolic array output incorrect at ("
                                << columnCount << ", " << rowCount
                                << "): Expected value: "
                                << matrixCEigen(rowCount, columnCount)
                                << " actual value: "
                                << resultMatrix[rowCount*sizeNConst + columnCount] << std::endl;

                    sanityCheckPassedDynamic = false;
                }
            }
        }
    }

    matrixProcessingUnit.printUnifiedBufferLayout();

    matrixProcessingUnit.resetMemoryManagementUnit();
    matrixProcessingUnit.setUnifiedBufferDynamicResize(false);
    matrixProcessingUnit.resetIterationCounts();
    matrixProcessingUnit.resetDataMovementAndFootprintMetrics();

    multiplicationCount = 0UL;

    std::cout << "MPU test 1: Static unified buffer size" << std::endl;

    for(size_t memoryManagementUnitTestCount{0UL}; memoryManagementUnitTestCount < 16UL;
                                                                    ++memoryManagementUnitTestCount)
    {
        ++multiplicationCount;

        std::cout << "Multiplication " << multiplicationCount << std::endl;

        size_t sizeM;
        size_t sizeN;
        size_t sizeK;

        do
        {
            sizeM = matrixDimensionDistribution(rng);
            sizeN = matrixDimensionDistribution(rng);
            sizeK = matrixDimensionDistribution(rng);
        }

        while((sizeM*sizeN*sizeK > (1UL << 24)) ||
                            ((sizeN <= systolicArrayWidth) !=
                                (sizeK <= systolicArrayHeight)));

        const size_t sizeMConst{sizeM};
        const size_t sizeNConst{sizeN};
        const size_t sizeKConst{sizeK};

        activationMatrix.clear();

        for(size_t rowCount{0}; rowCount < sizeMConst; ++rowCount)
        {
            for(size_t columnCount{0}; columnCount < sizeKConst; ++columnCount)
            {
                activationMatrix.emplace_back(static_cast<ActivationDatatype>(
                                                            matrixValueDistribution(rng)));
            }
        }

        weightMatrix.clear();

        for(size_t rowCount{0}; rowCount < sizeKConst; ++rowCount)
        {
            for(size_t columnCount{0}; columnCount < sizeNConst; ++columnCount)
            {
                weightMatrix.emplace_back(static_cast<WeightDatatype>(
                                                    matrixValueDistribution(rng)));
            }
        }

        matrixProcessingUnit.storeActivationMatrix(activationMatrix.data(),
                                                        sizeMConst, sizeKConst);

        const std::string weightMatrixNameString{"test" + std::to_string(multiplicationCount)};

        matrixProcessingUnit.storeWeightMatrix(weightMatrixNameString,
                                                    weightMatrix.data(),
                                                    sizeKConst, sizeNConst);

        matrixProcessingUnit.runMultiplication(weightMatrixNameString);

        resultMatrix.clear();
        resultMatrix.resize(sizeMConst*sizeNConst);

        matrixProcessingUnit.loadResultMatrix(resultMatrix.data(),
                                                resultMatrix.size());

        Eigen::Map<const RMatrix<ActivationDatatype>> matrixAEigen(
                                                            activationMatrix.data(),
                                                                        sizeMConst, sizeKConst);

        Eigen::Map<const RMatrix<WeightDatatype>> matrixBEigen(
                                                        weightMatrix.data(),
                                                                sizeKConst, sizeNConst);

        const RMatrix<AccumulatorDatatype> matrixCEigen{matrixAEigen.template cast<AccumulatorDatatype>()*
                                                        matrixBEigen.template cast<AccumulatorDatatype>()};

        for(size_t rowCount{0}; rowCount < sizeM; ++rowCount)
        {
            for(size_t columnCount{0}; columnCount < sizeN; ++columnCount)
            {
                if(resultMatrix[rowCount*sizeN + columnCount] !=
                                    matrixCEigen(rowCount, columnCount))
                {
                    std::cout << "Systolic array output incorrect at ("
                                << columnCount << ", " << rowCount
                                << "): Expected value: "
                                << matrixCEigen(rowCount, columnCount)
                                << " actual value: "
                                << resultMatrix[rowCount*sizeN + columnCount] << std::endl;

                    sanityCheckPassedStatic = false;
                }
            }
        }
    }
    
    matrixProcessingUnit.printUnifiedBufferLayout();
    
    std::cout << "================================ SUMMARY ================================\n\n";
    
    if(sanityCheckPassedDynamic)
    {
        std::cout << "Test 0: Matrix multiplication using dynamic unified buffer resizing\t\tPASSED\n\n";
    }
    
    else
    {
        std::cout << "Test 0: Matrix multiplication using dynamic unified buffer resizing\t\tFAILED\n\n";
    }
    
    if(sanityCheckPassedStatic)
    {
        std::cout << "Test 1: Matrix multiplication using static dynamic buffer size\t\tPASSED\n\n";
    }
    
    else
    {
        std::cout << "Test 1: Matrix multiplication using static dynamic buffer size\t\tFAILED\n\n";
    }
    
    if(!(sanityCheckPassedDynamic && sanityCheckPassedStatic))
    {
        return -1;
    }
    
    else
    {
        return 0;
    }
}
