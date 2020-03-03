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
 * @file        mpu_simulator_test.h
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   GNU Public License version 3 (GPLv3)
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

    constexpr size_t systolicArrayWidth{8UL};
    constexpr size_t systolicArrayHeight{256UL};
    constexpr size_t accumulatorArrayHeight{1024UL};

//    constexpr size_t systolicArrayWidth{77UL};
//    constexpr size_t systolicArrayHeight{17UL};
//    constexpr size_t accumulatorArrayHeight{190UL};

//    constexpr size_t systolicArrayWidth{256UL};
//    constexpr size_t systolicArrayHeight{256UL};
//    constexpr size_t accumulatorArrayHeight{4096UL};

    constexpr size_t activationFifoDepth{8UL};
    constexpr size_t unifiedBufferSizeByte{3UL*1024UL*1024UL*1024UL};

    MatrixProcessingUnit<WeightDatatype, ActivationDatatype, AccumulatorDatatype> matrixProcessingUnit(
                                                                                            systolicArrayWidth,
                                                                                            systolicArrayHeight,
                                                                                            activationFifoDepth,
                                                                                            accumulatorArrayHeight,
                                                                                            unifiedBufferSizeByte);


    matrixProcessingUnit.setDebugFlag(true);
//    matrixProcessingUnit.setDebugOutputVerboseFlag(true);

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

    std::cout << "MPU memory management unit test: "
                        "Dynamic unified buffer resize" << std::endl;

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

        bool sanityCheckPassed{true};

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

                    sanityCheckPassed = false;
                }
            }
        }

        assert(sanityCheckPassed);

        matrixProcessingUnit.resetIterationCounts();
        matrixProcessingUnit.resetDataMovementAndFootprintMetrics();
    }

    matrixProcessingUnit.printUnifiedBufferLayout();

//    matrixProcessingUnit.resetMemoryManagementUnit();
//    matrixProcessingUnit.setUnifiedBufferDynamicResize(false);
//    matrixProcessingUnit.resetIterationCounts();
//    matrixProcessingUnit.resetDataMovementAndFootprintMetrics();

//    multiplicationCount = 0UL;

//    std::cout << "MPU memory management unit test: "
//                        "Static unified buffer size" << std::endl;

//    for(size_t memoryManagementUnitTestCount{0UL}; memoryManagementUnitTestCount < 2UL;
//                                                                    ++memoryManagementUnitTestCount)
//    {
//        ++multiplicationCount;

//        std::cout << "Multiplication " << multiplicationCount << std::endl;

//        size_t sizeM;
//        size_t sizeN;
//        size_t sizeK;

//        do
//        {
//            sizeM = matrixDimensionDistribution(rng);
//            sizeN = matrixDimensionDistribution(rng);
//            sizeK = matrixDimensionDistribution(rng);
//        }

//        while((sizeM*sizeN*sizeK > (1UL << 24)) ||
//                            ((sizeN <= systolicArrayWidth) !=
//                                (sizeK <= systolicArrayHeight)));

//        const size_t sizeMConst{sizeM};
//        const size_t sizeNConst{sizeN};
//        const size_t sizeKConst{sizeK};

//        activationMatrix.clear();

//        for(size_t rowCount{0}; rowCount < sizeMConst; ++rowCount)
//        {
//            for(size_t columnCount{0}; columnCount < sizeKConst; ++columnCount)
//            {
//                activationMatrix.emplace_back(static_cast<ActivationDatatype>(
//                                                            matrixValueDistribution(rng)));
//            }
//        }

//        weightMatrix.clear();

//        for(size_t rowCount{0}; rowCount < sizeKConst; ++rowCount)
//        {
//            for(size_t columnCount{0}; columnCount < sizeNConst; ++columnCount)
//            {
//                weightMatrix.emplace_back(static_cast<WeightDatatype>(
//                                                    matrixValueDistribution(rng)));
//            }
//        }

//        matrixProcessingUnit.storeActivationMatrix(activationMatrix.data(),
//                                                        sizeMConst, sizeKConst);

//        const std::string weightMatrixNameString{"test" + std::to_string(multiplicationCount)};

//        matrixProcessingUnit.storeWeightMatrix(weightMatrixNameString,
//                                                    weightMatrix.data(),
//                                                    sizeKConst, sizeNConst);

//        matrixProcessingUnit.runMultiplication(weightMatrixNameString);

//        resultMatrix.clear();
//        resultMatrix.resize(sizeMConst*sizeNConst);

//        matrixProcessingUnit.loadResultMatrix(resultMatrix.data(),
//                                                resultMatrix.size());

//        Eigen::Map<const RMatrix<ActivationDatatype>> matrixAEigen(
//                                                            activationMatrix.data(),
//                                                                       sizeMConst, sizeKConst);

//        Eigen::Map<const RMatrix<WeightDatatype>> matrixBEigen(
//                                                        weightMatrix.data(),
//                                                                sizeKConst, sizeNConst);

//        const RMatrix<AccumulatorDatatype> matrixCEigen{matrixAEigen.template cast<AccumulatorDatatype>()*
//                                                        matrixBEigen.template cast<AccumulatorDatatype>()};

//        bool sanityCheckPassed{true};

//        for(size_t rowCount{0}; rowCount < sizeM; ++rowCount)
//        {
//            for(size_t columnCount{0}; columnCount < sizeN; ++columnCount)
//            {
//                if(resultMatrix[rowCount*sizeN + columnCount] !=
//                                    matrixCEigen(rowCount, columnCount))
//                {
//                    std::cout << "Systolic array output incorrect at ("
//                                << columnCount << ", " << rowCount
//                                << "): Expected value: "
//                                << matrixCEigen(rowCount, columnCount)
//                                << " actual value: "
//                                << resultMatrix[rowCount*sizeN + columnCount] << std::endl;

//                    sanityCheckPassed = false;
//                }
//            }
//        }

//        assert(sanityCheckPassed);
//    }

//    matrixProcessingUnit.printUnifiedBufferLayout();

//    matrixProcessingUnit.resetMemoryManagementUnit();
//    matrixProcessingUnit.setUnifiedBufferDynamicResize(false);

//    multiplicationCount = 0UL;

//    std::cout << "Starting infinite multiplication test loop..." << std::endl;

//    while(true)
//    {
//        ++multiplicationCount;

//        std::cout << "Multiplication " << multiplicationCount << std::endl;

//        size_t sizeM;
//        size_t sizeN;
//        size_t sizeK;

//        do
//        {
//            sizeM = matrixDimensionDistribution(rng);
//            sizeN = matrixDimensionDistribution(rng);
//            sizeK = matrixDimensionDistribution(rng);
//        }

//        while((sizeM*sizeN*sizeK > (1UL << 24)) ||
//                            ((sizeN <= systolicArrayWidth) !=
//                                (sizeK <= systolicArrayHeight)));

//        const size_t sizeMConst{sizeM};
//        const size_t sizeNConst{sizeN};
//        const size_t sizeKConst{sizeK};

//        activationMatrix.clear();

//        for(size_t rowCount{0}; rowCount < sizeMConst; ++rowCount)
//        {
//            for(size_t columnCount{0}; columnCount < sizeKConst; ++columnCount)
//            {
//                activationMatrix.emplace_back(static_cast<ActivationDatatype>(
//                                                            matrixValueDistribution(rng)));
//            }
//        }

//        weightMatrix.clear();

//        for(size_t rowCount{0}; rowCount < sizeKConst; ++rowCount)
//        {
//            for(size_t columnCount{0}; columnCount < sizeNConst; ++columnCount)
//            {
//                weightMatrix.emplace_back(static_cast<WeightDatatype>(
//                                                    matrixValueDistribution(rng)));
//            }
//        }

//        resultMatrix.clear();
//        resultMatrix.resize(sizeMConst*sizeNConst);

//        mpusim::byte* const activationMatrixDestAddress{
//                            matrixProcessingUnit.getUnifiedBufferAddress()};

//        mpusim::byte* const weightMatrixDestAddress{
//                            matrixProcessingUnit.getUnifiedBufferAddress() +
//                            activationMatrix.size()*sizeof(ActivationDatatype)};

//        mpusim::byte* const resultMatrixSrcAddress{
//                            matrixProcessingUnit.getUnifiedBufferAddress() +
//                            activationMatrix.size()*sizeof(ActivationDatatype) +
//                            weightMatrix.size()*sizeof(WeightDatatype)};

//        matrixProcessingUnit.storeToUnifiedBuffer(activationMatrixDestAddress,
//                                                    reinterpret_cast<mpusim::byte*>(activationMatrix.data()),
//                                                    activationMatrix.size()*sizeof(ActivationDatatype));

//        matrixProcessingUnit.storeToUnifiedBuffer(weightMatrixDestAddress,
//                                                    reinterpret_cast<mpusim::byte*>(weightMatrix.data()),
//                                                    weightMatrix.size()*sizeof(WeightDatatype));

//        matrixProcessingUnit.runMultiplication(sizeMConst, sizeNConst, sizeKConst,
//                                                reinterpret_cast<ActivationDatatype*>(
//                                                                    activationMatrixDestAddress),
//                                                reinterpret_cast<WeightDatatype*>(
//                                                                    weightMatrixDestAddress),
//                                                reinterpret_cast<AccumulatorDatatype*>(
//                                                                    resultMatrixSrcAddress));

//        matrixProcessingUnit.loadFromUnifiedBuffer(reinterpret_cast<mpusim::byte*>(resultMatrix.data()),
//                                                    resultMatrixSrcAddress, resultMatrix.size()*sizeof(AccumulatorDatatype));
//    }

    return 0;
}
