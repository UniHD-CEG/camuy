#ifndef MPUSIM_WRAPPER_H
#define MPUSIM_WRAPPER_H

#include <cstdint>

#include "matrix_processing_unit.h"
#include "mpu_statistics_logger.h"

class MpuSimWrapper final
{
public:

    static MpuSimWrapper& getInstance()
    {
        static MpuSimWrapper instance;
        return instance;
    }
    
    
void runMultiplication(const size_t activationsDatatypeSizeByte,
                                const size_t weightsDatatypeSizeByte,
                                const size_t resultsDatatypeSizeByte,
                                const size_t systolicArrayHeight,
                                const size_t systolicArrayWidth,
                                const size_t activationFifoDepth,
                                const size_t accumulatorArrayHeight,
                                const size_t sizeM,
                                const size_t sizeN,
                                const size_t sizeK,
                                const float* const activationMatrix,
                                const float* const weightMatrix,
                                float* const resultMatrix,
                                const std::string& logFileOutputDirString,
                                const std::string& modelNameString,
                                const std::string& operationNameString);

private:

    MpuSimWrapper()
    {
        std::cout << "Allocated MPU simulator wrapper object" << std::endl;
    }
    
    ~MpuSimWrapper();

    MpuSimWrapper(MpuSimWrapper& other) = delete;
    MpuSimWrapper(MpuSimWrapper&& other) = delete;

    void operator=(MpuSimWrapper& other) = delete;
    
    MpuStatisticsLogger* m_mpuStatisticsLoggerPtr{nullptr};
    
    union{
        MatrixProcessingUnit<int8_t, int8_t, int8_t>* m_matrixProcessingUnit8_8_8Ptr{nullptr};
        
        MatrixProcessingUnit<int8_t, int8_t, int16_t>* m_matrixProcessingUnit8_8_16Ptr;
        MatrixProcessingUnit<int16_t, int8_t, int16_t>* m_matrixProcessingUnit16_8_16Ptr;
        MatrixProcessingUnit<int8_t, int16_t, int16_t>* m_matrixProcessingUnit8_16_16Ptr;
        MatrixProcessingUnit<int16_t, int16_t, int16_t>* m_matrixProcessingUnit16_16_16Ptr;
        
        MatrixProcessingUnit<int8_t, int8_t, int32_t>* m_matrixProcessingUnit8_8_32Ptr;
        MatrixProcessingUnit<int16_t, int8_t, int32_t>* m_matrixProcessingUnit16_8_32Ptr;
        MatrixProcessingUnit<int32_t, int8_t, int32_t>* m_matrixProcessingUnit32_8_32Ptr;
        MatrixProcessingUnit<int8_t, int16_t, int32_t>* m_matrixProcessingUnit8_16_32Ptr;
        MatrixProcessingUnit<int16_t, int16_t, int32_t>* m_matrixProcessingUnit16_16_32Ptr;
        MatrixProcessingUnit<int32_t, int16_t, int32_t>* m_matrixProcessingUnit32_16_32Ptr;
        MatrixProcessingUnit<int8_t, int32_t, int32_t>* m_matrixProcessingUnit8_32_32Ptr;
        MatrixProcessingUnit<int16_t, int32_t, int32_t>* m_matrixProcessingUnit16_32_32Ptr;
        MatrixProcessingUnit<int32_t, int32_t, int32_t>* m_matrixProcessingUnit32_32_32Ptr;
        
        MatrixProcessingUnit<int8_t, int8_t, int64_t>* m_matrixProcessingUnit8_8_64Ptr;
        MatrixProcessingUnit<int16_t, int8_t, int64_t>* m_matrixProcessingUnit16_8_64Ptr;
        MatrixProcessingUnit<int32_t, int8_t, int64_t>* m_matrixProcessingUnit32_8_64Ptr;
        MatrixProcessingUnit<int64_t, int8_t, int64_t>* m_matrixProcessingUnit64_8_64Ptr;
        MatrixProcessingUnit<int8_t, int16_t, int64_t>* m_matrixProcessingUnit8_16_64Ptr;
        MatrixProcessingUnit<int16_t, int16_t, int64_t>* m_matrixProcessingUnit16_16_64Ptr;
        MatrixProcessingUnit<int32_t, int16_t, int64_t>* m_matrixProcessingUnit32_16_64Ptr;
        MatrixProcessingUnit<int64_t, int16_t, int64_t>* m_matrixProcessingUnit64_16_64Ptr;
        MatrixProcessingUnit<int8_t, int32_t, int64_t>* m_matrixProcessingUnit8_32_64Ptr;
        MatrixProcessingUnit<int16_t, int32_t, int64_t>* m_matrixProcessingUnit16_32_64Ptr;
        MatrixProcessingUnit<int32_t, int32_t, int64_t>* m_matrixProcessingUnit32_32_64Ptr;
        MatrixProcessingUnit<int64_t, int32_t, int64_t>* m_matrixProcessingUnit64_32_64Ptr;
        MatrixProcessingUnit<int8_t, int64_t, int64_t>* m_matrixProcessingUnit8_64_64Ptr;
        MatrixProcessingUnit<int16_t, int64_t, int64_t>* m_matrixProcessingUnit16_64_64Ptr;
        MatrixProcessingUnit<int32_t, int64_t, int64_t>* m_matrixProcessingUnit32_64_64Ptr;
        MatrixProcessingUnit<int64_t, int64_t, int64_t>* m_matrixProcessingUnit64_64_64Ptr;
    };
    
    size_t m_activationsDatatypeSizeByteCurrent{0UL};
    size_t m_weightsDatatypeSizeByteCurrent{0UL};
    size_t m_resultsDatatypeSizeByteCurrent{0UL};
    
    size_t m_systolicArrayHeightCurrent{0UL};
    size_t m_systolicArrayWidthCurrent{0UL};
    size_t m_activationFifoDepthCurrent{0UL};
    size_t m_accumulatorArrayHeightCurrent{0UL};

};


#endif
