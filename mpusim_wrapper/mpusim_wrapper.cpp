#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <limits>
#include <type_traits>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <climits>

#include "mpusim_wrapper.h"

#define CONSTRUCT_AND_CONFIGURE_MPU(mpuPtr, WeightsDatatype, ActivationsDatatype, ResultsDatatype) \
mpuPtr = new MatrixProcessingUnit<WeightsDatatype, ActivationsDatatype, ResultsDatatype>(\
                                                                                systolicArrayWidth,\
                                                                                systolicArrayHeight,\
                                                                                activationFifoDepth,\
                                                                                accumulatorArrayHeight,\
                                                                                unifiedBufferSizeMaxByte);\
mpuPtr->setDebugFlag(true);\
mpuPtr->registerLogEntryAvailableCallback([this](MpuStatisticsLogEntry&& mpuStatisticsLogEntry){\
    m_mpuStatisticsLoggerPtr->addMpuStatisticsLogEntry(std::move(mpuStatisticsLogEntry));\
})

#define QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(mpuPtr, WeightsDatatype, ActivationsDatatype, ResultsDatatype)\
if(!matrixNeedsPadding)\
{\
    WeightsDatatype* const weightMatrixQuantized{\
                            reinterpret_cast<WeightsDatatype*>(quantizationBufferPtr)};\
    const size_t weightMatrixSizeByte{sizeN*sizeK*sizeof(WeightsDatatype)};\
    quantizeLinear(weightMatrix, weightMatrixQuantized, sizeK*sizeN);\
    mpuPtr->storeWeightMatrix(operationNameString,\
                                weightMatrixQuantized,\
                                sizeK, sizeN);\
    ActivationsDatatype* const activationMatrixQuantized{\
                                    reinterpret_cast<ActivationsDatatype*>(\
                                                            quantizationBufferPtr +\
                                                            weightMatrixSizeByte)};\
    const size_t activationMatrixSizeByte{sizeM*sizeK*\
                                            sizeof(ActivationsDatatype)};\
    double activationMatrixMeanUnquantized;\
    double activationMatrixStdDevUnquantized;\
    getMeanAndStdDev(activationMatrix, sizeM*sizeK,\
                        activationMatrixMeanUnquantized,\
                        activationMatrixStdDevUnquantized);\
    std::cout << "Raw activations: Mean: " << activationMatrixMeanUnquantized\
                << "\tStdDev: " << activationMatrixStdDevUnquantized << std::endl;\
    const float scaleFactorResults{1.0F/quantizeLinear(activationMatrix,\
                                                        activationMatrixQuantized,\
                                                        sizeM*sizeK)};\
    double activationMatrixMeanQuantized;\
    double activationMatrixStdDevQuantized;\
    getMeanAndStdDev(activationMatrixQuantized, sizeM*sizeK,\
                                activationMatrixMeanQuantized,\
                                activationMatrixStdDevQuantized);\
    std::cout << "Quantized activations: Mean: " << activationMatrixMeanQuantized\
                    << "\tStdDev: " << activationMatrixStdDevQuantized << std::endl;\
    mpuPtr->storeActivationMatrix(activationMatrixQuantized, sizeM, sizeK);\
    mpuPtr->runMultiplication(operationNameString);\
    mpuPtr->loadResultMatrix(reinterpret_cast<ResultsDatatype*>(\
                                                        quantizationBufferPtr +\
                                                        weightMatrixSizeByte +\
                                                        activationMatrixSizeByte),\
                                                        sizeM*sizeN);\
    scaleToFactor(reinterpret_cast<ResultsDatatype*>(\
                                            quantizationBufferPtr +\
                                            weightMatrixSizeByte +\
                                            activationMatrixSizeByte),\
                                            resultMatrix,\
                                            scaleFactorResults,\
                                            sizeM*sizeN);\
}\
else\
{\
    WeightsDatatype* const weightMatrixQuantized{\
                            reinterpret_cast<WeightsDatatype*>(quantizationBufferPtr)};\
    const size_t weightMatrixSizeByte{sizeNPadded*sizeKPadded*sizeof(WeightsDatatype)};\
    quantizeLinearAndPad(weightMatrix,\
                            weightMatrixQuantized,\
                            sizeK,\
                            sizeN,\
                            sizeKPadded,\
                            sizeNPadded);\
    mpuPtr->storeWeightMatrix(operationNameString,\
                                weightMatrixQuantized,\
                                sizeKPadded, sizeNPadded);\
    ActivationsDatatype* const activationMatrixQuantized{\
                                    reinterpret_cast<ActivationsDatatype*>(\
                                                            quantizationBufferPtr +\
                                                            weightMatrixSizeByte)};\
    const size_t activationMatrixSizeByte{sizeM*sizeKPadded*\
                                            sizeof(ActivationsDatatype)};\
    double activationMatrixMeanUnquantized;\
    double activationMatrixStdDevUnquantized;\
    getMeanAndStdDev(activationMatrix, sizeM*sizeK,\
                        activationMatrixMeanUnquantized,\
                        activationMatrixStdDevUnquantized);\
    std::cout << "Raw activations: Mean: " << activationMatrixMeanUnquantized\
                << "\tStdDev: " << activationMatrixStdDevUnquantized << std::endl;\
    const float scaleFactorResults{1.0F/quantizeLinearAndPad(activationMatrix,\
                                                                activationMatrixQuantized,\
                                                                sizeM,\
                                                                sizeK,\
                                                                sizeM,\
                                                                sizeKPadded)};\
    double activationMatrixMeanQuantized;\
    double activationMatrixStdDevQuantized;\
    getMeanAndStdDevPadded(activationMatrix,\
                            sizeM,\
                            sizeK,\
                            sizeM,\
                            sizeKPadded,\
                            activationMatrixMeanQuantized,\
                            activationMatrixStdDevQuantized);\
    std::cout << "Quantized activations: Mean: " << activationMatrixMeanQuantized\
                    << "\tStdDev: " << activationMatrixStdDevQuantized << std::endl;\
    mpuPtr->storeActivationMatrix(activationMatrixQuantized, sizeM, sizeKPadded);\
    mpuPtr->runMultiplication(operationNameString);\
    mpuPtr->loadResultMatrix(reinterpret_cast<ResultsDatatype*>(\
                                                        quantizationBufferPtr +\
                                                        weightMatrixSizeByte +\
                                                        activationMatrixSizeByte),\
                                                        sizeM*sizeNPadded);\
    scaleToFactorAndCrop(reinterpret_cast<ResultsDatatype*>(\
                                            quantizationBufferPtr +\
                                            weightMatrixSizeByte +\
                                            activationMatrixSizeByte),\
                                            resultMatrix,\
                                            scaleFactorResults,\
                                            sizeM,\
                                            sizeNPadded,\
                                            sizeM,\
                                            sizeN);\
}\
mpuPtr->resetIterationCounts();\
mpuPtr->resetDataMovementAndFootprintMetrics();\
mpuPtr->printUnifiedBufferLayout();\
std::cout << "Unified buffer memory usage: "\
            << mpuPtr->getUnifiedBufferSizeMinBit() << std::endl\

namespace
{

constexpr size_t bufferSizeBegin{1024};

class QuantizationBufferSingleton final
{
public:

    static QuantizationBufferSingleton& getInstance()
    {
        static QuantizationBufferSingleton instance;
        return instance;
    }

    inline mpusim::byte* get(const size_t size)
    {
        if(size == 0)
        {
            throw;
        }
        
        if(size > m_size)
        {
            m_size = size;
            m_memory.resize(m_size);
            
            std::cout << "Resized quantization buffer, "
                                                "new size: "
                        << m_size/1024UL << " kB" << std::endl;
        }

        return m_memory.data();
    }

private:

    QuantizationBufferSingleton(): m_memory(bufferSizeBegin){}

    QuantizationBufferSingleton(QuantizationBufferSingleton& other) = delete;
    QuantizationBufferSingleton(QuantizationBufferSingleton&& other) = delete;

    void operator=(QuantizationBufferSingleton& other) = delete;

    std::vector<mpusim::byte> m_memory;

    size_t m_size{bufferSizeBegin};

};

constexpr size_t combineParameterDatatypeSizes(const size_t weightsDatatypeSizeByte,
                                                const size_t activationsDatatypeSizeByte,
                                                const size_t resultsDatatypeSizeByte)
{
    return (weightsDatatypeSizeByte << 16) |
            (activationsDatatypeSizeByte << 8) |
            resultsDatatypeSizeByte;
}

template<typename T> void getMeanAndStdDev(T* const data,
                                            const size_t size,
                                            double& mean,
                                            double& stdDev)
{
    if(std::is_floating_point<T>::value)
    {
        const T sum{std::accumulate(data, data + size, T{0})};
        mean = static_cast<double>(sum)/static_cast<double>(size);

        std::vector<double> diff(size);

        std::transform(data, data + size,
                        diff.begin(), [mean](const double x) { return x - mean; });

        const double squareSum{std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0)};
        stdDev = std::sqrt(squareSum/size);
    }

    else
    {
        const size_t sum{std::accumulate(data, data + size, 0UL)};
        mean = static_cast<double>(sum)/static_cast<double>(size);

        std::vector<double> diff(size);

        std::transform(data, data + size,
                        diff.begin(), [mean](const double x) { return x - mean; });

        const double squareSum{std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0)};
        stdDev = std::sqrt(squareSum/size);
    }
}

template<typename T> void getMeanAndStdDevPadded(T* const data,
                                                    const size_t heightOriginal,
                                                    const size_t widthOriginal,
                                                    const size_t heightPadded,
                                                    const size_t widthPadded,
                                                    double& mean,
                                                    double& stdDev)
{
    if(std::is_floating_point<T>::value)
    {
        const size_t elementCountOriginal{heightOriginal*widthOriginal};
        
        typename std::remove_const<T>::type sum{0UL};
        
        for(size_t rowCount{0UL}; rowCount < heightOriginal; ++rowCount)
        {
            sum += std::accumulate(data + rowCount*widthPadded,
                                    data + rowCount*widthPadded + widthOriginal, T{0});
        }
        
        mean = static_cast<double>(sum)/static_cast<double>(elementCountOriginal);

        std::vector<double> diff;
        
        diff.reserve(elementCountOriginal);

        for(size_t rowCount{0UL}; rowCount < heightOriginal; ++rowCount)
        {
            std::transform(data + rowCount*widthPadded,
                            data + rowCount*widthPadded + widthOriginal,
                            std::back_inserter(diff),
                            [mean](const double x) { return x - mean; });
        }

        const double squareSum{std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0)};
        stdDev = std::sqrt(squareSum/elementCountOriginal);
    }
    
    else
    {
        const size_t elementCountOriginal{heightOriginal*widthOriginal};
        
        size_t sum{0UL};
        
        for(size_t rowCount{0UL}; rowCount < heightOriginal; ++rowCount)
        {
            sum += std::accumulate(data + rowCount*widthPadded,
                                    data + rowCount*widthPadded + widthOriginal, 0UL);
        }
        
        mean = static_cast<double>(sum)/static_cast<double>(elementCountOriginal);

        std::vector<double> diff;
        
        diff.reserve(elementCountOriginal);

        for(size_t rowCount{0UL}; rowCount < heightOriginal; ++rowCount)
        {
            std::transform(data + rowCount*widthPadded,
                            data + rowCount*widthPadded + widthOriginal,
                            std::back_inserter(diff),
                            [mean](const double x) { return x - mean; });
        }

        const double squareSum{std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0)};
        stdDev = std::sqrt(squareSum/elementCountOriginal);
    }
}

template<typename T> float quantizeLinear(const float* const inputMatrix,
                                                    T* const outputMatrix,
                                                    const size_t size)
{
    const float inputValueMax{*(std::max_element(inputMatrix,
                                                    inputMatrix + size,
                                                    [](float lh, float rh){
        return std::fabs(lh) < std::fabs(rh);
    }))};
    
    const float scaleFactor{static_cast<float>(std::numeric_limits<T>::max())/
                                                                    inputValueMax};

    std::transform(inputMatrix,
                    inputMatrix + size,
                    outputMatrix,
                    [scaleFactor](const float input) -> T {
                        return static_cast<T>(input*scaleFactor);
    });
    
    return scaleFactor;
}

template<typename T> float quantizeLinearAndPad(const float* const inputMatrix,
                                                    T* const outputMatrix,
                                                    const size_t heightOriginal,
                                                    const size_t widthOriginal,
                                                    const size_t heightTarget,
                                                    const size_t widthTarget)
{
    
    if(heightTarget < heightOriginal)
    {
        throw std::invalid_argument("MpuSim Wrapper: quantizeLinearAndPad "
                                        "target height smaller than original height");
    }
    
    if(widthTarget < widthOriginal)
    {
        throw std::invalid_argument("MpuSim Wrapper: quantizeLinearAndPad "
                                        "target width smaller than original width");
    }
    
    const size_t elementCountOriginal{heightOriginal*widthOriginal};
    const size_t elementCountTarget{heightTarget*widthTarget};
    
    const float inputValueMax{*(std::max_element(inputMatrix,
                                                    inputMatrix +
                                                        elementCountOriginal,
                                                    [](float lh, float rh){
        return std::fabs(lh) < std::fabs(rh);
    }))};
    
    const float scaleFactor{static_cast<float>(std::numeric_limits<T>::max())/
                                                                    inputValueMax};
                                                                    
    std::fill(outputMatrix, outputMatrix + elementCountTarget, T{0});

    for(size_t rowCount{0UL}; rowCount < heightOriginal; ++rowCount)
    {
        for(size_t columnCount{0UL}; columnCount < widthOriginal; ++columnCount)
        {
            outputMatrix[rowCount*widthTarget + columnCount] = 
                            static_cast<T>(inputMatrix[rowCount*widthOriginal + columnCount]*
                                                                                    scaleFactor);
        }
    }
    
    return scaleFactor;
}

template<typename T> void scaleToFactor(const T* const inputMatrix,
                                                float* const outputMatrix,
                                                const float factor,
                                                const size_t size)

{
    std::transform(inputMatrix,
                    inputMatrix + size,
                    outputMatrix,
                    [factor](const decltype(*inputMatrix) input) -> float {
                        return static_cast<float>(input*factor);
    });
}

template<typename T> void scaleToFactorAndCrop(const T* const inputMatrix,
                                                float* const outputMatrix,
                                                const float factor,
                                                const size_t heightOriginal,
                                                const size_t widthOriginal,
                                                const size_t heightCropped,
                                                const size_t widthCropped)
{    
    if(heightOriginal < heightCropped)
    {
        throw std::invalid_argument("MpuSim Wrapper: scaleToFactorAndCrop "
                                        "target height larger than original height");
    }

    if(heightOriginal < heightCropped)
    {
        throw std::invalid_argument("MpuSim Wrapper: scaleToFactorAndCrop "
                                        "target width larger than original width");
    }

    for(size_t rowCount{0UL}; rowCount < heightCropped; ++rowCount)
    {
        std::transform(inputMatrix + rowCount*widthOriginal,
                        inputMatrix + rowCount*widthOriginal + widthCropped,
                        outputMatrix + rowCount*widthCropped,
                        [factor](const decltype(*inputMatrix) input) -> float {
                            return static_cast<float>(input*factor);
        });
    }
}


constexpr size_t unifiedBufferSizeMaxByte{1024UL*1024UL*1024UL};

constexpr size_t parameterDatatypeSizesCombined8_8_8{
                            combineParameterDatatypeSizes(1UL, 1UL, 1UL)};

constexpr size_t parameterDatatypeSizesCombined8_8_16{
                            combineParameterDatatypeSizes(1UL, 1UL, 2UL)};
constexpr size_t parameterDatatypeSizesCombined16_8_16{
                            combineParameterDatatypeSizes(2UL, 1UL, 2UL)};
constexpr size_t parameterDatatypeSizesCombined8_16_16{
                            combineParameterDatatypeSizes(1UL, 2UL, 2UL)};
constexpr size_t parameterDatatypeSizesCombined16_16_16{
                            combineParameterDatatypeSizes(2UL, 2UL, 2UL)};

constexpr size_t parameterDatatypeSizesCombined8_8_32{
                            combineParameterDatatypeSizes(1UL, 1UL, 4UL)};
constexpr size_t parameterDatatypeSizesCombined16_8_32{
                            combineParameterDatatypeSizes(2UL, 1UL, 4UL)};
constexpr size_t parameterDatatypeSizesCombined32_8_32{
                            combineParameterDatatypeSizes(4UL, 1UL, 4UL)};
constexpr size_t parameterDatatypeSizesCombined8_16_32{
                            combineParameterDatatypeSizes(1UL, 2UL, 4UL)};
constexpr size_t parameterDatatypeSizesCombined16_16_32{
                            combineParameterDatatypeSizes(2UL, 2UL, 4UL)};
constexpr size_t parameterDatatypeSizesCombined32_16_32{
                            combineParameterDatatypeSizes(4UL, 2UL, 4UL)};
constexpr size_t parameterDatatypeSizesCombined8_32_32{
                            combineParameterDatatypeSizes(1UL, 4UL, 4UL)};
constexpr size_t parameterDatatypeSizesCombined16_32_32{
                            combineParameterDatatypeSizes(2UL, 4UL, 4UL)};
constexpr size_t parameterDatatypeSizesCombined32_32_32{
                            combineParameterDatatypeSizes(4UL, 4UL, 4UL)};

constexpr size_t parameterDatatypeSizesCombined8_8_64{
                            combineParameterDatatypeSizes(1UL, 1UL, 8UL)};
constexpr size_t parameterDatatypeSizesCombined16_8_64{
                            combineParameterDatatypeSizes(2UL, 1UL, 8UL)};
constexpr size_t parameterDatatypeSizesCombined32_8_64{
                            combineParameterDatatypeSizes(4UL, 1UL, 8UL)};
constexpr size_t parameterDatatypeSizesCombined64_8_64{
                            combineParameterDatatypeSizes(8UL, 1UL, 8UL)};
constexpr size_t parameterDatatypeSizesCombined8_16_64{
                            combineParameterDatatypeSizes(1UL, 2UL, 8UL)};
constexpr size_t parameterDatatypeSizesCombined16_16_64{
                            combineParameterDatatypeSizes(2UL, 2UL, 8UL)};
constexpr size_t parameterDatatypeSizesCombined32_16_64{
                            combineParameterDatatypeSizes(4UL, 2UL, 8UL)};
constexpr size_t parameterDatatypeSizesCombined64_16_64{
                            combineParameterDatatypeSizes(8UL, 2UL, 8UL)};
constexpr size_t parameterDatatypeSizesCombined8_32_64{
                            combineParameterDatatypeSizes(1UL, 4UL, 8UL)};
constexpr size_t parameterDatatypeSizesCombined16_32_64{
                            combineParameterDatatypeSizes(2UL, 4UL, 8UL)};
constexpr size_t parameterDatatypeSizesCombined32_32_64{
                            combineParameterDatatypeSizes(4UL, 4UL, 8UL)};
constexpr size_t parameterDatatypeSizesCombined64_32_64{
                            combineParameterDatatypeSizes(8UL, 4UL, 8UL)};
constexpr size_t parameterDatatypeSizesCombined8_64_64{
                            combineParameterDatatypeSizes(1UL, 8UL, 8UL)};
constexpr size_t parameterDatatypeSizesCombined16_64_64{
                            combineParameterDatatypeSizes(2UL, 8UL, 8UL)};
constexpr size_t parameterDatatypeSizesCombined32_64_64{
                            combineParameterDatatypeSizes(4UL, 8UL, 8UL)};
constexpr size_t parameterDatatypeSizesCombined64_64_64{
                            combineParameterDatatypeSizes(8UL, 8UL, 8UL)};
    
}

void MpuSimWrapper::runMultiplication(const size_t activationsDatatypeSizeByte,
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
                                                const std::string& operationNameString)
{

    
    const bool matrixNeedsPadding{((sizeN > systolicArrayWidth) &&
                                    (sizeK <= systolicArrayHeight))};
    
//     const size_t sizeNPadded{matrixNeedsPadding &&
//                                 (sizeN <= systolicArrayWidth) ?
//                                                     systolicArrayWidth + 1UL : sizeN};
                                    
    const size_t sizeNPadded{sizeN};
                                                            
    const size_t sizeKPadded{matrixNeedsPadding &&
                                    (sizeK <= systolicArrayHeight) ?
                                                    systolicArrayHeight + 1UL : sizeK};

    if(matrixNeedsPadding)
    {
        std::cout << "MpuSim Wrapper: Padding matrix in dimension ";
        
        if(sizeN <= systolicArrayWidth)
        {
            std::cout << "N, old size: "
                        << sizeN
                        << ", new size: "
                        << sizeNPadded;
        }

        else
        {
            std::cout << "K, old size: "
                        << sizeK
                        << ", new size: "
                        << sizeKPadded;
        }
        std::cout << std::endl;
    }

    mpusim::byte* const quantizationBufferPtr{
                            QuantizationBufferSingleton::getInstance().get(
                                                sizeKPadded*sizeNPadded*weightsDatatypeSizeByte +
                                                sizeM*sizeKPadded*activationsDatatypeSizeByte + 
                                                sizeM*sizeNPadded*resultsDatatypeSizeByte)};

    const size_t parameterDatatypeSizesCombinedCurrent{
                            combineParameterDatatypeSizes(m_weightsDatatypeSizeByteCurrent,
                                                            m_activationsDatatypeSizeByteCurrent,
                                                            m_resultsDatatypeSizeByteCurrent)};


    const size_t parameterDatatypeSizesCombinedNext{
                            combineParameterDatatypeSizes(weightsDatatypeSizeByte,
                                                            activationsDatatypeSizeByte,
                                                            resultsDatatypeSizeByte)};

    if((parameterDatatypeSizesCombinedNext !=
                        parameterDatatypeSizesCombinedCurrent) ||
            (systolicArrayHeight != m_systolicArrayHeightCurrent) ||
            (systolicArrayWidth != m_systolicArrayWidthCurrent) || 
            (activationFifoDepth != m_activationFifoDepthCurrent) ||
            (accumulatorArrayHeight != m_accumulatorArrayHeightCurrent))
    {
        switch(parameterDatatypeSizesCombinedCurrent)
        {
            case parameterDatatypeSizesCombined8_8_8:
            {
                delete m_matrixProcessingUnit8_8_8Ptr;
                break;
            }
                
            case parameterDatatypeSizesCombined8_8_16:
            {
                delete m_matrixProcessingUnit8_8_16Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined16_8_16:
            {
                delete m_matrixProcessingUnit16_8_16Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined8_16_16:
            {
                delete m_matrixProcessingUnit8_16_16Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined16_16_16:
            {
                delete m_matrixProcessingUnit16_16_16Ptr;
                break;
            }
                
            case parameterDatatypeSizesCombined8_8_32:
            {
                delete m_matrixProcessingUnit8_8_32Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined16_8_32:
            {
                delete m_matrixProcessingUnit16_8_32Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined32_8_32:
            {
                delete m_matrixProcessingUnit32_8_32Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined8_16_32:
            {
                delete m_matrixProcessingUnit8_16_32Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined16_16_32:
            {
                delete m_matrixProcessingUnit16_16_32Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined32_16_32:
            {
                delete m_matrixProcessingUnit32_16_32Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined8_32_32:
            {
                delete m_matrixProcessingUnit8_32_32Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined16_32_32:
            {
                delete m_matrixProcessingUnit16_32_32Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined32_32_32:
            {
                delete m_matrixProcessingUnit32_32_32Ptr;
                break;
            }
                
            case parameterDatatypeSizesCombined8_8_64:
            {
                delete m_matrixProcessingUnit8_8_64Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined16_8_64:
            {
                delete m_matrixProcessingUnit16_8_64Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined32_8_64:
            {
                delete m_matrixProcessingUnit32_8_64Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined64_8_64:
            {
                delete m_matrixProcessingUnit64_8_64Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined8_16_64:
            {
                delete m_matrixProcessingUnit8_16_64Ptr;
                break;
            }
                
            case parameterDatatypeSizesCombined16_16_64:
            {
                delete m_matrixProcessingUnit16_16_64Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined32_16_64:
            {
                delete m_matrixProcessingUnit32_16_64Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined64_16_64:
            {
                delete m_matrixProcessingUnit64_16_64Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined8_32_64:
            {
                delete m_matrixProcessingUnit8_32_64Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined16_32_64:
            {
                delete m_matrixProcessingUnit16_32_64Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined32_32_64:
            {
                delete m_matrixProcessingUnit32_32_64Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined64_32_64:
            {
                delete m_matrixProcessingUnit64_32_64Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined8_64_64:
            {
                delete m_matrixProcessingUnit8_64_64Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined16_64_64:
            {
                delete m_matrixProcessingUnit16_64_64Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined32_64_64:
            {
                delete m_matrixProcessingUnit32_64_64Ptr;
                break;
            }
            
            case parameterDatatypeSizesCombined64_64_64:
            {
                delete m_matrixProcessingUnit64_64_64Ptr;
                break;
            }
        
            default:
                break;
        }
        
        if(parameterDatatypeSizesCombinedNext !=
                        parameterDatatypeSizesCombinedCurrent)
        {
            if(m_mpuStatisticsLoggerPtr)
            {
                delete m_mpuStatisticsLoggerPtr;
            }
            
            m_mpuStatisticsLoggerPtr = new MpuStatisticsLogger(
                                                std::string{logFileOutputDirString +
                                                                    std::string{"/"} +
                                                                    modelNameString},
                                                weightsDatatypeSizeByte,
                                                activationsDatatypeSizeByte,
                                                resultsDatatypeSizeByte);
        }
        
        switch(parameterDatatypeSizesCombinedNext)
        {
            case parameterDatatypeSizesCombined8_8_8:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit8_8_8Ptr, int8_t, int8_t, int8_t);
                break;
            }
                
            case parameterDatatypeSizesCombined8_8_16:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit8_8_16Ptr, int8_t, int8_t, int16_t);
                break;
            }
            
            case parameterDatatypeSizesCombined16_8_16:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit16_8_16Ptr, int16_t, int8_t, int16_t);
                break;
            }
            
            case parameterDatatypeSizesCombined8_16_16:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit8_16_16Ptr, int8_t, int16_t, int16_t);
                break;
            }
            
            case parameterDatatypeSizesCombined16_16_16:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit16_16_16Ptr, int16_t, int16_t, int16_t);
                break;
            }
                
            case parameterDatatypeSizesCombined8_8_32:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit8_8_32Ptr, int8_t, int8_t, int32_t);
                break;
            }
            
            case parameterDatatypeSizesCombined16_8_32:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit16_8_32Ptr, int16_t, int8_t, int32_t);
                break;
            }
            
            case parameterDatatypeSizesCombined32_8_32:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit32_8_32Ptr, int32_t, int8_t, int32_t);
                break;
            }
            
            case parameterDatatypeSizesCombined8_16_32:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit8_16_32Ptr, int8_t, int16_t, int32_t);
                break;
            }
            
            case parameterDatatypeSizesCombined16_16_32:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit16_16_32Ptr, int16_t, int16_t, int32_t);
                break;
            }
            
            case parameterDatatypeSizesCombined32_16_32:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit32_16_32Ptr, int32_t, int16_t, int32_t);
                break;
            }
            
            case parameterDatatypeSizesCombined8_32_32:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit8_32_32Ptr, int8_t, int32_t, int32_t);
                break;
            }
            
            case parameterDatatypeSizesCombined16_32_32:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit16_32_32Ptr, int16_t, int32_t, int32_t);
                break;
            }
            
            case parameterDatatypeSizesCombined32_32_32:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit32_32_32Ptr, int32_t, int32_t, int32_t);
                break;
            }
                
            case parameterDatatypeSizesCombined8_8_64:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit8_8_64Ptr, int8_t, int8_t, int64_t);
                break;
            }
            
            case parameterDatatypeSizesCombined16_8_64:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit16_8_64Ptr, int16_t, int8_t, int64_t);
                break;
            }
            
            case parameterDatatypeSizesCombined32_8_64:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit32_8_64Ptr, int32_t, int8_t, int64_t);
                break;
            }
            
            case parameterDatatypeSizesCombined64_8_64:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit64_8_64Ptr, int64_t, int8_t, int64_t);
                break;
            }
            
            case parameterDatatypeSizesCombined8_16_64:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit8_16_64Ptr, int8_t, int16_t, int64_t);
                break;
            }
                
            case parameterDatatypeSizesCombined16_16_64:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit16_16_64Ptr, int16_t, int16_t, int64_t);
                break;
            }
            
            case parameterDatatypeSizesCombined32_16_64:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit32_16_64Ptr, int32_t, int16_t, int64_t);
                break;
            }
            
            case parameterDatatypeSizesCombined64_16_64:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit64_16_64Ptr, int64_t, int16_t, int64_t);
                break;
            }
            
            case parameterDatatypeSizesCombined8_32_64:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit8_32_64Ptr, int8_t, int32_t, int64_t);
                break;
            }
            
            case parameterDatatypeSizesCombined16_32_64:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit16_32_64Ptr, int16_t, int32_t, int64_t);
                break;
            }
            
            case parameterDatatypeSizesCombined32_32_64:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit32_32_64Ptr, int32_t, int32_t, int64_t);
                break;
            }
            
            case parameterDatatypeSizesCombined64_32_64:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit64_32_64Ptr, int64_t, int32_t, int64_t);
                break;
            }
            
            case parameterDatatypeSizesCombined8_64_64:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit8_64_64Ptr, int8_t, int64_t, int64_t);
                break;
            }
            
            case parameterDatatypeSizesCombined16_64_64:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit16_64_64Ptr, int16_t, int64_t, int64_t);
                break;
            }
            
            case parameterDatatypeSizesCombined32_64_64:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit32_64_64Ptr, int32_t, int64_t, int64_t);
                break;
            }
            
            case parameterDatatypeSizesCombined64_64_64:
            {
                CONSTRUCT_AND_CONFIGURE_MPU(m_matrixProcessingUnit64_64_64Ptr, int64_t, int64_t, int64_t);
                break;
            }
        
            default:
            {
                throw std::invalid_argument("MpuSim Wrapper: One or more parameter datatype "
                                                    "size parameters have an unsupported value");
            }
        }
        
        m_weightsDatatypeSizeByteCurrent =
                            weightsDatatypeSizeByte;
        
        m_activationsDatatypeSizeByteCurrent =
                            activationsDatatypeSizeByte;
                            
        m_resultsDatatypeSizeByteCurrent =
                            resultsDatatypeSizeByte;
                            
        m_systolicArrayHeightCurrent =
                            systolicArrayHeight;
                            
        m_systolicArrayWidthCurrent =
                            systolicArrayWidth;
                        
        m_activationFifoDepthCurrent =
                            activationFifoDepth;
                            
        m_accumulatorArrayHeightCurrent =
                            accumulatorArrayHeight;
                            
    }
    
    switch(parameterDatatypeSizesCombinedNext)
    {
        case parameterDatatypeSizesCombined8_8_8:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit8_8_8Ptr,
                                                                int8_t, int8_t, int8_t);
            break;
        }
            
        case parameterDatatypeSizesCombined8_8_16:
        {            
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit8_8_16Ptr,
                                                                int8_t, int8_t, int16_t);
            break;
        }
        
        case parameterDatatypeSizesCombined16_8_16:
        {            
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit16_8_16Ptr,
                                                                int16_t, int8_t, int16_t);
            break;
        }
        
        case parameterDatatypeSizesCombined8_16_16:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit8_16_16Ptr,
                                                                int8_t, int16_t, int16_t);
            break;
        }
        
        case parameterDatatypeSizesCombined16_16_16:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit16_16_16Ptr,
                                                                int16_t, int16_t, int16_t);
            break;
        }
            
        case parameterDatatypeSizesCombined8_8_32:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit8_8_32Ptr,
                                                                int8_t, int8_t, int32_t);
            break;
        }
        
        case parameterDatatypeSizesCombined16_8_32:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit16_8_32Ptr,
                                                                int16_t, int8_t, int32_t);
            break;
        }
        
        case parameterDatatypeSizesCombined32_8_32:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit32_8_32Ptr,
                                                                int32_t, int8_t, int32_t);
            break;
        }
        
        case parameterDatatypeSizesCombined8_16_32:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit8_16_32Ptr,
                                                                int8_t, int16_t, int32_t);
            break;
        }
        
        case parameterDatatypeSizesCombined16_16_32:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit16_16_32Ptr,
                                                                int16_t, int16_t, int32_t);
            break;
        }
        
        case parameterDatatypeSizesCombined32_16_32:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit32_16_32Ptr,
                                                                int32_t, int16_t, int32_t);
            break;
        }
        
        case parameterDatatypeSizesCombined8_32_32:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit8_32_32Ptr,
                                                                int8_t, int32_t, int32_t);
            break;
        }
        
        case parameterDatatypeSizesCombined16_32_32:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit16_32_32Ptr,
                                                                int16_t, int32_t, int32_t);
            break;
        }
        
        case parameterDatatypeSizesCombined32_32_32:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit32_32_32Ptr,
                                                                int32_t, int32_t, int32_t);
            break;
        }
            
        case parameterDatatypeSizesCombined8_8_64:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit8_8_64Ptr,
                                                                int8_t, int8_t, int64_t);
            break;
        }
        
        case parameterDatatypeSizesCombined16_8_64:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit16_8_64Ptr,
                                                                int16_t, int8_t, int64_t);
            break;
        }
        
        case parameterDatatypeSizesCombined32_8_64:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit32_8_64Ptr,
                                                                int32_t, int8_t, int64_t);
            break;
        }
        
        case parameterDatatypeSizesCombined64_8_64:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit64_8_64Ptr,
                                                                int64_t, int8_t, int64_t);
            break;
        }
        
        case parameterDatatypeSizesCombined8_16_64:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit8_16_64Ptr,
                                                                int8_t, int16_t, int64_t);
            break;
        }
            
        case parameterDatatypeSizesCombined16_16_64:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit16_16_64Ptr,
                                                                int16_t, int16_t, int64_t);
            break;
        }
        
        case parameterDatatypeSizesCombined32_16_64:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit32_16_64Ptr,
                                                                int32_t, int16_t, int64_t);
            break;
        }
        
        case parameterDatatypeSizesCombined64_16_64:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit64_16_64Ptr,
                                                                int64_t, int16_t, int64_t);
            break;
        }
        
        case parameterDatatypeSizesCombined8_32_64:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit8_32_64Ptr,
                                                                int8_t, int32_t, int64_t);
            break;
        }
        
        case parameterDatatypeSizesCombined16_32_64:
        {
            QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit16_32_64Ptr,
                                                                int16_t, int32_t, int64_t);
            break;
        }
        
        case parameterDatatypeSizesCombined32_32_64:
        {
              QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit32_32_64Ptr,
                                                                int32_t, int32_t, int64_t);
            break;
        }
        
        case parameterDatatypeSizesCombined64_32_64:
        {
              QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit64_32_64Ptr,
                                                                int64_t, int32_t, int64_t);
            break;
        }
        
        case parameterDatatypeSizesCombined8_64_64:
        {
              QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit8_64_64Ptr,
                                                                int8_t, int64_t, int64_t);
            break;
        }
        
        case parameterDatatypeSizesCombined16_64_64:
        {
              QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit16_64_64Ptr,
                                                                int16_t, int64_t, int64_t);
            break;
        }
        
        case parameterDatatypeSizesCombined32_64_64:
        {
              QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit32_64_64Ptr,
                                                                int32_t, int64_t, int64_t);
            break;
        }
        
        case parameterDatatypeSizesCombined64_64_64:
        {
              QUANTIZE_AND_RUN_MATRIX_MULTIPLICATION(m_matrixProcessingUnit64_64_64Ptr,
                                                                int64_t, int64_t, int64_t);
            break;
        }
    
        default:
        {
            throw std::invalid_argument("MpuSim Wrapper: One or more parameter datatype "
                                                "size parameters have an unsupported value");
        }
    }
}

MpuSimWrapper::~MpuSimWrapper()
{
    switch(combineParameterDatatypeSizes(m_weightsDatatypeSizeByteCurrent,
                                            m_activationsDatatypeSizeByteCurrent,
                                            m_resultsDatatypeSizeByteCurrent))
    {
        case parameterDatatypeSizesCombined8_8_8:
        {
            m_matrixProcessingUnit8_8_8Ptr->printUnifiedBufferLayout();
            delete m_matrixProcessingUnit8_8_8Ptr;
            break;
        }
            
        case parameterDatatypeSizesCombined8_8_16:
        {
            delete m_matrixProcessingUnit8_8_16Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined16_8_16:
        {
            delete m_matrixProcessingUnit16_8_16Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined8_16_16:
        {
            delete m_matrixProcessingUnit8_16_16Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined16_16_16:
        {
            delete m_matrixProcessingUnit16_16_16Ptr;
            break;
        }
            
        case parameterDatatypeSizesCombined8_8_32:
        {
            delete m_matrixProcessingUnit8_8_32Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined16_8_32:
        {
            delete m_matrixProcessingUnit16_8_32Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined32_8_32:
        {
            delete m_matrixProcessingUnit32_8_32Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined8_16_32:
        {
            delete m_matrixProcessingUnit8_16_32Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined16_16_32:
        {
            delete m_matrixProcessingUnit16_16_32Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined32_16_32:
        {
            delete m_matrixProcessingUnit32_16_32Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined8_32_32:
        {
            delete m_matrixProcessingUnit8_32_32Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined16_32_32:
        {
            delete m_matrixProcessingUnit16_32_32Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined32_32_32:
        {
            delete m_matrixProcessingUnit32_32_32Ptr;
            break;
        }
            
        case parameterDatatypeSizesCombined8_8_64:
        {
            delete m_matrixProcessingUnit8_8_64Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined16_8_64:
        {
            delete m_matrixProcessingUnit16_8_64Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined32_8_64:
        {
            delete m_matrixProcessingUnit32_8_64Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined64_8_64:
        {
            delete m_matrixProcessingUnit64_8_64Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined8_16_64:
        {
            delete m_matrixProcessingUnit8_16_64Ptr;
            break;
        }
            
        case parameterDatatypeSizesCombined16_16_64:
        {
            delete m_matrixProcessingUnit16_16_64Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined32_16_64:
        {
            delete m_matrixProcessingUnit32_16_64Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined64_16_64:
        {
            delete m_matrixProcessingUnit64_16_64Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined8_32_64:
        {
            delete m_matrixProcessingUnit8_32_64Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined16_32_64:
        {
            delete m_matrixProcessingUnit16_32_64Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined32_32_64:
        {
            delete m_matrixProcessingUnit32_32_64Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined64_32_64:
        {
            delete m_matrixProcessingUnit64_32_64Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined8_64_64:
        {
            delete m_matrixProcessingUnit8_64_64Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined16_64_64:
        {
            delete m_matrixProcessingUnit16_64_64Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined32_64_64:
        {
            delete m_matrixProcessingUnit32_64_64Ptr;
            break;
        }
        
        case parameterDatatypeSizesCombined64_64_64:
        {
            delete m_matrixProcessingUnit64_64_64Ptr;
            break;
        }
    
        default:
            break;
    }
    
    if(m_mpuStatisticsLoggerPtr)
    {
        delete m_mpuStatisticsLoggerPtr;
    }

    std::cout << "Deleted MPU simulator wrapper object" << std::endl;
}
