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
 * @file        memory_management_unit.h
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   MIT License
 */

#ifndef MEMORY_MANAGEMENT_UNIT_H
#define MEMORY_MANAGEMENT_UNIT_H

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <tuple>
#include <utility>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <type_traits>
#include <cmath>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <climits>

#include "mpu_exception.h"

//#define MEMORY_MANAGEMENT_UNIT_DEBUG MEMORY_MANAGEMENT_UNIT_DEBUG

namespace mpusim
{
using byte = unsigned char;
}

/**
 * @struct  WeightMatrixDopeVector
 * @brief   
 */

struct WeightMatrixDopeVector
{

    /**
     * @brief
     * @param address
     * @param rows
     * @param columns
     */
    
    WeightMatrixDopeVector(const size_t address,
                                const size_t rows,
                                const size_t columns): address{address},
                                                                rows{rows},
                                                                columns{columns}
    {
    }

    const size_t address;

    const size_t rows;
    const size_t columns;
};


/**
 * @class                       MemoryManagementUnit
 * @tparam WeightDatatype       
 * @tparam ActivationDatatype   
 * @tparam ResultDatatype       
 */

template<typename WeightDatatype, typename ActivationDatatype, typename ResultDatatype> class MemoryManagementUnit
{

public:
    
    /**
     * @brief
     * @param unifiedBufferPtr
     * @param unifiedBufferSizeByteMax
     * @param unifiedBufferDynamicResize
     */

    MemoryManagementUnit(std::vector<mpusim::byte>* const unifiedBufferPtr,
                                            const size_t unifiedBufferSizeByteMax,
                                            const bool unifiedBufferDynamicResize = true):
                                                                    m_unifiedBufferPtr{unifiedBufferPtr},
                                                                    m_unifiedBufferSizeByteMax{
                                                                                unifiedBufferSizeByteMax}

    {
        setUnifiedBufferDynamicResize(unifiedBufferDynamicResize);
    }

    size_t getMemoryUsageMaxByte() const
    {
        return m_weightMatrixSpaceEnd +
                m_combinedActivationAndResultMatrixSpacesSizeMaxByte;
    }

    size_t getMemoryUsageMaxBit() const
    {
        return getMemoryUsageMaxByte()*CHAR_BIT;
    }

    void setUnifiedBufferDynamicResize(const bool unifiedBufferDynamicResize)
    {
        m_unifiedBufferDynamicResize = unifiedBufferDynamicResize;

        if(m_unifiedBufferDynamicResize)
        {
            m_unifiedBufferPtr->resize(m_resultMatrixSpaceEnd);
        }

        else
        {
            m_unifiedBufferPtr->resize(m_unifiedBufferSizeByteMax);
        }
    }

    void loadFromUnifiedBuffer(mpusim::byte* const dest,
                                const mpusim::byte* const src,
                                const size_t size) const
    {
        assert((src >= &(*m_unifiedBufferPtr->begin())) &&
                ((src + size) < &(*m_unifiedBufferPtr->end())));

        if((src < &(*m_unifiedBufferPtr->begin())) ||
            ((src + size) >= &(*m_unifiedBufferPtr->end())))
        {
            throw MpuException("Memory management unit: MPU unified "
                                "buffer load operation source address "
                                "outside MPU address space");
        }

        std::copy(src, src + size, dest);
    }

    void storeToUnifiedBuffer(mpusim::byte* const dest,
                                const mpusim::byte* const src,
                                const size_t size)
    {
        assert((dest >= &(*m_unifiedBufferPtr->begin())) &&
                ((dest + size) < &(*m_unifiedBufferPtr->end())));

        if((dest < &(*m_unifiedBufferPtr->begin())) ||
            ((dest + size) >= &(*m_unifiedBufferPtr->end())))
        {
            throw MpuException("Memory management unit: MPU unified "
                                "buffer store operation destination "
                                "address outside MPU address space");
        }

        std::copy(src, src + size, dest);
    }

    std::pair<const size_t, const size_t> getWeightMatrixDimensionsManaged(
                                                    const std::string& operationName) const
    {
        if(m_weightMatrixDopeVectorMap.find(operationName) ==
                                        m_weightMatrixDopeVectorMap.end())
        {
            throw MpuException("Memory management unit: Requested "
                                "weight matrix not present in unified buffer");
        }

        return std::pair<const size_t, const size_t>(
                            m_weightMatrixDopeVectorMap.at(operationName).rows,
                            m_weightMatrixDopeVectorMap.at(operationName).columns);

    }

    WeightDatatype* getWeightMatrixPtrManaged(const std::string& operationName) const
    {
        if(m_weightMatrixDopeVectorMap.find(operationName) ==
                                        m_weightMatrixDopeVectorMap.end())
        {
            throw MpuException("Memory management unit: Requested "
                                "weight matrix not present in unified buffer");
        }

        return reinterpret_cast<WeightDatatype*>(
                                        m_unifiedBufferPtr->data() +
                                        m_weightMatrixDopeVectorMap.at(operationName).address);

    }

    /**
     * @brief
     * @param oparationName
     * @param src
     * @param rows
     * @param columns
     */
    
    void storeWeightMatrixManaged(const std::string& operationName,
                                    const WeightDatatype* const src,
                                    const size_t rows,
                                    const size_t columns)
    {

        if(operationName.empty())
        {
            throw MpuException("Memory management unit: "
                                "Cannot use empty string as "
                                "weight matrix identifier");
        }

        if((rows == 0UL) || (columns == 0UL))
        {
            throw MpuException("Memory management unit: "
                                "Cannot store weight matrices"
                                "with a row count or column "
                                "count of zero");
        }

        const mpusim::byte* const srcPtrByte{
                                    reinterpret_cast<const mpusim::byte* const>(src)};

        const size_t sizeByte{rows*columns*sizeof(WeightDatatype)};

        if(m_weightMatrixDopeVectorMap.find(operationName) ==
                                        m_weightMatrixDopeVectorMap.end())
        {

            if((m_resultMatrixSpaceEnd + sizeByte) > m_unifiedBufferSizeByteMax)
            {
                throw MpuException("Memory management unit: Cannot store "
                                    "weight matrix to MPU unified buffer, "
                                    "as new unified buffer size would "
                                    "exceed maximum allowed size");
            }

            if(m_unifiedBufferDynamicResize)
            {

                m_unifiedBufferPtr->insert(m_unifiedBufferPtr->begin() +
                                                    m_weightMatrixSpaceEnd,
                                                    srcPtrByte,
                                                    srcPtrByte + sizeByte);
            }

            else
            {
                const std::vector<mpusim::byte> activationAndResultMatrixBuffer(
                                                            m_unifiedBufferPtr->begin() +
                                                                        m_weightMatrixSpaceEnd,
                                                            m_unifiedBufferPtr->begin() +
                                                                        m_resultMatrixSpaceEnd);

                std::copy(activationAndResultMatrixBuffer.begin(),
                            activationAndResultMatrixBuffer.end(),
                            m_unifiedBufferPtr->begin() +
                            m_weightMatrixSpaceEnd + sizeByte);

                std::copy(srcPtrByte,
                            srcPtrByte + sizeByte,
                            m_unifiedBufferPtr->begin() +
                                    m_weightMatrixSpaceEnd);

            }

#ifdef MEMORY_MANAGEMENT_UNIT_DEBUG
            std::cout << "Memory management unit:\n\tAdded "
                            "weight matrix for operation \""
                        << operationName
                        << "\"\n\tAddress: 0x"
                        <<  std::hex << m_weightMatrixSpaceEnd
                        << "\n\tSize: " << std::dec << sizeByte
                        << " byte\n\tActivation matrix "
                                "start address is now 0x"
                        << std::hex
                        << m_weightMatrixSpaceEnd + sizeByte
                        << "\n\rResult matrix space "
                                "start address is now 0x"
                        << m_activationMatrixSpaceEnd + sizeByte
                        << std::endl;
#endif

            m_weightMatrixDopeVectorMap.emplace(operationName,
                                                    WeightMatrixDopeVector(
                                                            m_weightMatrixSpaceEnd,
                                                                        rows, columns));

            m_weightMatrixSpaceEnd += sizeByte;
            m_activationMatrixSpaceEnd += sizeByte;
            m_resultMatrixSpaceEnd += sizeByte;

#ifdef MEMORY_MANAGEMENT_UNIT_DEBUG
            std::cout << "\n\tTotal memory size: "
                        << m_resultMatrixSpaceEnd << " byte"
                        << std::endl;
#endif

        }

        else
        {
#ifdef MEMORY_MANAGEMENT_UNIT_DEBUG
            std::cout << "Memory management unit: Weight matrix "
                            "for operation \"" << operationName
                        << "\" already present in unified buffer" << std::endl;
#endif
        }
    }

    std::pair<const size_t, const size_t> getActivationMatrixDimensionsManaged() const
    {
        return std::pair<const size_t, const size_t>(
                                    m_activationMatrixRows,
                                    m_activationMatrixColumns);
    }

    ActivationDatatype* getActivationMatrixPtrManaged() const
    {
        return reinterpret_cast<ActivationDatatype*>(m_unifiedBufferPtr->data() +
                                                            m_weightMatrixSpaceEnd);
    }
    
    /**
     * @brief
     * @param src
     * @param rows
     * @param columns
     */

    void storeActivationMatrixManaged(const ActivationDatatype* const src,
                                                            const size_t rows,
                                                            const size_t columns)
    {
        if((rows == 0UL) || (columns == 0UL))
        {
            throw MpuException("Memory management unit: "
                                "Cannot store activation "
                                "matrices with a row count "
                                "or column count of zero");
        }

        m_activationMatrixRows = rows;
        m_activationMatrixColumns = columns;

        const mpusim::byte* const srcPtrByte{
                                    reinterpret_cast<const mpusim::byte* const>(src)};

        const ssize_t sizeByte = rows*columns*sizeof(ActivationDatatype);

        const ssize_t activationMatrixSpaceGrowth = sizeByte -
                                                    m_activationMatrixSpaceEnd +
                                                    m_weightMatrixSpaceEnd;

        if((static_cast<ssize_t>(m_resultMatrixSpaceEnd) +
                                    activationMatrixSpaceGrowth) >
                                        static_cast<ssize_t>(m_unifiedBufferSizeByteMax))
        {
            throw MpuException("Memory management unit: Cannot store "
                                "activation matrix to MPU unified "
                                "buffer, as new unified buffer size "
                                "would exceed maximum allowed size");
        }

        if(m_unifiedBufferDynamicResize)
        {
            if(activationMatrixSpaceGrowth > 0L)
            {
                m_unifiedBufferPtr->insert(m_unifiedBufferPtr->begin() +
                                                m_activationMatrixSpaceEnd,
                                                activationMatrixSpaceGrowth,
                                                mpusim::byte{0});
            }

            else
            {
                m_unifiedBufferPtr->erase(m_unifiedBufferPtr->begin() +
                                                m_activationMatrixSpaceEnd +
                                                activationMatrixSpaceGrowth,
                                            m_unifiedBufferPtr->begin() +
                                                m_activationMatrixSpaceEnd);
            }
        }

        else
        {
            const std::vector<mpusim::byte> resultMatrixBuffer(
                                                m_unifiedBufferPtr->begin() +
                                                        m_activationMatrixSpaceEnd,
                                                m_unifiedBufferPtr->begin() +
                                                        m_resultMatrixSpaceEnd);

            std::copy(resultMatrixBuffer.begin(),
                            resultMatrixBuffer.end(),
                            m_unifiedBufferPtr->begin() +
                            m_activationMatrixSpaceEnd +
                            activationMatrixSpaceGrowth);
        }

        m_activationMatrixSpaceEnd += activationMatrixSpaceGrowth;
        m_resultMatrixSpaceEnd += activationMatrixSpaceGrowth;
        
        if(sizeByte)

#ifdef MEMORY_MANAGEMENT_UNIT_DEBUG
        if(activationMatrixSpaceGrowth != 0L)
        {
            std::cout << "Memory management unit:\nExtended "
                            "activation matrix space, new size: "
                        << sizeByte
                        << " byte\n\tGrowth from previous size: "
                        << activationMatrixSpaceGrowth
                        << "\n\tResult matrix space "
                                "start address is now 0x"
                        << std::hex
                        << m_activationMatrixSpaceEnd +
                               activationMatrixSpaceGrowth
                        << " byte\n\tTotal memory size: "
                        << std::dec
                        << m_resultMatrixSpaceEnd << " byte"
                        << std::endl;
        }
#endif

        std::copy(srcPtrByte,
                    srcPtrByte + sizeByte,
                    m_unifiedBufferPtr->begin() +
                            m_weightMatrixSpaceEnd);

#ifdef MEMORY_MANAGEMENT_UNIT_DEBUG
        std::cout << "Memory management unit:\n\tStored "
                        "activation matrix\n\tAddress: 0x"
                    << std::hex
                    << m_weightMatrixSpaceEnd
                    << "\n\tSize: " << std::dec
                    << sizeByte << " byte" << std::endl;
#endif

    }

    ResultDatatype* getResultMatrixPtrManaged() const
    {
        return reinterpret_cast<ResultDatatype*>(
                                        m_unifiedBufferPtr->data() +
                                        m_activationMatrixSpaceEnd);
    }

    /**
     * @brief
     * @param rows
     * @param columns
     */
    
    void setResultMatrixSizeManaged(const size_t rows,
                                    const size_t columns)
    {

        m_resultMatrixRows = rows;
        m_resultMatrixColumns = columns;

        const ssize_t sizeByte = m_resultMatrixRows*
                                    m_resultMatrixColumns*
                                    sizeof(ResultDatatype);

        const ssize_t resultMatrixSpaceGrowth = sizeByte -
                                                m_resultMatrixSpaceEnd +
                                                m_activationMatrixSpaceEnd;

        if(static_cast<ssize_t>(m_resultMatrixSpaceEnd +
                                    resultMatrixSpaceGrowth) >
                                        static_cast<ssize_t>(m_unifiedBufferSizeByteMax))
        {
            throw MpuException("Memory management unit: Cannot extend result "
                                "matrix size, as new MPU unified buffer "
                                "size would exceed maximum allowed size");
        }

        m_resultMatrixSpaceEnd += resultMatrixSpaceGrowth;

        if(m_unifiedBufferDynamicResize)
        {
            m_unifiedBufferPtr->resize(m_resultMatrixSpaceEnd);
            std::cout << "Resized unified buffer, new size: "
                        << m_resultMatrixSpaceEnd << " byte" << std::endl;
        }
        
        if(m_combinedActivationAndResultMatrixSpacesSizeMaxByte <
                                                    (m_resultMatrixSpaceEnd -
                                                        m_weightMatrixSpaceEnd))
        {
            m_combinedActivationAndResultMatrixSpacesSizeMaxByte =
                                                    m_resultMatrixSpaceEnd -
                                                        m_weightMatrixSpaceEnd;
            std::cout << "New max combined activation and "
                                    "result matrix space size: "
                        << m_combinedActivationAndResultMatrixSpacesSizeMaxByte 
                        << " byte" << std::endl;
        }

#ifdef MEMORY_MANAGEMENT_UNIT_DEBUG
        std::cout << "Memory management unit:\n\tExtended "
                            "result matrix space, new size: "
                    << sizeByte
                    << " byte\n\tGrowth from previous size: "
                    << resultMatrixSpaceGrowth
                    << " byte\n\tTotal memory size: "
                    << m_resultMatrixSpaceEnd << " byte" << std::endl;
#endif

    }

    /**
     * @brief
     * @param dest
     * @param size
     */
    
    void loadResultMatrixManaged(ResultDatatype* const dest,
                                            const size_t size) const
    {

        mpusim::byte* const destPtrByte{
                            reinterpret_cast<mpusim::byte* const>(dest)};

        const size_t sizeByte{size*sizeof(ResultDatatype)};

        std::copy(m_unifiedBufferPtr->begin() +
                        m_activationMatrixSpaceEnd,
                        m_unifiedBufferPtr->begin() +
                        m_activationMatrixSpaceEnd +
                        sizeByte, destPtrByte);

#ifdef MEMORY_MANAGEMENT_UNIT_DEBUG
        std::cout << "Memory management unit: "
                        "Loaded result matrix, size: "
                    << sizeByte  << " byte" << std::endl;
#endif
    }

    void printMemoryLayout() const
    {
        std::map<size_t, std::tuple<const std::string,
                                        const size_t, const size_t>> weightMatrixDopeVectorMapByAddress;

        size_t weightMatrixOperationNameLengthMax{0UL};

        for(const auto& element : m_weightMatrixDopeVectorMap)
        {
            weightMatrixDopeVectorMapByAddress.emplace(
                                        element.second.address,
                                        std::tuple<const std::string,
                                                    const size_t, const size_t>(
                                                                    element.first,
                                                                    element.second.rows,
                                                                    element.second.columns));

            if(weightMatrixOperationNameLengthMax <
                                            element.first.size())
            {
                weightMatrixOperationNameLengthMax =
                                            element.first.size();
            }
        }

        constexpr size_t textLengthHeader{34UL};
        constexpr size_t textLengthWeightMatrixInfoWithoutOperationName{97UL};

        const size_t weightMatrixOperationNameLength{
                        (weightMatrixOperationNameLengthMax < 9UL) ?
                                9UL : weightMatrixOperationNameLengthMax};

        const size_t lineWidth{
                        weightMatrixOperationNameLength +
                        textLengthWeightMatrixInfoWithoutOperationName};

        const size_t numberSignCountHeaderRight{(lineWidth - textLengthHeader)/2UL};

        const size_t numberSignCountHeaderLeft{
                            (((lineWidth - textLengthHeader) % 2UL) == 1UL) ?
                                                        numberSignCountHeaderRight + 1 :
                                                        numberSignCountHeaderRight};

        std::stringstream headerStringStream;

        headerStringStream << '\n';

        for(size_t charCount{0}; charCount < numberSignCountHeaderLeft;
                                                                ++charCount)
        {
            headerStringStream << '#';
        }

        headerStringStream << " MPU Unified buffer memory layout ";

        for(size_t charCount{0}; charCount < numberSignCountHeaderRight;
                                                               ++charCount)
        {
           headerStringStream << '#';
        }

        headerStringStream << '\n';

        std::stringstream weightMatrixSpaceHeaderStringStream;

        std::stringstream elementSeparatorStringStream;

        for(size_t charCount{0}; charCount < lineWidth; ++charCount)
        {
            elementSeparatorStringStream << '#';
        }

        std::cout << headerStringStream.str()
                    << elementSeparatorStringStream.str();

        for(const auto& element : weightMatrixDopeVectorMapByAddress)
        {
            const std::string weightMatrixSizeByteString{
                                                std::to_string(
                                                    ((std::get<1>(element.second)*
                                                    std::get<2>(element.second)*
                                                    sizeof(WeightDatatype)) > 1024UL) ?
                                                        (std::get<1>(element.second)*
                                                        std::get<2>(element.second)*
                                                        sizeof(WeightDatatype))/1024UL :
                                                        (std::get<1>(element.second)*
                                                        std::get<2>(element.second)*
                                                        sizeof(WeightDatatype))) +
                                                ((std::get<1>(element.second)*
                                                std::get<2>(element.second)*
                                                sizeof(WeightDatatype) > 1024UL) ?
                                                                        " kB" : " B")};

            std::cout << "\n#" << std::setw(lineWidth - 1UL)
                        << '#' << "\n# Address: 0x"
                        << std::left
                        << std::hex << std::setw(10)
                        << element.first
                        << "   Weight matrix "
                        << std::setw(weightMatrixOperationNameLength)
                        << std::get<0>(element.second)
                        << "   Size: "
                        << std::setw(13) << std::dec
                        << weightMatrixSizeByteString
                        << "    Rows: "
                        << std::setw(5)
                        << std::get<1>(element.second)
                        << "    Columns: "
                        << std::setw(5)
                        << std::get<2>(element.second)
                        << std::right
                        << " #\n#" << std::setw(lineWidth - 1UL)
                        << '#' << std::endl
                        << elementSeparatorStringStream.str();

        }

        const std::string activationMatrixSizeByteString{
                                                std::to_string(
                                                    ((m_activationMatrixSpaceEnd -
                                                    m_weightMatrixSpaceEnd) > 1024UL) ?
                                                        (m_activationMatrixSpaceEnd -
                                                        m_weightMatrixSpaceEnd)/1024UL :
                                                        m_activationMatrixSpaceEnd -
                                                        m_weightMatrixSpaceEnd) +
                                                ((m_activationMatrixSpaceEnd -
                                                m_weightMatrixSpaceEnd > 1024UL) ?
                                                                        " kB" : " B")};

        std::cout << "\n#" << std::setw(lineWidth - 1UL)
                    << '#' << "\n# Address: 0x"
                    << std::left
                    << std::hex << std::setw(10)
                    << m_weightMatrixSpaceEnd
                    << std::setw(weightMatrixOperationNameLength + 17)
                    << "   Activation matrix"
                    << "   Size: "
                    << std::setw(13) << std::dec
                    << activationMatrixSizeByteString
                    << "    Rows: "
                    << std::setw(5)
                    << m_activationMatrixRows
                    << "    Columns: "
                    << std::setw(5)
                    << m_activationMatrixColumns
                    << std::right
                    << " #\n#" << std::setw(lineWidth - 1UL)
                    << '#' << std::endl
                    << elementSeparatorStringStream.str();

        const std::string resultMatrixSizeByteString{
                                            std::to_string(
                                                ((m_resultMatrixSpaceEnd -
                                                m_activationMatrixSpaceEnd) > 1024UL) ?
                                                    (m_resultMatrixSpaceEnd -
                                                    m_activationMatrixSpaceEnd)/1024UL :
                                                    m_resultMatrixSpaceEnd -
                                                    m_activationMatrixSpaceEnd) +
                                            ((m_resultMatrixSpaceEnd -
                                            m_activationMatrixSpaceEnd > 1024UL) ?
                                                                        " kB" : " B")};


        std::cout << "\n#" << std::setw(lineWidth - 1UL)
                    << '#' << "\n# Address: 0x"
                    << std::left
                    << std::hex << std::setw(10)
                    << m_activationMatrixSpaceEnd
                    << std::setw(weightMatrixOperationNameLength + 17)
                    << "   Result matrix"
                    << "   Size: "
                    << std::setw(13) << std::dec
                    << resultMatrixSizeByteString
                    << "    Rows: "
                    << std::setw(5)
                    << m_resultMatrixRows
                    << "    Columns: "
                    << std::setw(5)
                    << m_resultMatrixColumns
                    << std::right
                    << " #\n#" << std::setw(lineWidth - 1UL)
                    << '#' << std::endl
                    << elementSeparatorStringStream.str();

        std::cout << '\n'
                    << elementSeparatorStringStream.str()
                    << std::endl;
    }

    void reset()
    {
        m_weightMatrixDopeVectorMap.clear();

        if(m_unifiedBufferDynamicResize)
        {
            m_unifiedBufferPtr->clear();
        }

        m_activationMatrixRows = 0UL;
        m_activationMatrixColumns = 0UL;

        m_resultMatrixRows = 0UL;
        m_resultMatrixColumns = 0UL;

        m_weightMatrixSpaceEnd = 0UL;
        m_activationMatrixSpaceEnd = 0UL;
        m_resultMatrixSpaceEnd = 0UL;

        m_combinedActivationAndResultMatrixSpacesSizeMaxByte = 0UL;
    }

private:

    std::vector<mpusim::byte>* const m_unifiedBufferPtr;

    const size_t m_unifiedBufferSizeByteMax;

    std::unordered_map<std::string, const WeightMatrixDopeVector> m_weightMatrixDopeVectorMap;

    size_t m_activationMatrixRows{0UL};
    size_t m_activationMatrixColumns{0UL};

    size_t m_resultMatrixRows{0UL};
    size_t m_resultMatrixColumns{0UL};

    size_t m_weightMatrixSpaceEnd{0UL};
    size_t m_activationMatrixSpaceEnd{0UL};
    size_t m_resultMatrixSpaceEnd{0UL};

    size_t m_combinedActivationAndResultMatrixSpacesSizeMaxByte{0UL};

    bool m_unifiedBufferDynamicResize{true};

};

#endif
