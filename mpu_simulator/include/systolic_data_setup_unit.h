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
 * @file        systolic_data_setup_unit.h
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   MIT License
 */

#ifndef SYSTOLIC_DATA_SETUP_UNIT_H
#define SYSTOLIC_DATA_SETUP_UNIT_H

#include <vector>
#include <iostream>
#include <cmath>

#include "activation_fifo.h"

//#define SYSTOLIC_DATA_SETUP_UNIT_DEBUG SYSTOLIC_DATA_SETUP_UNIT_DEBUG

namespace
{
constexpr bool matrix0{false};
constexpr bool matrix1{true};
}

/**
 * @class           SystolicDataSetupUnit
 * @brief           
 * @tparam Datatype 
 */

template<typename Datatype> class SystolicDataSetupUnit
{

public:

    /**
     * @brief                           
     * @param activationFifoArrayPtr    
     */
    
    SystolicDataSetupUnit(std::vector<ActivationFifo<Datatype>>* const activationFifoArrayPtr):
                                                                            m_activationFifoArrayPtr{activationFifoArrayPtr},
                                                                            m_activationFifoArraySize{m_activationFifoArrayPtr->size()},
                                                                            m_rowPtrArray0Current(m_activationFifoArraySize),
                                                                            m_rowPtrArray0Next(m_activationFifoArraySize),
                                                                            m_rowPtrArray1Current(m_activationFifoArraySize),
                                                                            m_rowPtrArray1Next(m_activationFifoArraySize),
                                                                            m_blockPtrArray0Current(m_activationFifoArraySize),
                                                                            m_blockPtrArray0Next(m_activationFifoArraySize),
                                                                            m_blockPtrArray1Current(m_activationFifoArraySize),
                                                                            m_blockPtrArray1Next(m_activationFifoArraySize),
                                                                            m_matrixReadRepetitionCountArray0Current(m_activationFifoArraySize),
                                                                            m_matrixReadRepetitionCountArray0Next(m_activationFifoArraySize),
                                                                            m_matrixReadRepetitionCountArray1Current(m_activationFifoArraySize),
                                                                            m_matrixReadRepetitionCountArray1Next(m_activationFifoArraySize),
                                                                            m_busyArray0Current(m_activationFifoArraySize),
                                                                            m_busyArray0Next(m_activationFifoArraySize),
                                                                            m_busyArray1Current(m_activationFifoArraySize),
                                                                            m_busyArray1Next(m_activationFifoArraySize)
    {
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

    size_t getBlockCountBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(m_blocksMax));
    }

    size_t getRepetitionsBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(m_matrixReadRepetitionsMax));
    }

    size_t getIdleRowsBitwidthRequiredMin() const
    {
        return std::ceil(std::log2(m_idleRowsMax));
    }

    size_t getControlRegisterBits(const size_t unifiedBufferSize) const
    {
        /* The matrix address, matrix dimension, idle row,
         * block count, and repetition count registers
         * as well as the busy flag bit are all duplicated,
         * as two matrices can be read  simultaneously,
         * as to prevent the activation FIFOs from running
         * empty.
         * The matrix address registers are modelled by
         * m_matrixPtr0Current, m_matrixPtr0Next,
         * m_matrixPtr1Current and m_matrixPtr1Next.
         * The matrix width registers are modelled by
         * m_matrix0WidthCurrent, m_matrix0WidthNext,
         * m_matrix1WidthCurrent and m_matrix1WidthNext.
         * The matrix height registers are modelled by
         * m_matrix0HeightCurrent, m_matrix0HeightNext,
         * m_matrix1HeightCurrent and m_matrix1HeightNext.
         * The block count registers are modelled by
         * m_blockCount0Current, m_blockCount0Next,
         * m_blockCount1Current, and m_blockCount1Next.
         * The matrix read repetition count registers
         * are modelled by m_matrixReadRepetitions0Current,
         * m_matrixReadRepetitions0Next,
         * m_matrixReadRepetitions1Current, and
         * m_matrixReadRepetitions1Next.
         * The idle row registers are modelled by
         * m_idleRowsLastBlock0Current,
         * m_idleRowsLastBlock0Next,
         * m_idleRowsLastBlock1Current, and
         * m_idleRowsLastBlock1Next.
         * The busy flag bits are modelled by
         * m_matrix0ReadBusyCurrent,
         * m_matrix0ReadBusyNext,
         * m_matrix1ReadBusyCurrent, and
         * m_matrix1ReadBusyNext.
         * Each activation FIFO requires its
         * own row pointer counter, block pointer counter,
         * repetition counter, and busy flag bit, which
         * also have to be duplicated to allow for
         * simultaneous reading of two arrays. The row
         * pointer counters are modelled by
         * m_rowPtrArray0Current, m_rowPtrArray0Next,
         * m_rowPtrArray1Current, and m_rowPtrArray1Next.
         * The block counter registers are modelled by
         * m_blockPtrArray0Current, m_blockPtrArray0Next,
         * m_blockPtrArray1Current and
         * m_blockPtrArray1Next.
         * The read repetition counters are modelled by
         * m_matrixReadRepetitionCountArray0Current,
         * m_matrixReadRepetitionCountArray0Next,
         * m_matrixReadRepetitionCountArray1Current, and
         * m_matrixReadRepetitionCountArray1 next.
         * The busy flag bits are modelled by
         * m_busyArray0Current, m_busyArray0Next,
         * m_busyArray1Current, and m_busyArray1Next.
         * The minimum bitwidths for these registers are
         * calculated using the maximum value they assume
         * during execution. These maximum values can be
         * reset using resetMaxRegisterValues().
         * In addition to the control registers and bit
         * flags required for each of the two
         * simultaneously read arrays, three flag bits
         * exist. The matrix precedent flag bit, modelled
         * by m_matrix1PrecedentCurrent and
         * m_matrix1PrecedentNext, determines which of the
         * matrices has precedent on filling the activation
         * FIFOs, as to keep the ensure the order in which
         * matrix read operation commands were send to the
         * systolic data setup unit is maintained when
         * pushing the activation data to the activation
         * FIFOs.
         * The active signal bit, modelled by
         * m_activeCurrent, and the busy signal bit,
         * modelled by m_busyCurrent, signal the active
         * state (is currently reading one or two matrices)
         * and busy state (is currently reading two
         * matrices) respectively. As their state only
         * changes in the updateState() function  depending
         * on the matrix0/matrix1 read busy  flags, only a
         * current state register was used  for these flag
         * bits, as the next state does not need to be
         * stored in between state updates.
         */

        return 2UL*(m_activationFifoArraySize*(
                    getMatrixHeightBitwidthRequiredMin() +
                    getBlockCountBitwidthRequiredMin() +
                    getRepetitionsBitwidthRequiredMin() + 1UL) +
                    getMatrixAddressBitwidthRequiredMin(unifiedBufferSize) +
                    getMatrixWidthBitwidthRequiredMin() +
                    getMatrixHeightBitwidthRequiredMin() +
                    getBlockCountBitwidthRequiredMin() +
                    getRepetitionsBitwidthRequiredMin() +
                    getIdleRowsBitwidthRequiredMin() + 1UL) + 3UL;
    }

    void resetMaxRegisterValues()
    {
        m_matrixWidthMax = 0UL;
        m_matrixHeightMax = 0UL;
        m_blocksMax = 0UL;
        m_matrixReadRepetitionsMax = 0UL;
        m_idleRowsMax = 0UL;
    }

    size_t getLoadCount() const
    {
        return m_loadCount;
    }

    void resetLoadCount()
    {
        m_loadCount = 0UL;
    }

    bool hasBusySignal() const
    {
        return m_busyCurrent;
    }

    bool hasActiveSignal() const
    {
        return m_activeCurrent;
    }

    /**
     * @brief                       
     * @param matrixPtr             
     * @param matrixWidth           
     * @param matrixHeight          
     * @param matrixReadRepetitions 
     */
    
    void addInputMatrix(const Datatype* const matrixPtr,
                                const size_t matrixWidth,
                                const size_t matrixHeight,
                                const size_t matrixReadRepetitions)
    {
        if(!m_busyCurrent)
        {
            if(!m_matrix0ReadBusyCurrent)
            {
                m_matrixPtr0Next = matrixPtr;

                m_matrix0WidthNext = matrixWidth;

                if(m_matrixWidthMax < m_matrix0WidthNext)
                {
                    m_matrixWidthMax = m_matrix0WidthNext;
                }

                m_matrix0HeightNext = matrixHeight;

                if(m_matrixHeightMax < m_matrix0HeightNext)
                {
                    m_matrixHeightMax = m_matrix0HeightNext;
                }

                m_blocksArray0Next = std::ceil(static_cast<float>(m_matrix0WidthNext)/
                                                static_cast<float>(m_activationFifoArraySize));

                if(m_blocksMax < m_blocksArray0Next)
                {
                    m_blocksMax = m_blocksArray0Next;
                }

                m_idleRowsLastBlock0Next = m_blocksArray0Next*m_activationFifoArraySize -
                                                                        m_matrix0WidthNext;

                if(m_idleRowsMax < m_idleRowsLastBlock0Next)
                {
                  m_idleRowsMax = m_idleRowsLastBlock0Next;
                }

                m_matrixReadRepetitions0Next = matrixReadRepetitions;

                m_matrix0ReadBusyNext = true;

                for(size_t elementCount = 0; elementCount < m_activationFifoArraySize;
                                                                            ++elementCount)
                {
                    m_busyArray0Next.at(elementCount) = true;
                }

#ifdef SYSTOLIC_DATA_SETUP_UNIT_DEBUG
                std::cout << "Set input for matrix 0" << std::endl;
#endif
            }

            else
            {
                m_matrixPtr1Next = matrixPtr;

                m_matrix1WidthNext = matrixWidth;

                if(m_matrixWidthMax < m_matrix1WidthNext)
                {
                    m_matrixWidthMax = m_matrix1WidthNext;
                }

                m_matrix1HeightNext = matrixHeight;

                if(m_matrixHeightMax < m_matrix1HeightNext)
                {
                    m_matrixHeightMax = m_matrix1HeightNext;
                }

                m_blocksArray1Next = std::ceil(static_cast<float>(m_matrix1WidthNext)/
                                                static_cast<float>(m_activationFifoArraySize));

                if(m_blocksMax < m_blocksArray1Next)
                {
                    m_blocksMax = m_blocksArray1Next;
                }

                m_idleRowsLastBlock1Next = m_blocksArray1Next*m_activationFifoArraySize -
                                                                        m_matrix1WidthNext;

                if(m_idleRowsMax < m_idleRowsLastBlock1Next)
                {
                  m_idleRowsMax = m_idleRowsLastBlock1Next;
                }

                m_matrixReadRepetitions1Next = matrixReadRepetitions;

                if(m_matrixReadRepetitionsMax < m_matrixReadRepetitions1Next)
                {
                    m_matrixReadRepetitionsMax = m_matrixReadRepetitions1Next;
                }

                m_matrix1ReadBusyNext = true;

                for(size_t elementCount = 0; elementCount < m_activationFifoArraySize;
                                                                            ++elementCount)
                {
                    m_busyArray1Next.at(elementCount) = true;
                }

#ifdef SYSTOLIC_DATA_SETUP_UNIT_DEBUG
                std::cout << "Set input for matrix 1" << std::endl;
#endif
            }
        }
    }

    void resetCounters(const bool matrixSelectBit)
    {
        if(matrixSelectBit == ::matrix0)
        {
            for(size_t elementCount{0}; elementCount < m_activationFifoArraySize;
                                                                        ++elementCount)
            {
                m_blockPtrArray0Next.at(elementCount) = 0;
                m_rowPtrArray0Next.at(elementCount) = 0;
                m_matrixReadRepetitionCountArray0Next.at(elementCount) = 0;
            }
        }

        else
        {
            for(size_t elementCount{0}; elementCount < m_activationFifoArraySize;
                                                                        ++elementCount)
            {
                m_blockPtrArray1Next.at(elementCount) = 0;
                m_rowPtrArray1Next.at(elementCount) = 0;
                m_matrixReadRepetitionCountArray1Next.at(elementCount) = 0;
            }
        }
    }

    /**
     * @brief
     */
    
    void runIteration()
    {
        if(m_activeCurrent)
        {
            for(size_t activationFifoCount = 0; activationFifoCount < m_activationFifoArraySize;
                                                                                ++activationFifoCount)
            {
                if(!(m_activationFifoArrayPtr->at(activationFifoCount).isFull()))
                {
                    if(!m_matrix1PrecedentCurrent)
                    {
                        if(m_matrix0ReadBusyCurrent ?
                                    !runIterationMatrix0(activationFifoCount) : true)
                        {
                            if(m_matrix1ReadBusyCurrent)
                            {
                                runIterationMatrix1(activationFifoCount);
                            }
                        }
                    }

                    else
                    {
                        if(m_matrix1ReadBusyCurrent ?
                                    !runIterationMatrix1(activationFifoCount) : true)
                        {
                            if(m_matrix0ReadBusyCurrent)
                            {
                                runIterationMatrix0(activationFifoCount);
                            }
                        }
                    }
                }
            }

            m_matrix0ReadBusyNext = false;

            for(const bool& element : m_busyArray0Next)
            {
                m_matrix0ReadBusyNext |= element;
            }

            if(!m_matrix0ReadBusyNext)
            {
                resetCounters(::matrix0);
            }

            m_matrix1ReadBusyNext = false;

            for(const bool& element : m_busyArray1Next)
            {
                m_matrix1ReadBusyNext |= element;
            }

            if(!m_matrix1ReadBusyNext)
            {
                resetCounters(::matrix1);
            }

            if(((m_matrix0ReadBusyNext == false) &&
                        (m_matrix0ReadBusyNext !=
                        m_matrix0ReadBusyCurrent)) ||
                    ((m_matrix1ReadBusyNext == false) &&
                        (m_matrix1ReadBusyNext !=
                        m_matrix1ReadBusyCurrent)))
            {
                m_matrix1PrecedentNext =
                            !m_matrix1PrecedentCurrent;
            }

            if(!(m_matrix0ReadBusyNext ||
                        m_matrix1ReadBusyNext))
            {
                m_matrix1PrecedentNext = false;
            }

#ifdef SYSTOLIC_DATA_SETUP_UNIT_DEBUG
            if((m_matrix0ReadBusyNext == false) &&
                                (m_matrix0ReadBusyNext !=
                                m_matrix0ReadBusyCurrent))
            {
                std::cout << "Systolic data setup unit: "
                            << "Matrix 0 output finished" << std::endl;
            }

            if((m_matrix1ReadBusyNext == false) &&
                                (m_matrix1ReadBusyNext !=
                                m_matrix1ReadBusyCurrent))
            {
               std::cout << "Systolic data setup unit: "
                           << "Matrix 1 output finished" << std::endl;
            }

            if(!(m_matrix0ReadBusyNext ||
                        m_matrix1ReadBusyNext))
            {
                std::cout << "Systolic data setup unit done" << std::endl;
            }
#endif
        }
    }

    void updateState()
    {
        for(size_t elementCount{0}; elementCount < m_activationFifoArraySize;
                                                                    ++elementCount)
        {
            m_blockPtrArray0Current.at(elementCount) =
                                m_blockPtrArray0Next.at(elementCount);
            m_blockPtrArray1Current.at(elementCount) =
                                m_blockPtrArray1Next.at(elementCount);

            m_rowPtrArray0Current.at(elementCount) =
                                m_rowPtrArray0Next.at(elementCount);
            m_rowPtrArray1Current.at(elementCount) =
                                m_rowPtrArray1Next.at(elementCount);

            m_busyArray0Current.at(elementCount) =
                                m_busyArray0Next.at(elementCount);
            m_busyArray1Current.at(elementCount) =
                                m_busyArray1Next.at(elementCount);

            m_matrixReadRepetitionCountArray0Current.at(elementCount) =
                                    m_matrixReadRepetitionCountArray0Next.at(elementCount);

            m_matrixReadRepetitionCountArray1Current.at(elementCount) =
                                    m_matrixReadRepetitionCountArray1Next.at(elementCount);
        }

        m_matrixPtr0Current = m_matrixPtr0Next;
        m_matrixPtr1Current = m_matrixPtr1Next;

        m_matrix0WidthCurrent = m_matrix0WidthNext;
        m_matrix1WidthCurrent = m_matrix1WidthNext;

        m_matrix0HeightCurrent = m_matrix0HeightNext;
        m_matrix1HeightCurrent = m_matrix1HeightNext;

        m_blocksArray0Current = m_blocksArray0Next;
        m_blocksArray1Current = m_blocksArray1Next;

        m_idleRowsLastBlock0Current = m_idleRowsLastBlock0Next;
        m_idleRowsLastBlock1Current = m_idleRowsLastBlock1Next;

        m_matrixReadRepetitions0Current = m_matrixReadRepetitions0Next;
        m_matrixReadRepetitions1Current = m_matrixReadRepetitions1Next;

        m_matrix1PrecedentCurrent =
                        m_matrix1PrecedentNext;

        m_matrix0ReadBusyCurrent =
                        m_matrix0ReadBusyNext;

        m_matrix1ReadBusyCurrent =
                        m_matrix1ReadBusyNext;

        m_activeCurrent = m_matrix0ReadBusyNext ||
                                m_matrix1ReadBusyNext;

        m_busyCurrent = m_matrix0ReadBusyNext &&
                                m_matrix1ReadBusyNext;

    }

private:
    
    /**
     * @brief                       
     * @param activationFifoCount   
     * @return                      
     */

    bool runIterationMatrix0(const size_t activationFifoCount)
    {
        if(m_busyArray0Current.at(activationFifoCount))
        {

            const size_t idleRows{(m_blockPtrArray0Current.at(activationFifoCount) !=
                                                                (m_blocksArray0Current - 1)) ? 0UL :
                                                                                                m_idleRowsLastBlock0Current};
            if(activationFifoCount >= idleRows)
            {

                m_activationFifoArrayPtr->at(activationFifoCount).push(m_matrixPtr0Current[
                                                                            m_blockPtrArray0Current.at(
                                                                            activationFifoCount)*
                                                                            m_activationFifoArraySize +
                                                                            m_rowPtrArray0Current.at(
                                                                            activationFifoCount)*
                                                                            m_matrix0WidthCurrent +
                                                                            activationFifoCount -
                                                                            idleRows]);
                ++m_loadCount;

            }

            else
            {
                m_activationFifoArrayPtr->at(activationFifoCount).push(Datatype{0});
            }

            if(m_rowPtrArray0Current.at(activationFifoCount) < (m_matrix0HeightCurrent - 1))
            {
                m_rowPtrArray0Next.at(activationFifoCount) =
                                        m_rowPtrArray0Current.at(activationFifoCount) + 1;
            }

            else
            {
                m_rowPtrArray0Next.at(activationFifoCount) = 0;

                if(m_blockPtrArray0Current.at(activationFifoCount) <
                                                    (m_blocksArray0Current - 1))
                {
                    m_blockPtrArray0Next.at(activationFifoCount) =
                                            m_blockPtrArray0Current.at(activationFifoCount) + 1;
                }

                else
                {
                    if(m_matrixReadRepetitionCountArray0Current.at(activationFifoCount) <
                                                            (m_matrixReadRepetitions0Current - 1))
                    {
                        m_blockPtrArray0Next.at(activationFifoCount) = 0;

                        m_matrixReadRepetitionCountArray0Next.at(activationFifoCount) =
                                        m_matrixReadRepetitionCountArray0Current.at(activationFifoCount) + 1;
                    }

                    else
                    {
                        m_busyArray0Next.at(activationFifoCount) = false;
                    }
                }
            }

            return true;
        }

        else
        {
            return false;
        }
    }
    
    /**
     * @brief                       
     * @param activationFifoCount   
     * @return                      
     */

    bool runIterationMatrix1(const size_t activationFifoCount)
    {
        if(m_busyArray1Current.at(activationFifoCount))
        {

            const size_t idleRows{(m_blockPtrArray1Current.at(activationFifoCount) !=
                                                                (m_blocksArray1Current - 1)) ? 0UL :
                                                                                                m_idleRowsLastBlock1Current};

            if(activationFifoCount >= idleRows)
            {

                m_activationFifoArrayPtr->at(activationFifoCount).push(m_matrixPtr1Current[
                                                                            m_blockPtrArray1Current.at(
                                                                            activationFifoCount)*
                                                                            m_activationFifoArraySize +
                                                                            m_rowPtrArray1Current.at(
                                                                            activationFifoCount)*
                                                                            m_matrix1WidthCurrent +
                                                                            activationFifoCount -
                                                                            idleRows]);
                ++m_loadCount;

            }

            else
            {
                m_activationFifoArrayPtr->at(activationFifoCount).push(Datatype{0});
            }

            if(m_rowPtrArray1Current.at(activationFifoCount) < (m_matrix1HeightCurrent - 1))
            {
                m_rowPtrArray1Next.at(activationFifoCount) =
                                        m_rowPtrArray1Current.at(activationFifoCount) + 1;
            }

            else
            {
                m_rowPtrArray1Next.at(activationFifoCount) = 0;

                if(m_blockPtrArray1Current.at(activationFifoCount) <
                                                    (m_blocksArray1Current - 1))
                {
                    m_blockPtrArray1Next.at(activationFifoCount) =
                                            m_blockPtrArray1Current.at(activationFifoCount) + 1;
                }

                else
                {
                    if(m_matrixReadRepetitionCountArray1Current.at(activationFifoCount) <
                                                            (m_matrixReadRepetitions1Current - 1))
                    {
                        m_blockPtrArray1Next.at(activationFifoCount) = 0;

                        m_matrixReadRepetitionCountArray1Next.at(activationFifoCount) =
                                        m_matrixReadRepetitionCountArray1Current.at(activationFifoCount) + 1;
                    }

                    else
                    {
                        m_busyArray1Next.at(activationFifoCount) = false;
                    }
                }
            }

            return true;
        }

        else
        {
            return false;
        }
    }

    std::vector<ActivationFifo<Datatype>>* const m_activationFifoArrayPtr;

    const size_t m_activationFifoArraySize;

    std::vector<size_t> m_rowPtrArray0Current;
    std::vector<size_t> m_rowPtrArray0Next;
    std::vector<size_t> m_rowPtrArray1Current;
    std::vector<size_t> m_rowPtrArray1Next;

    std::vector<size_t> m_blockPtrArray0Current;
    std::vector<size_t> m_blockPtrArray0Next;
    std::vector<size_t> m_blockPtrArray1Current;
    std::vector<size_t> m_blockPtrArray1Next;

    std::vector<size_t> m_matrixReadRepetitionCountArray0Current;
    std::vector<size_t> m_matrixReadRepetitionCountArray0Next;
    std::vector<size_t> m_matrixReadRepetitionCountArray1Current;
    std::vector<size_t> m_matrixReadRepetitionCountArray1Next;

    std::vector<bool> m_busyArray0Current;
    std::vector<bool> m_busyArray0Next;
    std::vector<bool> m_busyArray1Current;
    std::vector<bool> m_busyArray1Next;

    const Datatype* m_matrixPtr0Current{nullptr};
    const Datatype* m_matrixPtr0Next{nullptr};
    const Datatype* m_matrixPtr1Current{nullptr};
    const Datatype* m_matrixPtr1Next{nullptr};

    size_t m_matrix0WidthCurrent{0UL};
    size_t m_matrix0WidthNext{0UL};
    size_t m_matrix1WidthCurrent{0UL};
    size_t m_matrix1WidthNext{0UL};

    size_t m_matrixWidthMax{0UL};

    size_t m_matrix0HeightCurrent{0UL};
    size_t m_matrix0HeightNext{0UL};
    size_t m_matrix1HeightCurrent{0UL};
    size_t m_matrix1HeightNext{0UL};

    size_t m_matrixHeightMax{0UL};

    size_t m_blocksArray0Current{0UL};
    size_t m_blocksArray0Next{0UL};
    size_t m_blocksArray1Current{0UL};
    size_t m_blocksArray1Next{0UL};

    size_t m_blocksMax{0UL};

    size_t m_matrixReadRepetitions0Current{0UL};
    size_t m_matrixReadRepetitions0Next{0UL};
    size_t m_matrixReadRepetitions1Current{0UL};
    size_t m_matrixReadRepetitions1Next{0UL};

    size_t m_matrixReadRepetitionsMax{0UL};

    size_t m_idleRowsLastBlock0Current{0UL};
    size_t m_idleRowsLastBlock0Next{0UL};
    size_t m_idleRowsLastBlock1Current{0UL};
    size_t m_idleRowsLastBlock1Next{0UL};

    size_t m_idleRowsMax{0UL};

    size_t m_loadCount{0UL};

    bool m_matrix1PrecedentCurrent{false};
    bool m_matrix1PrecedentNext{false};

    bool m_matrix0ReadBusyCurrent{false};
    bool m_matrix0ReadBusyNext{false};
    bool m_matrix1ReadBusyCurrent{false};
    bool m_matrix1ReadBusyNext{false};

    bool m_activeCurrent{false};

    bool m_busyCurrent{false};

};

#endif
