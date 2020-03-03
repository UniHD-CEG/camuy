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
 * @file        mpu_statistics_log_entry.h
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   GNU Public License version 3 (GPLv3)
 */


#ifndef MPU_LOG_ENTRY_H
#define MPU_LOG_ENTRY_H

#include <string>
#include <sstream>
#include <cstddef>

class MpuStatisticsLogEntry
{

public:

    MpuStatisticsLogEntry(const std::string& operationNameString,
                            const size_t sizeM,
                            const size_t sizeN,
                            const size_t sizeK,
                            const size_t systolicArrayHeight,
                            const size_t systolicArrayWidth,
                            const size_t fifoDepth,
                            const size_t accumulatorArrayHeight,
                            const size_t mpuControlRegisterBits,
                            const size_t systolicDataSetupUnitControlRegisterBits,
                            const size_t activationFifoControlRegisterBits,
                            const size_t weightFetcherControlRegisterBits,
                            const size_t systolicArrayControlRegisterBits,
                            const size_t accumulatorArrayControlRegisterBits,
                            const size_t activationFifoDataRegisterBits,
                            const size_t systolicArrayDataRegisterBits,
                            const size_t accumulatorArrayDataRegisterBits,
                            const size_t unifiedBufferBits,
                            const size_t intraPeDataMovementsTotal,
                            const size_t interPeDataMovementsTotal,
                            const size_t systolicDataSetupUnitLoadCountTotal,
                            const size_t weightFetcherLoadCountTotal,
                            const size_t weightFetcherConcurrentLoadsMax,
                            const size_t weightFetcherConcurrentLoadsPerColumnMax,
                            const size_t accumulatorArrayLoadCountTotal,
                            const size_t accumulatorArrayConcurrentLoadsMax,
                            const size_t accumulatorArrayConcurrentLoadsPerColumnMax,
                            const size_t iterationsTotal,
                            const size_t iterationsStalled,
                            const size_t multiplicationsWithWeightZeroCountTotal):
                                                                m_operationNameString{operationNameString},
                                                                m_sizeM{sizeM},
                                                                m_sizeN{sizeN},
                                                                m_sizeK{sizeK},
                                                                m_systolicArrayHeight{systolicArrayHeight},
                                                                m_systolicArrayWidth{systolicArrayWidth},
                                                                m_activationFifoDepth{fifoDepth},
                                                                m_accumulatorArrayHeight{accumulatorArrayHeight},
                                                                m_mpuControlRegisterBits{mpuControlRegisterBits},
                                                                m_systolicDataSetupUnitControlRegisterBits{
                                                                                systolicDataSetupUnitControlRegisterBits},
                                                                m_activationFifoControlRegisterBits{
                                                                                activationFifoControlRegisterBits},
                                                                m_weightFetcherControlRegisterBits{
                                                                                weightFetcherControlRegisterBits},
                                                                m_systolicArrayControlRegisterBits{
                                                                                systolicArrayControlRegisterBits},
                                                                m_accumulatorArrayControlRegisterBits{
                                                                                accumulatorArrayControlRegisterBits},
                                                                m_activationFifoDataRegisterBits{
                                                                                activationFifoDataRegisterBits},
                                                                m_systolicArrayDataRegisterBits{
                                                                                systolicArrayDataRegisterBits},
                                                                m_accumulatorArrayDataRegisterBits{
                                                                                accumulatorArrayDataRegisterBits},
                                                                m_unifiedBufferBits{unifiedBufferBits},
                                                                m_intraPeDataMovementsTotal{
                                                                                intraPeDataMovementsTotal},
                                                                m_interPeDataMovementsTotal{
                                                                                interPeDataMovementsTotal},
                                                                m_systolicDataSetupUnitLoadCountTotal{
                                                                                systolicDataSetupUnitLoadCountTotal},
                                                                m_weightFetcherLoadCountTotal{
                                                                                weightFetcherLoadCountTotal},
                                                                m_weightFetcherConcurrentLoadsMax{
                                                                                weightFetcherConcurrentLoadsMax},
                                                                m_weightFetcherConcurrentLoadsPerColumnMax{
                                                                                weightFetcherConcurrentLoadsPerColumnMax},
                                                                m_accumulatorArrayLoadCountTotal{
                                                                                accumulatorArrayLoadCountTotal},
                                                                m_accumulatorArrayConcurrentLoadsMax{
                                                                                accumulatorArrayConcurrentLoadsMax},
                                                                m_accumulatorArrayConcurrentLoadsPerColumnMax{
                                                                                accumulatorArrayConcurrentLoadsPerColumnMax},
                                                                m_iterationsTotal{iterationsTotal},
                                                                m_iterationsStalled{iterationsStalled},
                                                                m_multiplicationsWithWeightZeroCountTotal{
                                                                                multiplicationsWithWeightZeroCountTotal}
    {
    }

    MpuStatisticsLogEntry() = default;
    MpuStatisticsLogEntry(MpuStatisticsLogEntry& other) = default;
    MpuStatisticsLogEntry(MpuStatisticsLogEntry&& other) = default;

    MpuStatisticsLogEntry& operator=(MpuStatisticsLogEntry& other) = default;

    std::string getString() const
    {

        std::stringstream logEntryStringStream;

        logEntryStringStream << '\"' << m_operationNameString << "\"\t"
                                << m_sizeM << '\t'
                                << m_sizeN << '\t'
                                << m_sizeK << '\t'
                                << m_systolicArrayHeight << '\t'
                                << m_systolicArrayWidth << '\t'
                                << m_activationFifoDepth << '\t'
                                << m_accumulatorArrayHeight << '\t'
                                << m_mpuControlRegisterBits << '\t'
                                << m_systolicDataSetupUnitControlRegisterBits << '\t'
                                << m_activationFifoControlRegisterBits << '\t'
                                << m_weightFetcherControlRegisterBits << '\t'
                                << m_systolicArrayControlRegisterBits << '\t'
                                << m_accumulatorArrayControlRegisterBits << '\t'
                                << m_activationFifoDataRegisterBits << '\t'
                                << m_systolicArrayDataRegisterBits << '\t'
                                << m_accumulatorArrayDataRegisterBits << '\t'
                                << m_unifiedBufferBits << '\t'
                                << m_intraPeDataMovementsTotal << '\t'
                                << m_interPeDataMovementsTotal << '\t'
                                << m_systolicDataSetupUnitLoadCountTotal << '\t'
                                << m_weightFetcherLoadCountTotal << '\t'
                                << m_weightFetcherConcurrentLoadsMax << '\t'
                                << m_weightFetcherConcurrentLoadsPerColumnMax << '\t'
                                << m_accumulatorArrayLoadCountTotal << '\t'
                                << m_accumulatorArrayConcurrentLoadsMax << '\t'
                                << m_accumulatorArrayConcurrentLoadsPerColumnMax << '\t'
                                << m_iterationsTotal << '\t'
                                << m_iterationsStalled << '\t'
                                << m_multiplicationsWithWeightZeroCountTotal << '\n';

        return logEntryStringStream.str();
    }

private:

    std::string m_operationNameString;
    
    size_t m_sizeM{0UL};
    size_t m_sizeN{0UL};
    size_t m_sizeK{0UL};

    size_t m_systolicArrayHeight{0UL};
    size_t m_systolicArrayWidth{0UL};
    size_t m_activationFifoDepth{0UL};
    size_t m_accumulatorArrayHeight{0UL};

    size_t m_mpuControlRegisterBits{0UL};
    size_t m_systolicDataSetupUnitControlRegisterBits{0UL};
    size_t m_activationFifoControlRegisterBits{0UL};
    size_t m_weightFetcherControlRegisterBits{0UL};
    size_t m_systolicArrayControlRegisterBits{0UL};
    size_t m_accumulatorArrayControlRegisterBits{0UL};

    size_t m_activationFifoDataRegisterBits{0UL};
    size_t m_systolicArrayDataRegisterBits{0UL};
    size_t m_accumulatorArrayDataRegisterBits{0UL};

    size_t m_unifiedBufferBits{0UL};

    size_t m_intraPeDataMovementsTotal{0UL};
    size_t m_interPeDataMovementsTotal{0UL};
    size_t m_systolicDataSetupUnitLoadCountTotal{0UL};
    size_t m_weightFetcherLoadCountTotal{0UL};
    size_t m_weightFetcherConcurrentLoadsMax{0UL};
    size_t m_weightFetcherConcurrentLoadsPerColumnMax{0UL};
    size_t m_accumulatorArrayLoadCountTotal{0UL};
    size_t m_accumulatorArrayConcurrentLoadsMax{0UL};
    size_t m_accumulatorArrayConcurrentLoadsPerColumnMax{0UL};
    
    size_t m_iterationsTotal{0UL};
    size_t m_iterationsStalled{0UL};
    
    size_t m_multiplicationsWithWeightZeroCountTotal{0UL};


};

#endif
