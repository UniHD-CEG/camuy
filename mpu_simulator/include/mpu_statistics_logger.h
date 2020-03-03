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
 * @file        mpu_statistics_logger.h
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   GNU Public License version 3 (GPLv3)
 */


#ifndef MPU_STATISTICS_LOGGER_H
#define MPU_STATISTICS_LOGGER_H

#include <vector>
#include <string>
#include <fstream>

#include "matrix_processing_unit.h"
#include "mpu_statistics_log_entry.h"

class MpuStatisticsLogger
{

public:

    MpuStatisticsLogger(const std::string& outputFilenameString,
                                const size_t weightDatatypeSizeByte,
                                const size_t activationDatatypeSizeByte,
                                const size_t accumulatorDatatypeSizeByte):
                                                        m_outputFilenameString{outputFilenameString},
                                                        m_weightDatatypeSizeByte{weightDatatypeSizeByte},
                                                        m_activationDatatypeSizeByte{activationDatatypeSizeByte},
                                                        m_accumulatorDatatypeSizeByte{accumulatorDatatypeSizeByte}
    {
    }

    std::string getColumnHeaderString() const
    {
        return std::string{"Operation\t"
                            "\"GEMM Size M\"\t"
                            "\"GEMM Size N\"\t"
                            "\"GEMM Size K\"\t"
                            "\"Systolic Array Height\"\t"
                            "\"Systolic Array Width\"\t"
                            "\"Activation FIFO Depth\"\t"
                            "\"Accumulator Array Height\"\t"
                            "\"MPU Control Register Bits\"\t"
                            "\"Systolic Data Setup Unit Control Register Bits\"\t"
                            "\"Activation FIFO Control Register Bits\"\t"
                            "\"Weight Fetcher Control Register Bits\"\t"
                            "\"Systolic Array Control Register Bits\"\t"
                            "\"Accumulator Array Control Register Bits\"\t"
                            "\"Activation FIFO Data Register Bits\"\t"
                            "\"Systolic Array Data Register Bits\"\t"
                            "\"Accumulator Array Data Register Bits\"\t"
                            "\"Unified Buffer Bits\"\t"
                            "\"Intra PE Data Movements\"\t"
                            "\"Inter PE Data Movements\"\t"
                            "\"Systolic Data Setup Unit Load Count Total\"\t"
                            "\"Weight Fetcher Load Count Total\"\t"
                            "\"Weight Fetcher Concurrent Load Count Max\"\t"
                            "\"Weight Fetcher Concurrent Load Count Per Column Max\"\t"
                            "\"Accumulator Array Load Count Total\"\t"
                            "\"Accumulator Array Concurrent Load Count Max\"\t"
                            "\"Accumulator Array Concurrent Load Count Per Column Max\"\t"
                            "\"Iterations Total\"\t"
                            "\"Iterations Stalled\"\t"
                            "\"Multiplications With Weight Zero Count Total\"\n"};
    }

    void addMpuStatisticsLogEntry(MpuStatisticsLogEntry&& mpuStatisticsLogEntry)
    {
        m_mpuStatisticsLogEntryVector.emplace_back(mpuStatisticsLogEntry);
    }

    ~MpuStatisticsLogger()
    {
        
        
        const std::string outputFilenameStringComplete{m_outputFilenameString +
                                                            std::string{"_W_"} +
                                                            std::to_string(m_weightDatatypeSizeByte) +
                                                            std::string{"_ACT_"} +
                                                            std::to_string(m_activationDatatypeSizeByte) +
                                                            std::string{"_ACC_"} +
                                                            std::to_string(m_accumulatorDatatypeSizeByte) +
                                                            std::string{".csv"}};
        
        std::cout << "Writing log file " << outputFilenameStringComplete << std::endl;
        
        m_outputFileStream.open(outputFilenameStringComplete);

        m_outputFileStream << getColumnHeaderString();

        for(const MpuStatisticsLogEntry& mpuStatisticsLogEntry :
                                                m_mpuStatisticsLogEntryVector)
        {
            m_outputFileStream << mpuStatisticsLogEntry.getString();
        }
    }

private:

    std::ofstream m_outputFileStream;

    std::vector<MpuStatisticsLogEntry> m_mpuStatisticsLogEntryVector;

    const std::string m_outputFilenameString;

    const size_t m_weightDatatypeSizeByte;
    const size_t m_activationDatatypeSizeByte;
    const size_t m_accumulatorDatatypeSizeByte;

};

#endif
