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
 * @file        mpu_exception.h
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   GNU Public License version 3 (GPLv3)
 */


#ifndef MPU_EXCEPTION_H
#define MPU_EXCEPTION_H

#include <exception>
#include <string>

class MpuException: public std::exception
{
public:

    MpuException(const std::string& whatMessage): m_whatMessage{whatMessage} {   }

    const char* what() const noexcept override
    {
        return m_whatMessage.c_str();
    }

private:

    std::string m_whatMessage;
};

#endif
