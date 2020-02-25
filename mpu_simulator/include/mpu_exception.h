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
