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
 * @file        processing_element.h
 * @author      Kevin Stehle (stehle@stud.uni-heidelberg.de)
 * @date        2019-2020
 * @copyright   MIT License
 */

#ifndef PROCESSING_ELEMENT_H
#define PROCESSING_ELEMENT_H

#include <mutex>
#include <iostream>
#include <cstdio>
#include <cmath>

#include <opencv2/opencv.hpp>

#define PE_COLOR_INACTIVE cv::Scalar(0xE0, 0xE0 , 0xE0)
#define PE_COLOR_HAS_VALID_SIGNAL cv::Scalar(0x25, 0xA8 , 0x00)
#define PE_COLOR_HAS_UPDATE_WEIGHTS_SIGNAL cv::Scalar(0xCC, 0x40, 0x00)

constexpr size_t peSize{47UL};

struct PEPosition
{
    PEPosition(size_t x, size_t y): x{x},
                                        y{y}
    {
    }

    size_t x;
    size_t y;
};

template<typename WeightDatatype,
            typename ActivationDatatype,
            typename SumDatatype> class ProcessingElement
{

public:

    ProcessingElement(const PEPosition position): m_position{position}
    {
    }

    WeightDatatype loadWeight() const
    {
        return (*m_weightRegisterReadPtrCurrent);
    }

    void storeWeight(const WeightDatatype weight)
    {
        *m_weightRegisterWritePtrCurrent = weight;
    }

    void updateWeight()
    {
        m_weightRegisterReadPtrNext =
                            m_weightRegisterWritePtrCurrent;

        m_weightRegisterWritePtrNext =
                            m_weightRegisterReadPtrCurrent;
    }

    SumDatatype getSum() const
    {
        return m_sumCurrent;
    }

    ActivationDatatype getActivation() const
    {
        return m_activationCurrent;
    }

    PEPosition getPosition() const
    {
         return m_position;
    }

    bool hasValidSignal() const
    {
        return m_validCurrent;
    }

    bool hasUpdateWeightSignal() const
    {
        return m_updateWeightCurrent;
    }

    virtual void readUpdateWeightSignals() = 0;

    virtual void computeSum() = 0;

    virtual void updateState()
    {
        if(m_updateWeightNext)
        {
            updateWeight();
        }

        m_sumCurrent = m_sumNext;

        m_activationCurrent = m_activationNext;

        m_weightRegisterReadPtrCurrent =
                        m_weightRegisterReadPtrNext;

        m_weightRegisterWritePtrCurrent =
                        m_weightRegisterWritePtrNext;

        m_validCurrent = m_validNext;
        m_updateWeightCurrent = m_updateWeightNext;

        m_validNext = false;
        m_updateWeightNext = false;

    }

    void paintState(cv::Mat& outputFrameMat,
                        const cv::Point& outputFrameOffset) const
    {
        const cv::Point rectanglePointTopLeft(outputFrameOffset.x +
                                                ProcessingElement<WeightDatatype,
                                                                    ActivationDatatype,
                                                                    SumDatatype>::m_position.x*
                                                peSize,
                                                outputFrameOffset.y +
                                                    ProcessingElement<WeightDatatype,
                                                                        ActivationDatatype,
                                                                        SumDatatype>::m_position.y*
                                                peSize);

        const cv::Point rectanglePointTopRight(outputFrameOffset.x +
                                                ProcessingElement<WeightDatatype,
                                                                    ActivationDatatype,
                                                                    SumDatatype>::m_position.x*
                                                peSize +
                                                peSize,
                                                outputFrameOffset.y +
                                                    ProcessingElement<WeightDatatype,
                                                                        ActivationDatatype,
                                                                        SumDatatype>::m_position.y*
                                                peSize);

        const cv::Point rectanglePointBottomLeft(outputFrameOffset.x +
                                                    ProcessingElement<WeightDatatype,
                                                                        ActivationDatatype,
                                                                        SumDatatype>::m_position.x*
                                                    peSize,
                                                    outputFrameOffset.y +
                                                    ProcessingElement<WeightDatatype,
                                                                        ActivationDatatype,
                                                                        SumDatatype>::m_position.y*
                                                    peSize +
                                                    peSize);

        const cv::Point rectanglePointBottomRight(outputFrameOffset.x +
                                                    ProcessingElement<WeightDatatype,
                                                                        ActivationDatatype,
                                                                        SumDatatype>::m_position.x*
                                                    peSize +
                                                    peSize,
                                                    outputFrameOffset.y +
                                                    ProcessingElement<WeightDatatype,
                                                                        ActivationDatatype,
                                                                        SumDatatype>::m_position.y*
                                                    peSize +
                                                    peSize);


        if(m_validCurrent && m_updateWeightCurrent)
        {
            cv::fillConvexPoly(outputFrameMat,
                                cv::InputArray(std::vector<cv::Point>{
                                                    rectanglePointTopLeft,
                                                    rectanglePointTopRight,
                                                    rectanglePointBottomLeft}),
                                PE_COLOR_HAS_VALID_SIGNAL);

            cv::fillConvexPoly(outputFrameMat,
                                cv::InputArray(std::vector<cv::Point>{
                                                    rectanglePointTopRight,
                                                    rectanglePointBottomRight,
                                                    rectanglePointBottomLeft}),
                                PE_COLOR_HAS_UPDATE_WEIGHTS_SIGNAL);
        }

        else if(m_validCurrent)
        {
            cv::rectangle(outputFrameMat,
                            rectanglePointTopLeft,
                            rectanglePointBottomRight,
                            PE_COLOR_HAS_VALID_SIGNAL,
                            cv::FILLED);
        }

        else if(m_updateWeightCurrent)
        {
            cv::rectangle(outputFrameMat,
                            rectanglePointTopLeft,
                            rectanglePointBottomRight,
                            PE_COLOR_HAS_UPDATE_WEIGHTS_SIGNAL,
                            cv::FILLED);
        }

        else
        {
            cv::rectangle(outputFrameMat,
                            rectanglePointTopLeft,
                            rectanglePointBottomRight,
                            PE_COLOR_INACTIVE,
                            cv::FILLED);
        }

        const std::string weightString{"W: " + std::to_string(loadWeight())};
        const std::string activationString{"A: " + std::to_string(m_activationCurrent)};
        const std::string sumString{"S: " + std::to_string(getSum())};

        constexpr double fontScale{0.59};

        cv::putText(outputFrameMat,
                        weightString,
                        cv::Point(rectanglePointTopLeft.x + 2,
                                    rectanglePointTopLeft.y + 13),
                        cv::FONT_HERSHEY_PLAIN,
                        fontScale,
                        cv::Scalar(0, 0, 0), 1);

        cv::putText(outputFrameMat,
                        activationString,
                        cv::Point(rectanglePointTopLeft.x + 2,
                                    rectanglePointTopLeft.y + 26),
                        cv::FONT_HERSHEY_PLAIN,
                        fontScale,
                        cv::Scalar(0, 0, 0), 1);

        cv::putText(outputFrameMat,
                            sumString,
                            cv::Point(rectanglePointTopLeft.x + 2,
                                        rectanglePointTopLeft.y + 39),
                            cv::FONT_HERSHEY_PLAIN,
                            fontScale,
                            cv::Scalar(0, 0, 0), 1);

        cv::rectangle(outputFrameMat,
                        rectanglePointTopLeft,
                        rectanglePointBottomRight,
                        cv::Scalar(0, 0, 0));
    }

    virtual ~ProcessingElement()
    {
    }

protected:

    const PEPosition m_position;

    SumDatatype m_sumCurrent{0};
    SumDatatype m_sumNext{0};
    ActivationDatatype m_activationCurrent{0};
    ActivationDatatype m_activationNext{0};

    bool m_validCurrent{false};
    bool m_validNext{false};
    bool m_updateWeightCurrent{false};
    bool m_updateWeightNext{false};

private:

    WeightDatatype m_weightRegister0{0};
    WeightDatatype m_weightRegister1{0};

    WeightDatatype* m_weightRegisterReadPtrCurrent{&m_weightRegister0};
    WeightDatatype* m_weightRegisterReadPtrNext{&m_weightRegister0};
    WeightDatatype* m_weightRegisterWritePtrCurrent{&m_weightRegister1};
    WeightDatatype* m_weightRegisterWritePtrNext{&m_weightRegister1};



};

#endif
