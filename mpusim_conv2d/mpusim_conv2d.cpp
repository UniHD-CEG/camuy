/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 * Modifications copyright (c) 2020 Computing Systems Group
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file        mpusim_conv2d.cpp
 * @author      The TensorFlow Authors
 * @author      Kevin Stehle
 * @date        2019-2020
 * @copyright   Apache License, Version 2.0
 */

#define EIGEN_USE_THREADS

#include <stdexcept>
#include <mutex>
#include <map>
#include <vector>
#include <string>
#include <regex>
#include <sstream>
#include <ios>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/kernels/gemm_functors.h"
#include "tensorflow/core/kernels/image_resizer_state.h"
#include "tensorflow/core/util/mirror_pad_mode.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

#include "mpusim_wrapper.h"

constexpr size_t maxChunkSizeMpuSimConv2DFunctor{4UL*1024UL*1024UL*1024UL};

using namespace tensorflow;


class MpuSimConv2DFunctor
{

public:

    void operator()(OpKernelContext* opKernelContext,
                        const int64 activationsDatatypeSizeByte,
                        const int64 weightsDatatypeSizeByte,
                        const int64 resultsDatatypeSizeByte,
                        const int64 systolicArrayHeight,
                        const int64 systolicArrayWidth,
                        const int64 activationFifoDepth,
                        const int64 accumulatorArrayHeight,
                        const float* inputData,
                        int batchSize,
                        int inputHeight,
                        int inputWidth,
                        int inputDepth,
                        const float* filterData,
                        int filterHeight,
                        int filterWidth,
                        int filterCount,
                        int strideRows,
                        int strideCols,
                        Padding padding,
                        float* outputData,
                        int outputHeight,
                        int outputWidth,
                        const std::string& logFileOutputDirString,
                        const std::string& modelNameString)
    {
        if((batchSize <= 0) || (inputWidth <= 0) || (inputHeight <= 0) ||
                                                                    (inputDepth <= 0))
        {
            LOG(WARNING) << "MpuSimConv2D was called with bad input dimensions: "
                         << batchSize << ", " << inputHeight << ", "
                         << inputWidth << ", " << inputDepth;
            return;
        }

        if((filterWidth <= 0) || (filterHeight <= 0) || (filterCount <= 0))
        {
            LOG(WARNING) << "MpuSimConv2D was called with bad filter dimensions: "
                             << filterWidth << ", " << filterHeight << ", " << filterCount;
            return;
        }

        if((outputWidth <= 0) || (outputHeight <= 0))
        {
            LOG(WARNING) << "MpuSimConv2D was called with bad output width or height: "
                            << outputWidth << ", " << outputHeight;
            return;
        }

        if(filterHeight == 1 && filterWidth == 1 && strideRows == 1 &&
                                                            strideCols == 1)
        {
            /* The kernel is 1x1 */

			std::string opKernelString{opKernelContext->op_kernel().name()};

			opKernelString.erase(opKernelString.begin() + opKernelString.find_last_of("/"),
																			opKernelString.end());
            
            const std::regex slashRegex("/");
            
            const std::string opKernelStringSlashesReplacedWithUnderscores{
                                                            std::regex_replace(opKernelString,
                                                                                slashRegex, "_")};
            
            MpuSimWrapper::getInstance().runMultiplication(
                                                    activationsDatatypeSizeByte,
                                                    weightsDatatypeSizeByte,
                                                    resultsDatatypeSizeByte,
                                                    systolicArrayHeight,
                                                    systolicArrayWidth,
                                                    activationFifoDepth,
                                                    accumulatorArrayHeight,
                                                    batchSize*inputHeight*inputWidth,
                                                    filterCount,
                                                    inputDepth,
                                                    inputData,
                                                    filterData,
                                                    outputData,
                                                    logFileOutputDirString,
                                                    modelNameString,
                                                    opKernelStringSlashesReplacedWithUnderscores);
            
            
            return;
        }

        else if(filterHeight == inputHeight && filterWidth == inputWidth &&
                                                                    padding == VALID)
        {
            /* The input data and filter have the same height/width */

			std::string opKernelString{opKernelContext->op_kernel().name()};

			opKernelString.erase(opKernelString.begin() + opKernelString.find_last_of("/"),
																			opKernelString.end());
            
            const std::regex slashRegex("/");

            const std::string opKernelStringSlashesReplacedWithUnderscores{
                                                            std::regex_replace(opKernelString,
                                                                                slashRegex, "_")};
                                                                                
            MpuSimWrapper::getInstance().runMultiplication(
                                                    activationsDatatypeSizeByte,
                                                    weightsDatatypeSizeByte,
                                                    resultsDatatypeSizeByte,
                                                    systolicArrayHeight,
                                                    systolicArrayWidth,
                                                    activationFifoDepth,
                                                    accumulatorArrayHeight,
                                                    batchSize,
                                                    filterCount,
                                                    inputHeight*inputWidth*inputDepth,
                                                    inputData,
                                                    filterData,
                                                    outputData,
                                                    logFileOutputDirString,
                                                    modelNameString,
                                                    opKernelStringSlashesReplacedWithUnderscores);

            return;
        }

        int filterLeftOffset{0};
        int filterTopOffset{0};

        if(padding == VALID)
        {
            filterLeftOffset = ((outputWidth - 1)*strideCols +
                                            filterWidth - inputWidth + 1)/2;
            filterTopOffset = ((outputHeight - 1)*strideRows +
                                            filterHeight - inputHeight + 1)/2;
        }

        else
        {
            filterLeftOffset = ((outputWidth - 1)*strideCols +
                                            filterWidth - inputWidth)/2;
            filterTopOffset = ((outputHeight - 1)*strideRows +
                                            filterHeight - inputHeight)/2;
        }

        const int filterValueCount{filterWidth*filterHeight*inputDepth};

        OP_REQUIRES(opKernelContext, (filterValueCount*sizeof(float)) <= maxChunkSizeMpuSimConv2DFunctor,
													errors::InvalidArgument("Im2Col patch too large for buffer"));

        const int64 patchesPerChunk = maxChunkSizeMpuSimConv2DFunctor/(filterValueCount*sizeof(float));
        const int64 chunkValueCount{(maxChunkSizeMpuSimConv2DFunctor + (sizeof(float) - 1))/sizeof(float)};

        Im2ColBufferResource<float, chunkValueCount>* im2colBufferResource;

        std::function<Status(Im2ColBufferResource<float, chunkValueCount>**)> creator =
                                [](Im2ColBufferResource<float, chunkValueCount>** resource) {
            *resource = new Im2ColBufferResource<float, chunkValueCount>();
            return Status::OK();
        };

        OP_REQUIRES_OK(opKernelContext, opKernelContext->resource_manager()->LookupOrCreate(
                                                       "MpuSimConv2d", "im2col_buffer",
                                                       &im2colBufferResource, creator));

        mutex_lock lockBuffer(im2colBufferResource->mu);
        core::ScopedUnref unrefBuffer(im2colBufferResource);
        float* im2ColBuffer{im2colBufferResource->data};

        const int64 patchCount{batchSize*outputHeight*outputWidth};
        const int64 chunkCount{(patchCount + (patchesPerChunk - 1))/patchesPerChunk};
        
        if(chunkCount > 1L)
        {
            throw std::runtime_error("MpuSimConv2D implementation does not allow "
                                                "for chunked im2col/conv execution");
        }
        
        const int64 patchIndexStart{std::min(patchesPerChunk, patchCount)};

        for(int64 patchIndex{0L}; patchIndex < patchIndexStart; ++patchIndex)
        {
            const int64 batch{patchIndex/(outputHeight*outputWidth)};
            const int64 outputX{patchIndex % outputWidth};
            const int64 outputY{(patchIndex/outputWidth) % outputHeight};

            const float* inputBatchStart{inputData + batch*inputHeight*inputWidth*inputDepth};
            const int inputXOrigin = outputX*strideCols - filterLeftOffset;
            const int inputYOrigin = outputY*strideRows - filterTopOffset;
            const int patchIndexWithinChunk = patchIndex % patchesPerChunk;

            float* im2colPatchStart{im2ColBuffer + patchIndexWithinChunk*filterValueCount};

            for(int filterY = 0; filterY < filterHeight; ++filterY)
            {
                const int inputY{inputYOrigin + filterY};

                float* im2colRowStart{im2colPatchStart + filterY*filterWidth*inputDepth};

                if((inputY < 0) || (inputY >= inputHeight))
                {
                    float* im2colRowEnd{im2colRowStart + filterWidth*inputDepth};
                    std::fill(im2colRowStart, im2colRowEnd, float(0));
                }

                else
                {

                    const int inputXEnd{inputXOrigin + filterWidth};
                    const int leftZeroCount{std::max(0, 0 - inputXOrigin)};
                    const int rightZeroCount{std::max(0, inputXEnd - inputWidth)};
                    const int centerCopyCount{filterWidth - leftZeroCount + rightZeroCount};

                    if(leftZeroCount > 0)
                    {
                        float* im2colLeftStart{im2colRowStart};
                        float* im2colLeftEnd{im2colLeftStart + leftZeroCount*inputDepth};
                        std::fill(im2colLeftStart, im2colLeftEnd, float(0));
                    }

                    if(centerCopyCount > 0)
                    {
                        const float* inputRowStart{inputBatchStart + inputY*inputWidth*inputDepth +
                                                                    (std::max(0, inputXOrigin)*inputDepth)};
                        const float* inputRowEnd{inputRowStart + centerCopyCount*inputDepth};
                        float* im2colCenterStart{im2colRowStart + leftZeroCount*inputDepth};

                        std::copy(inputRowStart, inputRowEnd, im2colCenterStart);
                    }

                    if(rightZeroCount > 0)
                    {
                        float* im2colRightStart{im2colRowStart +
                                                    ((leftZeroCount + centerCopyCount) * inputDepth)};
                        float* im2colRightEnd{im2colRightStart + rightZeroCount*inputDepth};

                        std::fill(im2colRightStart, im2colRightEnd, float(0));
                    }
                }
            }
        }

        std::string opKernelString{opKernelContext->op_kernel().name()};

        opKernelString.erase(opKernelString.begin() + opKernelString.find_last_of("/"),
                                                                        opKernelString.end());
        
        const std::regex slashRegex("/");
        
        const std::string opKernelStringSlashesReplacedWithUnderscores{
                                                        std::regex_replace(opKernelString,
                                                                            slashRegex, "_")};
                                                                            
        MpuSimWrapper::getInstance().runMultiplication(
                                                activationsDatatypeSizeByte,
                                                weightsDatatypeSizeByte,
                                                resultsDatatypeSizeByte,
                                                systolicArrayHeight,
                                                systolicArrayWidth,
                                                activationFifoDepth,
                                                accumulatorArrayHeight,
                                                patchIndexStart,
                                                filterCount,
                                                filterValueCount,
                                                im2ColBuffer,
                                                filterData,
                                                outputData,
                                                logFileOutputDirString,
                                                modelNameString,
                                                opKernelStringSlashesReplacedWithUnderscores);

    }
};

class MpuSimConv2D : public BinaryOp<float>
{

public:

    explicit MpuSimConv2D(OpKernelConstruction* opKernelConstruction): BinaryOp<float>(opKernelConstruction)
    {
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr("strides", &m_strides));

        std::string dataFormat;

        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr("data_format", &dataFormat));

        OP_REQUIRES(opKernelConstruction, FormatFromString(dataFormat, &m_dataFormat),
                                        errors::InvalidArgument("Invalid data format"));

        OP_REQUIRES(opKernelConstruction, m_dataFormat == FORMAT_NHWC, errors::InvalidArgument(
                                            "Data format not supported by this kernel", dataFormat));

        OP_REQUIRES(opKernelConstruction, m_strides.size() == 4, errors::InvalidArgument(
                                    "Sliding window strides field must specify 4 dimensions"));

        const int64 strideN{GetTensorDim(m_strides, m_dataFormat, 'N')};
        const int64 strideC{GetTensorDim(m_strides, m_dataFormat, 'C')};

        OP_REQUIRES(opKernelConstruction, strideN == 1 && strideC == 1,
                        errors::InvalidArgument("Current implementation does not yet support "
                                                    "strides in the batch and depth dimensions."));

        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr("padding", &m_padding));
        
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr(
                                                                "activationsDatatypeSizeByte",
                                                                &m_activationsDatatypeSizeByte));
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr(
                                                                "weightsDatatypeSizeByte",
                                                                &m_weightsDatatypeSizeByte));
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr(
                                                                "resultsDatatypeSizeByte",
                                                                &m_resultsDatatypeSizeByte));
        
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr(
                                                                "systolicArrayHeight",
                                                                &m_systolicArrayHeight));
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr(
                                                                "systolicArrayWidth",
                                                                &m_systolicArrayWidth));
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr(
                                                                "activationFifoDepth",
                                                                &m_activationFifoDepth));
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr(
                                                                "accumulatorArrayHeight",
                                                                &m_accumulatorArrayHeight));
        
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr(
                                                                "modelName",
                                                                &m_modelNameString));
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr(
                                                                "logFileOutputDir",
                                                                &m_logFileOutputDirString));
        
//         std::cout << "Added mpusim-conv2d operation\n\tActivations size: "
//                     << m_activationsDatatypeSizeByte << " byte\tWeights size: "
//                     << m_weightsDatatypeSizeByte << " byte\tResults size: "
//                     << m_resultsDatatypeSizeByte << " byte\n\tSystolic array height: "
//                     << m_systolicArrayHeight << "\tSystolic array width: "
//                     << m_systolicArrayWidth << "\tActivation FIFO depth: "
//                     << m_activationFifoDepth << "\tAccumulator Array Height: "
//                     << m_accumulatorArrayHeight << "\n\tModel name: "
//                     << m_modelNameString << "\tLog file output dir: " 
//                     << m_logFileOutputDirString << std::endl;
        
    }

    void Compute(OpKernelContext* opKernelContext) override
    {
        /* Input tensor is of the following
         * dimensions: [ batch, in_rows, in_cols, in_depth ] */
        const Tensor& input{opKernelContext->input(0)};

        /* Input filter is of the following dimensions:
         * [ filter_rows, filter_cols, in_depth, out_depth] */
        const Tensor& filter{opKernelContext->input(1)};

        /* For 2D convolution, there should be 4 dimensions. */
        OP_REQUIRES(opKernelContext, input.dims() == 4, errors::InvalidArgument(
                            "input must be 4-dimensional", input.shape().DebugString()));
        OP_REQUIRES(opKernelContext, filter.dims() == 4, errors::InvalidArgument(
                            "filter must be 4-dimensional: ", filter.shape().DebugString()));

        for(size_t dimCount{0UL}; dimCount < 3; ++dimCount)
        {
            OP_REQUIRES(opKernelContext, FastBoundsCheck(filter.dim_size(dimCount),
                                                    std::numeric_limits<int>::max()),
                                                    errors::InvalidArgument("filter too large"));
        }

        /* The last dimension for input is in_depth.
         * It must be the same as the filter's in_depth. */
        const int64 inputDepth{GetTensorDim(input, m_dataFormat, 'C')};
        OP_REQUIRES(opKernelContext, inputDepth == filter.dim_size(2),
                        errors::InvalidArgument("input and filter must have the same depth: ",
                                                            inputDepth, " vs ", filter.dim_size(2)));

        /* The last dimension for filter is out_depth. */
        const int outputDepth{static_cast<int>(filter.dim_size(3))};

        /* The second dimension for input is rows/height.
         * The first dimension for filter is rows/height. */

        const int64 inputRowsRaw{GetTensorDim(input, m_dataFormat, 'H')};
        OP_REQUIRES(opKernelContext, FastBoundsCheck(inputRowsRaw, std::numeric_limits<int>::max()),
                                                        errors::InvalidArgument("Input rows too large"));

        const int inputRows{static_cast<int>(inputRowsRaw)};
        const int filterRows{static_cast<int>(filter.dim_size(0))};

        /* The third dimension for input is columns/width.
         * The second dimension for filter is columns/width. */

        const int64 inputColsRaw{GetTensorDim(input, m_dataFormat, 'W')};
        OP_REQUIRES(opKernelContext, FastBoundsCheck(inputColsRaw, std::numeric_limits<int>::max()),
                                                        errors::InvalidArgument("Input cols too large"));

        const int inputCols{static_cast<int>(inputColsRaw)};
        const int filterCols{static_cast<int>(filter.dim_size(1))};

        /* The first dimension for input is batch. */

        const int64 batchRaw{GetTensorDim(input, m_dataFormat, 'N')};
        OP_REQUIRES(opKernelContext, FastBoundsCheck(batchRaw, std::numeric_limits<int>::max()),
                                                        errors::InvalidArgument("batch is too large"));

        const int batch{static_cast<int>(batchRaw)};

        /* For now we take the stride from the second and third dimensions
         * only (we do not support striding on the batch or depth dimension). */

        const int strideRows{GetTensorDim(m_strides, m_dataFormat, 'H')};
        const int strideCols{GetTensorDim(m_strides, m_dataFormat, 'W')};

        int64 outputRows{0};
        int64 outputCols{0};
        int64 padRows{0};
        int64 padCols{0};

        OP_REQUIRES_OK(opKernelContext, GetWindowedOutputSize(inputRows, filterRows, strideRows,
                                                                    m_padding, &outputRows, &padRows));

        OP_REQUIRES_OK(opKernelContext, GetWindowedOutputSize(inputCols, filterCols, strideCols,
                                                                    m_padding, &outputCols, &padCols));

        TensorShape outputShape{ShapeFromFormat(m_dataFormat, batch,
                                                    outputRows, outputCols, outputDepth)};

        /* Output tensor is of the following
         * dimensions: [ in_batch, out_rows, out_cols, out_depth ] */

        Tensor* output{nullptr};

        OP_REQUIRES_OK(opKernelContext, opKernelContext->allocate_output(0, outputShape, &output));

        VLOG(2) << "MpuSimConv2D: inputDepth = " << inputDepth
                << ", inputCols = " << inputCols
                << ", filterCols = " << filterCols
                << ", inputRows = " << inputRows
                << ", filterRows = " << filterRows
                << ", strideRows = " << strideRows
                << ", strideCols = " << strideCols
                << ", outputDepth = " << outputDepth;

        /* If there is nothing to compute, return. */

        if(outputShape.num_elements() == 0)
        {
            return;
        }

        MpuSimConv2DFunctor mpuSimConv2DFunctor;

        mpuSimConv2DFunctor(opKernelContext, m_activationsDatatypeSizeByte, m_weightsDatatypeSizeByte,
                                m_resultsDatatypeSizeByte, m_systolicArrayHeight, m_systolicArrayWidth,
                                m_activationFifoDepth, m_accumulatorArrayHeight,input.flat<float>().data(),
                                batch, inputRows,  inputCols, inputDepth, filter.flat<float>().data(),
                                filterRows, filterCols, outputDepth, strideRows, strideCols, m_padding,
                                output->flat<float>().data(),  outputRows, outputCols,
                                m_logFileOutputDirString, m_modelNameString);
    }

private:

    std::vector<int32> m_strides;
    Padding m_padding;
    TensorFormat m_dataFormat;

    std::string m_logFileOutputDirString;
    std::string m_modelNameString;

    int64 m_activationsDatatypeSizeByte;
    int64 m_weightsDatatypeSizeByte;
    int64 m_resultsDatatypeSizeByte;
    
    int64 m_systolicArrayHeight;
    int64 m_systolicArrayWidth;
    int64 m_activationFifoDepth;
    int64 m_accumulatorArrayHeight;

    TF_DISALLOW_COPY_AND_ASSIGN(MpuSimConv2D);
};


REGISTER_OP("MpuSimConv2D")
                .Input("input: T")
                .Input("filter: T")
                .Output("output: T")
                .Attr("T: {float}")
                .Attr("activationsDatatypeSizeByte: int >= 1")
                .Attr("weightsDatatypeSizeByte: int >= 1")
                .Attr("resultsDatatypeSizeByte: int >= 1")
                .Attr("systolicArrayHeight: int >= 2")
                .Attr("systolicArrayWidth: int >= 2")
                .Attr("activationFifoDepth: int >= 4")
                .Attr("accumulatorArrayHeight: int >= 4")
                .Attr("logFileOutputDir: string")
                .Attr("modelName: string")
                .Attr("strides: list(int)")
                .Attr(GetPaddingAttrString())
                .Attr(GetConvnetDataFormatAttrString())
                .Attr("dilations: list(int) = [1, 1, 1, 1]")
                .SetShapeFn(shape_inference::Conv2DShape);

REGISTER_KERNEL_BUILDER(Name("MpuSimConv2D") \
                            .Device(DEVICE_CPU).TypeConstraint<float>("T"), \
                            MpuSimConv2D);
