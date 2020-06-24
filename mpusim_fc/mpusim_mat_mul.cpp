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

#define EIGEN_USE_THREADS

#include <stdexcept>
#include <string>
#include <regex>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include "mpusim_wrapper.h"

using namespace tensorflow;

struct MpuSimMatMulFunctor
{
    void operator()(OpKernelContext* opKernelContext,
                        const int64 activationsDatatypeSizeByte,
                        const int64 weightsDatatypeSizeByte,
                        const int64 resultsDatatypeSizeByte,
                        const int64 systolicArrayHeight,
                        const int64 systolicArrayWidth,
                        const int64 activationFifoDepth,
                        const int64 accumulatorArrayHeight,
                        const int64 sizeM,
                        const int64 sizeN,
                        const int64 sizeK,
                        const float* matrixA,
                        const float* matrixB,
                        float* matrixC,
                        const std::string& logFileOutputDirString,
                        const std::string& modelNameString)
    {
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
                                                    sizeM,
                                                    sizeN,
                                                    sizeK,
                                                    matrixA,
                                                    matrixB,
                                                    matrixC,
                                                    logFileOutputDirString,
                                                    modelNameString,
                                                    opKernelStringSlashesReplacedWithUnderscores);
    }
};

class MpuSimMatMulOp : public OpKernel
{

public:
    
    explicit MpuSimMatMulOp(OpKernelConstruction* opKernelConstruction): OpKernel(opKernelConstruction)
    {
        
        bool transposeA;
        bool transposeB;
        
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr(
                                                                    "transpose_a", &transposeA));
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr(
                                                                    "transpose_b", &transposeB));
        
        
        if(transposeA || transposeB)
        {
            throw std::invalid_argument("mpu_sim_mat_mul does not support "
                                            "transposition of input matrices");
        }
      
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr(
                                                                "activations_datatype_size_byte",
                                                                &m_activationsDatatypeSizeByte));
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr(
                                                                "weights_datatype_size_byte",
                                                                &m_weightsDatatypeSizeByte));
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr(
                                                                "results_datatype_size_byte",
                                                                &m_resultsDatatypeSizeByte));
        
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr(
                                                                "systolic_array_height",
                                                                &m_systolicArrayHeight));
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr(
                                                                "systolic_array_width",
                                                                &m_systolicArrayWidth));
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr(
                                                                "activation_fifo_depth",
                                                                &m_activationFifoDepth));
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr(
                                                                "accumulator_array_height",
                                                                &m_accumulatorArrayHeight));
        
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr(
                                                                "model_name",
                                                                &m_modelNameString));
        OP_REQUIRES_OK(opKernelConstruction, opKernelConstruction->GetAttr(
                                                                "log_file_output_dir",
                                                                &m_logFileOutputDirString));
                                                                                                                                
        std::cout << "Added mpu_sim_mat_mul operation\n\tActivations size: "
                    << m_activationsDatatypeSizeByte << " byte\tWeights size: "
                    << m_weightsDatatypeSizeByte << " byte\tResults size: "
                    << m_resultsDatatypeSizeByte << " byte\n\tSystolic array height: "
                    << m_systolicArrayHeight << "\tSystolic array width: "
                    << m_systolicArrayWidth << "\tActivation FIFO depth: "
                    << m_activationFifoDepth << "\tAccumulator Array Height: "
                    << m_accumulatorArrayHeight << "\n\tModel name: "
                    << m_modelNameString << "\tLog file output dir: " 
                    << m_logFileOutputDirString << std::endl;
    }

    void Compute(OpKernelContext* opKernelContext) override
    {
        const Tensor& tensorA = opKernelContext->input(0);
        const Tensor& tensorB = opKernelContext->input(1);

        // Check that the dimensions of the two matrices are valid.

        OP_REQUIRES(opKernelContext,
                        TensorShapeUtils::IsMatrix(tensorA.shape()),
                        errors::InvalidArgument(
                                    "In[0] is not a matrix. Instead it has shape ",
                                    tensorA.shape().DebugString()));
                        
        OP_REQUIRES(opKernelContext,
                        TensorShapeUtils::IsMatrix(tensorB.shape()),
                        errors::InvalidArgument(
                                    "In[1] is not a matrix. Instead it has shape ",
                                    tensorB.shape().DebugString()));
                        
        Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dimPair;

        dimPair[0].first = 1;
        dimPair[0].second = 0;

        OP_REQUIRES(opKernelContext,
                        tensorA.dim_size(dimPair[0].first) ==
                                    tensorB.dim_size(dimPair[0].second),
                        errors::InvalidArgument(
                            "Matrix size-incompatible: In[0]: ",
                            tensorA.shape().DebugString(),
                            ", In[1]: ", tensorB.shape().DebugString()));
                
        int dimARemaining = 1 - dimPair[0].first;
        int dimBRemaining = 1 - dimPair[0].second;

        TensorShape outShape({tensorA.dim_size(dimARemaining),
                                    tensorB.dim_size(dimBRemaining)});
            
        Tensor* tensorC{nullptr};

        OP_REQUIRES_OK(opKernelContext, opKernelContext->allocate_output(0, outShape, &tensorC));

        if(tensorC->NumElements() == 0)
        {
            return;
        }

        if((tensorA.NumElements() == 0) && (tensorB.NumElements() == 0))
        {
            return;
        }
        
//         std::cout << "mpu_sim_mat_mul: Tensor A shape: " <<  tensorA.shape().DebugString()
//             << "\nTensor B shape: "  <<  tensorB.shape().DebugString()
//             <<  "\nTensor C shape: "  <<  tensorC->shape().DebugString() <<  std::endl;


        std::cout << "mpu_sim_mat_mul:\n\tTensor A: Rows: "
                    << tensorA.dim_size(0)
                    << "\tcolumns: "
                    << tensorA.dim_size(1)
                    << "\n\tTensor B: Rows: "
                    << tensorB.dim_size(0)
                    << "\tcolumns: "
                    << tensorB.dim_size(1)
                    << "\n\tTensor C: Rows: "
                    << tensorC->dim_size(0)
                    << "\tcolumns: "
                    << tensorC->dim_size(1) << std::endl;

        MpuSimMatMulFunctor mpuSimMatMulFunctor;
        
        mpuSimMatMulFunctor(opKernelContext,
                                m_activationsDatatypeSizeByte,
                                m_weightsDatatypeSizeByte,
                                m_resultsDatatypeSizeByte,
                                m_systolicArrayHeight,
                                m_systolicArrayWidth,
                                m_activationFifoDepth,
                                m_accumulatorArrayHeight,
                                tensorA.dim_size(0),
                                tensorB.dim_size(1),
                                tensorA.dim_size(1),
                                tensorA.flat<float>().data(),
                                tensorB.flat<float>().data(),
                                tensorC->flat<float>().data(),
                                m_logFileOutputDirString,
                                m_modelNameString);
    }

 private:

    std::string m_logFileOutputDirString;
    std::string m_modelNameString;

    int64 m_activationsDatatypeSizeByte;
    int64 m_weightsDatatypeSizeByte;
    int64 m_resultsDatatypeSizeByte;
    
    int64 m_systolicArrayHeight;
    int64 m_systolicArrayWidth;
    int64 m_activationFifoDepth;
    int64 m_accumulatorArrayHeight;
};

REGISTER_OP("MpuSimMatMul")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("T: {float}")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("activations_datatype_size_byte: int >= 1")
    .Attr("weights_datatype_size_byte: int >= 1")
    .Attr("results_datatype_size_byte: int >= 1")
    .Attr("systolic_array_height: int >= 2")
    .Attr("systolic_array_width: int >= 2")
    .Attr("activation_fifo_depth: int >= 4")
    .Attr("accumulator_array_height: int >= 4")
    .Attr("log_file_output_dir: string")
    .Attr("model_name: string")
    .SetShapeFn(shape_inference::MatMulShape);
  
REGISTER_KERNEL_BUILDER(Name("MpuSimMatMul") \
                            .Device(DEVICE_CPU).TypeConstraint<float>("T"), \
                            MpuSimMatMulOp);
