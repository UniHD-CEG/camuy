/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Global Instruction Selector for the X86 target                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_GLOBALISEL_PREDICATE_BITSET
const unsigned MAX_SUBTARGET_PREDICATES = 113;
using PredicateBitset = llvm::PredicateBitsetImpl<MAX_SUBTARGET_PREDICATES>;
#endif // ifdef GET_GLOBALISEL_PREDICATE_BITSET

#ifdef GET_GLOBALISEL_TEMPORARIES_DECL
  mutable MatcherState State;
  typedef ComplexRendererFns(X86InstructionSelector::*ComplexMatcherMemFn)(MachineOperand &) const;
  typedef void(X86InstructionSelector::*CustomRendererFn)(MachineInstrBuilder &, const MachineInstr&) const;
  const ISelInfoTy<PredicateBitset, ComplexMatcherMemFn, CustomRendererFn> ISelInfo;
  static X86InstructionSelector::ComplexMatcherMemFn ComplexPredicateFns[];
  static X86InstructionSelector::CustomRendererFn CustomRenderers[];
  bool testImmPredicate_I64(unsigned PredicateID, int64_t Imm) const override;
  bool testImmPredicate_APInt(unsigned PredicateID, const APInt &Imm) const override;
  bool testImmPredicate_APFloat(unsigned PredicateID, const APFloat &Imm) const override;
  const int64_t *getMatchTable() const override;
  bool testMIPredicate_MI(unsigned PredicateID, const MachineInstr &MI) const override;
#endif // ifdef GET_GLOBALISEL_TEMPORARIES_DECL

#ifdef GET_GLOBALISEL_TEMPORARIES_INIT
, State(0),
ISelInfo(TypeObjects, NumTypeObjects, FeatureBitsets, ComplexPredicateFns, CustomRenderers)
#endif // ifdef GET_GLOBALISEL_TEMPORARIES_INIT

#ifdef GET_GLOBALISEL_IMPL
// Bits for subtarget features that participate in instruction matching.
enum SubtargetFeatureBits : uint8_t {
  Feature_TruePredicateBit = 48,
  Feature_HasCMovBit = 20,
  Feature_NoCMovBit = 100,
  Feature_HasMMXBit = 81,
  Feature_Has3DNowBit = 83,
  Feature_HasSSE1Bit = 34,
  Feature_UseSSE1Bit = 42,
  Feature_HasSSE2Bit = 35,
  Feature_UseSSE2Bit = 43,
  Feature_HasSSE3Bit = 26,
  Feature_UseSSE3Bit = 51,
  Feature_HasSSSE3Bit = 82,
  Feature_UseSSSE3Bit = 52,
  Feature_UseSSE41Bit = 49,
  Feature_HasSSE42Bit = 56,
  Feature_UseSSE42Bit = 55,
  Feature_HasSSE4ABit = 65,
  Feature_NoAVXBit = 61,
  Feature_HasAVXBit = 44,
  Feature_HasAVX2Bit = 38,
  Feature_HasAVX1OnlyBit = 36,
  Feature_HasAVX512Bit = 69,
  Feature_UseAVXBit = 40,
  Feature_UseAVX2Bit = 67,
  Feature_NoAVX512Bit = 31,
  Feature_HasCDIBit = 73,
  Feature_HasVPOPCNTDQBit = 77,
  Feature_HasERIBit = 76,
  Feature_HasDQIBit = 71,
  Feature_NoDQIBit = 53,
  Feature_HasBWIBit = 72,
  Feature_NoBWIBit = 50,
  Feature_HasVLXBit = 70,
  Feature_NoVLXBit = 30,
  Feature_NoVLX_Or_NoBWIBit = 47,
  Feature_NoVLX_Or_NoDQIBit = 104,
  Feature_HasVNNIBit = 79,
  Feature_HasBITALGBit = 80,
  Feature_HasPOPCNTBit = 54,
  Feature_HasAESBit = 58,
  Feature_HasVAESBit = 60,
  Feature_NoVLX_Or_NoVAESBit = 59,
  Feature_HasFXSRBit = 27,
  Feature_HasXSAVEBit = 89,
  Feature_HasXSAVEOPTBit = 90,
  Feature_HasXSAVECBit = 91,
  Feature_HasXSAVESBit = 92,
  Feature_HasPCLMULBit = 62,
  Feature_NoVLX_Or_NoVPCLMULQDQBit = 63,
  Feature_HasVPCLMULQDQBit = 64,
  Feature_HasGFNIBit = 68,
  Feature_HasFMABit = 28,
  Feature_HasFMA4Bit = 32,
  Feature_NoFMA4Bit = 29,
  Feature_HasXOPBit = 33,
  Feature_HasTBMBit = 9,
  Feature_NoTBMBit = 112,
  Feature_HasLWPBit = 10,
  Feature_HasMOVBEBit = 3,
  Feature_HasRDRANDBit = 4,
  Feature_HasF16CBit = 66,
  Feature_HasFSGSBaseBit = 93,
  Feature_HasLZCNTBit = 6,
  Feature_HasBMIBit = 7,
  Feature_HasBMI2Bit = 8,
  Feature_NoBMI2Bit = 111,
  Feature_HasVBMIBit = 74,
  Feature_HasVBMI2Bit = 78,
  Feature_HasIFMABit = 75,
  Feature_HasRTMBit = 87,
  Feature_HasSHABit = 57,
  Feature_HasRDSEEDBit = 5,
  Feature_HasSSEPrefetchBit = 45,
  Feature_NoSSEPrefetchBit = 84,
  Feature_HasPrefetchWBit = 85,
  Feature_HasPREFETCHWT1Bit = 86,
  Feature_HasLAHFSAHFBit = 2,
  Feature_HasMWAITXBit = 11,
  Feature_HasCLZEROBit = 15,
  Feature_HasCLDEMOTEBit = 18,
  Feature_HasMOVDIRIBit = 13,
  Feature_HasMOVDIR64BBit = 14,
  Feature_HasPTWRITEBit = 96,
  Feature_FPStackf32Bit = 24,
  Feature_FPStackf64Bit = 25,
  Feature_HasCLFLUSHOPTBit = 16,
  Feature_HasCLWBBit = 17,
  Feature_HasWBNOINVDBit = 88,
  Feature_HasRDPIDBit = 95,
  Feature_HasWAITPKGBit = 12,
  Feature_HasINVPCIDBit = 94,
  Feature_HasCmpxchg16bBit = 101,
  Feature_Not64BitModeBit = 0,
  Feature_In64BitModeBit = 1,
  Feature_IsLP64Bit = 98,
  Feature_NotLP64Bit = 97,
  Feature_NotWin64WithoutFPBit = 99,
  Feature_IsPS4Bit = 106,
  Feature_NotPS4Bit = 105,
  Feature_KernelCodeBit = 107,
  Feature_NearDataBit = 109,
  Feature_IsNotPICBit = 108,
  Feature_OptForSizeBit = 39,
  Feature_OptForMinSizeBit = 37,
  Feature_OptForSpeedBit = 103,
  Feature_UseIncDecBit = 19,
  Feature_NoSSE41_Or_OptForSizeBit = 41,
  Feature_CallImmAddrBit = 110,
  Feature_FavorMemIndirectCallBit = 21,
  Feature_HasFastSHLDRotateBit = 102,
  Feature_HasMFenceBit = 46,
  Feature_UseRetpolineIndirectCallsBit = 23,
  Feature_NotUseRetpolineIndirectCallsBit = 22,
};

PredicateBitset X86InstructionSelector::
computeAvailableModuleFeatures(const X86Subtarget *Subtarget) const {
  PredicateBitset Features;
  if (true)
    Features[Feature_TruePredicateBit] = 1;
  if (Subtarget->hasCMov())
    Features[Feature_HasCMovBit] = 1;
  if (!Subtarget->hasCMov())
    Features[Feature_NoCMovBit] = 1;
  if (Subtarget->hasMMX())
    Features[Feature_HasMMXBit] = 1;
  if (Subtarget->has3DNow())
    Features[Feature_Has3DNowBit] = 1;
  if (Subtarget->hasSSE1())
    Features[Feature_HasSSE1Bit] = 1;
  if (Subtarget->hasSSE1() && !Subtarget->hasAVX())
    Features[Feature_UseSSE1Bit] = 1;
  if (Subtarget->hasSSE2())
    Features[Feature_HasSSE2Bit] = 1;
  if (Subtarget->hasSSE2() && !Subtarget->hasAVX())
    Features[Feature_UseSSE2Bit] = 1;
  if (Subtarget->hasSSE3())
    Features[Feature_HasSSE3Bit] = 1;
  if (Subtarget->hasSSE3() && !Subtarget->hasAVX())
    Features[Feature_UseSSE3Bit] = 1;
  if (Subtarget->hasSSSE3())
    Features[Feature_HasSSSE3Bit] = 1;
  if (Subtarget->hasSSSE3() && !Subtarget->hasAVX())
    Features[Feature_UseSSSE3Bit] = 1;
  if (Subtarget->hasSSE41() && !Subtarget->hasAVX())
    Features[Feature_UseSSE41Bit] = 1;
  if (Subtarget->hasSSE42())
    Features[Feature_HasSSE42Bit] = 1;
  if (Subtarget->hasSSE42() && !Subtarget->hasAVX())
    Features[Feature_UseSSE42Bit] = 1;
  if (Subtarget->hasSSE4A())
    Features[Feature_HasSSE4ABit] = 1;
  if (!Subtarget->hasAVX())
    Features[Feature_NoAVXBit] = 1;
  if (Subtarget->hasAVX())
    Features[Feature_HasAVXBit] = 1;
  if (Subtarget->hasAVX2())
    Features[Feature_HasAVX2Bit] = 1;
  if (Subtarget->hasAVX() && !Subtarget->hasAVX2())
    Features[Feature_HasAVX1OnlyBit] = 1;
  if (Subtarget->hasAVX512())
    Features[Feature_HasAVX512Bit] = 1;
  if (Subtarget->hasAVX() && !Subtarget->hasAVX512())
    Features[Feature_UseAVXBit] = 1;
  if (Subtarget->hasAVX2() && !Subtarget->hasAVX512())
    Features[Feature_UseAVX2Bit] = 1;
  if (!Subtarget->hasAVX512())
    Features[Feature_NoAVX512Bit] = 1;
  if (Subtarget->hasCDI())
    Features[Feature_HasCDIBit] = 1;
  if (Subtarget->hasVPOPCNTDQ())
    Features[Feature_HasVPOPCNTDQBit] = 1;
  if (Subtarget->hasERI())
    Features[Feature_HasERIBit] = 1;
  if (Subtarget->hasDQI())
    Features[Feature_HasDQIBit] = 1;
  if (!Subtarget->hasDQI())
    Features[Feature_NoDQIBit] = 1;
  if (Subtarget->hasBWI())
    Features[Feature_HasBWIBit] = 1;
  if (!Subtarget->hasBWI())
    Features[Feature_NoBWIBit] = 1;
  if (Subtarget->hasVLX())
    Features[Feature_HasVLXBit] = 1;
  if (!Subtarget->hasVLX())
    Features[Feature_NoVLXBit] = 1;
  if (!Subtarget->hasVLX() || !Subtarget->hasBWI())
    Features[Feature_NoVLX_Or_NoBWIBit] = 1;
  if (!Subtarget->hasVLX() || !Subtarget->hasDQI())
    Features[Feature_NoVLX_Or_NoDQIBit] = 1;
  if (Subtarget->hasVNNI())
    Features[Feature_HasVNNIBit] = 1;
  if (Subtarget->hasBITALG())
    Features[Feature_HasBITALGBit] = 1;
  if (Subtarget->hasPOPCNT())
    Features[Feature_HasPOPCNTBit] = 1;
  if (Subtarget->hasAES())
    Features[Feature_HasAESBit] = 1;
  if (Subtarget->hasVAES())
    Features[Feature_HasVAESBit] = 1;
  if (!Subtarget->hasVLX() || !Subtarget->hasVAES())
    Features[Feature_NoVLX_Or_NoVAESBit] = 1;
  if (Subtarget->hasFXSR())
    Features[Feature_HasFXSRBit] = 1;
  if (Subtarget->hasXSAVE())
    Features[Feature_HasXSAVEBit] = 1;
  if (Subtarget->hasXSAVEOPT())
    Features[Feature_HasXSAVEOPTBit] = 1;
  if (Subtarget->hasXSAVEC())
    Features[Feature_HasXSAVECBit] = 1;
  if (Subtarget->hasXSAVES())
    Features[Feature_HasXSAVESBit] = 1;
  if (Subtarget->hasPCLMUL())
    Features[Feature_HasPCLMULBit] = 1;
  if (!Subtarget->hasVLX() || !Subtarget->hasVPCLMULQDQ())
    Features[Feature_NoVLX_Or_NoVPCLMULQDQBit] = 1;
  if (Subtarget->hasVPCLMULQDQ())
    Features[Feature_HasVPCLMULQDQBit] = 1;
  if (Subtarget->hasGFNI())
    Features[Feature_HasGFNIBit] = 1;
  if (Subtarget->hasFMA())
    Features[Feature_HasFMABit] = 1;
  if (Subtarget->hasFMA4())
    Features[Feature_HasFMA4Bit] = 1;
  if (!Subtarget->hasFMA4())
    Features[Feature_NoFMA4Bit] = 1;
  if (Subtarget->hasXOP())
    Features[Feature_HasXOPBit] = 1;
  if (Subtarget->hasTBM())
    Features[Feature_HasTBMBit] = 1;
  if (!Subtarget->hasTBM())
    Features[Feature_NoTBMBit] = 1;
  if (Subtarget->hasLWP())
    Features[Feature_HasLWPBit] = 1;
  if (Subtarget->hasMOVBE())
    Features[Feature_HasMOVBEBit] = 1;
  if (Subtarget->hasRDRAND())
    Features[Feature_HasRDRANDBit] = 1;
  if (Subtarget->hasF16C())
    Features[Feature_HasF16CBit] = 1;
  if (Subtarget->hasFSGSBase())
    Features[Feature_HasFSGSBaseBit] = 1;
  if (Subtarget->hasLZCNT())
    Features[Feature_HasLZCNTBit] = 1;
  if (Subtarget->hasBMI())
    Features[Feature_HasBMIBit] = 1;
  if (Subtarget->hasBMI2())
    Features[Feature_HasBMI2Bit] = 1;
  if (!Subtarget->hasBMI2())
    Features[Feature_NoBMI2Bit] = 1;
  if (Subtarget->hasVBMI())
    Features[Feature_HasVBMIBit] = 1;
  if (Subtarget->hasVBMI2())
    Features[Feature_HasVBMI2Bit] = 1;
  if (Subtarget->hasIFMA())
    Features[Feature_HasIFMABit] = 1;
  if (Subtarget->hasRTM())
    Features[Feature_HasRTMBit] = 1;
  if (Subtarget->hasSHA())
    Features[Feature_HasSHABit] = 1;
  if (Subtarget->hasRDSEED())
    Features[Feature_HasRDSEEDBit] = 1;
  if (Subtarget->hasSSEPrefetch())
    Features[Feature_HasSSEPrefetchBit] = 1;
  if (!Subtarget->hasSSEPrefetch())
    Features[Feature_NoSSEPrefetchBit] = 1;
  if (Subtarget->hasPRFCHW())
    Features[Feature_HasPrefetchWBit] = 1;
  if (Subtarget->hasPREFETCHWT1())
    Features[Feature_HasPREFETCHWT1Bit] = 1;
  if (Subtarget->hasLAHFSAHF())
    Features[Feature_HasLAHFSAHFBit] = 1;
  if (Subtarget->hasMWAITX())
    Features[Feature_HasMWAITXBit] = 1;
  if (Subtarget->hasCLZERO())
    Features[Feature_HasCLZEROBit] = 1;
  if (Subtarget->hasCLDEMOTE())
    Features[Feature_HasCLDEMOTEBit] = 1;
  if (Subtarget->hasMOVDIRI())
    Features[Feature_HasMOVDIRIBit] = 1;
  if (Subtarget->hasMOVDIR64B())
    Features[Feature_HasMOVDIR64BBit] = 1;
  if (Subtarget->hasPTWRITE())
    Features[Feature_HasPTWRITEBit] = 1;
  if (!Subtarget->hasSSE1())
    Features[Feature_FPStackf32Bit] = 1;
  if (!Subtarget->hasSSE2())
    Features[Feature_FPStackf64Bit] = 1;
  if (Subtarget->hasCLFLUSHOPT())
    Features[Feature_HasCLFLUSHOPTBit] = 1;
  if (Subtarget->hasCLWB())
    Features[Feature_HasCLWBBit] = 1;
  if (Subtarget->hasWBNOINVD())
    Features[Feature_HasWBNOINVDBit] = 1;
  if (Subtarget->hasRDPID())
    Features[Feature_HasRDPIDBit] = 1;
  if (Subtarget->hasWAITPKG())
    Features[Feature_HasWAITPKGBit] = 1;
  if (Subtarget->hasINVPCID())
    Features[Feature_HasINVPCIDBit] = 1;
  if (Subtarget->hasCmpxchg16b())
    Features[Feature_HasCmpxchg16bBit] = 1;
  if (!Subtarget->is64Bit())
    Features[Feature_Not64BitModeBit] = 1;
  if (Subtarget->is64Bit())
    Features[Feature_In64BitModeBit] = 1;
  if (Subtarget->isTarget64BitLP64())
    Features[Feature_IsLP64Bit] = 1;
  if (!Subtarget->isTarget64BitLP64())
    Features[Feature_NotLP64Bit] = 1;
  if (Subtarget->isTargetPS4())
    Features[Feature_IsPS4Bit] = 1;
  if (!Subtarget->isTargetPS4())
    Features[Feature_NotPS4Bit] = 1;
  if (TM.getCodeModel() == CodeModel::Kernel)
    Features[Feature_KernelCodeBit] = 1;
  if (TM.getCodeModel() == CodeModel::Small ||TM.getCodeModel() == CodeModel::Kernel)
    Features[Feature_NearDataBit] = 1;
  if (!TM.isPositionIndependent())
    Features[Feature_IsNotPICBit] = 1;
  if (Subtarget->isLegalToCallImmediateAddr())
    Features[Feature_CallImmAddrBit] = 1;
  if (!Subtarget->slowTwoMemOps())
    Features[Feature_FavorMemIndirectCallBit] = 1;
  if (Subtarget->hasFastSHLDRotate())
    Features[Feature_HasFastSHLDRotateBit] = 1;
  if (Subtarget->hasMFence())
    Features[Feature_HasMFenceBit] = 1;
  if (Subtarget->useRetpolineIndirectCalls())
    Features[Feature_UseRetpolineIndirectCallsBit] = 1;
  if (!Subtarget->useRetpolineIndirectCalls())
    Features[Feature_NotUseRetpolineIndirectCallsBit] = 1;
  return Features;
}

PredicateBitset X86InstructionSelector::
computeAvailableFunctionFeatures(const X86Subtarget *Subtarget, const MachineFunction *MF) const {
  PredicateBitset Features;
  if (!Subtarget->isTargetWin64() ||Subtarget->getFrameLowering()->hasFP(*MF))
    Features[Feature_NotWin64WithoutFPBit] = 1;
  if (MF->getFunction().optForSize())
    Features[Feature_OptForSizeBit] = 1;
  if (MF->getFunction().optForMinSize())
    Features[Feature_OptForMinSizeBit] = 1;
  if (!MF->getFunction().optForSize())
    Features[Feature_OptForSpeedBit] = 1;
  if (!Subtarget->slowIncDec() || MF->getFunction().optForSize())
    Features[Feature_UseIncDecBit] = 1;
  if (MF->getFunction().optForSize() || !Subtarget->hasSSE41())
    Features[Feature_NoSSE41_Or_OptForSizeBit] = 1;
  return Features;
}

// LLT Objects.
enum {
  GILLT_s1,
  GILLT_s8,
  GILLT_s16,
  GILLT_s32,
  GILLT_s64,
  GILLT_s80,
  GILLT_s128,
  GILLT_v2s1,
  GILLT_v2s64,
  GILLT_v4s1,
  GILLT_v4s32,
  GILLT_v4s64,
  GILLT_v8s1,
  GILLT_v8s16,
  GILLT_v8s32,
  GILLT_v8s64,
  GILLT_v16s1,
  GILLT_v16s8,
  GILLT_v16s16,
  GILLT_v16s32,
  GILLT_v32s1,
  GILLT_v32s8,
  GILLT_v32s16,
  GILLT_v64s1,
  GILLT_v64s8,
};
const static size_t NumTypeObjects = 25;
const static LLT TypeObjects[] = {
  LLT::scalar(1),
  LLT::scalar(8),
  LLT::scalar(16),
  LLT::scalar(32),
  LLT::scalar(64),
  LLT::scalar(80),
  LLT::scalar(128),
  LLT::vector(2, 1),
  LLT::vector(2, 64),
  LLT::vector(4, 1),
  LLT::vector(4, 32),
  LLT::vector(4, 64),
  LLT::vector(8, 1),
  LLT::vector(8, 16),
  LLT::vector(8, 32),
  LLT::vector(8, 64),
  LLT::vector(16, 1),
  LLT::vector(16, 8),
  LLT::vector(16, 16),
  LLT::vector(16, 32),
  LLT::vector(32, 1),
  LLT::vector(32, 8),
  LLT::vector(32, 16),
  LLT::vector(64, 1),
  LLT::vector(64, 8),
};

// Feature bitsets.
enum {
  GIFBS_Invalid,
  GIFBS_FPStackf32,
  GIFBS_FPStackf64,
  GIFBS_Has3DNow,
  GIFBS_HasAVX,
  GIFBS_HasAVX1Only,
  GIFBS_HasAVX2,
  GIFBS_HasAVX512,
  GIFBS_HasBITALG,
  GIFBS_HasBMI,
  GIFBS_HasBMI2,
  GIFBS_HasBWI,
  GIFBS_HasCDI,
  GIFBS_HasDQI,
  GIFBS_HasLWP,
  GIFBS_HasMFence,
  GIFBS_HasMMX,
  GIFBS_HasMOVBE,
  GIFBS_HasPTWRITE,
  GIFBS_HasRTM,
  GIFBS_HasSHA,
  GIFBS_HasSSE1,
  GIFBS_HasSSE2,
  GIFBS_HasSSE42,
  GIFBS_HasSSE4A,
  GIFBS_HasTBM,
  GIFBS_HasVLX,
  GIFBS_HasVPOPCNTDQ,
  GIFBS_HasWAITPKG,
  GIFBS_HasWBNOINVD,
  GIFBS_HasXOP,
  GIFBS_In64BitMode,
  GIFBS_NoDQI,
  GIFBS_Not64BitMode,
  GIFBS_UseAVX,
  GIFBS_UseIncDec,
  GIFBS_UseSSE1,
  GIFBS_UseSSE2,
  GIFBS_UseSSE41,
  GIFBS_UseSSSE3,
  GIFBS_HasAES_HasAVX,
  GIFBS_HasAES_NoAVX,
  GIFBS_HasAVX_NoVLX,
  GIFBS_HasAVX_NoVLX_Or_NoBWI,
  GIFBS_HasAVX2_NoVLX,
  GIFBS_HasAVX2_NoVLX_Or_NoBWI,
  GIFBS_HasAVX512_HasVAES,
  GIFBS_HasAVX512_HasVLX,
  GIFBS_HasAVX512_HasVPCLMULQDQ,
  GIFBS_HasAVX512_NoBWI,
  GIFBS_HasBITALG_HasVLX,
  GIFBS_HasBWI_HasVLX,
  GIFBS_HasCDI_HasVLX,
  GIFBS_HasDQI_HasVLX,
  GIFBS_HasDQI_NoBWI,
  GIFBS_HasFSGSBase_In64BitMode,
  GIFBS_HasPCLMUL_NoAVX,
  GIFBS_HasPTWRITE_In64BitMode,
  GIFBS_HasRDPID_Not64BitMode,
  GIFBS_HasSSE2_NoAVX512,
  GIFBS_HasVAES_HasVLX,
  GIFBS_HasVAES_NoVLX,
  GIFBS_HasVLX_HasVPCLMULQDQ,
  GIFBS_HasVLX_HasVPOPCNTDQ,
  GIFBS_HasVPCLMULQDQ_NoVLX,
  GIFBS_HasWAITPKG_In64BitMode,
  GIFBS_HasWAITPKG_Not64BitMode,
  GIFBS_Not64BitMode_OptForSize,
  GIFBS_NotWin64WithoutFP_OptForMinSize,
  GIFBS_HasAES_HasAVX_NoVLX_Or_NoVAES,
  GIFBS_HasAVX_HasPCLMUL_NoVLX_Or_NoVPCLMULQDQ,
  GIFBS_HasDQI_HasVLX_NoBWI,
};
const static PredicateBitset FeatureBitsets[] {
  {}, // GIFBS_Invalid
  {Feature_FPStackf32Bit, },
  {Feature_FPStackf64Bit, },
  {Feature_Has3DNowBit, },
  {Feature_HasAVXBit, },
  {Feature_HasAVX1OnlyBit, },
  {Feature_HasAVX2Bit, },
  {Feature_HasAVX512Bit, },
  {Feature_HasBITALGBit, },
  {Feature_HasBMIBit, },
  {Feature_HasBMI2Bit, },
  {Feature_HasBWIBit, },
  {Feature_HasCDIBit, },
  {Feature_HasDQIBit, },
  {Feature_HasLWPBit, },
  {Feature_HasMFenceBit, },
  {Feature_HasMMXBit, },
  {Feature_HasMOVBEBit, },
  {Feature_HasPTWRITEBit, },
  {Feature_HasRTMBit, },
  {Feature_HasSHABit, },
  {Feature_HasSSE1Bit, },
  {Feature_HasSSE2Bit, },
  {Feature_HasSSE42Bit, },
  {Feature_HasSSE4ABit, },
  {Feature_HasTBMBit, },
  {Feature_HasVLXBit, },
  {Feature_HasVPOPCNTDQBit, },
  {Feature_HasWAITPKGBit, },
  {Feature_HasWBNOINVDBit, },
  {Feature_HasXOPBit, },
  {Feature_In64BitModeBit, },
  {Feature_NoDQIBit, },
  {Feature_Not64BitModeBit, },
  {Feature_UseAVXBit, },
  {Feature_UseIncDecBit, },
  {Feature_UseSSE1Bit, },
  {Feature_UseSSE2Bit, },
  {Feature_UseSSE41Bit, },
  {Feature_UseSSSE3Bit, },
  {Feature_HasAESBit, Feature_HasAVXBit, },
  {Feature_HasAESBit, Feature_NoAVXBit, },
  {Feature_HasAVXBit, Feature_NoVLXBit, },
  {Feature_HasAVXBit, Feature_NoVLX_Or_NoBWIBit, },
  {Feature_HasAVX2Bit, Feature_NoVLXBit, },
  {Feature_HasAVX2Bit, Feature_NoVLX_Or_NoBWIBit, },
  {Feature_HasAVX512Bit, Feature_HasVAESBit, },
  {Feature_HasAVX512Bit, Feature_HasVLXBit, },
  {Feature_HasAVX512Bit, Feature_HasVPCLMULQDQBit, },
  {Feature_HasAVX512Bit, Feature_NoBWIBit, },
  {Feature_HasBITALGBit, Feature_HasVLXBit, },
  {Feature_HasBWIBit, Feature_HasVLXBit, },
  {Feature_HasCDIBit, Feature_HasVLXBit, },
  {Feature_HasDQIBit, Feature_HasVLXBit, },
  {Feature_HasDQIBit, Feature_NoBWIBit, },
  {Feature_HasFSGSBaseBit, Feature_In64BitModeBit, },
  {Feature_HasPCLMULBit, Feature_NoAVXBit, },
  {Feature_HasPTWRITEBit, Feature_In64BitModeBit, },
  {Feature_HasRDPIDBit, Feature_Not64BitModeBit, },
  {Feature_HasSSE2Bit, Feature_NoAVX512Bit, },
  {Feature_HasVAESBit, Feature_HasVLXBit, },
  {Feature_HasVAESBit, Feature_NoVLXBit, },
  {Feature_HasVLXBit, Feature_HasVPCLMULQDQBit, },
  {Feature_HasVLXBit, Feature_HasVPOPCNTDQBit, },
  {Feature_HasVPCLMULQDQBit, Feature_NoVLXBit, },
  {Feature_HasWAITPKGBit, Feature_In64BitModeBit, },
  {Feature_HasWAITPKGBit, Feature_Not64BitModeBit, },
  {Feature_Not64BitModeBit, Feature_OptForSizeBit, },
  {Feature_NotWin64WithoutFPBit, Feature_OptForMinSizeBit, },
  {Feature_HasAESBit, Feature_HasAVXBit, Feature_NoVLX_Or_NoVAESBit, },
  {Feature_HasAVXBit, Feature_HasPCLMULBit, Feature_NoVLX_Or_NoVPCLMULQDQBit, },
  {Feature_HasDQIBit, Feature_HasVLXBit, Feature_NoBWIBit, },
};

// ComplexPattern predicates.
enum {
  GICP_Invalid,
};
// See constructor for table contents

// PatFrag predicates.
enum {
  GIPFP_I64_Predicate_AndMask64 = GIPFP_I64_Invalid + 1,
  GIPFP_I64_Predicate_BTCBTSMask64,
  GIPFP_I64_Predicate_BTRMask64,
  GIPFP_I64_Predicate_PrefetchWT1Level,
  GIPFP_I64_Predicate_i16immSExt8,
  GIPFP_I64_Predicate_i32immSExt8,
  GIPFP_I64_Predicate_i64immSExt32,
  GIPFP_I64_Predicate_i64immSExt8,
  GIPFP_I64_Predicate_i64immZExt32,
  GIPFP_I64_Predicate_i64immZExt32SExt8,
  GIPFP_I64_Predicate_immShift16,
  GIPFP_I64_Predicate_immShift32,
  GIPFP_I64_Predicate_immShift64,
  GIPFP_I64_Predicate_immShift8,
  GIPFP_I64_Predicate_immff00_ffff,
};
bool X86InstructionSelector::testImmPredicate_I64(unsigned PredicateID, int64_t Imm) const {
  switch (PredicateID) {
  case GIPFP_I64_Predicate_AndMask64: {
    
  return isMask_64(Imm) && !isUInt<32>(Imm);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_BTCBTSMask64: {
    
  return !isInt<32>(Imm) && isPowerOf2_64(Imm);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_BTRMask64: {
    
  return !isUInt<32>(Imm) && !isInt<32>(Imm) && isPowerOf2_64(~Imm);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_PrefetchWT1Level: {
    
  return Imm < 3;

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_i16immSExt8: {
     return isInt<8>(Imm); 
    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_i32immSExt8: {
     return isInt<8>(Imm); 
    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_i64immSExt32: {
     return isInt<32>(Imm); 
    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_i64immSExt8: {
     return isInt<8>(Imm); 
    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_i64immZExt32: {
     return isUInt<32>(Imm); 
    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_i64immZExt32SExt8: {
    
  return isUInt<32>(Imm) && isInt<8>(static_cast<int32_t>(Imm));

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_immShift16: {
    
  return countTrailingOnes<uint64_t>(Imm) >= 4;

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_immShift32: {
    
  return countTrailingOnes<uint64_t>(Imm) >= 5;

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_immShift64: {
    
  return countTrailingOnes<uint64_t>(Imm) >= 6;

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_immShift8: {
    
  return countTrailingOnes<uint64_t>(Imm) >= 3;

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_I64_Predicate_immff00_ffff: {
    
  return Imm >= 0xff00 && Imm <= 0xffff;

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  }
  llvm_unreachable("Unknown predicate");
  return false;
}
// PatFrag predicates.
enum {
  GIPFP_APFloat_Predicate_fpimm0 = GIPFP_APFloat_Invalid + 1,
  GIPFP_APFloat_Predicate_fpimm1,
  GIPFP_APFloat_Predicate_fpimmneg0,
  GIPFP_APFloat_Predicate_fpimmneg1,
};
bool X86InstructionSelector::testImmPredicate_APFloat(unsigned PredicateID, const APFloat & Imm) const {
  switch (PredicateID) {
  case GIPFP_APFloat_Predicate_fpimm0: {
    
  return Imm.isExactlyValue(+0.0);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_APFloat_Predicate_fpimm1: {
    
  return Imm.isExactlyValue(+1.0);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_APFloat_Predicate_fpimmneg0: {
    
  return Imm.isExactlyValue(-0.0);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  case GIPFP_APFloat_Predicate_fpimmneg1: {
    
  return Imm.isExactlyValue(-1.0);

    llvm_unreachable("ImmediateCode should have returned");
    return false;
  }
  }
  llvm_unreachable("Unknown predicate");
  return false;
}
bool X86InstructionSelector::testImmPredicate_APInt(unsigned PredicateID, const APInt & Imm) const {
  llvm_unreachable("Unknown predicate");
  return false;
}
bool X86InstructionSelector::testMIPredicate_MI(unsigned PredicateID, const MachineInstr & MI) const {
  const MachineFunction &MF = *MI.getParent()->getParent();
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  (void)MRI;
  llvm_unreachable("Unknown predicate");
  return false;
}

X86InstructionSelector::ComplexMatcherMemFn
X86InstructionSelector::ComplexPredicateFns[] = {
  nullptr, // GICP_Invalid
};

// Custom renderers.
enum {
  GICR_Invalid,
};
X86InstructionSelector::CustomRendererFn
X86InstructionSelector::CustomRenderers[] = {
  nullptr, // GICP_Invalid
};

bool X86InstructionSelector::selectImpl(MachineInstr &I, CodeGenCoverage &CoverageInfo) const {
  MachineFunction &MF = *I.getParent()->getParent();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  // FIXME: This should be computed on a per-function basis rather than per-insn.
  AvailableFunctionFeatures = computeAvailableFunctionFeatures(&STI, &MF);
  const PredicateBitset AvailableFeatures = getAvailableFeatures();
  NewMIVector OutMIs;
  State.MIs.clear();
  State.MIs.push_back(&I);

  if (executeMatchTable(*this, OutMIs, State, ISelInfo, getMatchTable(), TII, MRI, TRI, RBI, AvailableFeatures, CoverageInfo)) {
    return true;
  }

  return false;
}

const int64_t *X86InstructionSelector::getMatchTable() const {
  constexpr static int64_t MatchTable0[] = {
    GIM_SwitchOpcode, /*MI*/0, /*[*/34, 139, /*)*//*default:*//*Label 34*/ 32281,
    /*TargetOpcode::G_ADD*//*Label 0*/ 110,
    /*TargetOpcode::G_SUB*//*Label 1*/ 1727,
    /*TargetOpcode::G_MUL*//*Label 2*/ 2808, 0, 0, 0, 0,
    /*TargetOpcode::G_AND*//*Label 3*/ 3574,
    /*TargetOpcode::G_OR*//*Label 4*/ 7079,
    /*TargetOpcode::G_XOR*//*Label 5*/ 10333, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    /*TargetOpcode::G_BITCAST*//*Label 6*/ 12889, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    /*TargetOpcode::G_INTRINSIC*//*Label 7*/ 13289,
    /*TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS*//*Label 8*/ 19872,
    /*TargetOpcode::G_ANYEXT*//*Label 9*/ 21303,
    /*TargetOpcode::G_TRUNC*//*Label 10*/ 22111,
    /*TargetOpcode::G_CONSTANT*//*Label 11*/ 22542,
    /*TargetOpcode::G_FCONSTANT*//*Label 12*/ 22745, 0, 0,
    /*TargetOpcode::G_SEXT*//*Label 13*/ 22938,
    /*TargetOpcode::G_ZEXT*//*Label 14*/ 23743,
    /*TargetOpcode::G_SHL*//*Label 15*/ 24157,
    /*TargetOpcode::G_LSHR*//*Label 16*/ 24882,
    /*TargetOpcode::G_ASHR*//*Label 17*/ 25591, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    /*TargetOpcode::G_FADD*//*Label 18*/ 26248,
    /*TargetOpcode::G_FSUB*//*Label 19*/ 26861,
    /*TargetOpcode::G_FMUL*//*Label 20*/ 27474, 0,
    /*TargetOpcode::G_FDIV*//*Label 21*/ 28087, 0, 0, 0, 0, 0, 0, 0,
    /*TargetOpcode::G_FNEG*//*Label 22*/ 28700,
    /*TargetOpcode::G_FPEXT*//*Label 23*/ 28789,
    /*TargetOpcode::G_FPTRUNC*//*Label 24*/ 29065,
    /*TargetOpcode::G_FPTOSI*//*Label 25*/ 29350,
    /*TargetOpcode::G_FPTOUI*//*Label 26*/ 29980,
    /*TargetOpcode::G_SITOFP*//*Label 27*/ 30334,
    /*TargetOpcode::G_UITOFP*//*Label 28*/ 31164, 0, 0, 0,
    /*TargetOpcode::G_BR*//*Label 29*/ 31618, 0, 0, 0, 0,
    /*TargetOpcode::G_CTTZ_ZERO_UNDEF*//*Label 30*/ 31631,
    /*TargetOpcode::G_CTLZ*//*Label 31*/ 31716, 0,
    /*TargetOpcode::G_CTPOP*//*Label 32*/ 31879,
    /*TargetOpcode::G_BSWAP*//*Label 33*/ 32191,
    // Label 0: @110
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 25, /*)*//*default:*//*Label 51*/ 1726,
    /*GILLT_s8*//*Label 35*/ 140,
    /*GILLT_s16*//*Label 36*/ 254,
    /*GILLT_s32*//*Label 37*/ 426,
    /*GILLT_s64*//*Label 38*/ 598, 0, 0, 0,
    /*GILLT_v2s64*//*Label 39*/ 798, 0,
    /*GILLT_v4s32*//*Label 40*/ 879,
    /*GILLT_v4s64*//*Label 41*/ 1082, 0,
    /*GILLT_v8s16*//*Label 42*/ 1140,
    /*GILLT_v8s32*//*Label 43*/ 1343,
    /*GILLT_v8s64*//*Label 44*/ 1401, 0,
    /*GILLT_v16s8*//*Label 45*/ 1433,
    /*GILLT_v16s16*//*Label 46*/ 1514,
    /*GILLT_v16s32*//*Label 47*/ 1572, 0,
    /*GILLT_v32s8*//*Label 48*/ 1604,
    /*GILLT_v32s16*//*Label 49*/ 1662, 0,
    /*GILLT_v64s8*//*Label 50*/ 1694,
    // Label 35: @140
    GIM_Try, /*On fail goto*//*Label 52*/ 253,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 53*/ 182, // Rule ID 17424 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (add:{ *:[i8] } GR8:{ *:[i8] }:$src, 1:{ *:[i8] })  =>  (INC8r:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INC8r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17424,
        GIR_Done,
      // Label 53: @182
      GIM_Try, /*On fail goto*//*Label 54*/ 206, // Rule ID 17428 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (add:{ *:[i8] } GR8:{ *:[i8] }:$src, -1:{ *:[i8] })  =>  (DEC8r:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::DEC8r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17428,
        GIR_Done,
      // Label 54: @206
      GIM_Try, /*On fail goto*//*Label 55*/ 236, // Rule ID 17378 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (ADD8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17378,
        GIR_Done,
      // Label 55: @236
      GIM_Try, /*On fail goto*//*Label 56*/ 252, // Rule ID 17370 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR8RegClassID,
        // (add:{ *:[i8] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)  =>  (ADD8rr:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD8rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17370,
        GIR_Done,
      // Label 56: @252
      GIM_Reject,
    // Label 52: @253
    GIM_Reject,
    // Label 36: @254
    GIM_Try, /*On fail goto*//*Label 57*/ 425,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 58*/ 297, // Rule ID 17236 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 128,
        // (add:{ *:[i16] } GR16:{ *:[i16] }:$src1, 128:{ *:[i16] })  =>  (SUB16ri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, -128:{ *:[i16] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB16ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_AddImm, /*InsnID*/0, /*Imm*/-128,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17236,
        GIR_Done,
      // Label 58: @297
      GIM_Try, /*On fail goto*//*Label 59*/ 321, // Rule ID 17425 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (add:{ *:[i16] } GR16:{ *:[i16] }:$src, 1:{ *:[i16] })  =>  (INC16r:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INC16r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17425,
        GIR_Done,
      // Label 59: @321
      GIM_Try, /*On fail goto*//*Label 60*/ 345, // Rule ID 17429 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (add:{ *:[i16] } GR16:{ *:[i16] }:$src, -1:{ *:[i16] })  =>  (DEC16r:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::DEC16r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17429,
        GIR_Done,
      // Label 60: @345
      GIM_Try, /*On fail goto*//*Label 61*/ 378, // Rule ID 17381 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i16immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)  =>  (ADD16ri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD16ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17381,
        GIR_Done,
      // Label 61: @378
      GIM_Try, /*On fail goto*//*Label 62*/ 408, // Rule ID 17379 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)  =>  (ADD16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17379,
        GIR_Done,
      // Label 62: @408
      GIM_Try, /*On fail goto*//*Label 63*/ 424, // Rule ID 17371 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR16RegClassID,
        // (add:{ *:[i16] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (ADD16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD16rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17371,
        GIR_Done,
      // Label 63: @424
      GIM_Reject,
    // Label 57: @425
    GIM_Reject,
    // Label 37: @426
    GIM_Try, /*On fail goto*//*Label 64*/ 597,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      GIM_Try, /*On fail goto*//*Label 65*/ 469, // Rule ID 17238 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 128,
        // (add:{ *:[i32] } GR32:{ *:[i32] }:$src1, 128:{ *:[i32] })  =>  (SUB32ri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, -128:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB32ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_AddImm, /*InsnID*/0, /*Imm*/-128,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17238,
        GIR_Done,
      // Label 65: @469
      GIM_Try, /*On fail goto*//*Label 66*/ 493, // Rule ID 17426 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] })  =>  (INC32r:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INC32r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17426,
        GIR_Done,
      // Label 66: @493
      GIM_Try, /*On fail goto*//*Label 67*/ 517, // Rule ID 17430 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] })  =>  (DEC32r:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::DEC32r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17430,
        GIR_Done,
      // Label 67: @517
      GIM_Try, /*On fail goto*//*Label 68*/ 550, // Rule ID 17382 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)  =>  (ADD32ri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD32ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17382,
        GIR_Done,
      // Label 68: @550
      GIM_Try, /*On fail goto*//*Label 69*/ 580, // Rule ID 17380 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)  =>  (ADD32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17380,
        GIR_Done,
      // Label 69: @580
      GIM_Try, /*On fail goto*//*Label 70*/ 596, // Rule ID 17372 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (add:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (ADD32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD32rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17372,
        GIR_Done,
      // Label 70: @596
      GIM_Reject,
    // Label 64: @597
    GIM_Reject,
    // Label 38: @598
    GIM_Try, /*On fail goto*//*Label 71*/ 797,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      GIM_Try, /*On fail goto*//*Label 72*/ 641, // Rule ID 17240 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 128,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src1, 128:{ *:[i64] })  =>  (SUB64ri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, -128:{ *:[i64] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB64ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_AddImm, /*InsnID*/0, /*Imm*/-128,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17240,
        GIR_Done,
      // Label 72: @641
      GIM_Try, /*On fail goto*//*Label 73*/ 666, // Rule ID 17242 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 2147483648,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src1, 2147483648:{ *:[i64] })  =>  (SUB64ri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, -2147483648:{ *:[i64] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB64ri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_AddImm, /*InsnID*/0, /*Imm*/-2147483648,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17242,
        GIR_Done,
      // Label 73: @666
      GIM_Try, /*On fail goto*//*Label 74*/ 690, // Rule ID 17427 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] })  =>  (INC64r:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INC64r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17427,
        GIR_Done,
      // Label 74: @690
      GIM_Try, /*On fail goto*//*Label 75*/ 714, // Rule ID 17431 //
        GIM_CheckFeatures, GIFBS_UseIncDec,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] })  =>  (DEC64r:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::DEC64r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17431,
        GIR_Done,
      // Label 75: @714
      GIM_Try, /*On fail goto*//*Label 76*/ 747, // Rule ID 17383 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)  =>  (ADD64ri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD64ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17383,
        GIR_Done,
      // Label 76: @747
      GIM_Try, /*On fail goto*//*Label 77*/ 780, // Rule ID 17384 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)  =>  (ADD64ri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD64ri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17384,
        GIR_Done,
      // Label 77: @780
      GIM_Try, /*On fail goto*//*Label 78*/ 796, // Rule ID 17373 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (add:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (ADD64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD64rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17373,
        GIR_Done,
      // Label 78: @796
      GIM_Reject,
    // Label 71: @797
    GIM_Reject,
    // Label 39: @798
    GIM_Try, /*On fail goto*//*Label 79*/ 878,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 80*/ 831, // Rule ID 1819 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPADDQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDQrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1819,
        GIR_Done,
      // Label 80: @831
      GIM_Try, /*On fail goto*//*Label 81*/ 854, // Rule ID 1821 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (PADDQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PADDQrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1821,
        GIR_Done,
      // Label 81: @854
      GIM_Try, /*On fail goto*//*Label 82*/ 877, // Rule ID 3833 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (add:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPADDQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3833,
        GIR_Done,
      // Label 82: @877
      GIM_Reject,
    // Label 79: @878
    GIM_Reject,
    // Label 40: @879
    GIM_Try, /*On fail goto*//*Label 83*/ 1081,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 84*/ 950, // Rule ID 12361 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_MUL,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[v4i32] } (mul:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2), VR128:{ *:[v4i32] }:$src3)  =>  (VPMACSDDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSDDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12361,
        GIR_Done,
      // Label 84: @950
      GIM_Try, /*On fail goto*//*Label 85*/ 1011, // Rule ID 19488 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_MUL,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src3, (mul:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2))  =>  (VPMACSDDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSDDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 19488,
        GIR_Done,
      // Label 85: @1011
      GIM_Try, /*On fail goto*//*Label 86*/ 1034, // Rule ID 1813 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPADDDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1813,
        GIR_Done,
      // Label 86: @1034
      GIM_Try, /*On fail goto*//*Label 87*/ 1057, // Rule ID 1815 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (PADDDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PADDDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1815,
        GIR_Done,
      // Label 87: @1057
      GIM_Try, /*On fail goto*//*Label 88*/ 1080, // Rule ID 3860 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (add:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPADDDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3860,
        GIR_Done,
      // Label 88: @1080
      GIM_Reject,
    // Label 83: @1081
    GIM_Reject,
    // Label 41: @1082
    GIM_Try, /*On fail goto*//*Label 89*/ 1139,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 90*/ 1115, // Rule ID 1823 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (add:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPADDQYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDQYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1823,
        GIR_Done,
      // Label 90: @1115
      GIM_Try, /*On fail goto*//*Label 91*/ 1138, // Rule ID 3824 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (add:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPADDQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3824,
        GIR_Done,
      // Label 91: @1138
      GIM_Reject,
    // Label 89: @1139
    GIM_Reject,
    // Label 42: @1140
    GIM_Try, /*On fail goto*//*Label 92*/ 1342,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_Try, /*On fail goto*//*Label 93*/ 1211, // Rule ID 12360 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_MUL,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[v8i16] } (mul:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2), VR128:{ *:[v8i16] }:$src3)  =>  (VPMACSWWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v8i16] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSWWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12360,
        GIR_Done,
      // Label 93: @1211
      GIM_Try, /*On fail goto*//*Label 94*/ 1272, // Rule ID 19487 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_MUL,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (add:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src3, (mul:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2))  =>  (VPMACSWWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v8i16] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSWWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 19487,
        GIR_Done,
      // Label 94: @1272
      GIM_Try, /*On fail goto*//*Label 95*/ 1295, // Rule ID 1807 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPADDWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDWrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1807,
        GIR_Done,
      // Label 95: @1295
      GIM_Try, /*On fail goto*//*Label 96*/ 1318, // Rule ID 1809 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PADDWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PADDWrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1809,
        GIR_Done,
      // Label 96: @1318
      GIM_Try, /*On fail goto*//*Label 97*/ 1341, // Rule ID 3881 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (add:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPADDWZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDWZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3881,
        GIR_Done,
      // Label 97: @1341
      GIM_Reject,
    // Label 92: @1342
    GIM_Reject,
    // Label 43: @1343
    GIM_Try, /*On fail goto*//*Label 98*/ 1400,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 99*/ 1376, // Rule ID 1817 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (add:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPADDDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1817,
        GIR_Done,
      // Label 99: @1376
      GIM_Try, /*On fail goto*//*Label 100*/ 1399, // Rule ID 3851 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (add:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPADDDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3851,
        GIR_Done,
      // Label 100: @1399
      GIM_Reject,
    // Label 98: @1400
    GIM_Reject,
    // Label 44: @1401
    GIM_Try, /*On fail goto*//*Label 101*/ 1432, // Rule ID 3815 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (add:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPADDQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3815,
      GIR_Done,
    // Label 101: @1432
    GIM_Reject,
    // Label 45: @1433
    GIM_Try, /*On fail goto*//*Label 102*/ 1513,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
      GIM_Try, /*On fail goto*//*Label 103*/ 1466, // Rule ID 1801 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (VPADDBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1801,
        GIR_Done,
      // Label 103: @1466
      GIM_Try, /*On fail goto*//*Label 104*/ 1489, // Rule ID 1803 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (add:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (PADDBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PADDBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1803,
        GIR_Done,
      // Label 104: @1489
      GIM_Try, /*On fail goto*//*Label 105*/ 1512, // Rule ID 3899 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (add:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1, VR128X:{ *:[v16i8] }:$src2)  =>  (VPADDBZ128rr:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1, VR128X:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDBZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3899,
        GIR_Done,
      // Label 105: @1512
      GIM_Reject,
    // Label 102: @1513
    GIM_Reject,
    // Label 46: @1514
    GIM_Try, /*On fail goto*//*Label 106*/ 1571,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_Try, /*On fail goto*//*Label 107*/ 1547, // Rule ID 1811 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (add:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPADDWYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDWYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1811,
        GIR_Done,
      // Label 107: @1547
      GIM_Try, /*On fail goto*//*Label 108*/ 1570, // Rule ID 3875 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (add:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPADDWZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDWZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3875,
        GIR_Done,
      // Label 108: @1570
      GIM_Reject,
    // Label 106: @1571
    GIM_Reject,
    // Label 47: @1572
    GIM_Try, /*On fail goto*//*Label 109*/ 1603, // Rule ID 3842 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (add:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPADDDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3842,
      GIR_Done,
    // Label 109: @1603
    GIM_Reject,
    // Label 48: @1604
    GIM_Try, /*On fail goto*//*Label 110*/ 1661,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s8,
      GIM_Try, /*On fail goto*//*Label 111*/ 1637, // Rule ID 1805 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (add:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)  =>  (VPADDBYrr:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDBYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1805,
        GIR_Done,
      // Label 111: @1637
      GIM_Try, /*On fail goto*//*Label 112*/ 1660, // Rule ID 3893 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (add:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1, VR256X:{ *:[v32i8] }:$src2)  =>  (VPADDBZ256rr:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1, VR256X:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDBZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3893,
        GIR_Done,
      // Label 112: @1660
      GIM_Reject,
    // Label 110: @1661
    GIM_Reject,
    // Label 49: @1662
    GIM_Try, /*On fail goto*//*Label 113*/ 1693, // Rule ID 3869 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (add:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPADDWZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3869,
      GIR_Done,
    // Label 113: @1693
    GIM_Reject,
    // Label 50: @1694
    GIM_Try, /*On fail goto*//*Label 114*/ 1725, // Rule ID 3887 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v64s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (add:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1, VR512:{ *:[v64i8] }:$src2)  =>  (VPADDBZrr:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1, VR512:{ *:[v64i8] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPADDBZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3887,
      GIR_Done,
    // Label 114: @1725
    GIM_Reject,
    // Label 51: @1726
    GIM_Reject,
    // Label 1: @1727
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 25, /*)*//*default:*//*Label 131*/ 2807,
    /*GILLT_s8*//*Label 115*/ 1757,
    /*GILLT_s16*//*Label 116*/ 1823,
    /*GILLT_s32*//*Label 117*/ 1922,
    /*GILLT_s64*//*Label 118*/ 2021, 0, 0, 0,
    /*GILLT_v2s64*//*Label 119*/ 2123, 0,
    /*GILLT_v4s32*//*Label 120*/ 2204,
    /*GILLT_v4s64*//*Label 121*/ 2285, 0,
    /*GILLT_v8s16*//*Label 122*/ 2343,
    /*GILLT_v8s32*//*Label 123*/ 2424,
    /*GILLT_v8s64*//*Label 124*/ 2482, 0,
    /*GILLT_v16s8*//*Label 125*/ 2514,
    /*GILLT_v16s16*//*Label 126*/ 2595,
    /*GILLT_v16s32*//*Label 127*/ 2653, 0,
    /*GILLT_v32s8*//*Label 128*/ 2685,
    /*GILLT_v32s16*//*Label 129*/ 2743, 0,
    /*GILLT_v64s8*//*Label 130*/ 2775,
    // Label 115: @1757
    GIM_Try, /*On fail goto*//*Label 132*/ 1822,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 133*/ 1805, // Rule ID 17393 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SUB8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17393,
        GIR_Done,
      // Label 133: @1805
      GIM_Try, /*On fail goto*//*Label 134*/ 1821, // Rule ID 17385 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR8RegClassID,
        // (sub:{ *:[i8] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)  =>  (SUB8rr:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB8rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17385,
        GIR_Done,
      // Label 134: @1821
      GIM_Reject,
    // Label 132: @1822
    GIM_Reject,
    // Label 116: @1823
    GIM_Try, /*On fail goto*//*Label 135*/ 1921,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 136*/ 1874, // Rule ID 17396 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i16immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)  =>  (SUB16ri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB16ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17396,
        GIR_Done,
      // Label 136: @1874
      GIM_Try, /*On fail goto*//*Label 137*/ 1904, // Rule ID 17394 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)  =>  (SUB16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17394,
        GIR_Done,
      // Label 137: @1904
      GIM_Try, /*On fail goto*//*Label 138*/ 1920, // Rule ID 17386 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR16RegClassID,
        // (sub:{ *:[i16] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (SUB16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB16rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17386,
        GIR_Done,
      // Label 138: @1920
      GIM_Reject,
    // Label 135: @1921
    GIM_Reject,
    // Label 117: @1922
    GIM_Try, /*On fail goto*//*Label 139*/ 2020,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      GIM_Try, /*On fail goto*//*Label 140*/ 1973, // Rule ID 17397 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)  =>  (SUB32ri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB32ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17397,
        GIR_Done,
      // Label 140: @1973
      GIM_Try, /*On fail goto*//*Label 141*/ 2003, // Rule ID 17395 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)  =>  (SUB32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17395,
        GIR_Done,
      // Label 141: @2003
      GIM_Try, /*On fail goto*//*Label 142*/ 2019, // Rule ID 17387 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (sub:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (SUB32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB32rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17387,
        GIR_Done,
      // Label 142: @2019
      GIM_Reject,
    // Label 139: @2020
    GIM_Reject,
    // Label 118: @2021
    GIM_Try, /*On fail goto*//*Label 143*/ 2122,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      GIM_Try, /*On fail goto*//*Label 144*/ 2072, // Rule ID 17398 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)  =>  (SUB64ri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB64ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17398,
        GIR_Done,
      // Label 144: @2072
      GIM_Try, /*On fail goto*//*Label 145*/ 2105, // Rule ID 17399 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sub:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)  =>  (SUB64ri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SUB64ri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17399,
        GIR_Done,
      // Label 145: @2105
      GIM_Try, /*On fail goto*//*Label 146*/ 2121, // Rule ID 17388 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (sub:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (SUB64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB64rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17388,
        GIR_Done,
      // Label 146: @2121
      GIM_Reject,
    // Label 143: @2122
    GIM_Reject,
    // Label 119: @2123
    GIM_Try, /*On fail goto*//*Label 147*/ 2203,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 148*/ 2156, // Rule ID 1885 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPSUBQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBQrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1885,
        GIR_Done,
      // Label 148: @2156
      GIM_Try, /*On fail goto*//*Label 149*/ 2179, // Rule ID 1887 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (PSUBQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PSUBQrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1887,
        GIR_Done,
      // Label 149: @2179
      GIM_Try, /*On fail goto*//*Label 150*/ 2202, // Rule ID 3923 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (sub:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPSUBQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3923,
        GIR_Done,
      // Label 150: @2202
      GIM_Reject,
    // Label 147: @2203
    GIM_Reject,
    // Label 120: @2204
    GIM_Try, /*On fail goto*//*Label 151*/ 2284,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 152*/ 2237, // Rule ID 1879 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPSUBDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1879,
        GIR_Done,
      // Label 152: @2237
      GIM_Try, /*On fail goto*//*Label 153*/ 2260, // Rule ID 1881 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (PSUBDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PSUBDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1881,
        GIR_Done,
      // Label 153: @2260
      GIM_Try, /*On fail goto*//*Label 154*/ 2283, // Rule ID 3950 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (sub:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPSUBDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3950,
        GIR_Done,
      // Label 154: @2283
      GIM_Reject,
    // Label 151: @2284
    GIM_Reject,
    // Label 121: @2285
    GIM_Try, /*On fail goto*//*Label 155*/ 2342,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 156*/ 2318, // Rule ID 1889 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (sub:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPSUBQYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBQYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1889,
        GIR_Done,
      // Label 156: @2318
      GIM_Try, /*On fail goto*//*Label 157*/ 2341, // Rule ID 3914 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (sub:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPSUBQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3914,
        GIR_Done,
      // Label 157: @2341
      GIM_Reject,
    // Label 155: @2342
    GIM_Reject,
    // Label 122: @2343
    GIM_Try, /*On fail goto*//*Label 158*/ 2423,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_Try, /*On fail goto*//*Label 159*/ 2376, // Rule ID 1873 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPSUBWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBWrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1873,
        GIR_Done,
      // Label 159: @2376
      GIM_Try, /*On fail goto*//*Label 160*/ 2399, // Rule ID 1875 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PSUBWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PSUBWrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1875,
        GIR_Done,
      // Label 160: @2399
      GIM_Try, /*On fail goto*//*Label 161*/ 2422, // Rule ID 3971 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (sub:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPSUBWZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBWZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3971,
        GIR_Done,
      // Label 161: @2422
      GIM_Reject,
    // Label 158: @2423
    GIM_Reject,
    // Label 123: @2424
    GIM_Try, /*On fail goto*//*Label 162*/ 2481,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 163*/ 2457, // Rule ID 1883 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (sub:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPSUBDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1883,
        GIR_Done,
      // Label 163: @2457
      GIM_Try, /*On fail goto*//*Label 164*/ 2480, // Rule ID 3941 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (sub:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPSUBDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3941,
        GIR_Done,
      // Label 164: @2480
      GIM_Reject,
    // Label 162: @2481
    GIM_Reject,
    // Label 124: @2482
    GIM_Try, /*On fail goto*//*Label 165*/ 2513, // Rule ID 3905 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sub:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPSUBQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3905,
      GIR_Done,
    // Label 165: @2513
    GIM_Reject,
    // Label 125: @2514
    GIM_Try, /*On fail goto*//*Label 166*/ 2594,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
      GIM_Try, /*On fail goto*//*Label 167*/ 2547, // Rule ID 1867 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (VPSUBBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1867,
        GIR_Done,
      // Label 167: @2547
      GIM_Try, /*On fail goto*//*Label 168*/ 2570, // Rule ID 1869 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sub:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (PSUBBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PSUBBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1869,
        GIR_Done,
      // Label 168: @2570
      GIM_Try, /*On fail goto*//*Label 169*/ 2593, // Rule ID 3989 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (sub:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1, VR128X:{ *:[v16i8] }:$src2)  =>  (VPSUBBZ128rr:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1, VR128X:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBBZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3989,
        GIR_Done,
      // Label 169: @2593
      GIM_Reject,
    // Label 166: @2594
    GIM_Reject,
    // Label 126: @2595
    GIM_Try, /*On fail goto*//*Label 170*/ 2652,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_Try, /*On fail goto*//*Label 171*/ 2628, // Rule ID 1877 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (sub:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPSUBWYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBWYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1877,
        GIR_Done,
      // Label 171: @2628
      GIM_Try, /*On fail goto*//*Label 172*/ 2651, // Rule ID 3965 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (sub:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPSUBWZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBWZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3965,
        GIR_Done,
      // Label 172: @2651
      GIM_Reject,
    // Label 170: @2652
    GIM_Reject,
    // Label 127: @2653
    GIM_Try, /*On fail goto*//*Label 173*/ 2684, // Rule ID 3932 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sub:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPSUBDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3932,
      GIR_Done,
    // Label 173: @2684
    GIM_Reject,
    // Label 128: @2685
    GIM_Try, /*On fail goto*//*Label 174*/ 2742,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s8,
      GIM_Try, /*On fail goto*//*Label 175*/ 2718, // Rule ID 1871 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (sub:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)  =>  (VPSUBBYrr:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBBYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1871,
        GIR_Done,
      // Label 175: @2718
      GIM_Try, /*On fail goto*//*Label 176*/ 2741, // Rule ID 3983 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (sub:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1, VR256X:{ *:[v32i8] }:$src2)  =>  (VPSUBBZ256rr:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1, VR256X:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBBZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3983,
        GIR_Done,
      // Label 176: @2741
      GIM_Reject,
    // Label 174: @2742
    GIM_Reject,
    // Label 129: @2743
    GIM_Try, /*On fail goto*//*Label 177*/ 2774, // Rule ID 3959 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sub:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPSUBWZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3959,
      GIR_Done,
    // Label 177: @2774
    GIM_Reject,
    // Label 130: @2775
    GIM_Try, /*On fail goto*//*Label 178*/ 2806, // Rule ID 3977 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v64s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sub:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1, VR512:{ *:[v64i8] }:$src2)  =>  (VPSUBBZrr:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1, VR512:{ *:[v64i8] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSUBBZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3977,
      GIR_Done,
    // Label 178: @2806
    GIM_Reject,
    // Label 131: @2807
    GIM_Reject,
    // Label 2: @2808
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/2, 23, /*)*//*default:*//*Label 191*/ 3573,
    /*GILLT_s16*//*Label 179*/ 2835,
    /*GILLT_s32*//*Label 180*/ 2934,
    /*GILLT_s64*//*Label 181*/ 3033, 0, 0, 0,
    /*GILLT_v2s64*//*Label 182*/ 3135, 0,
    /*GILLT_v4s32*//*Label 183*/ 3167,
    /*GILLT_v4s64*//*Label 184*/ 3248, 0,
    /*GILLT_v8s16*//*Label 185*/ 3280,
    /*GILLT_v8s32*//*Label 186*/ 3361,
    /*GILLT_v8s64*//*Label 187*/ 3419, 0, 0,
    /*GILLT_v16s16*//*Label 188*/ 3451,
    /*GILLT_v16s32*//*Label 189*/ 3509, 0, 0,
    /*GILLT_v32s16*//*Label 190*/ 3541,
    // Label 179: @2835
    GIM_Try, /*On fail goto*//*Label 192*/ 2933,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 193*/ 2886, // Rule ID 17414 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i16immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (mul:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)  =>  (IMUL16rri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::IMUL16rri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17414,
        GIR_Done,
      // Label 193: @2886
      GIM_Try, /*On fail goto*//*Label 194*/ 2916, // Rule ID 17412 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (mul:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)  =>  (IMUL16rri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::IMUL16rri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17412,
        GIR_Done,
      // Label 194: @2916
      GIM_Try, /*On fail goto*//*Label 195*/ 2932, // Rule ID 17406 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR16RegClassID,
        // (mul:{ *:[i16] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (IMUL16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::IMUL16rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17406,
        GIR_Done,
      // Label 195: @2932
      GIM_Reject,
    // Label 192: @2933
    GIM_Reject,
    // Label 180: @2934
    GIM_Try, /*On fail goto*//*Label 196*/ 3032,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      GIM_Try, /*On fail goto*//*Label 197*/ 2985, // Rule ID 17415 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (mul:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)  =>  (IMUL32rri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::IMUL32rri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17415,
        GIR_Done,
      // Label 197: @2985
      GIM_Try, /*On fail goto*//*Label 198*/ 3015, // Rule ID 17413 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (mul:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)  =>  (IMUL32rri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::IMUL32rri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17413,
        GIR_Done,
      // Label 198: @3015
      GIM_Try, /*On fail goto*//*Label 199*/ 3031, // Rule ID 17407 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (mul:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (IMUL32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::IMUL32rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17407,
        GIR_Done,
      // Label 199: @3031
      GIM_Reject,
    // Label 196: @3032
    GIM_Reject,
    // Label 181: @3033
    GIM_Try, /*On fail goto*//*Label 200*/ 3134,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      GIM_Try, /*On fail goto*//*Label 201*/ 3084, // Rule ID 17416 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (mul:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)  =>  (IMUL64rri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::IMUL64rri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17416,
        GIR_Done,
      // Label 201: @3084
      GIM_Try, /*On fail goto*//*Label 202*/ 3117, // Rule ID 17417 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (mul:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)  =>  (IMUL64rri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::IMUL64rri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17417,
        GIR_Done,
      // Label 202: @3117
      GIM_Try, /*On fail goto*//*Label 203*/ 3133, // Rule ID 17408 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (mul:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (IMUL64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::IMUL64rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17408,
        GIR_Done,
      // Label 203: @3133
      GIM_Reject,
    // Label 200: @3134
    GIM_Reject,
    // Label 182: @3135
    GIM_Try, /*On fail goto*//*Label 204*/ 3166, // Rule ID 4202 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
      // (mul:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPMULLQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4202,
      GIR_Done,
    // Label 204: @3166
    GIM_Reject,
    // Label 183: @3167
    GIM_Try, /*On fail goto*//*Label 205*/ 3247,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 206*/ 3200, // Rule ID 2507 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (mul:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPMULLDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2507,
        GIR_Done,
      // Label 206: @3200
      GIM_Try, /*On fail goto*//*Label 207*/ 3223, // Rule ID 2515 //
        GIM_CheckFeatures, GIFBS_UseSSE41,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (mul:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (PMULLDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PMULLDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2515,
        GIR_Done,
      // Label 207: @3223
      GIM_Try, /*On fail goto*//*Label 208*/ 3246, // Rule ID 4157 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (mul:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPMULLDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4157,
        GIR_Done,
      // Label 208: @3246
      GIM_Reject,
    // Label 205: @3247
    GIM_Reject,
    // Label 184: @3248
    GIM_Try, /*On fail goto*//*Label 209*/ 3279, // Rule ID 4193 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
      // (mul:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPMULLQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4193,
      GIR_Done,
    // Label 209: @3279
    GIM_Reject,
    // Label 185: @3280
    GIM_Try, /*On fail goto*//*Label 210*/ 3360,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_Try, /*On fail goto*//*Label 211*/ 3313, // Rule ID 1849 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (mul:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPMULLWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLWrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1849,
        GIR_Done,
      // Label 211: @3313
      GIM_Try, /*On fail goto*//*Label 212*/ 3336, // Rule ID 1851 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (mul:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PMULLWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PMULLWrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1851,
        GIR_Done,
      // Label 212: @3336
      GIM_Try, /*On fail goto*//*Label 213*/ 3359, // Rule ID 4178 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (mul:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPMULLWZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLWZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4178,
        GIR_Done,
      // Label 213: @3359
      GIM_Reject,
    // Label 210: @3360
    GIM_Reject,
    // Label 186: @3361
    GIM_Try, /*On fail goto*//*Label 214*/ 3418,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 215*/ 3394, // Rule ID 2511 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (mul:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPMULLDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2511,
        GIR_Done,
      // Label 215: @3394
      GIM_Try, /*On fail goto*//*Label 216*/ 3417, // Rule ID 4148 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (mul:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPMULLDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4148,
        GIR_Done,
      // Label 216: @3417
      GIM_Reject,
    // Label 214: @3418
    GIM_Reject,
    // Label 187: @3419
    GIM_Try, /*On fail goto*//*Label 217*/ 3450, // Rule ID 4184 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (mul:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPMULLQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4184,
      GIR_Done,
    // Label 217: @3450
    GIM_Reject,
    // Label 188: @3451
    GIM_Try, /*On fail goto*//*Label 218*/ 3508,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_Try, /*On fail goto*//*Label 219*/ 3484, // Rule ID 1853 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (mul:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPMULLWYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLWYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1853,
        GIR_Done,
      // Label 219: @3484
      GIM_Try, /*On fail goto*//*Label 220*/ 3507, // Rule ID 4172 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (mul:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPMULLWZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLWZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4172,
        GIR_Done,
      // Label 220: @3507
      GIM_Reject,
    // Label 218: @3508
    GIM_Reject,
    // Label 189: @3509
    GIM_Try, /*On fail goto*//*Label 221*/ 3540, // Rule ID 4139 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (mul:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPMULLDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4139,
      GIR_Done,
    // Label 221: @3540
    GIM_Reject,
    // Label 190: @3541
    GIM_Try, /*On fail goto*//*Label 222*/ 3572, // Rule ID 4166 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (mul:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPMULLWZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMULLWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4166,
      GIR_Done,
    // Label 222: @3572
    GIM_Reject,
    // Label 191: @3573
    GIM_Reject,
    // Label 3: @3574
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/0, 25, /*)*//*default:*//*Label 246*/ 7078,
    /*GILLT_s1*//*Label 223*/ 3605,
    /*GILLT_s8*//*Label 224*/ 3903,
    /*GILLT_s16*//*Label 225*/ 3969,
    /*GILLT_s32*//*Label 226*/ 4068,
    /*GILLT_s64*//*Label 227*/ 5039, 0, 0,
    /*GILLT_v2s1*//*Label 228*/ 5911,
    /*GILLT_v2s64*//*Label 229*/ 6004,
    /*GILLT_v4s1*//*Label 230*/ 6085,
    /*GILLT_v4s32*//*Label 231*/ 6178,
    /*GILLT_v4s64*//*Label 232*/ 6259,
    /*GILLT_v8s1*//*Label 233*/ 6340,
    /*GILLT_v8s16*//*Label 234*/ 6449,
    /*GILLT_v8s32*//*Label 235*/ 6530,
    /*GILLT_v8s64*//*Label 236*/ 6611,
    /*GILLT_v16s1*//*Label 237*/ 6643,
    /*GILLT_v16s8*//*Label 238*/ 6675,
    /*GILLT_v16s16*//*Label 239*/ 6756,
    /*GILLT_v16s32*//*Label 240*/ 6837,
    /*GILLT_v32s1*//*Label 241*/ 6869,
    /*GILLT_v32s8*//*Label 242*/ 6901,
    /*GILLT_v32s16*//*Label 243*/ 6982,
    /*GILLT_v64s1*//*Label 244*/ 7014,
    /*GILLT_v64s8*//*Label 245*/ 7046,
    // Label 223: @3605
    GIM_Try, /*On fail goto*//*Label 247*/ 3902,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_Try, /*On fail goto*//*Label 248*/ 3720, // Rule ID 13877 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s1,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s1,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK1RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[v1i1] } (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, -1:{ *:[v1i1] }), VK1:{ *:[v1i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KANDNWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KANDNWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
        // GIR_Coverage, 13877,
        GIR_Done,
      // Label 248: @3720
      GIM_Try, /*On fail goto*//*Label 249*/ 3821, // Rule ID 19632 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s1,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s1,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src2, (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, -1:{ *:[v1i1] }))  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KANDNWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/1, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KANDNWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
        // GIR_Coverage, 19632,
        GIR_Done,
      // Label 249: @3821
      GIM_Try, /*On fail goto*//*Label 250*/ 3901, // Rule ID 13873 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK1RegClassID,
        // (and:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, VK1:{ *:[v1i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KANDWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KANDWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
        // GIR_Coverage, 13873,
        GIR_Done,
      // Label 250: @3901
      GIM_Reject,
    // Label 247: @3902
    GIM_Reject,
    // Label 224: @3903
    GIM_Try, /*On fail goto*//*Label 251*/ 3968,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 252*/ 3951, // Rule ID 17470 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (AND8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17470,
        GIR_Done,
      // Label 252: @3951
      GIM_Try, /*On fail goto*//*Label 253*/ 3967, // Rule ID 17462 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR8RegClassID,
        // (and:{ *:[i8] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)  =>  (AND8rr:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::AND8rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17462,
        GIR_Done,
      // Label 253: @3967
      GIM_Reject,
    // Label 251: @3968
    GIM_Reject,
    // Label 225: @3969
    GIM_Try, /*On fail goto*//*Label 254*/ 4067,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 255*/ 4020, // Rule ID 17473 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i16immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)  =>  (AND16ri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND16ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17473,
        GIR_Done,
      // Label 255: @4020
      GIM_Try, /*On fail goto*//*Label 256*/ 4050, // Rule ID 17471 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)  =>  (AND16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17471,
        GIR_Done,
      // Label 256: @4050
      GIM_Try, /*On fail goto*//*Label 257*/ 4066, // Rule ID 17463 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR16RegClassID,
        // (and:{ *:[i16] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (AND16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::AND16rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17463,
        GIR_Done,
      // Label 257: @4066
      GIM_Reject,
    // Label 254: @4067
    GIM_Reject,
    // Label 226: @4068
    GIM_Try, /*On fail goto*//*Label 258*/ 5038,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 259*/ 4149, // Rule ID 19441 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (TZMSK32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::TZMSK32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 19441,
        GIR_Done,
      // Label 259: @4149
      GIM_Try, /*On fail goto*//*Label 260*/ 4224, // Rule ID 20690 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (BLCIC32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCIC32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 20690,
        GIR_Done,
      // Label 260: @4224
      GIM_Try, /*On fail goto*//*Label 261*/ 4299, // Rule ID 12236 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i32] } (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (TZMSK32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::TZMSK32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12236,
        GIR_Done,
      // Label 261: @4299
      GIM_Try, /*On fail goto*//*Label 262*/ 4374, // Rule ID 17693 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i32] } (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }))  =>  (BLCIC32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCIC32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17693,
        GIR_Done,
      // Label 262: @4374
      GIM_Try, /*On fail goto*//*Label 263*/ 4428, // Rule ID 20678 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLSR32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSR32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 20678,
        GIR_Done,
      // Label 263: @4428
      GIM_Try, /*On fail goto*//*Label 264*/ 4482, // Rule ID 20684 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLCFILL32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCFILL32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 20684,
        GIR_Done,
      // Label 264: @4482
      GIM_Try, /*On fail goto*//*Label 265*/ 4536, // Rule ID 20682 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, 0,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/2,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } (sub:{ *:[i32] } 0:{ *:[i32] }, GR32:{ *:[i32] }:$src), GR32:{ *:[i32] }:$src)  =>  (BLSI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 20682,
        GIR_Done,
      // Label 265: @4536
      GIM_Try, /*On fail goto*//*Label 266*/ 4590, // Rule ID 17675 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src, (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (BLSR32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSR32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17675,
        GIR_Done,
      // Label 266: @4590
      GIM_Try, /*On fail goto*//*Label 267*/ 4644, // Rule ID 17687 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src, (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }))  =>  (BLCFILL32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCFILL32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17687,
        GIR_Done,
      // Label 267: @4644
      GIM_Try, /*On fail goto*//*Label 268*/ 4698, // Rule ID 17679 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, 0,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/2, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src, (sub:{ *:[i32] } 0:{ *:[i32] }, GR32:{ *:[i32] }:$src))  =>  (BLSI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17679,
        GIR_Done,
      // Label 268: @4698
      GIM_Try, /*On fail goto*//*Label 269*/ 4749, // Rule ID 17246 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 65535,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src1, 65535:{ *:[i32] })  =>  (MOVZX32rr16:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i16] } GR32:{ *:[i32] }:$src1, sub_16bit:{ *:[i32] }))
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s16,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_CopySubReg, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, /*SubRegIdx*/4, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOVZX32rr16,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17246,
        GIR_Done,
      // Label 269: @4749
      GIM_Try, /*On fail goto*//*Label 270*/ 4800, // Rule ID 17247 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 255,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src1, 255:{ *:[i32] })  =>  (MOVZX32rr8:{ *:[i32] } (EXTRACT_SUBREG:{ *:[i8] } GR32:{ *:[i32] }:$src1, sub_8bit:{ *:[i32] }))
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s8,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_CopySubReg, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, /*SubRegIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOVZX32rr8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17247,
        GIR_Done,
      // Label 270: @4800
      GIM_Try, /*On fail goto*//*Label 271*/ 4845, // Rule ID 17474 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)  =>  (AND32ri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND32ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17474,
        GIR_Done,
      // Label 271: @4845
      GIM_Try, /*On fail goto*//*Label 272*/ 4887, // Rule ID 17472 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)  =>  (AND32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17472,
        GIR_Done,
      // Label 272: @4887
      GIM_Try, /*On fail goto*//*Label 273*/ 4948, // Rule ID 12246 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } (xor:{ *:[i32] } GR32:{ *:[i32] }:$src1, -1:{ *:[i32] }), GR32:{ *:[i32] }:$src2)  =>  (ANDN32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ANDN32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12246,
        GIR_Done,
      // Label 273: @4948
      GIM_Try, /*On fail goto*//*Label 274*/ 5009, // Rule ID 19443 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src2, (xor:{ *:[i32] } GR32:{ *:[i32] }:$src1, -1:{ *:[i32] }))  =>  (ANDN32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ANDN32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 19443,
        GIR_Done,
      // Label 274: @5009
      GIM_Try, /*On fail goto*//*Label 275*/ 5037, // Rule ID 17464 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (and:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (AND32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::AND32rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17464,
        GIR_Done,
      // Label 275: @5037
      GIM_Reject,
    // Label 258: @5038
    GIM_Reject,
    // Label 227: @5039
    GIM_Try, /*On fail goto*//*Label 276*/ 5910,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 277*/ 5120, // Rule ID 19442 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (TZMSK64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::TZMSK64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 19442,
        GIR_Done,
      // Label 277: @5120
      GIM_Try, /*On fail goto*//*Label 278*/ 5195, // Rule ID 20691 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (BLCIC64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCIC64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 20691,
        GIR_Done,
      // Label 278: @5195
      GIM_Try, /*On fail goto*//*Label 279*/ 5270, // Rule ID 12237 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i64] } (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (TZMSK64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::TZMSK64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12237,
        GIR_Done,
      // Label 279: @5270
      GIM_Try, /*On fail goto*//*Label 280*/ 5345, // Rule ID 17694 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (and:{ *:[i64] } (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }))  =>  (BLCIC64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCIC64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17694,
        GIR_Done,
      // Label 280: @5345
      GIM_Try, /*On fail goto*//*Label 281*/ 5399, // Rule ID 20679 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLSR64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSR64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 20679,
        GIR_Done,
      // Label 281: @5399
      GIM_Try, /*On fail goto*//*Label 282*/ 5453, // Rule ID 20685 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLCFILL64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCFILL64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 20685,
        GIR_Done,
      // Label 282: @5453
      GIM_Try, /*On fail goto*//*Label 283*/ 5507, // Rule ID 20683 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, 0,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/2,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } (sub:{ *:[i64] } 0:{ *:[i64] }, GR64:{ *:[i64] }:$src), GR64:{ *:[i64] }:$src)  =>  (BLSI64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSI64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 20683,
        GIR_Done,
      // Label 283: @5507
      GIM_Try, /*On fail goto*//*Label 284*/ 5561, // Rule ID 17676 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src, (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (BLSR64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSR64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17676,
        GIR_Done,
      // Label 284: @5561
      GIM_Try, /*On fail goto*//*Label 285*/ 5615, // Rule ID 17688 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src, (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }))  =>  (BLCFILL64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCFILL64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17688,
        GIR_Done,
      // Label 285: @5615
      GIM_Try, /*On fail goto*//*Label 286*/ 5669, // Rule ID 17680 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, 0,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/2, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src, (sub:{ *:[i64] } 0:{ *:[i64] }, GR64:{ *:[i64] }:$src))  =>  (BLSI64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSI64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17680,
        GIR_Done,
      // Label 286: @5669
      GIM_Try, /*On fail goto*//*Label 287*/ 5714, // Rule ID 17475 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)  =>  (AND64ri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND64ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17475,
        GIR_Done,
      // Label 287: @5714
      GIM_Try, /*On fail goto*//*Label 288*/ 5759, // Rule ID 17476 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)  =>  (AND64ri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AND64ri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17476,
        GIR_Done,
      // Label 288: @5759
      GIM_Try, /*On fail goto*//*Label 289*/ 5820, // Rule ID 12247 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } (xor:{ *:[i64] } GR64:{ *:[i64] }:$src1, -1:{ *:[i64] }), GR64:{ *:[i64] }:$src2)  =>  (ANDN64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ANDN64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12247,
        GIR_Done,
      // Label 289: @5820
      GIM_Try, /*On fail goto*//*Label 290*/ 5881, // Rule ID 19444 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src2, (xor:{ *:[i64] } GR64:{ *:[i64] }:$src1, -1:{ *:[i64] }))  =>  (ANDN64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ANDN64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 19444,
        GIR_Done,
      // Label 290: @5881
      GIM_Try, /*On fail goto*//*Label 291*/ 5909, // Rule ID 17465 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (and:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (AND64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::AND64rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17465,
        GIR_Done,
      // Label 291: @5909
      GIM_Reject,
    // Label 276: @5910
    GIM_Reject,
    // Label 228: @5911
    GIM_Try, /*On fail goto*//*Label 292*/ 6003, // Rule ID 13874 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK2RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK2RegClassID,
      // (and:{ *:[v2i1] } VK2:{ *:[v2i1] }:$src1, VK2:{ *:[v2i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v2i1] } (KANDWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK2:{ *:[v2i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK2:{ *:[v2i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KANDWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
      // GIR_Coverage, 13874,
      GIR_Done,
    // Label 292: @6003
    GIM_Reject,
    // Label 229: @6004
    GIM_Try, /*On fail goto*//*Label 293*/ 6084,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 294*/ 6037, // Rule ID 1541 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (and:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPANDrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1541,
        GIR_Done,
      // Label 294: @6037
      GIM_Try, /*On fail goto*//*Label 295*/ 6060, // Rule ID 1543 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (and:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (PANDrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PANDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1543,
        GIR_Done,
      // Label 295: @6060
      GIM_Try, /*On fail goto*//*Label 296*/ 6083, // Rule ID 4886 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (and:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPANDQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4886,
        GIR_Done,
      // Label 296: @6083
      GIM_Reject,
    // Label 293: @6084
    GIM_Reject,
    // Label 230: @6085
    GIM_Try, /*On fail goto*//*Label 297*/ 6177, // Rule ID 13875 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK4RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK4RegClassID,
      // (and:{ *:[v4i1] } VK4:{ *:[v4i1] }:$src1, VK4:{ *:[v4i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v4i1] } (KANDWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK4:{ *:[v4i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK4:{ *:[v4i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KANDWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
      // GIR_Coverage, 13875,
      GIR_Done,
    // Label 297: @6177
    GIM_Reject,
    // Label 231: @6178
    GIM_Try, /*On fail goto*//*Label 298*/ 6258,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 299*/ 6211, // Rule ID 4913 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (and:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPANDDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4913,
        GIR_Done,
      // Label 299: @6211
      GIM_Try, /*On fail goto*//*Label 300*/ 6234, // Rule ID 12608 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (and:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPANDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12608,
        GIR_Done,
      // Label 300: @6234
      GIM_Try, /*On fail goto*//*Label 301*/ 6257, // Rule ID 12632 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (and:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (PANDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PANDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12632,
        GIR_Done,
      // Label 301: @6257
      GIM_Reject,
    // Label 298: @6258
    GIM_Reject,
    // Label 232: @6259
    GIM_Try, /*On fail goto*//*Label 302*/ 6339,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 303*/ 6292, // Rule ID 1545 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (and:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPANDYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1545,
        GIR_Done,
      // Label 303: @6292
      GIM_Try, /*On fail goto*//*Label 304*/ 6315, // Rule ID 4877 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (and:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPANDQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4877,
        GIR_Done,
      // Label 304: @6315
      GIM_Try, /*On fail goto*//*Label 305*/ 6338, // Rule ID 12561 //
        GIM_CheckFeatures, GIFBS_HasAVX1Only,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (and:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VANDPSYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VANDPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12561,
        GIR_Done,
      // Label 305: @6338
      GIM_Reject,
    // Label 302: @6339
    GIM_Reject,
    // Label 233: @6340
    GIM_Try, /*On fail goto*//*Label 306*/ 6448,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK8RegClassID,
      GIM_Try, /*On fail goto*//*Label 307*/ 6373, // Rule ID 3601 //
        GIM_CheckFeatures, GIFBS_HasDQI,
        // (and:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)  =>  (KANDBrr:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KANDBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3601,
        GIR_Done,
      // Label 307: @6373
      GIM_Try, /*On fail goto*//*Label 308*/ 6447, // Rule ID 13872 //
        GIM_CheckFeatures, GIFBS_NoDQI,
        // (and:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v8i1] } (KANDWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK8:{ *:[v8i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK8:{ *:[v8i1] }:$src2, VK16:{ *:[i32] })), VK8:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KANDWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK8*/12,
        // GIR_Coverage, 13872,
        GIR_Done,
      // Label 308: @6447
      GIM_Reject,
    // Label 306: @6448
    GIM_Reject,
    // Label 234: @6449
    GIM_Try, /*On fail goto*//*Label 309*/ 6529,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_Try, /*On fail goto*//*Label 310*/ 6482, // Rule ID 12607 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (and:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPANDrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12607,
        GIR_Done,
      // Label 310: @6482
      GIM_Try, /*On fail goto*//*Label 311*/ 6505, // Rule ID 12631 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (and:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PANDrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PANDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12631,
        GIR_Done,
      // Label 311: @6505
      GIM_Try, /*On fail goto*//*Label 312*/ 6528, // Rule ID 14391 //
        GIM_CheckFeatures, GIFBS_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (and:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPANDQZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14391,
        GIR_Done,
      // Label 312: @6528
      GIM_Reject,
    // Label 309: @6529
    GIM_Reject,
    // Label 235: @6530
    GIM_Try, /*On fail goto*//*Label 313*/ 6610,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 314*/ 6563, // Rule ID 4904 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (and:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPANDDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4904,
        GIR_Done,
      // Label 314: @6563
      GIM_Try, /*On fail goto*//*Label 315*/ 6586, // Rule ID 12536 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (and:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPANDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12536,
        GIR_Done,
      // Label 315: @6586
      GIM_Try, /*On fail goto*//*Label 316*/ 6609, // Rule ID 12560 //
        GIM_CheckFeatures, GIFBS_HasAVX1Only,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (and:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VANDPSYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VANDPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12560,
        GIR_Done,
      // Label 316: @6609
      GIM_Reject,
    // Label 313: @6610
    GIM_Reject,
    // Label 236: @6611
    GIM_Try, /*On fail goto*//*Label 317*/ 6642, // Rule ID 4868 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (and:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPANDQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4868,
      GIR_Done,
    // Label 317: @6642
    GIM_Reject,
    // Label 237: @6643
    GIM_Try, /*On fail goto*//*Label 318*/ 6674, // Rule ID 3602 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK16RegClassID,
      // (and:{ *:[v16i1] } VK16:{ *:[v16i1] }:$src1, VK16:{ *:[v16i1] }:$src2)  =>  (KANDWrr:{ *:[v16i1] } VK16:{ *:[v16i1] }:$src1, VK16:{ *:[v16i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KANDWrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3602,
      GIR_Done,
    // Label 318: @6674
    GIM_Reject,
    // Label 238: @6675
    GIM_Try, /*On fail goto*//*Label 319*/ 6755,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
      GIM_Try, /*On fail goto*//*Label 320*/ 6708, // Rule ID 12606 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (and:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (VPANDrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12606,
        GIR_Done,
      // Label 320: @6708
      GIM_Try, /*On fail goto*//*Label 321*/ 6731, // Rule ID 12630 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (and:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (PANDrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PANDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12630,
        GIR_Done,
      // Label 321: @6731
      GIM_Try, /*On fail goto*//*Label 322*/ 6754, // Rule ID 14390 //
        GIM_CheckFeatures, GIFBS_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (and:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1, VR128X:{ *:[v16i8] }:$src2)  =>  (VPANDQZ128rr:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1, VR128X:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14390,
        GIR_Done,
      // Label 322: @6754
      GIM_Reject,
    // Label 319: @6755
    GIM_Reject,
    // Label 239: @6756
    GIM_Try, /*On fail goto*//*Label 323*/ 6836,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_Try, /*On fail goto*//*Label 324*/ 6789, // Rule ID 12535 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (and:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPANDYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12535,
        GIR_Done,
      // Label 324: @6789
      GIM_Try, /*On fail goto*//*Label 325*/ 6812, // Rule ID 12559 //
        GIM_CheckFeatures, GIFBS_HasAVX1Only,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (and:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VANDPSYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VANDPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12559,
        GIR_Done,
      // Label 325: @6812
      GIM_Try, /*On fail goto*//*Label 326*/ 6835, // Rule ID 14415 //
        GIM_CheckFeatures, GIFBS_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (and:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPANDQZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14415,
        GIR_Done,
      // Label 326: @6835
      GIM_Reject,
    // Label 323: @6836
    GIM_Reject,
    // Label 240: @6837
    GIM_Try, /*On fail goto*//*Label 327*/ 6868, // Rule ID 4895 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (and:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPANDDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4895,
      GIR_Done,
    // Label 327: @6868
    GIM_Reject,
    // Label 241: @6869
    GIM_Try, /*On fail goto*//*Label 328*/ 6900, // Rule ID 3603 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK32RegClassID,
      // (and:{ *:[v32i1] } VK32:{ *:[v32i1] }:$src1, VK32:{ *:[v32i1] }:$src2)  =>  (KANDDrr:{ *:[v32i1] } VK32:{ *:[v32i1] }:$src1, VK32:{ *:[v32i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KANDDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3603,
      GIR_Done,
    // Label 328: @6900
    GIM_Reject,
    // Label 242: @6901
    GIM_Try, /*On fail goto*//*Label 329*/ 6981,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s8,
      GIM_Try, /*On fail goto*//*Label 330*/ 6934, // Rule ID 12534 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (and:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)  =>  (VPANDYrr:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12534,
        GIR_Done,
      // Label 330: @6934
      GIM_Try, /*On fail goto*//*Label 331*/ 6957, // Rule ID 12558 //
        GIM_CheckFeatures, GIFBS_HasAVX1Only,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (and:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)  =>  (VANDPSYrr:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VANDPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12558,
        GIR_Done,
      // Label 331: @6957
      GIM_Try, /*On fail goto*//*Label 332*/ 6980, // Rule ID 14414 //
        GIM_CheckFeatures, GIFBS_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (and:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1, VR256X:{ *:[v32i8] }:$src2)  =>  (VPANDQZ256rr:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1, VR256X:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14414,
        GIR_Done,
      // Label 332: @6980
      GIM_Reject,
    // Label 329: @6981
    GIM_Reject,
    // Label 243: @6982
    GIM_Try, /*On fail goto*//*Label 333*/ 7013, // Rule ID 14439 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (and:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPANDQZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14439,
      GIR_Done,
    // Label 333: @7013
    GIM_Reject,
    // Label 244: @7014
    GIM_Try, /*On fail goto*//*Label 334*/ 7045, // Rule ID 3604 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v64s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK64RegClassID,
      // (and:{ *:[v64i1] } VK64:{ *:[v64i1] }:$src1, VK64:{ *:[v64i1] }:$src2)  =>  (KANDQrr:{ *:[v64i1] } VK64:{ *:[v64i1] }:$src1, VK64:{ *:[v64i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KANDQrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3604,
      GIR_Done,
    // Label 334: @7045
    GIM_Reject,
    // Label 245: @7046
    GIM_Try, /*On fail goto*//*Label 335*/ 7077, // Rule ID 14438 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v64s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (and:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1, VR512:{ *:[v64i8] }:$src2)  =>  (VPANDQZrr:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1, VR512:{ *:[v64i8] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPANDQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14438,
      GIR_Done,
    // Label 335: @7077
    GIM_Reject,
    // Label 246: @7078
    GIM_Reject,
    // Label 4: @7079
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/0, 25, /*)*//*default:*//*Label 359*/ 10332,
    /*GILLT_s1*//*Label 336*/ 7110,
    /*GILLT_s8*//*Label 337*/ 7203,
    /*GILLT_s16*//*Label 338*/ 7269,
    /*GILLT_s32*//*Label 339*/ 7368,
    /*GILLT_s64*//*Label 340*/ 8265, 0, 0,
    /*GILLT_v2s1*//*Label 341*/ 9165,
    /*GILLT_v2s64*//*Label 342*/ 9258,
    /*GILLT_v4s1*//*Label 343*/ 9339,
    /*GILLT_v4s32*//*Label 344*/ 9432,
    /*GILLT_v4s64*//*Label 345*/ 9513,
    /*GILLT_v8s1*//*Label 346*/ 9594,
    /*GILLT_v8s16*//*Label 347*/ 9703,
    /*GILLT_v8s32*//*Label 348*/ 9784,
    /*GILLT_v8s64*//*Label 349*/ 9865,
    /*GILLT_v16s1*//*Label 350*/ 9897,
    /*GILLT_v16s8*//*Label 351*/ 9929,
    /*GILLT_v16s16*//*Label 352*/ 10010,
    /*GILLT_v16s32*//*Label 353*/ 10091,
    /*GILLT_v32s1*//*Label 354*/ 10123,
    /*GILLT_v32s8*//*Label 355*/ 10155,
    /*GILLT_v32s16*//*Label 356*/ 10236,
    /*GILLT_v64s1*//*Label 357*/ 10268,
    /*GILLT_v64s8*//*Label 358*/ 10300,
    // Label 336: @7110
    GIM_Try, /*On fail goto*//*Label 360*/ 7202, // Rule ID 13881 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK1RegClassID,
      // (or:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, VK1:{ *:[v1i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KORWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
      // GIR_Coverage, 13881,
      GIR_Done,
    // Label 360: @7202
    GIM_Reject,
    // Label 337: @7203
    GIM_Try, /*On fail goto*//*Label 361*/ 7268,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 362*/ 7251, // Rule ID 17440 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (OR8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17440,
        GIR_Done,
      // Label 362: @7251
      GIM_Try, /*On fail goto*//*Label 363*/ 7267, // Rule ID 17432 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR8RegClassID,
        // (or:{ *:[i8] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)  =>  (OR8rr:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::OR8rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17432,
        GIR_Done,
      // Label 363: @7267
      GIM_Reject,
    // Label 361: @7268
    GIM_Reject,
    // Label 338: @7269
    GIM_Try, /*On fail goto*//*Label 364*/ 7367,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 365*/ 7320, // Rule ID 17443 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i16immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)  =>  (OR16ri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR16ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17443,
        GIR_Done,
      // Label 365: @7320
      GIM_Try, /*On fail goto*//*Label 366*/ 7350, // Rule ID 17441 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)  =>  (OR16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17441,
        GIR_Done,
      // Label 366: @7350
      GIM_Try, /*On fail goto*//*Label 367*/ 7366, // Rule ID 17433 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR16RegClassID,
        // (or:{ *:[i16] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (OR16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::OR16rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17433,
        GIR_Done,
      // Label 367: @7366
      GIM_Reject,
    // Label 364: @7367
    GIM_Reject,
    // Label 339: @7368
    GIM_Try, /*On fail goto*//*Label 368*/ 8264,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 369*/ 7449, // Rule ID 19437 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (BLSIC32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSIC32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 19437,
        GIR_Done,
      // Label 369: @7449
      GIM_Try, /*On fail goto*//*Label 370*/ 7524, // Rule ID 19439 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (T1MSKC32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::T1MSKC32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 19439,
        GIR_Done,
      // Label 370: @7524
      GIM_Try, /*On fail goto*//*Label 371*/ 7599, // Rule ID 20686 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/2, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/2, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i32] } (xor:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), -1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLCI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 20686,
        GIR_Done,
      // Label 371: @7599
      GIM_Try, /*On fail goto*//*Label 372*/ 7674, // Rule ID 12232 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i32] } (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (BLSIC32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSIC32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12232,
        GIR_Done,
      // Label 372: @7674
      GIM_Try, /*On fail goto*//*Label 373*/ 7749, // Rule ID 12234 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i32] } (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }))  =>  (T1MSKC32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::T1MSKC32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12234,
        GIR_Done,
      // Label 373: @7749
      GIM_Try, /*On fail goto*//*Label 374*/ 7824, // Rule ID 17689 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src, (xor:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), -1:{ *:[i32] }))  =>  (BLCI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17689,
        GIR_Done,
      // Label 374: @7824
      GIM_Try, /*On fail goto*//*Label 375*/ 7878, // Rule ID 19433 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLCS32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCS32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 19433,
        GIR_Done,
      // Label 375: @7878
      GIM_Try, /*On fail goto*//*Label 376*/ 7932, // Rule ID 19435 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLSFILL32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSFILL32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 19435,
        GIR_Done,
      // Label 376: @7932
      GIM_Try, /*On fail goto*//*Label 377*/ 7986, // Rule ID 20688 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, -2,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/2,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } (sub:{ *:[i32] } -2:{ *:[i32] }, GR32:{ *:[i32] }:$src), GR32:{ *:[i32] }:$src)  =>  (BLCI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 20688,
        GIR_Done,
      // Label 377: @7986
      GIM_Try, /*On fail goto*//*Label 378*/ 8040, // Rule ID 12228 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src, (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }))  =>  (BLCS32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCS32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12228,
        GIR_Done,
      // Label 378: @8040
      GIM_Try, /*On fail goto*//*Label 379*/ 8094, // Rule ID 12230 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src, (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (BLSFILL32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSFILL32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12230,
        GIR_Done,
      // Label 379: @8094
      GIM_Try, /*On fail goto*//*Label 380*/ 8148, // Rule ID 17691 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, -2,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/2, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src, (sub:{ *:[i32] } -2:{ *:[i32] }, GR32:{ *:[i32] }:$src))  =>  (BLCI32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17691,
        GIR_Done,
      // Label 380: @8148
      GIM_Try, /*On fail goto*//*Label 381*/ 8193, // Rule ID 17444 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)  =>  (OR32ri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR32ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17444,
        GIR_Done,
      // Label 381: @8193
      GIM_Try, /*On fail goto*//*Label 382*/ 8235, // Rule ID 17442 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)  =>  (OR32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17442,
        GIR_Done,
      // Label 382: @8235
      GIM_Try, /*On fail goto*//*Label 383*/ 8263, // Rule ID 17434 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (or:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (OR32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::OR32rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17434,
        GIR_Done,
      // Label 383: @8263
      GIM_Reject,
    // Label 368: @8264
    GIM_Reject,
    // Label 340: @8265
    GIM_Try, /*On fail goto*//*Label 384*/ 9164,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 385*/ 8346, // Rule ID 19438 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (BLSIC64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSIC64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 19438,
        GIR_Done,
      // Label 385: @8346
      GIM_Try, /*On fail goto*//*Label 386*/ 8421, // Rule ID 19440 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (T1MSKC64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::T1MSKC64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 19440,
        GIR_Done,
      // Label 386: @8421
      GIM_Try, /*On fail goto*//*Label 387*/ 8496, // Rule ID 20687 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/2, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/2, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i64] } (xor:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), -1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLCI64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/2, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 20687,
        GIR_Done,
      // Label 387: @8496
      GIM_Try, /*On fail goto*//*Label 388*/ 8571, // Rule ID 12233 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i64] } (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (BLSIC64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSIC64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12233,
        GIR_Done,
      // Label 388: @8571
      GIM_Try, /*On fail goto*//*Label 389*/ 8646, // Rule ID 12235 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/0, /*OpIdx*/2, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i64] } (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }))  =>  (T1MSKC64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::T1MSKC64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12235,
        GIR_Done,
      // Label 389: @8646
      GIM_Try, /*On fail goto*//*Label 390*/ 8721, // Rule ID 17690 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_RecordInsn, /*DefineMI*/2, /*MI*/1, /*OpIdx*/1, // MIs[2]
        GIM_CheckOpcode, /*MI*/2, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/2, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[2] src
        GIM_CheckIsSameOperand, /*MI*/2, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/2, /*Op*/2, 1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/2,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src, (xor:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), -1:{ *:[i64] }))  =>  (BLCI64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17690,
        GIR_Done,
      // Label 390: @8721
      GIM_Try, /*On fail goto*//*Label 391*/ 8775, // Rule ID 19434 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLCS64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCS64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 19434,
        GIR_Done,
      // Label 391: @8775
      GIM_Try, /*On fail goto*//*Label 392*/ 8829, // Rule ID 19436 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLSFILL64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSFILL64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 19436,
        GIR_Done,
      // Label 392: @8829
      GIM_Try, /*On fail goto*//*Label 393*/ 8883, // Rule ID 20689 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, -2,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/2,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } (sub:{ *:[i64] } -2:{ *:[i64] }, GR64:{ *:[i64] }:$src), GR64:{ *:[i64] }:$src)  =>  (BLCI64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 20689,
        GIR_Done,
      // Label 393: @8883
      GIM_Try, /*On fail goto*//*Label 394*/ 8937, // Rule ID 12229 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src, (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }))  =>  (BLCS64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCS64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12229,
        GIR_Done,
      // Label 394: @8937
      GIM_Try, /*On fail goto*//*Label 395*/ 8991, // Rule ID 12231 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src, (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (BLSFILL64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSFILL64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12231,
        GIR_Done,
      // Label 395: @8991
      GIM_Try, /*On fail goto*//*Label 396*/ 9045, // Rule ID 17692 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_SUB,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/1, -2,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/2, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src, (sub:{ *:[i64] } -2:{ *:[i64] }, GR64:{ *:[i64] }:$src))  =>  (BLCI64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCI64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17692,
        GIR_Done,
      // Label 396: @9045
      GIM_Try, /*On fail goto*//*Label 397*/ 9090, // Rule ID 17445 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)  =>  (OR64ri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR64ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17445,
        GIR_Done,
      // Label 397: @9090
      GIM_Try, /*On fail goto*//*Label 398*/ 9135, // Rule ID 17446 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)  =>  (OR64ri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::OR64ri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17446,
        GIR_Done,
      // Label 398: @9135
      GIM_Try, /*On fail goto*//*Label 399*/ 9163, // Rule ID 17435 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (or:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (OR64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::OR64rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17435,
        GIR_Done,
      // Label 399: @9163
      GIM_Reject,
    // Label 384: @9164
    GIM_Reject,
    // Label 341: @9165
    GIM_Try, /*On fail goto*//*Label 400*/ 9257, // Rule ID 13882 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK2RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK2RegClassID,
      // (or:{ *:[v2i1] } VK2:{ *:[v2i1] }:$src1, VK2:{ *:[v2i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v2i1] } (KORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK2:{ *:[v2i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK2:{ *:[v2i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KORWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
      // GIR_Coverage, 13882,
      GIR_Done,
    // Label 400: @9257
    GIM_Reject,
    // Label 342: @9258
    GIM_Try, /*On fail goto*//*Label 401*/ 9338,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 402*/ 9291, // Rule ID 1547 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (or:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPORrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1547,
        GIR_Done,
      // Label 402: @9291
      GIM_Try, /*On fail goto*//*Label 403*/ 9314, // Rule ID 1549 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (or:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (PORrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1549,
        GIR_Done,
      // Label 403: @9314
      GIM_Try, /*On fail goto*//*Label 404*/ 9337, // Rule ID 4940 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (or:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPORQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4940,
        GIR_Done,
      // Label 404: @9337
      GIM_Reject,
    // Label 401: @9338
    GIM_Reject,
    // Label 343: @9339
    GIM_Try, /*On fail goto*//*Label 405*/ 9431, // Rule ID 13883 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK4RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK4RegClassID,
      // (or:{ *:[v4i1] } VK4:{ *:[v4i1] }:$src1, VK4:{ *:[v4i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v4i1] } (KORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK4:{ *:[v4i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK4:{ *:[v4i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KORWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
      // GIR_Coverage, 13883,
      GIR_Done,
    // Label 405: @9431
    GIM_Reject,
    // Label 344: @9432
    GIM_Try, /*On fail goto*//*Label 406*/ 9512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 407*/ 9465, // Rule ID 4967 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (or:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPORDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4967,
        GIR_Done,
      // Label 407: @9465
      GIM_Try, /*On fail goto*//*Label 408*/ 9488, // Rule ID 12611 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (or:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPORrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12611,
        GIR_Done,
      // Label 408: @9488
      GIM_Try, /*On fail goto*//*Label 409*/ 9511, // Rule ID 12635 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (or:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (PORrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12635,
        GIR_Done,
      // Label 409: @9511
      GIM_Reject,
    // Label 406: @9512
    GIM_Reject,
    // Label 345: @9513
    GIM_Try, /*On fail goto*//*Label 410*/ 9593,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 411*/ 9546, // Rule ID 1551 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (or:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPORYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1551,
        GIR_Done,
      // Label 411: @9546
      GIM_Try, /*On fail goto*//*Label 412*/ 9569, // Rule ID 4931 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (or:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPORQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4931,
        GIR_Done,
      // Label 412: @9569
      GIM_Try, /*On fail goto*//*Label 413*/ 9592, // Rule ID 12565 //
        GIM_CheckFeatures, GIFBS_HasAVX1Only,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (or:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VORPSYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VORPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12565,
        GIR_Done,
      // Label 413: @9592
      GIM_Reject,
    // Label 410: @9593
    GIM_Reject,
    // Label 346: @9594
    GIM_Try, /*On fail goto*//*Label 414*/ 9702,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK8RegClassID,
      GIM_Try, /*On fail goto*//*Label 415*/ 9627, // Rule ID 3605 //
        GIM_CheckFeatures, GIFBS_HasDQI,
        // (or:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)  =>  (KORBrr:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KORBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3605,
        GIR_Done,
      // Label 415: @9627
      GIM_Try, /*On fail goto*//*Label 416*/ 9701, // Rule ID 13880 //
        GIM_CheckFeatures, GIFBS_NoDQI,
        // (or:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v8i1] } (KORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK8:{ *:[v8i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK8:{ *:[v8i1] }:$src2, VK16:{ *:[i32] })), VK8:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KORWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK8*/12,
        // GIR_Coverage, 13880,
        GIR_Done,
      // Label 416: @9701
      GIM_Reject,
    // Label 414: @9702
    GIM_Reject,
    // Label 347: @9703
    GIM_Try, /*On fail goto*//*Label 417*/ 9783,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_Try, /*On fail goto*//*Label 418*/ 9736, // Rule ID 12610 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (or:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPORrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12610,
        GIR_Done,
      // Label 418: @9736
      GIM_Try, /*On fail goto*//*Label 419*/ 9759, // Rule ID 12634 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (or:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PORrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12634,
        GIR_Done,
      // Label 419: @9759
      GIM_Try, /*On fail goto*//*Label 420*/ 9782, // Rule ID 14393 //
        GIM_CheckFeatures, GIFBS_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (or:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPORQZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14393,
        GIR_Done,
      // Label 420: @9782
      GIM_Reject,
    // Label 417: @9783
    GIM_Reject,
    // Label 348: @9784
    GIM_Try, /*On fail goto*//*Label 421*/ 9864,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 422*/ 9817, // Rule ID 4958 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (or:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPORDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4958,
        GIR_Done,
      // Label 422: @9817
      GIM_Try, /*On fail goto*//*Label 423*/ 9840, // Rule ID 12539 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (or:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPORYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12539,
        GIR_Done,
      // Label 423: @9840
      GIM_Try, /*On fail goto*//*Label 424*/ 9863, // Rule ID 12564 //
        GIM_CheckFeatures, GIFBS_HasAVX1Only,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (or:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VORPSYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VORPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12564,
        GIR_Done,
      // Label 424: @9863
      GIM_Reject,
    // Label 421: @9864
    GIM_Reject,
    // Label 349: @9865
    GIM_Try, /*On fail goto*//*Label 425*/ 9896, // Rule ID 4922 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (or:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPORQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4922,
      GIR_Done,
    // Label 425: @9896
    GIM_Reject,
    // Label 350: @9897
    GIM_Try, /*On fail goto*//*Label 426*/ 9928, // Rule ID 3606 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK16RegClassID,
      // (or:{ *:[v16i1] } VK16:{ *:[v16i1] }:$src1, VK16:{ *:[v16i1] }:$src2)  =>  (KORWrr:{ *:[v16i1] } VK16:{ *:[v16i1] }:$src1, VK16:{ *:[v16i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KORWrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3606,
      GIR_Done,
    // Label 426: @9928
    GIM_Reject,
    // Label 351: @9929
    GIM_Try, /*On fail goto*//*Label 427*/ 10009,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
      GIM_Try, /*On fail goto*//*Label 428*/ 9962, // Rule ID 12609 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (or:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (VPORrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12609,
        GIR_Done,
      // Label 428: @9962
      GIM_Try, /*On fail goto*//*Label 429*/ 9985, // Rule ID 12633 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (or:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (PORrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12633,
        GIR_Done,
      // Label 429: @9985
      GIM_Try, /*On fail goto*//*Label 430*/ 10008, // Rule ID 14392 //
        GIM_CheckFeatures, GIFBS_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (or:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1, VR128X:{ *:[v16i8] }:$src2)  =>  (VPORQZ128rr:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1, VR128X:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14392,
        GIR_Done,
      // Label 430: @10008
      GIM_Reject,
    // Label 427: @10009
    GIM_Reject,
    // Label 352: @10010
    GIM_Try, /*On fail goto*//*Label 431*/ 10090,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_Try, /*On fail goto*//*Label 432*/ 10043, // Rule ID 12538 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (or:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPORYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12538,
        GIR_Done,
      // Label 432: @10043
      GIM_Try, /*On fail goto*//*Label 433*/ 10066, // Rule ID 12563 //
        GIM_CheckFeatures, GIFBS_HasAVX1Only,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (or:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VORPSYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VORPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12563,
        GIR_Done,
      // Label 433: @10066
      GIM_Try, /*On fail goto*//*Label 434*/ 10089, // Rule ID 14417 //
        GIM_CheckFeatures, GIFBS_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (or:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPORQZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14417,
        GIR_Done,
      // Label 434: @10089
      GIM_Reject,
    // Label 431: @10090
    GIM_Reject,
    // Label 353: @10091
    GIM_Try, /*On fail goto*//*Label 435*/ 10122, // Rule ID 4949 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (or:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPORDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4949,
      GIR_Done,
    // Label 435: @10122
    GIM_Reject,
    // Label 354: @10123
    GIM_Try, /*On fail goto*//*Label 436*/ 10154, // Rule ID 3607 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK32RegClassID,
      // (or:{ *:[v32i1] } VK32:{ *:[v32i1] }:$src1, VK32:{ *:[v32i1] }:$src2)  =>  (KORDrr:{ *:[v32i1] } VK32:{ *:[v32i1] }:$src1, VK32:{ *:[v32i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KORDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3607,
      GIR_Done,
    // Label 436: @10154
    GIM_Reject,
    // Label 355: @10155
    GIM_Try, /*On fail goto*//*Label 437*/ 10235,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s8,
      GIM_Try, /*On fail goto*//*Label 438*/ 10188, // Rule ID 12537 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (or:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)  =>  (VPORYrr:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12537,
        GIR_Done,
      // Label 438: @10188
      GIM_Try, /*On fail goto*//*Label 439*/ 10211, // Rule ID 12562 //
        GIM_CheckFeatures, GIFBS_HasAVX1Only,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (or:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)  =>  (VORPSYrr:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VORPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12562,
        GIR_Done,
      // Label 439: @10211
      GIM_Try, /*On fail goto*//*Label 440*/ 10234, // Rule ID 14416 //
        GIM_CheckFeatures, GIFBS_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (or:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1, VR256X:{ *:[v32i8] }:$src2)  =>  (VPORQZ256rr:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1, VR256X:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14416,
        GIR_Done,
      // Label 440: @10234
      GIM_Reject,
    // Label 437: @10235
    GIM_Reject,
    // Label 356: @10236
    GIM_Try, /*On fail goto*//*Label 441*/ 10267, // Rule ID 14441 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (or:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPORQZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14441,
      GIR_Done,
    // Label 441: @10267
    GIM_Reject,
    // Label 357: @10268
    GIM_Try, /*On fail goto*//*Label 442*/ 10299, // Rule ID 3608 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v64s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK64RegClassID,
      // (or:{ *:[v64i1] } VK64:{ *:[v64i1] }:$src1, VK64:{ *:[v64i1] }:$src2)  =>  (KORQrr:{ *:[v64i1] } VK64:{ *:[v64i1] }:$src1, VK64:{ *:[v64i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KORQrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3608,
      GIR_Done,
    // Label 442: @10299
    GIM_Reject,
    // Label 358: @10300
    GIM_Try, /*On fail goto*//*Label 443*/ 10331, // Rule ID 14440 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v64s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (or:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1, VR512:{ *:[v64i8] }:$src2)  =>  (VPORQZrr:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1, VR512:{ *:[v64i8] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPORQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14440,
      GIR_Done,
    // Label 443: @10331
    GIM_Reject,
    // Label 359: @10332
    GIM_Reject,
    // Label 5: @10333
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/0, 25, /*)*//*default:*//*Label 467*/ 12888,
    /*GILLT_s1*//*Label 444*/ 10364,
    /*GILLT_s8*//*Label 445*/ 10763,
    /*GILLT_s16*//*Label 446*/ 10851,
    /*GILLT_s32*//*Label 447*/ 10972,
    /*GILLT_s64*//*Label 448*/ 11345, 0, 0,
    /*GILLT_v2s1*//*Label 449*/ 11721,
    /*GILLT_v2s64*//*Label 450*/ 11814,
    /*GILLT_v4s1*//*Label 451*/ 11895,
    /*GILLT_v4s32*//*Label 452*/ 11988,
    /*GILLT_v4s64*//*Label 453*/ 12069,
    /*GILLT_v8s1*//*Label 454*/ 12150,
    /*GILLT_v8s16*//*Label 455*/ 12259,
    /*GILLT_v8s32*//*Label 456*/ 12340,
    /*GILLT_v8s64*//*Label 457*/ 12421,
    /*GILLT_v16s1*//*Label 458*/ 12453,
    /*GILLT_v16s8*//*Label 459*/ 12485,
    /*GILLT_v16s16*//*Label 460*/ 12566,
    /*GILLT_v16s32*//*Label 461*/ 12647,
    /*GILLT_v32s1*//*Label 462*/ 12679,
    /*GILLT_v32s8*//*Label 463*/ 12711,
    /*GILLT_v32s16*//*Label 464*/ 12792,
    /*GILLT_v64s1*//*Label 465*/ 12824,
    /*GILLT_v64s8*//*Label 466*/ 12856,
    // Label 444: @10364
    GIM_Try, /*On fail goto*//*Label 468*/ 10762,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_Try, /*On fail goto*//*Label 469*/ 10479, // Rule ID 19644 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s1,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s1,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK1RegClassID,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[v1i1] } (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, -1:{ *:[v1i1] }), VK1:{ *:[v1i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KXNORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXNORWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
        // GIR_Coverage, 19644,
        GIR_Done,
      // Label 469: @10479
      GIM_Try, /*On fail goto*//*Label 470*/ 10580, // Rule ID 13885 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s1,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s1,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/2, /*RC*/X86::VK1RegClassID,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[v1i1] } (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, VK1:{ *:[v1i1] }:$src2), -1:{ *:[v1i1] })  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KXNORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/1, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXNORWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
        // GIR_Coverage, 13885,
        GIR_Done,
      // Label 470: @10580
      GIM_Try, /*On fail goto*//*Label 471*/ 10681, // Rule ID 19645 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_XOR,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s1,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s1,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src2, (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, -1:{ *:[v1i1] }))  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KXNORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/1, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/1, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXNORWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
        // GIR_Coverage, 19645,
        GIR_Done,
      // Label 471: @10681
      GIM_Try, /*On fail goto*//*Label 472*/ 10761, // Rule ID 13889 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK1RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK1RegClassID,
        // (xor:{ *:[v1i1] } VK1:{ *:[v1i1] }:$src1, VK1:{ *:[v1i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v1i1] } (KXORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK1:{ *:[v1i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXORWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
        // GIR_Coverage, 13889,
        GIR_Done,
      // Label 472: @10761
      GIM_Reject,
    // Label 468: @10762
    GIM_Reject,
    // Label 445: @10763
    GIM_Try, /*On fail goto*//*Label 473*/ 10850,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 474*/ 10803, // Rule ID 152 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (xor:{ *:[i8] } GR8:{ *:[i8] }:$src1, -1:{ *:[i8] })  =>  (NOT8r:{ *:[i8] } GR8:{ *:[i8] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::NOT8r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 152,
        GIR_Done,
      // Label 474: @10803
      GIM_Try, /*On fail goto*//*Label 475*/ 10833, // Rule ID 17455 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (XOR8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17455,
        GIR_Done,
      // Label 475: @10833
      GIM_Try, /*On fail goto*//*Label 476*/ 10849, // Rule ID 17447 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR8RegClassID,
        // (xor:{ *:[i8] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)  =>  (XOR8rr:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::XOR8rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17447,
        GIR_Done,
      // Label 476: @10849
      GIM_Reject,
    // Label 473: @10850
    GIM_Reject,
    // Label 446: @10851
    GIM_Try, /*On fail goto*//*Label 477*/ 10971,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 478*/ 10891, // Rule ID 153 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (xor:{ *:[i16] } GR16:{ *:[i16] }:$src1, -1:{ *:[i16] })  =>  (NOT16r:{ *:[i16] } GR16:{ *:[i16] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::NOT16r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 153,
        GIR_Done,
      // Label 478: @10891
      GIM_Try, /*On fail goto*//*Label 479*/ 10924, // Rule ID 17458 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i16immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)  =>  (XOR16ri8:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] })<<P:Predicate_i16immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR16ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17458,
        GIR_Done,
      // Label 479: @10924
      GIM_Try, /*On fail goto*//*Label 480*/ 10954, // Rule ID 17456 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)  =>  (XOR16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i16] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17456,
        GIR_Done,
      // Label 480: @10954
      GIM_Try, /*On fail goto*//*Label 481*/ 10970, // Rule ID 17448 //
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR16RegClassID,
        // (xor:{ *:[i16] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (XOR16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::XOR16rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17448,
        GIR_Done,
      // Label 481: @10970
      GIM_Reject,
    // Label 477: @10971
    GIM_Reject,
    // Label 447: @10972
    GIM_Try, /*On fail goto*//*Label 482*/ 11344,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 483*/ 11032, // Rule ID 19431 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLCMSK32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCMSK32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 19431,
        GIR_Done,
      // Label 483: @11032
      GIM_Try, /*On fail goto*//*Label 484*/ 11086, // Rule ID 20680 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i32] } (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }), GR32:{ *:[i32] }:$src)  =>  (BLSMSK32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSMSK32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 20680,
        GIR_Done,
      // Label 484: @11086
      GIM_Try, /*On fail goto*//*Label 485*/ 11140, // Rule ID 12226 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, (add:{ *:[i32] } GR32:{ *:[i32] }:$src, 1:{ *:[i32] }))  =>  (BLCMSK32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCMSK32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12226,
        GIR_Done,
      // Label 485: @11140
      GIM_Try, /*On fail goto*//*Label 486*/ 11194, // Rule ID 17677 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s32,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i32] } GR32:{ *:[i32] }:$src, (add:{ *:[i32] } GR32:{ *:[i32] }:$src, -1:{ *:[i32] }))  =>  (BLSMSK32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSMSK32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17677,
        GIR_Done,
      // Label 486: @11194
      GIM_Try, /*On fail goto*//*Label 487*/ 11228, // Rule ID 154 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (xor:{ *:[i32] } GR32:{ *:[i32] }:$src1, -1:{ *:[i32] })  =>  (NOT32r:{ *:[i32] } GR32:{ *:[i32] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::NOT32r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 154,
        GIR_Done,
      // Label 487: @11228
      GIM_Try, /*On fail goto*//*Label 488*/ 11273, // Rule ID 17459 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)  =>  (XOR32ri8:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR32ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17459,
        GIR_Done,
      // Label 488: @11273
      GIM_Try, /*On fail goto*//*Label 489*/ 11315, // Rule ID 17457 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)  =>  (XOR32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17457,
        GIR_Done,
      // Label 489: @11315
      GIM_Try, /*On fail goto*//*Label 490*/ 11343, // Rule ID 17449 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (xor:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (XOR32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::XOR32rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17449,
        GIR_Done,
      // Label 490: @11343
      GIM_Reject,
    // Label 482: @11344
    GIM_Reject,
    // Label 448: @11345
    GIM_Try, /*On fail goto*//*Label 491*/ 11720,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 492*/ 11405, // Rule ID 19432 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLCMSK64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCMSK64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 19432,
        GIR_Done,
      // Label 492: @11405
      GIM_Try, /*On fail goto*//*Label 493*/ 11459, // Rule ID 20681 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        // MIs[0] src
        GIM_CheckIsSameOperand, /*MI*/0, /*OpIdx*/2, /*OtherMI*/1, /*OtherOpIdx*/1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i64] } (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }), GR64:{ *:[i64] }:$src)  =>  (BLSMSK64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSMSK64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 20681,
        GIR_Done,
      // Label 493: @11459
      GIM_Try, /*On fail goto*//*Label 494*/ 11513, // Rule ID 12227 //
        GIM_CheckFeatures, GIFBS_HasTBM,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, 1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, (add:{ *:[i64] } GR64:{ *:[i64] }:$src, 1:{ *:[i64] }))  =>  (BLCMSK64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLCMSK64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12227,
        GIR_Done,
      // Label 494: @11513
      GIM_Try, /*On fail goto*//*Label 495*/ 11567, // Rule ID 17678 //
        GIM_CheckFeatures, GIFBS_HasBMI,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_ADD,
        GIM_CheckType, /*MI*/1, /*Op*/2, /*Type*/GILLT_s64,
        // MIs[1] src
        GIM_CheckIsSameOperand, /*MI*/1, /*OpIdx*/1, /*OtherMI*/0, /*OtherOpIdx*/1,
        GIM_CheckConstantInt, /*MI*/1, /*Op*/2, -1,
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i64] } GR64:{ *:[i64] }:$src, (add:{ *:[i64] } GR64:{ *:[i64] }:$src, -1:{ *:[i64] }))  =>  (BLSMSK64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::BLSMSK64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17678,
        GIR_Done,
      // Label 495: @11567
      GIM_Try, /*On fail goto*//*Label 496*/ 11601, // Rule ID 155 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, -1,
        // (xor:{ *:[i64] } GR64:{ *:[i64] }:$src1, -1:{ *:[i64] })  =>  (NOT64r:{ *:[i64] } GR64:{ *:[i64] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::NOT64r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 155,
        GIR_Done,
      // Label 496: @11601
      GIM_Try, /*On fail goto*//*Label 497*/ 11646, // Rule ID 17460 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)  =>  (XOR64ri8:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR64ri8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17460,
        GIR_Done,
      // Label 497: @11646
      GIM_Try, /*On fail goto*//*Label 498*/ 11691, // Rule ID 17461 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        GIM_CheckI64ImmPredicate, /*MI*/1, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (xor:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)  =>  (XOR64ri32:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XOR64ri32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17461,
        GIR_Done,
      // Label 498: @11691
      GIM_Try, /*On fail goto*//*Label 499*/ 11719, // Rule ID 17450 //
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (xor:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (XOR64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::XOR64rr,
        GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17450,
        GIR_Done,
      // Label 499: @11719
      GIM_Reject,
    // Label 491: @11720
    GIM_Reject,
    // Label 449: @11721
    GIM_Try, /*On fail goto*//*Label 500*/ 11813, // Rule ID 13890 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK2RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK2RegClassID,
      // (xor:{ *:[v2i1] } VK2:{ *:[v2i1] }:$src1, VK2:{ *:[v2i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v2i1] } (KXORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK2:{ *:[v2i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK2:{ *:[v2i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXORWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
      // GIR_Coverage, 13890,
      GIR_Done,
    // Label 500: @11813
    GIM_Reject,
    // Label 450: @11814
    GIM_Try, /*On fail goto*//*Label 501*/ 11894,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 502*/ 11847, // Rule ID 1553 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (xor:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPXORrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1553,
        GIR_Done,
      // Label 502: @11847
      GIM_Try, /*On fail goto*//*Label 503*/ 11870, // Rule ID 1555 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (xor:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (PXORrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PXORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1555,
        GIR_Done,
      // Label 503: @11870
      GIM_Try, /*On fail goto*//*Label 504*/ 11893, // Rule ID 4994 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (xor:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPXORQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4994,
        GIR_Done,
      // Label 504: @11893
      GIM_Reject,
    // Label 501: @11894
    GIM_Reject,
    // Label 451: @11895
    GIM_Try, /*On fail goto*//*Label 505*/ 11987, // Rule ID 13891 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK1RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK4RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK4RegClassID,
      // (xor:{ *:[v4i1] } VK4:{ *:[v4i1] }:$src1, VK4:{ *:[v4i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v4i1] } (KXORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK4:{ *:[v4i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK4:{ *:[v4i1] }:$src2, VK16:{ *:[i32] })), VK1:{ *:[i32] })
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
      GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
      GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
      GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
      GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
      GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
      GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXORWrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK1*/8,
      // GIR_Coverage, 13891,
      GIR_Done,
    // Label 505: @11987
    GIM_Reject,
    // Label 452: @11988
    GIM_Try, /*On fail goto*//*Label 506*/ 12068,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 507*/ 12021, // Rule ID 5021 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (xor:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPXORDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5021,
        GIR_Done,
      // Label 507: @12021
      GIM_Try, /*On fail goto*//*Label 508*/ 12044, // Rule ID 12614 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (xor:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPXORrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12614,
        GIR_Done,
      // Label 508: @12044
      GIM_Try, /*On fail goto*//*Label 509*/ 12067, // Rule ID 12638 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (xor:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (PXORrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PXORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12638,
        GIR_Done,
      // Label 509: @12067
      GIM_Reject,
    // Label 506: @12068
    GIM_Reject,
    // Label 453: @12069
    GIM_Try, /*On fail goto*//*Label 510*/ 12149,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 511*/ 12102, // Rule ID 1557 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (xor:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPXORYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1557,
        GIR_Done,
      // Label 511: @12102
      GIM_Try, /*On fail goto*//*Label 512*/ 12125, // Rule ID 4985 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (xor:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPXORQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4985,
        GIR_Done,
      // Label 512: @12125
      GIM_Try, /*On fail goto*//*Label 513*/ 12148, // Rule ID 12569 //
        GIM_CheckFeatures, GIFBS_HasAVX1Only,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (xor:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VXORPSYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VXORPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12569,
        GIR_Done,
      // Label 513: @12148
      GIM_Reject,
    // Label 510: @12149
    GIM_Reject,
    // Label 454: @12150
    GIM_Try, /*On fail goto*//*Label 514*/ 12258,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK8RegClassID,
      GIM_Try, /*On fail goto*//*Label 515*/ 12183, // Rule ID 3613 //
        GIM_CheckFeatures, GIFBS_HasDQI,
        // (xor:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)  =>  (KXORBrr:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KXORBrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3613,
        GIR_Done,
      // Label 515: @12183
      GIM_Try, /*On fail goto*//*Label 516*/ 12257, // Rule ID 13888 //
        GIM_CheckFeatures, GIFBS_NoDQI,
        // (xor:{ *:[v8i1] } VK8:{ *:[v8i1] }:$src1, VK8:{ *:[v8i1] }:$src2)  =>  (COPY_TO_REGCLASS:{ *:[v8i1] } (KXORWrr:{ *:[v16i1] } (COPY_TO_REGCLASS:{ *:[v16i1] } VK8:{ *:[v8i1] }:$src1, VK16:{ *:[i32] }), (COPY_TO_REGCLASS:{ *:[v16i1] } VK8:{ *:[v8i1] }:$src2, VK16:{ *:[i32] })), VK8:{ *:[i32] })
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/1, /*TypeID*/GILLT_v16s1,
        GIR_MakeTempReg, /*TempRegID*/2, /*TypeID*/GILLT_v16s1,
        GIR_BuildMI, /*InsnID*/3, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/3, /*TempRegID*/2, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/3, /*OldInsnID*/0, /*OpIdx*/2, // src2
        GIR_ConstrainSelectedInstOperands, /*InsnID*/3,
        GIR_BuildMI, /*InsnID*/2, /*Opcode*/TargetOpcode::COPY,
        GIR_AddTempRegister, /*InsnID*/2, /*TempRegID*/1, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/2, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_ConstrainSelectedInstOperands, /*InsnID*/2,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::KXORWrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/1, /*TempRegFlags*/0,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/2, /*TempRegFlags*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK8*/12,
        // GIR_Coverage, 13888,
        GIR_Done,
      // Label 516: @12257
      GIM_Reject,
    // Label 514: @12258
    GIM_Reject,
    // Label 455: @12259
    GIM_Try, /*On fail goto*//*Label 517*/ 12339,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_Try, /*On fail goto*//*Label 518*/ 12292, // Rule ID 12613 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (xor:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPXORrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12613,
        GIR_Done,
      // Label 518: @12292
      GIM_Try, /*On fail goto*//*Label 519*/ 12315, // Rule ID 12637 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (xor:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PXORrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PXORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12637,
        GIR_Done,
      // Label 519: @12315
      GIM_Try, /*On fail goto*//*Label 520*/ 12338, // Rule ID 14395 //
        GIM_CheckFeatures, GIFBS_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (xor:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPXORQZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14395,
        GIR_Done,
      // Label 520: @12338
      GIM_Reject,
    // Label 517: @12339
    GIM_Reject,
    // Label 456: @12340
    GIM_Try, /*On fail goto*//*Label 521*/ 12420,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 522*/ 12373, // Rule ID 5012 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (xor:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPXORDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5012,
        GIR_Done,
      // Label 522: @12373
      GIM_Try, /*On fail goto*//*Label 523*/ 12396, // Rule ID 12542 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (xor:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPXORYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12542,
        GIR_Done,
      // Label 523: @12396
      GIM_Try, /*On fail goto*//*Label 524*/ 12419, // Rule ID 12568 //
        GIM_CheckFeatures, GIFBS_HasAVX1Only,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (xor:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VXORPSYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VXORPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12568,
        GIR_Done,
      // Label 524: @12419
      GIM_Reject,
    // Label 521: @12420
    GIM_Reject,
    // Label 457: @12421
    GIM_Try, /*On fail goto*//*Label 525*/ 12452, // Rule ID 4976 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (xor:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPXORQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 4976,
      GIR_Done,
    // Label 525: @12452
    GIM_Reject,
    // Label 458: @12453
    GIM_Try, /*On fail goto*//*Label 526*/ 12484, // Rule ID 3614 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK16RegClassID,
      // (xor:{ *:[v16i1] } VK16:{ *:[v16i1] }:$src1, VK16:{ *:[v16i1] }:$src2)  =>  (KXORWrr:{ *:[v16i1] } VK16:{ *:[v16i1] }:$src1, VK16:{ *:[v16i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KXORWrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3614,
      GIR_Done,
    // Label 526: @12484
    GIM_Reject,
    // Label 459: @12485
    GIM_Try, /*On fail goto*//*Label 527*/ 12565,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
      GIM_Try, /*On fail goto*//*Label 528*/ 12518, // Rule ID 12612 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (xor:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (VPXORrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12612,
        GIR_Done,
      // Label 528: @12518
      GIM_Try, /*On fail goto*//*Label 529*/ 12541, // Rule ID 12636 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (xor:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (PXORrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::PXORrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12636,
        GIR_Done,
      // Label 529: @12541
      GIM_Try, /*On fail goto*//*Label 530*/ 12564, // Rule ID 14394 //
        GIM_CheckFeatures, GIFBS_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (xor:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1, VR128X:{ *:[v16i8] }:$src2)  =>  (VPXORQZ128rr:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1, VR128X:{ *:[v16i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14394,
        GIR_Done,
      // Label 530: @12564
      GIM_Reject,
    // Label 527: @12565
    GIM_Reject,
    // Label 460: @12566
    GIM_Try, /*On fail goto*//*Label 531*/ 12646,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_Try, /*On fail goto*//*Label 532*/ 12599, // Rule ID 12541 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (xor:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPXORYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12541,
        GIR_Done,
      // Label 532: @12599
      GIM_Try, /*On fail goto*//*Label 533*/ 12622, // Rule ID 12567 //
        GIM_CheckFeatures, GIFBS_HasAVX1Only,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (xor:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VXORPSYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VXORPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12567,
        GIR_Done,
      // Label 533: @12622
      GIM_Try, /*On fail goto*//*Label 534*/ 12645, // Rule ID 14419 //
        GIM_CheckFeatures, GIFBS_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (xor:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPXORQZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14419,
        GIR_Done,
      // Label 534: @12645
      GIM_Reject,
    // Label 531: @12646
    GIM_Reject,
    // Label 461: @12647
    GIM_Try, /*On fail goto*//*Label 535*/ 12678, // Rule ID 5003 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (xor:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPXORDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5003,
      GIR_Done,
    // Label 535: @12678
    GIM_Reject,
    // Label 462: @12679
    GIM_Try, /*On fail goto*//*Label 536*/ 12710, // Rule ID 3615 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK32RegClassID,
      // (xor:{ *:[v32i1] } VK32:{ *:[v32i1] }:$src1, VK32:{ *:[v32i1] }:$src2)  =>  (KXORDrr:{ *:[v32i1] } VK32:{ *:[v32i1] }:$src1, VK32:{ *:[v32i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KXORDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3615,
      GIR_Done,
    // Label 536: @12710
    GIM_Reject,
    // Label 463: @12711
    GIM_Try, /*On fail goto*//*Label 537*/ 12791,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s8,
      GIM_Try, /*On fail goto*//*Label 538*/ 12744, // Rule ID 12540 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (xor:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)  =>  (VPXORYrr:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12540,
        GIR_Done,
      // Label 538: @12744
      GIM_Try, /*On fail goto*//*Label 539*/ 12767, // Rule ID 12566 //
        GIM_CheckFeatures, GIFBS_HasAVX1Only,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (xor:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)  =>  (VXORPSYrr:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VXORPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12566,
        GIR_Done,
      // Label 539: @12767
      GIM_Try, /*On fail goto*//*Label 540*/ 12790, // Rule ID 14418 //
        GIM_CheckFeatures, GIFBS_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (xor:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1, VR256X:{ *:[v32i8] }:$src2)  =>  (VPXORQZ256rr:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1, VR256X:{ *:[v32i8] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 14418,
        GIR_Done,
      // Label 540: @12790
      GIM_Reject,
    // Label 537: @12791
    GIM_Reject,
    // Label 464: @12792
    GIM_Try, /*On fail goto*//*Label 541*/ 12823, // Rule ID 14443 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (xor:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPXORQZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14443,
      GIR_Done,
    // Label 541: @12823
    GIM_Reject,
    // Label 465: @12824
    GIM_Try, /*On fail goto*//*Label 542*/ 12855, // Rule ID 3616 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s1,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v64s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VK64RegClassID,
      // (xor:{ *:[v64i1] } VK64:{ *:[v64i1] }:$src1, VK64:{ *:[v64i1] }:$src2)  =>  (KXORQrr:{ *:[v64i1] } VK64:{ *:[v64i1] }:$src1, VK64:{ *:[v64i1] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::KXORQrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3616,
      GIR_Done,
    // Label 542: @12855
    GIM_Reject,
    // Label 466: @12856
    GIM_Try, /*On fail goto*//*Label 543*/ 12887, // Rule ID 14442 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v64s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (xor:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1, VR512:{ *:[v64i8] }:$src2)  =>  (VPXORQZrr:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1, VR512:{ *:[v64i8] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPXORQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 14442,
      GIR_Done,
    // Label 543: @12887
    GIM_Reject,
    // Label 467: @12888
    GIM_Reject,
    // Label 6: @12889
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 24, /*)*//*default:*//*Label 548*/ 13288,
    /*GILLT_s32*//*Label 544*/ 12916,
    /*GILLT_s64*//*Label 545*/ 13078, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    /*GILLT_v32s1*//*Label 546*/ 13240, 0, 0,
    /*GILLT_v64s1*//*Label 547*/ 13264,
    // Label 544: @12916
    GIM_Try, /*On fail goto*//*Label 549*/ 12939, // Rule ID 2194 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (bitconvert:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (VMOVDI2SSrr:{ *:[f32] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOVDI2SSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2194,
      GIR_Done,
    // Label 549: @12939
    GIM_Try, /*On fail goto*//*Label 550*/ 12962, // Rule ID 2196 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (bitconvert:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (MOVDI2SSrr:{ *:[f32] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVDI2SSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2196,
      GIR_Done,
    // Label 550: @12962
    GIM_Try, /*On fail goto*//*Label 551*/ 12985, // Rule ID 2210 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
      // (bitconvert:{ *:[i32] } FR32:{ *:[f32] }:$src)  =>  (VMOVSS2DIrr:{ *:[i32] } FR32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOVSS2DIrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2210,
      GIR_Done,
    // Label 551: @12985
    GIM_Try, /*On fail goto*//*Label 552*/ 13008, // Rule ID 2212 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
      // (bitconvert:{ *:[i32] } FR32:{ *:[f32] }:$src)  =>  (MOVSS2DIrr:{ *:[i32] } FR32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSS2DIrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2212,
      GIR_Done,
    // Label 552: @13008
    GIM_Try, /*On fail goto*//*Label 553*/ 13031, // Rule ID 3786 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (bitconvert:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (VMOVDI2SSZrr:{ *:[f32] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOVDI2SSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3786,
      GIR_Done,
    // Label 553: @13031
    GIM_Try, /*On fail goto*//*Label 554*/ 13054, // Rule ID 3792 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
      // (bitconvert:{ *:[i32] } FR32X:{ *:[f32] }:$src)  =>  (VMOVSS2DIZrr:{ *:[i32] } FR32X:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOVSS2DIZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3792,
      GIR_Done,
    // Label 554: @13054
    GIM_Try, /*On fail goto*//*Label 555*/ 13077, // Rule ID 13846 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK32RegClassID,
      // (bitconvert:{ *:[i32] } VK32:{ *:[v32i1] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[i32] } VK32:{ *:[v32i1] }:$src, GR32:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR32*/27,
      // GIR_Coverage, 13846,
      GIR_Done,
    // Label 555: @13077
    GIM_Reject,
    // Label 545: @13078
    GIM_Try, /*On fail goto*//*Label 556*/ 13101, // Rule ID 2189 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (bitconvert:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (VMOV64toSDrr:{ *:[f64] } GR64:{ *:[i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOV64toSDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2189,
      GIR_Done,
    // Label 556: @13101
    GIM_Try, /*On fail goto*//*Label 557*/ 13124, // Rule ID 2193 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (bitconvert:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (MOV64toSDrr:{ *:[f64] } GR64:{ *:[i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOV64toSDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2193,
      GIR_Done,
    // Label 557: @13124
    GIM_Try, /*On fail goto*//*Label 558*/ 13147, // Rule ID 2205 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (bitconvert:{ *:[i64] } FR64:{ *:[f64] }:$src)  =>  (VMOVSDto64rr:{ *:[i64] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOVSDto64rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2205,
      GIR_Done,
    // Label 558: @13147
    GIM_Try, /*On fail goto*//*Label 559*/ 13170, // Rule ID 2208 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (bitconvert:{ *:[i64] } FR64:{ *:[f64] }:$src)  =>  (MOVSDto64rr:{ *:[i64] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSDto64rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2208,
      GIR_Done,
    // Label 559: @13170
    GIM_Try, /*On fail goto*//*Label 560*/ 13193, // Rule ID 3782 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (bitconvert:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (VMOV64toSDZrr:{ *:[f64] } GR64:{ *:[i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOV64toSDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3782,
      GIR_Done,
    // Label 560: @13193
    GIM_Try, /*On fail goto*//*Label 561*/ 13216, // Rule ID 3784 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
      // (bitconvert:{ *:[i64] } FR64X:{ *:[f64] }:$src)  =>  (VMOVSDto64Zrr:{ *:[i64] } FR64X:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMOVSDto64Zrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 3784,
      GIR_Done,
    // Label 561: @13216
    GIM_Try, /*On fail goto*//*Label 562*/ 13239, // Rule ID 13848 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK64RegClassID,
      // (bitconvert:{ *:[i64] } VK64:{ *:[v64i1] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[i64] } VK64:{ *:[v64i1] }:$src, GR64:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR64*/59,
      // GIR_Coverage, 13848,
      GIR_Done,
    // Label 562: @13239
    GIM_Reject,
    // Label 546: @13240
    GIM_Try, /*On fail goto*//*Label 563*/ 13263, // Rule ID 13845 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (bitconvert:{ *:[v32i1] } GR32:{ *:[i32] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[v32i1] } GR32:{ *:[i32] }:$src, VK32:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK32*/31,
      // GIR_Coverage, 13845,
      GIR_Done,
    // Label 563: @13263
    GIM_Reject,
    // Label 547: @13264
    GIM_Try, /*On fail goto*//*Label 564*/ 13287, // Rule ID 13847 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VK64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (bitconvert:{ *:[v64i1] } GR64:{ *:[i64] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[v64i1] } GR64:{ *:[i64] }:$src, VK64:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC VK64*/72,
      // GIR_Coverage, 13847,
      GIR_Done,
    // Label 564: @13287
    GIM_Reject,
    // Label 548: @13288
    GIM_Reject,
    // Label 7: @13289
    GIM_Try, /*On fail goto*//*Label 565*/ 14391,
      GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
      GIM_Try, /*On fail goto*//*Label 566*/ 13334, // Rule ID 1129 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphsubwd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6758:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src)  =>  (VPHSUBWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHSUBWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1129,
        GIR_Done,
      // Label 566: @13334
      GIM_Try, /*On fail goto*//*Label 567*/ 13374, // Rule ID 1131 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphsubdq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6757:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src)  =>  (VPHSUBDQrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHSUBDQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1131,
        GIR_Done,
      // Label 567: @13374
      GIM_Try, /*On fail goto*//*Label 568*/ 13414, // Rule ID 1133 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphsubbw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6756:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHSUBBWrr:{ *:[v8i16] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHSUBBWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1133,
        GIR_Done,
      // Label 568: @13414
      GIM_Try, /*On fail goto*//*Label 569*/ 13454, // Rule ID 1135 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddwq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6755:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src)  =>  (VPHADDWQrr:{ *:[v2i64] } VR128:{ *:[v8i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDWQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1135,
        GIR_Done,
      // Label 569: @13454
      GIM_Try, /*On fail goto*//*Label 570*/ 13494, // Rule ID 1137 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddwd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6754:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src)  =>  (VPHADDWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1137,
        GIR_Done,
      // Label 570: @13494
      GIM_Try, /*On fail goto*//*Label 571*/ 13534, // Rule ID 1139 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphadduwq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6753:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src)  =>  (VPHADDUWQrr:{ *:[v2i64] } VR128:{ *:[v8i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDUWQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1139,
        GIR_Done,
      // Label 571: @13534
      GIM_Try, /*On fail goto*//*Label 572*/ 13574, // Rule ID 1141 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphadduwd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6752:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src)  =>  (VPHADDUWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDUWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1141,
        GIR_Done,
      // Label 572: @13574
      GIM_Try, /*On fail goto*//*Label 573*/ 13614, // Rule ID 1143 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddudq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6751:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src)  =>  (VPHADDUDQrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDUDQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1143,
        GIR_Done,
      // Label 573: @13614
      GIM_Try, /*On fail goto*//*Label 574*/ 13654, // Rule ID 1145 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddubw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6750:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHADDUBWrr:{ *:[v8i16] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDUBWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1145,
        GIR_Done,
      // Label 574: @13654
      GIM_Try, /*On fail goto*//*Label 575*/ 13694, // Rule ID 1147 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddubq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6749:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHADDUBQrr:{ *:[v2i64] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDUBQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1147,
        GIR_Done,
      // Label 575: @13694
      GIM_Try, /*On fail goto*//*Label 576*/ 13734, // Rule ID 1149 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddubd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6748:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHADDUBDrr:{ *:[v4i32] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDUBDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1149,
        GIR_Done,
      // Label 576: @13734
      GIM_Try, /*On fail goto*//*Label 577*/ 13774, // Rule ID 1151 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphadddq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6747:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src)  =>  (VPHADDDQrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDDQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1151,
        GIR_Done,
      // Label 577: @13774
      GIM_Try, /*On fail goto*//*Label 578*/ 13814, // Rule ID 1153 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddbw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6746:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHADDBWrr:{ *:[v8i16] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDBWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1153,
        GIR_Done,
      // Label 578: @13814
      GIM_Try, /*On fail goto*//*Label 579*/ 13854, // Rule ID 1155 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddbq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6745:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHADDBQrr:{ *:[v2i64] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDBQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1155,
        GIR_Done,
      // Label 579: @13854
      GIM_Try, /*On fail goto*//*Label 580*/ 13894, // Rule ID 1157 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vphaddbd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6744:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src)  =>  (VPHADDBDrr:{ *:[v4i32] } VR128:{ *:[v16i8] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDBDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1157,
        GIR_Done,
      // Label 580: @13894
      GIM_Try, /*On fail goto*//*Label 581*/ 13934, // Rule ID 1159 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vfrcz_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6731:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (VFRCZSSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VFRCZSSrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1159,
        GIR_Done,
      // Label 581: @13934
      GIM_Try, /*On fail goto*//*Label 582*/ 13974, // Rule ID 1161 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vfrcz_ps,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6728:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (VFRCZPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VFRCZPSrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1161,
        GIR_Done,
      // Label 582: @13974
      GIM_Try, /*On fail goto*//*Label 583*/ 14014, // Rule ID 1163 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vfrcz_ps_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v8f32] } 6729:{ *:[iPTR] }, VR256:{ *:[v8f32] }:$src)  =>  (VFRCZPSYrr:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VFRCZPSYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1163,
        GIR_Done,
      // Label 583: @14014
      GIM_Try, /*On fail goto*//*Label 584*/ 14054, // Rule ID 1165 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vfrcz_sd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6730:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src)  =>  (VFRCZSDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VFRCZSDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1165,
        GIR_Done,
      // Label 584: @14054
      GIM_Try, /*On fail goto*//*Label 585*/ 14094, // Rule ID 1167 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vfrcz_pd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6726:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src)  =>  (VFRCZPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VFRCZPDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1167,
        GIR_Done,
      // Label 585: @14094
      GIM_Try, /*On fail goto*//*Label 586*/ 14134, // Rule ID 1169 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vfrcz_pd_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f64] } 6727:{ *:[iPTR] }, VR256:{ *:[v4f64] }:$src)  =>  (VFRCZPDYrr:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VFRCZPDYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1169,
        GIR_Done,
      // Label 586: @14134
      GIM_Try, /*On fail goto*//*Label 587*/ 14174, // Rule ID 2623 //
        GIM_CheckFeatures, GIFBS_HasAES_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesimc,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5588:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1)  =>  (VAESIMCrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESIMCrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2623,
        GIR_Done,
      // Label 587: @14174
      GIM_Try, /*On fail goto*//*Label 588*/ 14214, // Rule ID 2625 //
        GIM_CheckFeatures, GIFBS_HasAES_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesimc,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5588:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1)  =>  (AESIMCrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AESIMCrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2625,
        GIR_Done,
      // Label 588: @14214
      GIM_Try, /*On fail goto*//*Label 589*/ 14258, // Rule ID 12678 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_rsqrt_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6546:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (RSQRTSSr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src, VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RSQRTSSr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12678,
        GIR_Done,
      // Label 589: @14258
      GIM_Try, /*On fail goto*//*Label 590*/ 14302, // Rule ID 12680 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_rsqrt_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6546:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (VRSQRTSSr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src, VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VRSQRTSSr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12680,
        GIR_Done,
      // Label 590: @14302
      GIM_Try, /*On fail goto*//*Label 591*/ 14346, // Rule ID 12690 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_rcp_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6544:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (RCPSSr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src, VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RCPSSr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12690,
        GIR_Done,
      // Label 591: @14346
      GIM_Try, /*On fail goto*//*Label 592*/ 14390, // Rule ID 12692 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_rcp_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6544:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src)  =>  (VRCPSSr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src, VR128:{ *:[v4f32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VRCPSSr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12692,
        GIR_Done,
      // Label 592: @14390
      GIM_Reject,
    // Label 565: @14391
    GIM_Try, /*On fail goto*//*Label 593*/ 17421,
      GIM_CheckNumOperands, /*MI*/0, /*Expected*/4,
      GIM_Try, /*On fail goto*//*Label 594*/ 14452, // Rule ID 2627 //
        GIM_CheckFeatures, GIFBS_HasAES_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aeskeygenassist,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/3, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5589:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (VAESKEYGENASSIST128rr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESKEYGENASSIST128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2627,
        GIR_Done,
      // Label 594: @14452
      GIM_Try, /*On fail goto*//*Label 595*/ 14508, // Rule ID 2629 //
        GIM_CheckFeatures, GIFBS_HasAES_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aeskeygenassist,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/3, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5589:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (AESKEYGENASSIST128rr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AESKEYGENASSIST128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2629,
        GIR_Done,
      // Label 595: @14508
      GIM_Try, /*On fail goto*//*Label 596*/ 14560, // Rule ID 82 //
        GIM_CheckFeatures, GIFBS_HasBMI2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_bmi_pdep_32,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR32RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6379:{ *:[iPTR] }, GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (PDEP32rr:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PDEP32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 82,
        GIR_Done,
      // Label 596: @14560
      GIM_Try, /*On fail goto*//*Label 597*/ 14612, // Rule ID 84 //
        GIM_CheckFeatures, GIFBS_HasBMI2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_bmi_pdep_64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR64RegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6380:{ *:[iPTR] }, GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (PDEP64rr:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PDEP64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 84,
        GIR_Done,
      // Label 597: @14612
      GIM_Try, /*On fail goto*//*Label 598*/ 14664, // Rule ID 86 //
        GIM_CheckFeatures, GIFBS_HasBMI2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_bmi_pext_32,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR32RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6381:{ *:[iPTR] }, GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (PEXT32rr:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PEXT32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 86,
        GIR_Done,
      // Label 598: @14664
      GIM_Try, /*On fail goto*//*Label 599*/ 14716, // Rule ID 88 //
        GIM_CheckFeatures, GIFBS_HasBMI2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_bmi_pext_64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR64RegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6382:{ *:[iPTR] }, GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (PEXT64rr:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PEXT64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 88,
        GIR_Done,
      // Label 599: @14716
      GIM_Try, /*On fail goto*//*Label 600*/ 14768, // Rule ID 1399 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cvtsd2ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6569:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (VCVTSD2SSrr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSD2SSrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1399,
        GIR_Done,
      // Label 600: @14768
      GIM_Try, /*On fail goto*//*Label 601*/ 14820, // Rule ID 1401 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cvtsd2ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6569:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (CVTSD2SSrr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CVTSD2SSrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1401,
        GIR_Done,
      // Label 601: @14820
      GIM_Try, /*On fail goto*//*Label 602*/ 14872, // Rule ID 2309 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_psign_b_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v16i8] } 6686:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (VPSIGNBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPSIGNBrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2309,
        GIR_Done,
      // Label 602: @14872
      GIM_Try, /*On fail goto*//*Label 603*/ 14924, // Rule ID 2311 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_psign_w_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6690:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPSIGNWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPSIGNWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2311,
        GIR_Done,
      // Label 603: @14924
      GIM_Try, /*On fail goto*//*Label 604*/ 14976, // Rule ID 2313 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_psign_d_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6688:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPSIGNDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPSIGNDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2313,
        GIR_Done,
      // Label 604: @14976
      GIM_Try, /*On fail goto*//*Label 605*/ 15028, // Rule ID 2315 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_phadd_sw_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6670:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPHADDSWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDSWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2315,
        GIR_Done,
      // Label 605: @15028
      GIM_Try, /*On fail goto*//*Label 606*/ 15080, // Rule ID 2317 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_phsub_sw_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6676:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (VPHSUBSWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHSUBSWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2317,
        GIR_Done,
      // Label 606: @15080
      GIM_Try, /*On fail goto*//*Label 607*/ 15132, // Rule ID 2333 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_psign_b,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v32s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v32i8] } 5694:{ *:[iPTR] }, VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)  =>  (VPSIGNBYrr:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPSIGNBYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2333,
        GIR_Done,
      // Label 607: @15132
      GIM_Try, /*On fail goto*//*Label 608*/ 15184, // Rule ID 2335 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_psign_w,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v16i16] } 5696:{ *:[iPTR] }, VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPSIGNWYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPSIGNWYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2335,
        GIR_Done,
      // Label 608: @15184
      GIM_Try, /*On fail goto*//*Label 609*/ 15236, // Rule ID 2337 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_psign_d,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i32] } 5695:{ *:[iPTR] }, VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPSIGNDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPSIGNDYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2337,
        GIR_Done,
      // Label 609: @15236
      GIM_Try, /*On fail goto*//*Label 610*/ 15288, // Rule ID 2339 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_phadd_sw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v16i16] } 5681:{ *:[iPTR] }, VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPHADDSWYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHADDSWYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2339,
        GIR_Done,
      // Label 610: @15288
      GIM_Try, /*On fail goto*//*Label 611*/ 15340, // Rule ID 2341 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_phsub_sw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v16i16] } 5684:{ *:[iPTR] }, VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)  =>  (VPHSUBSWYrr:{ *:[v16i16] } VR256:{ *:[v16i16] }:$src1, VR256:{ *:[v16i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPHSUBSWYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2341,
        GIR_Done,
      // Label 611: @15340
      GIM_Try, /*On fail goto*//*Label 612*/ 15392, // Rule ID 2351 //
        GIM_CheckFeatures, GIFBS_UseSSSE3,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_psign_b_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v16i8] } 6686:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)  =>  (PSIGNBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PSIGNBrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2351,
        GIR_Done,
      // Label 612: @15392
      GIM_Try, /*On fail goto*//*Label 613*/ 15444, // Rule ID 2353 //
        GIM_CheckFeatures, GIFBS_UseSSSE3,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_psign_w_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6690:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PSIGNWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PSIGNWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2353,
        GIR_Done,
      // Label 613: @15444
      GIM_Try, /*On fail goto*//*Label 614*/ 15496, // Rule ID 2355 //
        GIM_CheckFeatures, GIFBS_UseSSSE3,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_psign_d_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6688:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (PSIGNDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PSIGNDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2355,
        GIR_Done,
      // Label 614: @15496
      GIM_Try, /*On fail goto*//*Label 615*/ 15548, // Rule ID 2359 //
        GIM_CheckFeatures, GIFBS_UseSSSE3,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_phadd_sw_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6670:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PHADDSWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PHADDSWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2359,
        GIR_Done,
      // Label 615: @15548
      GIM_Try, /*On fail goto*//*Label 616*/ 15600, // Rule ID 2361 //
        GIM_CheckFeatures, GIFBS_UseSSSE3,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_ssse3_phsub_sw_128,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6676:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)  =>  (PHSUBSWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PHSUBSWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2361,
        GIR_Done,
      // Label 616: @15600
      GIM_Try, /*On fail goto*//*Label 617*/ 15652, // Rule ID 2578 //
        GIM_CheckFeatures, GIFBS_HasSSE42,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse42_crc32_32_8,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR8RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6644:{ *:[iPTR] }, GR32:{ *:[i32] }:$src1, GR8:{ *:[i8] }:$src2)  =>  (CRC32r32r8:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR8:{ *:[i8] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CRC32r32r8,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2578,
        GIR_Done,
      // Label 617: @15652
      GIM_Try, /*On fail goto*//*Label 618*/ 15704, // Rule ID 2580 //
        GIM_CheckFeatures, GIFBS_HasSSE42,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse42_crc32_32_16,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR16RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6642:{ *:[iPTR] }, GR32:{ *:[i32] }:$src1, GR16:{ *:[i16] }:$src2)  =>  (CRC32r32r16:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR16:{ *:[i16] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CRC32r32r16,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2580,
        GIR_Done,
      // Label 618: @15704
      GIM_Try, /*On fail goto*//*Label 619*/ 15756, // Rule ID 2582 //
        GIM_CheckFeatures, GIFBS_HasSSE42,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse42_crc32_32_32,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR32RegClassID,
        // (intrinsic_wo_chain:{ *:[i32] } 6643:{ *:[iPTR] }, GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)  =>  (CRC32r32r32:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CRC32r32r32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2582,
        GIR_Done,
      // Label 619: @15756
      GIM_Try, /*On fail goto*//*Label 620*/ 15808, // Rule ID 2584 //
        GIM_CheckFeatures, GIFBS_HasSSE42,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse42_crc32_64_64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::GR64RegClassID,
        // (intrinsic_wo_chain:{ *:[i64] } 6645:{ *:[iPTR] }, GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)  =>  (CRC32r64r64:{ *:[i64] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CRC32r64r64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2584,
        GIR_Done,
      // Label 620: @15808
      GIM_Try, /*On fail goto*//*Label 621*/ 15860, // Rule ID 2587 //
        GIM_CheckFeatures, GIFBS_HasSHA,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sha1nexte,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6512:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (SHA1NEXTErr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHA1NEXTErr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2587,
        GIR_Done,
      // Label 621: @15860
      GIM_Try, /*On fail goto*//*Label 622*/ 15912, // Rule ID 2589 //
        GIM_CheckFeatures, GIFBS_HasSHA,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sha1msg1,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6510:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (SHA1MSG1rr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHA1MSG1rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2589,
        GIR_Done,
      // Label 622: @15912
      GIM_Try, /*On fail goto*//*Label 623*/ 15964, // Rule ID 2591 //
        GIM_CheckFeatures, GIFBS_HasSHA,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sha1msg2,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6511:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (SHA1MSG2rr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHA1MSG2rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2591,
        GIR_Done,
      // Label 623: @15964
      GIM_Try, /*On fail goto*//*Label 624*/ 16016, // Rule ID 2595 //
        GIM_CheckFeatures, GIFBS_HasSHA,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sha256msg1,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6514:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (SHA256MSG1rr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHA256MSG1rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2595,
        GIR_Done,
      // Label 624: @16016
      GIM_Try, /*On fail goto*//*Label 625*/ 16068, // Rule ID 2597 //
        GIM_CheckFeatures, GIFBS_HasSHA,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sha256msg2,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6515:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (SHA256MSG2rr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHA256MSG2rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2597,
        GIR_Done,
      // Label 625: @16068
      GIM_Try, /*On fail goto*//*Label 626*/ 16120, // Rule ID 2599 //
        GIM_CheckFeatures, GIFBS_HasAES_HasAVX_NoVLX_Or_NoVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenc,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5582:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VAESENCrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2599,
        GIR_Done,
      // Label 626: @16120
      GIM_Try, /*On fail goto*//*Label 627*/ 16172, // Rule ID 2601 //
        GIM_CheckFeatures, GIFBS_HasAES_HasAVX_NoVLX_Or_NoVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenclast,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5585:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VAESENCLASTrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCLASTrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2601,
        GIR_Done,
      // Label 627: @16172
      GIM_Try, /*On fail goto*//*Label 628*/ 16224, // Rule ID 2603 //
        GIM_CheckFeatures, GIFBS_HasAES_HasAVX_NoVLX_Or_NoVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdec,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5576:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VAESDECrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2603,
        GIR_Done,
      // Label 628: @16224
      GIM_Try, /*On fail goto*//*Label 629*/ 16276, // Rule ID 2605 //
        GIM_CheckFeatures, GIFBS_HasAES_HasAVX_NoVLX_Or_NoVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdeclast,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5579:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VAESDECLASTrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECLASTrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2605,
        GIR_Done,
      // Label 629: @16276
      GIM_Try, /*On fail goto*//*Label 630*/ 16328, // Rule ID 2607 //
        GIM_CheckFeatures, GIFBS_HasVAES_NoVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenc_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5583:{ *:[iPTR] }, VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VAESENCYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2607,
        GIR_Done,
      // Label 630: @16328
      GIM_Try, /*On fail goto*//*Label 631*/ 16380, // Rule ID 2609 //
        GIM_CheckFeatures, GIFBS_HasVAES_NoVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenclast_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5586:{ *:[iPTR] }, VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VAESENCLASTYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCLASTYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2609,
        GIR_Done,
      // Label 631: @16380
      GIM_Try, /*On fail goto*//*Label 632*/ 16432, // Rule ID 2611 //
        GIM_CheckFeatures, GIFBS_HasVAES_NoVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdec_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5577:{ *:[iPTR] }, VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VAESDECYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2611,
        GIR_Done,
      // Label 632: @16432
      GIM_Try, /*On fail goto*//*Label 633*/ 16484, // Rule ID 2613 //
        GIM_CheckFeatures, GIFBS_HasVAES_NoVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdeclast_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5580:{ *:[iPTR] }, VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VAESDECLASTYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECLASTYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2613,
        GIR_Done,
      // Label 633: @16484
      GIM_Try, /*On fail goto*//*Label 634*/ 16536, // Rule ID 2615 //
        GIM_CheckFeatures, GIFBS_HasAES_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenc,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5582:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (AESENCrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AESENCrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2615,
        GIR_Done,
      // Label 634: @16536
      GIM_Try, /*On fail goto*//*Label 635*/ 16588, // Rule ID 2617 //
        GIM_CheckFeatures, GIFBS_HasAES_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenclast,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5585:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (AESENCLASTrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AESENCLASTrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2617,
        GIR_Done,
      // Label 635: @16588
      GIM_Try, /*On fail goto*//*Label 636*/ 16640, // Rule ID 2619 //
        GIM_CheckFeatures, GIFBS_HasAES_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdec,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5576:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (AESDECrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AESDECrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2619,
        GIR_Done,
      // Label 636: @16640
      GIM_Try, /*On fail goto*//*Label 637*/ 16692, // Rule ID 2621 //
        GIM_CheckFeatures, GIFBS_HasAES_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdeclast,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5579:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (AESDECLASTrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AESDECLASTrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2621,
        GIR_Done,
      // Label 637: @16692
      GIM_Try, /*On fail goto*//*Label 638*/ 16744, // Rule ID 2638 //
        GIM_CheckFeatures, GIFBS_HasSSE4A,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse4a_extrq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6660:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src, VR128:{ *:[v16i8] }:$mask)  =>  (EXTRQ:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src, VR128:{ *:[v16i8] }:$mask)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::EXTRQ,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // mask
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2638,
        GIR_Done,
      // Label 638: @16744
      GIM_Try, /*On fail goto*//*Label 639*/ 16796, // Rule ID 2640 //
        GIM_CheckFeatures, GIFBS_HasSSE4A,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse4a_insertq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6662:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src, VR128:{ *:[v2i64] }:$mask)  =>  (INSERTQ:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src, VR128:{ *:[v2i64] }:$mask)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INSERTQ,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // mask
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2640,
        GIR_Done,
      // Label 639: @16796
      GIM_Try, /*On fail goto*//*Label 640*/ 16848, // Rule ID 11186 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenc,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5582:{ *:[iPTR] }, VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VAESENCZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCZ128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11186,
        GIR_Done,
      // Label 640: @16848
      GIM_Try, /*On fail goto*//*Label 641*/ 16900, // Rule ID 11188 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenc_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256XRegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5583:{ *:[iPTR] }, VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VAESENCZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCZ256rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11188,
        GIR_Done,
      // Label 641: @16900
      GIM_Try, /*On fail goto*//*Label 642*/ 16952, // Rule ID 11190 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenc_512,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR512RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i64] } 5584:{ *:[iPTR] }, VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VAESENCZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11190,
        GIR_Done,
      // Label 642: @16952
      GIM_Try, /*On fail goto*//*Label 643*/ 17004, // Rule ID 11192 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenclast,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5585:{ *:[iPTR] }, VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VAESENCLASTZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCLASTZ128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11192,
        GIR_Done,
      // Label 643: @17004
      GIM_Try, /*On fail goto*//*Label 644*/ 17056, // Rule ID 11194 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenclast_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256XRegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5586:{ *:[iPTR] }, VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VAESENCLASTZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCLASTZ256rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11194,
        GIR_Done,
      // Label 644: @17056
      GIM_Try, /*On fail goto*//*Label 645*/ 17108, // Rule ID 11196 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesenclast_512,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR512RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i64] } 5587:{ *:[iPTR] }, VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VAESENCLASTZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESENCLASTZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11196,
        GIR_Done,
      // Label 645: @17108
      GIM_Try, /*On fail goto*//*Label 646*/ 17160, // Rule ID 11198 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdec,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5576:{ *:[iPTR] }, VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VAESDECZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECZ128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11198,
        GIR_Done,
      // Label 646: @17160
      GIM_Try, /*On fail goto*//*Label 647*/ 17212, // Rule ID 11200 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdec_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256XRegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5577:{ *:[iPTR] }, VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VAESDECZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECZ256rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11200,
        GIR_Done,
      // Label 647: @17212
      GIM_Try, /*On fail goto*//*Label 648*/ 17264, // Rule ID 11202 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdec_512,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR512RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i64] } 5578:{ *:[iPTR] }, VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VAESDECZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11202,
        GIR_Done,
      // Label 648: @17264
      GIM_Try, /*On fail goto*//*Label 649*/ 17316, // Rule ID 11204 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdeclast,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128XRegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 5579:{ *:[iPTR] }, VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VAESDECLASTZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECLASTZ128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11204,
        GIR_Done,
      // Label 649: @17316
      GIM_Try, /*On fail goto*//*Label 650*/ 17368, // Rule ID 11206 //
        GIM_CheckFeatures, GIFBS_HasVAES_HasVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdeclast_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256XRegClassID,
        // (intrinsic_wo_chain:{ *:[v4i64] } 5580:{ *:[iPTR] }, VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VAESDECLASTZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECLASTZ256rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11206,
        GIR_Done,
      // Label 650: @17368
      GIM_Try, /*On fail goto*//*Label 651*/ 17420, // Rule ID 11208 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVAES,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_aesni_aesdeclast_512,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR512RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i64] } 5581:{ *:[iPTR] }, VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VAESDECLASTZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VAESDECLASTZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11208,
        GIR_Done,
      // Label 651: @17420
      GIM_Reject,
    // Label 593: @17421
    GIM_Try, /*On fail goto*//*Label 652*/ 19871,
      GIM_CheckNumOperands, /*MI*/0, /*Expected*/5,
      GIM_Try, /*On fail goto*//*Label 653*/ 17494, // Rule ID 1455 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_cmp_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6519:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src, (imm:{ *:[i8] }):$cc)  =>  (VCMPSSrr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src, (imm:{ *:[i8] }):$cc)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCMPSSrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // cc
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1455,
        GIR_Done,
      // Label 653: @17494
      GIM_Try, /*On fail goto*//*Label 654*/ 17562, // Rule ID 1457 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cmp_sd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6557:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src, (imm:{ *:[i8] }):$cc)  =>  (VCMPSDrr_Int:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src, (imm:{ *:[i8] }):$cc)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCMPSDrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // cc
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1457,
        GIR_Done,
      // Label 654: @17562
      GIM_Try, /*On fail goto*//*Label 655*/ 17630, // Rule ID 1459 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse_cmp_ss,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6519:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src, (imm:{ *:[i8] }):$cc)  =>  (CMPSSrr_Int:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src, (imm:{ *:[i8] }):$cc)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CMPSSrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // cc
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1459,
        GIR_Done,
      // Label 655: @17630
      GIM_Try, /*On fail goto*//*Label 656*/ 17698, // Rule ID 1461 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse2_cmp_sd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6557:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src, (imm:{ *:[i8] }):$cc)  =>  (CMPSDrr_Int:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src, (imm:{ *:[i8] }):$cc)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::CMPSDrr_Int,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // cc
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1461,
        GIR_Done,
      // Label 656: @17698
      GIM_Try, /*On fail goto*//*Label 657*/ 17766, // Rule ID 2519 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_mpsadbw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6631:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VMPSADBWrri:{ *:[v8i16] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VMPSADBWrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2519,
        GIR_Done,
      // Label 657: @17766
      GIM_Try, /*On fail goto*//*Label 658*/ 17834, // Rule ID 2521 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_dpps,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6629:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VDPPSrri:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VDPPSrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2521,
        GIR_Done,
      // Label 658: @17834
      GIM_Try, /*On fail goto*//*Label 659*/ 17902, // Rule ID 2523 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_dppd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6628:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VDPPDrri:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VDPPDrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2523,
        GIR_Done,
      // Label 659: @17902
      GIM_Try, /*On fail goto*//*Label 660*/ 17970, // Rule ID 2525 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx_dp_ps_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v8f32] } 5601:{ *:[iPTR] }, VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VDPPSYrri:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VDPPSYrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2525,
        GIR_Done,
      // Label 660: @17970
      GIM_Try, /*On fail goto*//*Label 661*/ 18038, // Rule ID 2527 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_mpsadbw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v16i16] } 5670:{ *:[iPTR] }, VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VMPSADBWYrri:{ *:[v16i16] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VMPSADBWYrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2527,
        GIR_Done,
      // Label 661: @18038
      GIM_Try, /*On fail goto*//*Label 662*/ 18106, // Rule ID 2529 //
        GIM_CheckFeatures, GIFBS_UseSSE41,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_mpsadbw,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6631:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (MPSADBWrri:{ *:[v8i16] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MPSADBWrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2529,
        GIR_Done,
      // Label 662: @18106
      GIM_Try, /*On fail goto*//*Label 663*/ 18174, // Rule ID 2531 //
        GIM_CheckFeatures, GIFBS_UseSSE41,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_dpps,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6629:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (DPPSrri:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::DPPSrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2531,
        GIR_Done,
      // Label 663: @18174
      GIM_Try, /*On fail goto*//*Label 664*/ 18242, // Rule ID 2533 //
        GIM_CheckFeatures, GIFBS_UseSSE41,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_dppd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6628:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (DPPDrri:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::DPPDrri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2533,
        GIR_Done,
      // Label 664: @18242
      GIM_Try, /*On fail goto*//*Label 665*/ 18310, // Rule ID 2585 //
        GIM_CheckFeatures, GIFBS_HasSHA,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sha1rnds4,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6513:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (SHA1RNDS4rri:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHA1RNDS4rri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2585,
        GIR_Done,
      // Label 665: @18310
      GIM_Try, /*On fail goto*//*Label 666*/ 18378, // Rule ID 2631 //
        GIM_CheckFeatures, GIFBS_HasPCLMUL_NoAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_pclmulqdq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6481:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (PCLMULQDQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PCLMULQDQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2631,
        GIR_Done,
      // Label 666: @18378
      GIM_Try, /*On fail goto*//*Label 667*/ 18446, // Rule ID 2633 //
        GIM_CheckFeatures, GIFBS_HasAVX_HasPCLMUL_NoVLX_Or_NoVPCLMULQDQ,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_pclmulqdq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6481:{ *:[iPTR] }, VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VPCLMULQDQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPCLMULQDQrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2633,
        GIR_Done,
      // Label 667: @18446
      GIM_Try, /*On fail goto*//*Label 668*/ 18514, // Rule ID 2635 //
        GIM_CheckFeatures, GIFBS_HasVPCLMULQDQ_NoVLX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_pclmulqdq_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4i64] } 6482:{ *:[iPTR] }, VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VPCLMULQDQYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPCLMULQDQYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2635,
        GIR_Done,
      // Label 668: @18514
      GIM_Try, /*On fail goto*//*Label 669*/ 18582, // Rule ID 11210 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVPCLMULQDQ,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_pclmulqdq_512,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR512RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v8i64] } 6483:{ *:[iPTR] }, VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VPCLMULQDQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPCLMULQDQZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11210,
        GIR_Done,
      // Label 669: @18582
      GIM_Try, /*On fail goto*//*Label 670*/ 18650, // Rule ID 11212 //
        GIM_CheckFeatures, GIFBS_HasVLX_HasVPCLMULQDQ,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_pclmulqdq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128XRegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6481:{ *:[iPTR] }, VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VPCLMULQDQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPCLMULQDQZ128rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11212,
        GIR_Done,
      // Label 670: @18650
      GIM_Try, /*On fail goto*//*Label 671*/ 18718, // Rule ID 11214 //
        GIM_CheckFeatures, GIFBS_HasVLX_HasVPCLMULQDQ,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_pclmulqdq_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256XRegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/4, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_wo_chain:{ *:[v4i64] } 6482:{ *:[iPTR] }, VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2, (imm:{ *:[i8] }):$src3)  =>  (VPCLMULQDQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2, (imm:{ *:[i8] }):$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPCLMULQDQZ256rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 11214,
        GIR_Done,
      // Label 671: @18718
      GIM_Try, /*On fail goto*//*Label 672*/ 18782, // Rule ID 1215 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmadcswd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6770:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)  =>  (VPMADCSWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMADCSWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1215,
        GIR_Done,
      // Label 672: @18782
      GIM_Try, /*On fail goto*//*Label 673*/ 18846, // Rule ID 1217 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmadcsswd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6769:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)  =>  (VPMADCSSWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMADCSSWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1217,
        GIR_Done,
      // Label 673: @18846
      GIM_Try, /*On fail goto*//*Label 674*/ 18910, // Rule ID 1219 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacsww,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6768:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v8i16] }:$src3)  =>  (VPMACSWWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v8i16] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSWWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1219,
        GIR_Done,
      // Label 674: @18910
      GIM_Try, /*On fail goto*//*Label 675*/ 18974, // Rule ID 1221 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacswd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6767:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)  =>  (VPMACSWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1221,
        GIR_Done,
      // Label 675: @18974
      GIM_Try, /*On fail goto*//*Label 676*/ 19038, // Rule ID 1223 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacssww,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v8s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v8i16] } 6766:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v8i16] }:$src3)  =>  (VPMACSSWWrr:{ *:[v8i16] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v8i16] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSSWWrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1223,
        GIR_Done,
      // Label 676: @19038
      GIM_Try, /*On fail goto*//*Label 677*/ 19102, // Rule ID 1225 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacsswd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s16,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6765:{ *:[iPTR] }, VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)  =>  (VPMACSSWDrr:{ *:[v4i32] } VR128:{ *:[v8i16] }:$src1, VR128:{ *:[v8i16] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSSWDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1225,
        GIR_Done,
      // Label 677: @19102
      GIM_Try, /*On fail goto*//*Label 678*/ 19166, // Rule ID 1227 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacssdql,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6764:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)  =>  (VPMACSSDQLrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSSDQLrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1227,
        GIR_Done,
      // Label 678: @19166
      GIM_Try, /*On fail goto*//*Label 679*/ 19230, // Rule ID 1229 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacssdqh,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6763:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)  =>  (VPMACSSDQHrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSSDQHrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1229,
        GIR_Done,
      // Label 679: @19230
      GIM_Try, /*On fail goto*//*Label 680*/ 19294, // Rule ID 1231 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacssdd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6762:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v4i32] }:$src3)  =>  (VPMACSSDDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSSDDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1231,
        GIR_Done,
      // Label 680: @19294
      GIM_Try, /*On fail goto*//*Label 681*/ 19358, // Rule ID 1233 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacsdql,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6761:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)  =>  (VPMACSDQLrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSDQLrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1233,
        GIR_Done,
      // Label 681: @19358
      GIM_Try, /*On fail goto*//*Label 682*/ 19422, // Rule ID 1235 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacsdqh,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2i64] } 6760:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)  =>  (VPMACSDQHrr:{ *:[v2i64] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v2i64] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSDQHrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1235,
        GIR_Done,
      // Label 682: @19422
      GIM_Try, /*On fail goto*//*Label 683*/ 19486, // Rule ID 1237 //
        GIM_CheckFeatures, GIFBS_HasXOP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xop_vpmacsdd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4i32] } 6759:{ *:[iPTR] }, VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v4i32] }:$src3)  =>  (VPMACSDDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2, VR128:{ *:[v4i32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMACSDDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1237,
        GIR_Done,
      // Label 683: @19486
      GIM_Try, /*On fail goto*//*Label 684*/ 19550, // Rule ID 2553 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_blendvpd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v2s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v2s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v2f64] } 6626:{ *:[iPTR] }, VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2, VR128:{ *:[v2f64] }:$src3)  =>  (VBLENDVPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2, VR128:{ *:[v2f64] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VBLENDVPDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2553,
        GIR_Done,
      // Label 684: @19550
      GIM_Try, /*On fail goto*//*Label 685*/ 19614, // Rule ID 2555 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx_blendv_pd_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s64,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f64] } 5592:{ *:[iPTR] }, VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2, VR256:{ *:[v4f64] }:$src3)  =>  (VBLENDVPDYrr:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2, VR256:{ *:[v4f64] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VBLENDVPDYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2555,
        GIR_Done,
      // Label 685: @19614
      GIM_Try, /*On fail goto*//*Label 686*/ 19678, // Rule ID 2557 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_blendvps,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v4s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v4s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v4f32] } 6627:{ *:[iPTR] }, VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2, VR128:{ *:[v4f32] }:$src3)  =>  (VBLENDVPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2, VR128:{ *:[v4f32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VBLENDVPSrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2557,
        GIR_Done,
      // Label 686: @19678
      GIM_Try, /*On fail goto*//*Label 687*/ 19742, // Rule ID 2559 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx_blendv_ps_256,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v8s32,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v8s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v8f32] } 5593:{ *:[iPTR] }, VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2, VR256:{ *:[v8f32] }:$src3)  =>  (VBLENDVPSYrr:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2, VR256:{ *:[v8f32] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VBLENDVPSYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2559,
        GIR_Done,
      // Label 687: @19742
      GIM_Try, /*On fail goto*//*Label 688*/ 19806, // Rule ID 2561 //
        GIM_CheckFeatures, GIFBS_HasAVX,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_sse41_pblendvb,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v16s8,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v16s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR128RegClassID,
        // (intrinsic_wo_chain:{ *:[v16i8] } 6633:{ *:[iPTR] }, VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2, VR128:{ *:[v16i8] }:$src3)  =>  (VPBLENDVBrr:{ *:[v16i8] } VR128:{ *:[v16i8] }:$src1, VR128:{ *:[v16i8] }:$src2, VR128:{ *:[v16i8] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPBLENDVBrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2561,
        GIR_Done,
      // Label 688: @19806
      GIM_Try, /*On fail goto*//*Label 689*/ 19870, // Rule ID 2563 //
        GIM_CheckFeatures, GIFBS_HasAVX2,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_avx2_pblendvb,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_v32s8,
        GIM_CheckType, /*MI*/0, /*Op*/4, /*Type*/GILLT_v32s8,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/3, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/4, /*RC*/X86::VR256RegClassID,
        // (intrinsic_wo_chain:{ *:[v32i8] } 5677:{ *:[iPTR] }, VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2, VR256:{ *:[v32i8] }:$src3)  =>  (VPBLENDVBYrr:{ *:[v32i8] } VR256:{ *:[v32i8] }:$src1, VR256:{ *:[v32i8] }:$src2, VR256:{ *:[v32i8] }:$src3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPBLENDVBYrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/3, // src2
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/4, // src3
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2563,
        GIR_Done,
      // Label 689: @19870
      GIM_Reject,
    // Label 652: @19871
    GIM_Reject,
    // Label 8: @19872
    GIM_Try, /*On fail goto*//*Label 690*/ 19890, // Rule ID 1789 //
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_sse2_pause,
      // (intrinsic_void 6587:{ *:[iPTR] })  =>  (PAUSE)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PAUSE,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1789,
      GIR_Done,
    // Label 690: @19890
    GIM_Try, /*On fail goto*//*Label 691*/ 19910, // Rule ID 1790 //
      GIM_CheckFeatures, GIFBS_HasSSE1,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_sse_sfence,
      // (intrinsic_void 6547:{ *:[iPTR] })  =>  (SFENCE)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SFENCE,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1790,
      GIR_Done,
    // Label 691: @19910
    GIM_Try, /*On fail goto*//*Label 692*/ 19930, // Rule ID 1791 //
      GIM_CheckFeatures, GIFBS_HasSSE2,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_sse2_lfence,
      // (intrinsic_void 6574:{ *:[iPTR] })  =>  (LFENCE)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LFENCE,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1791,
      GIR_Done,
    // Label 692: @19930
    GIM_Try, /*On fail goto*//*Label 693*/ 19950, // Rule ID 1792 //
      GIM_CheckFeatures, GIFBS_HasMFence,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_sse2_mfence,
      // (intrinsic_void 6578:{ *:[iPTR] })  =>  (MFENCE)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MFENCE,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1792,
      GIR_Done,
    // Label 693: @19950
    GIM_Try, /*On fail goto*//*Label 694*/ 19970, // Rule ID 2673 //
      GIM_CheckFeatures, GIFBS_HasAVX,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_avx_vzeroall,
      // (intrinsic_void 5644:{ *:[iPTR] })  =>  (VZEROALL)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VZEROALL,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2673,
      GIR_Done,
    // Label 694: @19970
    GIM_Try, /*On fail goto*//*Label 695*/ 19990, // Rule ID 2674 //
      GIM_CheckFeatures, GIFBS_HasAVX,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_avx_vzeroupper,
      // (intrinsic_void 5645:{ *:[iPTR] })  =>  (VZEROUPPER)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VZEROUPPER,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2674,
      GIR_Done,
    // Label 695: @19990
    GIM_Try, /*On fail goto*//*Label 696*/ 20010, // Rule ID 11786 //
      GIM_CheckFeatures, GIFBS_HasMMX,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_mmx_emms,
      // (intrinsic_void 6407:{ *:[iPTR] })  =>  (MMX_EMMS)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MMX_EMMS,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 11786,
      GIR_Done,
    // Label 696: @20010
    GIM_Try, /*On fail goto*//*Label 697*/ 20030, // Rule ID 12005 //
      GIM_CheckFeatures, GIFBS_Has3DNow,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_mmx_femms,
      // (intrinsic_void 6408:{ *:[iPTR] })  =>  (FEMMS)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::FEMMS,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12005,
      GIR_Done,
    // Label 697: @20030
    GIM_Try, /*On fail goto*//*Label 698*/ 20050, // Rule ID 12020 //
      GIM_CheckFeatures, GIFBS_HasRTM,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_xend,
      // (intrinsic_void 6724:{ *:[iPTR] })  =>  (XEND)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XEND,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12020,
      GIR_Done,
    // Label 698: @20050
    GIM_Try, /*On fail goto*//*Label 699*/ 20068, // Rule ID 12029 //
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wbinvd,
      // (intrinsic_void 6711:{ *:[iPTR] })  =>  (WBINVD)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WBINVD,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12029,
      GIR_Done,
    // Label 699: @20068
    GIM_Try, /*On fail goto*//*Label 700*/ 20088, // Rule ID 12030 //
      GIM_CheckFeatures, GIFBS_HasWBNOINVD,
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wbnoinvd,
      // (intrinsic_void 6712:{ *:[iPTR] })  =>  (WBNOINVD)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WBNOINVD,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12030,
      GIR_Done,
    // Label 700: @20088
    GIM_Try, /*On fail goto*//*Label 701*/ 20106, // Rule ID 12035 //
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_saveprevssp,
      // (intrinsic_void 6504:{ *:[iPTR] })  =>  (SAVEPREVSSP)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAVEPREVSSP,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12035,
      GIR_Done,
    // Label 701: @20106
    GIM_Try, /*On fail goto*//*Label 702*/ 20124, // Rule ID 12041 //
      GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_setssbsy,
      // (intrinsic_void 6509:{ *:[iPTR] })  =>  (SETSSBSY)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SETSSBSY,
      GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12041,
      GIR_Done,
    // Label 702: @20124
    GIM_Try, /*On fail goto*//*Label 703*/ 21084,
      GIM_CheckNumOperands, /*MI*/0, /*Expected*/2,
      GIM_Try, /*On fail goto*//*Label 704*/ 20155, // Rule ID 12026 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_int,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
        GIM_CheckConstantInt, /*MI*/0, /*Op*/1, 3,
        // (intrinsic_void 6400:{ *:[iPTR] }, 3:{ *:[i8] })  =>  (INT3)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INT3,
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12026,
        GIR_Done,
      // Label 704: @20155
      GIM_Try, /*On fail goto*//*Label 705*/ 20192, // Rule ID 12022 //
        GIM_CheckFeatures, GIFBS_HasRTM,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_xabort,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_void 6722:{ *:[iPTR] }, (imm:{ *:[i8] }):$imm)  =>  (XABORT (imm:{ *:[i8] }):$imm)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XABORT,
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // imm
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, 1, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12022,
        GIR_Done,
      // Label 705: @20192
      GIM_Try, /*On fail goto*//*Label 706*/ 20227, // Rule ID 12027 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_int,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_void 6400:{ *:[iPTR] }, (imm:{ *:[i8] }):$trap)  =>  (INT (imm:{ *:[i8] }):$trap)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INT,
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // trap
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, 1, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12027,
        GIR_Done,
      // Label 706: @20227
      GIM_Try, /*On fail goto*//*Label 707*/ 20259, // Rule ID 1 //
        GIM_CheckFeatures, GIFBS_Not64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_flags_read_u32,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6390:{ *:[iPTR] })  =>  (RDFLAGS32:{ *:[i32] }:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDFLAGS32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1,
        GIR_Done,
      // Label 707: @20259
      GIM_Try, /*On fail goto*//*Label 708*/ 20291, // Rule ID 2 //
        GIM_CheckFeatures, GIFBS_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_flags_read_u64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        // (intrinsic_w_chain:{ *:[i64] } 6391:{ *:[iPTR] })  =>  (RDFLAGS64:{ *:[i64] }:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDFLAGS64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2,
        GIR_Done,
      // Label 708: @20291
      GIM_Try, /*On fail goto*//*Label 709*/ 20323, // Rule ID 95 //
        GIM_CheckFeatures, GIFBS_HasLWP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_slwpcb,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6517:{ *:[iPTR] })  =>  (SLWPCB:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SLWPCB,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 95,
        GIR_Done,
      // Label 709: @20323
      GIM_Try, /*On fail goto*//*Label 710*/ 20355, // Rule ID 97 //
        GIM_CheckFeatures, GIFBS_HasLWP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_slwpcb,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        // (intrinsic_w_chain:{ *:[i64] } 6517:{ *:[iPTR] })  =>  (SLWPCB64:{ *:[i64] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SLWPCB64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 97,
        GIR_Done,
      // Label 710: @20355
      GIM_Try, /*On fail goto*//*Label 711*/ 20387, // Rule ID 12019 //
        GIM_CheckFeatures, GIFBS_HasRTM,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_xbegin,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6723:{ *:[iPTR] })  =>  (XBEGIN:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::XBEGIN,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12019,
        GIR_Done,
      // Label 711: @20387
      GIM_Try, /*On fail goto*//*Label 712*/ 20417, // Rule ID 12057 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdpkru,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6491:{ *:[iPTR] })  =>  (RDPKRU:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDPKRU,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12057,
        GIR_Done,
      // Label 712: @20417
      GIM_Try, /*On fail goto*//*Label 713*/ 20449, // Rule ID 12058 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdfsbase_32,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6486:{ *:[iPTR] })  =>  (RDFSBASE:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDFSBASE,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12058,
        GIR_Done,
      // Label 713: @20449
      GIM_Try, /*On fail goto*//*Label 714*/ 20481, // Rule ID 12059 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdfsbase_64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        // (intrinsic_w_chain:{ *:[i64] } 6487:{ *:[iPTR] })  =>  (RDFSBASE64:{ *:[i64] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDFSBASE64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12059,
        GIR_Done,
      // Label 714: @20481
      GIM_Try, /*On fail goto*//*Label 715*/ 20513, // Rule ID 12060 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdgsbase_32,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6488:{ *:[iPTR] })  =>  (RDGSBASE:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDGSBASE,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12060,
        GIR_Done,
      // Label 715: @20513
      GIM_Try, /*On fail goto*//*Label 716*/ 20545, // Rule ID 12061 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdgsbase_64,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        // (intrinsic_w_chain:{ *:[i64] } 6489:{ *:[iPTR] })  =>  (RDGSBASE64:{ *:[i64] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDGSBASE64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12061,
        GIR_Done,
      // Label 716: @20545
      GIM_Try, /*On fail goto*//*Label 717*/ 20577, // Rule ID 12067 //
        GIM_CheckFeatures, GIFBS_HasRDPID_Not64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdpid,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6490:{ *:[iPTR] })  =>  (RDPID32:{ *:[i32] })
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDPID32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12067,
        GIR_Done,
      // Label 717: @20577
      GIM_Try, /*On fail goto*//*Label 718*/ 20609, // Rule ID 3 //
        GIM_CheckFeatures, GIFBS_Not64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_flags_write_u32,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6392:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (WRFLAGS32:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRFLAGS32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 3,
        GIR_Done,
      // Label 718: @20609
      GIM_Try, /*On fail goto*//*Label 719*/ 20641, // Rule ID 4 //
        GIM_CheckFeatures, GIFBS_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_flags_write_u64,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6393:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (WRFLAGS64:{ *:[i64] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRFLAGS64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 4,
        GIR_Done,
      // Label 719: @20641
      GIM_Try, /*On fail goto*//*Label 720*/ 20673, // Rule ID 94 //
        GIM_CheckFeatures, GIFBS_HasLWP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_llwpcb,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6402:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (LLWPCB GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LLWPCB,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 94,
        GIR_Done,
      // Label 720: @20673
      GIM_Try, /*On fail goto*//*Label 721*/ 20705, // Rule ID 96 //
        GIM_CheckFeatures, GIFBS_HasLWP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_llwpcb,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6402:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (LLWPCB64 GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LLWPCB64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 96,
        GIR_Done,
      // Label 721: @20705
      GIM_Try, /*On fail goto*//*Label 722*/ 20737, // Rule ID 108 //
        GIM_CheckFeatures, GIFBS_HasWAITPKG_Not64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_umonitor,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
        // (intrinsic_void 6696:{ *:[iPTR] }, GR16:{ *:[i16] }:$src)  =>  (UMONITOR16 GR16:{ *:[i16] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::UMONITOR16,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 108,
        GIR_Done,
      // Label 722: @20737
      GIM_Try, /*On fail goto*//*Label 723*/ 20769, // Rule ID 109 //
        GIM_CheckFeatures, GIFBS_HasWAITPKG,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_umonitor,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6696:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (UMONITOR32 GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::UMONITOR32,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 109,
        GIR_Done,
      // Label 723: @20769
      GIM_Try, /*On fail goto*//*Label 724*/ 20801, // Rule ID 110 //
        GIM_CheckFeatures, GIFBS_HasWAITPKG_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_umonitor,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6696:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (UMONITOR64 GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::UMONITOR64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 110,
        GIR_Done,
      // Label 724: @20801
      GIM_Try, /*On fail goto*//*Label 725*/ 20831, // Rule ID 12031 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_incsspd,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6398:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (INCSSPD GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INCSSPD,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12031,
        GIR_Done,
      // Label 725: @20831
      GIM_Try, /*On fail goto*//*Label 726*/ 20861, // Rule ID 12032 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_incsspq,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6399:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (INCSSPQ GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::INCSSPQ,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12032,
        GIR_Done,
      // Label 726: @20861
      GIM_Try, /*On fail goto*//*Label 727*/ 20891, // Rule ID 12056 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wrpkru,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6717:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (WRPKRU GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRPKRU,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12056,
        GIR_Done,
      // Label 727: @20891
      GIM_Try, /*On fail goto*//*Label 728*/ 20923, // Rule ID 12062 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wrfsbase_32,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6713:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (WRFSBASE GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRFSBASE,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12062,
        GIR_Done,
      // Label 728: @20923
      GIM_Try, /*On fail goto*//*Label 729*/ 20955, // Rule ID 12063 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wrfsbase_64,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6714:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (WRFSBASE64 GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRFSBASE64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12063,
        GIR_Done,
      // Label 729: @20955
      GIM_Try, /*On fail goto*//*Label 730*/ 20987, // Rule ID 12064 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wrgsbase_32,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6715:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (WRGSBASE GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRGSBASE,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12064,
        GIR_Done,
      // Label 730: @20987
      GIM_Try, /*On fail goto*//*Label 731*/ 21019, // Rule ID 12065 //
        GIM_CheckFeatures, GIFBS_HasFSGSBase_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_wrgsbase_64,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6716:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (WRGSBASE64 GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::WRGSBASE64,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12065,
        GIR_Done,
      // Label 731: @21019
      GIM_Try, /*On fail goto*//*Label 732*/ 21051, // Rule ID 12070 //
        GIM_CheckFeatures, GIFBS_HasPTWRITE,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_ptwrite32,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        // (intrinsic_void 6484:{ *:[iPTR] }, GR32:{ *:[i32] }:$dst)  =>  (PTWRITEr GR32:{ *:[i32] }:$dst)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PTWRITEr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12070,
        GIR_Done,
      // Label 732: @21051
      GIM_Try, /*On fail goto*//*Label 733*/ 21083, // Rule ID 12071 //
        GIM_CheckFeatures, GIFBS_HasPTWRITE_In64BitMode,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_ptwrite64,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        // (intrinsic_void 6485:{ *:[iPTR] }, GR64:{ *:[i64] }:$dst)  =>  (PTWRITE64r GR64:{ *:[i64] }:$dst)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::PTWRITE64r,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // dst
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12071,
        GIR_Done,
      // Label 733: @21083
      GIM_Reject,
    // Label 703: @21084
    GIM_Try, /*On fail goto*//*Label 734*/ 21174,
      GIM_CheckNumOperands, /*MI*/0, /*Expected*/3,
      GIM_Try, /*On fail goto*//*Label 735*/ 21131, // Rule ID 12033 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdsspd,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        // (intrinsic_w_chain:{ *:[i32] } 6499:{ *:[iPTR] }, GR32:{ *:[i32] }:$src)  =>  (RDSSPD:{ *:[i32] } GR32:{ *:[i32] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDSSPD,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12033,
        GIR_Done,
      // Label 735: @21131
      GIM_Try, /*On fail goto*//*Label 736*/ 21173, // Rule ID 12034 //
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/1, Intrinsic::x86_rdsspq,
        GIM_CheckType, /*MI*/0, /*Op*/0, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR64RegClassID,
        // (intrinsic_w_chain:{ *:[i64] } 6500:{ *:[iPTR] }, GR64:{ *:[i64] }:$src)  =>  (RDSSPQ:{ *:[i64] } GR64:{ *:[i64] }:$src)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::RDSSPQ,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12034,
        GIR_Done,
      // Label 736: @21173
      GIM_Reject,
    // Label 734: @21174
    GIM_Try, /*On fail goto*//*Label 737*/ 21302,
      GIM_CheckNumOperands, /*MI*/0, /*Expected*/4,
      GIM_Try, /*On fail goto*//*Label 738*/ 21240, // Rule ID 102 //
        GIM_CheckFeatures, GIFBS_HasLWP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_lwpval32,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/3, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_void 6405:{ *:[iPTR] }, GR32:{ *:[i32] }:$src0, GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$cntl)  =>  (LWPVAL32rri GR32:{ *:[i32] }:$src0, GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$cntl)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LWPVAL32rri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src0
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // cntl
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, 1, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 102,
        GIR_Done,
      // Label 738: @21240
      GIM_Try, /*On fail goto*//*Label 739*/ 21301, // Rule ID 104 //
        GIM_CheckFeatures, GIFBS_HasLWP,
        GIM_CheckIntrinsicID, /*MI*/0, /*Op*/0, Intrinsic::x86_lwpval64,
        GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
        GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
        GIM_CheckType, /*MI*/0, /*Op*/3, /*Type*/GILLT_s32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::GR32RegClassID,
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/3, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (intrinsic_void 6406:{ *:[iPTR] }, GR64:{ *:[i64] }:$src0, GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$cntl)  =>  (LWPVAL64rri GR64:{ *:[i64] }:$src0, GR32:{ *:[i32] }:$src1, (imm:{ *:[i32] }):$cntl)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LWPVAL64rri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src0
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/2, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // cntl
        GIR_MergeMemOperands, /*InsnID*/0, /*MergeInsnID's*/0, 1, GIU_MergeMemOperands_EndOfList,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 104,
        GIR_Done,
      // Label 739: @21301
      GIM_Reject,
    // Label 737: @21302
    GIM_Reject,
    // Label 9: @21303
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 25, /*)*//*default:*//*Label 753*/ 22110,
    /*GILLT_s32*//*Label 740*/ 21331, 0, 0, 0, 0,
    /*GILLT_v2s64*//*Label 741*/ 21443, 0,
    /*GILLT_v4s32*//*Label 742*/ 21467,
    /*GILLT_v4s64*//*Label 743*/ 21491, 0,
    /*GILLT_v8s16*//*Label 744*/ 21561,
    /*GILLT_v8s32*//*Label 745*/ 21624,
    /*GILLT_v8s64*//*Label 746*/ 21694, 0,
    /*GILLT_v16s8*//*Label 747*/ 21764,
    /*GILLT_v16s16*//*Label 748*/ 21827,
    /*GILLT_v16s32*//*Label 749*/ 21945, 0,
    /*GILLT_v32s8*//*Label 750*/ 22015,
    /*GILLT_v32s16*//*Label 751*/ 22039, 0,
    /*GILLT_v64s8*//*Label 752*/ 22086,
    // Label 740: @21331
    GIM_Try, /*On fail goto*//*Label 754*/ 21376, // Rule ID 13842 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
      GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_BITCAST,
      GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK16RegClassID,
      GIM_CheckIsSafeToFold, /*InsnID*/1,
      // (anyext:{ *:[i32] } (bitconvert:{ *:[i16] } VK16:{ *:[v16i1] }:$src))  =>  (COPY_TO_REGCLASS:{ *:[i32] } VK16:{ *:[v16i1] }:$src, GR32:{ *:[i32] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR32*/27,
      // GIR_Coverage, 13842,
      GIR_Done,
    // Label 754: @21376
    GIM_Try, /*On fail goto*//*Label 755*/ 21421, // Rule ID 13844 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
      GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_BITCAST,
      GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK8RegClassID,
      GIM_CheckIsSafeToFold, /*InsnID*/1,
      // (anyext:{ *:[i32] } (bitconvert:{ *:[i8] } VK8:{ *:[v8i1] }:$src))  =>  (COPY_TO_REGCLASS:{ *:[i32] } VK8:{ *:[v8i1] }:$src, GR32:{ *:[i32] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR32*/27,
      // GIR_Coverage, 13844,
      GIR_Done,
    // Label 755: @21421
    GIM_Try, /*On fail goto*//*Label 756*/ 21442, // Rule ID 17222 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      // (anyext:{ *:[i32] } GR8:{ *:[i8] }:$src)  =>  (MOVZX32rr8:{ *:[i32] } GR8:{ *:[i8] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVZX32rr8,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 17222,
      GIR_Done,
    // Label 756: @21442
    GIM_Reject,
    // Label 741: @21443
    GIM_Try, /*On fail goto*//*Label 757*/ 21466, // Rule ID 16403 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK2RegClassID,
      // (anyext:{ *:[v2i64] } VK2:{ *:[v2i1] }:$src)  =>  (VPMOVM2QZ128rr:{ *:[v2i64] } VK2:{ *:[v2i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2QZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16403,
      GIR_Done,
    // Label 757: @21466
    GIM_Reject,
    // Label 742: @21467
    GIM_Try, /*On fail goto*//*Label 758*/ 21490, // Rule ID 16400 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK4RegClassID,
      // (anyext:{ *:[v4i32] } VK4:{ *:[v4i1] }:$src)  =>  (VPMOVM2DZ128rr:{ *:[v4i32] } VK4:{ *:[v4i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2DZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16400,
      GIR_Done,
    // Label 758: @21490
    GIM_Reject,
    // Label 743: @21491
    GIM_Try, /*On fail goto*//*Label 759*/ 21514, // Rule ID 12832 //
      GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (anyext:{ *:[v4i64] } VR128:{ *:[v4i32] }:$src)  =>  (VPMOVZXDQYrr:{ *:[v4i64] } VR128:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXDQYrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12832,
      GIR_Done,
    // Label 759: @21514
    GIM_Try, /*On fail goto*//*Label 760*/ 21537, // Rule ID 16270 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (anyext:{ *:[v4i64] } VR128X:{ *:[v4i32] }:$src)  =>  (VPMOVZXDQZ256rr:{ *:[v4i64] } VR128X:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXDQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16270,
      GIR_Done,
    // Label 760: @21537
    GIM_Try, /*On fail goto*//*Label 761*/ 21560, // Rule ID 16402 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK4RegClassID,
      // (anyext:{ *:[v4i64] } VK4:{ *:[v4i1] }:$src)  =>  (VPMOVM2QZ256rr:{ *:[v4i64] } VK4:{ *:[v4i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2QZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16402,
      GIR_Done,
    // Label 761: @21560
    GIM_Reject,
    // Label 744: @21561
    GIM_Try, /*On fail goto*//*Label 762*/ 21623,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      GIM_Try, /*On fail goto*//*Label 763*/ 21586, // Rule ID 16397 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        // (anyext:{ *:[v8i16] } VK8:{ *:[v8i1] }:$src)  =>  (VPMOVM2WZ128rr:{ *:[v8i16] } VK8:{ *:[v8i1] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2WZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16397,
        GIR_Done,
      // Label 763: @21586
      GIM_Try, /*On fail goto*//*Label 764*/ 21622, // Rule ID 16417 //
        GIM_CheckFeatures, GIFBS_HasDQI_HasVLX_NoBWI,
        // (anyext:{ *:[v8i16] } VK8:{ *:[v8i1] }:$src)  =>  (VPMOVDWZ256rr:{ *:[v8i16] } (VPMOVM2DZ256rr:{ *:[v8i32] } VK8:{ *:[v8i1] }:$src))
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v8s32,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::VPMOVM2DZ256rr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMOVDWZ256rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16417,
        GIR_Done,
      // Label 764: @21622
      GIM_Reject,
    // Label 762: @21623
    GIM_Reject,
    // Label 745: @21624
    GIM_Try, /*On fail goto*//*Label 765*/ 21647, // Rule ID 12831 //
      GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (anyext:{ *:[v8i32] } VR128:{ *:[v8i16] }:$src)  =>  (VPMOVZXWDYrr:{ *:[v8i32] } VR128:{ *:[v8i16] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXWDYrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12831,
      GIR_Done,
    // Label 765: @21647
    GIM_Try, /*On fail goto*//*Label 766*/ 21670, // Rule ID 16269 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (anyext:{ *:[v8i32] } VR128X:{ *:[v8i16] }:$src)  =>  (VPMOVZXWDZ256rr:{ *:[v8i32] } VR128X:{ *:[v8i16] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXWDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16269,
      GIR_Done,
    // Label 766: @21670
    GIM_Try, /*On fail goto*//*Label 767*/ 21693, // Rule ID 16399 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      // (anyext:{ *:[v8i32] } VK8:{ *:[v8i1] }:$src)  =>  (VPMOVM2DZ256rr:{ *:[v8i32] } VK8:{ *:[v8i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2DZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16399,
      GIR_Done,
    // Label 767: @21693
    GIM_Reject,
    // Label 746: @21694
    GIM_Try, /*On fail goto*//*Label 768*/ 21717, // Rule ID 16274 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (anyext:{ *:[v8i64] } VR128X:{ *:[v8i16] }:$src)  =>  (VPMOVZXWQZrr:{ *:[v8i64] } VR128X:{ *:[v8i16] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXWQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16274,
      GIR_Done,
    // Label 768: @21717
    GIM_Try, /*On fail goto*//*Label 769*/ 21740, // Rule ID 16275 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (anyext:{ *:[v8i64] } VR256X:{ *:[v8i32] }:$src)  =>  (VPMOVZXDQZrr:{ *:[v8i64] } VR256X:{ *:[v8i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXDQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16275,
      GIR_Done,
    // Label 769: @21740
    GIM_Try, /*On fail goto*//*Label 770*/ 21763, // Rule ID 16401 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      // (anyext:{ *:[v8i64] } VK8:{ *:[v8i1] }:$src)  =>  (VPMOVM2QZrr:{ *:[v8i64] } VK8:{ *:[v8i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2QZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16401,
      GIR_Done,
    // Label 770: @21763
    GIM_Reject,
    // Label 747: @21764
    GIM_Try, /*On fail goto*//*Label 771*/ 21826,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      GIM_Try, /*On fail goto*//*Label 772*/ 21789, // Rule ID 16394 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        // (anyext:{ *:[v16i8] } VK16:{ *:[v16i1] }:$src)  =>  (VPMOVM2BZ128rr:{ *:[v16i8] } VK16:{ *:[v16i1] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2BZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16394,
        GIR_Done,
      // Label 772: @21789
      GIM_Try, /*On fail goto*//*Label 773*/ 21825, // Rule ID 16414 //
        GIM_CheckFeatures, GIFBS_HasDQI_NoBWI,
        // (anyext:{ *:[v16i8] } VK16:{ *:[v16i1] }:$src)  =>  (VPMOVDBZrr:{ *:[v16i8] } (VPMOVM2DZrr:{ *:[v16i32] } VK16:{ *:[v16i1] }:$src))
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s32,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::VPMOVM2DZrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMOVDBZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16414,
        GIR_Done,
      // Label 773: @21825
      GIM_Reject,
    // Label 771: @21826
    GIM_Reject,
    // Label 748: @21827
    GIM_Try, /*On fail goto*//*Label 774*/ 21850, // Rule ID 12830 //
      GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX_Or_NoBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (anyext:{ *:[v16i16] } VR128:{ *:[v16i8] }:$src)  =>  (VPMOVZXBWYrr:{ *:[v16i16] } VR128:{ *:[v16i8] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXBWYrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12830,
      GIR_Done,
    // Label 774: @21850
    GIM_Try, /*On fail goto*//*Label 775*/ 21873, // Rule ID 16268 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (anyext:{ *:[v16i16] } VR128X:{ *:[v16i8] }:$src)  =>  (VPMOVZXBWZ256rr:{ *:[v16i16] } VR128X:{ *:[v16i8] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXBWZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16268,
      GIR_Done,
    // Label 775: @21873
    GIM_Try, /*On fail goto*//*Label 776*/ 21896, // Rule ID 16396 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      // (anyext:{ *:[v16i16] } VK16:{ *:[v16i1] }:$src)  =>  (VPMOVM2WZ256rr:{ *:[v16i16] } VK16:{ *:[v16i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2WZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16396,
      GIR_Done,
    // Label 776: @21896
    GIM_Try, /*On fail goto*//*Label 777*/ 21944, // Rule ID 16415 //
      GIM_CheckFeatures, GIFBS_HasDQI_NoBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      // (anyext:{ *:[v16i16] } VK16:{ *:[v16i1] }:$src)  =>  (VPMOVDWZrr:{ *:[v16i16] } (VPMOVM2DZrr:{ *:[v16i32] } VK16:{ *:[v16i1] }:$src))
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::VPMOVM2DZrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMOVDWZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16415,
      GIR_Done,
    // Label 777: @21944
    GIM_Reject,
    // Label 749: @21945
    GIM_Try, /*On fail goto*//*Label 778*/ 21968, // Rule ID 16272 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (anyext:{ *:[v16i32] } VR128X:{ *:[v16i8] }:$src)  =>  (VPMOVZXBDZrr:{ *:[v16i32] } VR128X:{ *:[v16i8] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXBDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16272,
      GIR_Done,
    // Label 778: @21968
    GIM_Try, /*On fail goto*//*Label 779*/ 21991, // Rule ID 16273 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (anyext:{ *:[v16i32] } VR256X:{ *:[v16i16] }:$src)  =>  (VPMOVZXWDZrr:{ *:[v16i32] } VR256X:{ *:[v16i16] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXWDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16273,
      GIR_Done,
    // Label 779: @21991
    GIM_Try, /*On fail goto*//*Label 780*/ 22014, // Rule ID 16398 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      // (anyext:{ *:[v16i32] } VK16:{ *:[v16i1] }:$src)  =>  (VPMOVM2DZrr:{ *:[v16i32] } VK16:{ *:[v16i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2DZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16398,
      GIR_Done,
    // Label 780: @22014
    GIM_Reject,
    // Label 750: @22015
    GIM_Try, /*On fail goto*//*Label 781*/ 22038, // Rule ID 16393 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK32RegClassID,
      // (anyext:{ *:[v32i8] } VK32:{ *:[v32i1] }:$src)  =>  (VPMOVM2BZ256rr:{ *:[v32i8] } VK32:{ *:[v32i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2BZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16393,
      GIR_Done,
    // Label 781: @22038
    GIM_Reject,
    // Label 751: @22039
    GIM_Try, /*On fail goto*//*Label 782*/ 22062, // Rule ID 16271 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (anyext:{ *:[v32i16] } VR256X:{ *:[v32i8] }:$src)  =>  (VPMOVZXBWZrr:{ *:[v32i16] } VR256X:{ *:[v32i8] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXBWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16271,
      GIR_Done,
    // Label 782: @22062
    GIM_Try, /*On fail goto*//*Label 783*/ 22085, // Rule ID 16395 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK32RegClassID,
      // (anyext:{ *:[v32i16] } VK32:{ *:[v32i1] }:$src)  =>  (VPMOVM2WZrr:{ *:[v32i16] } VK32:{ *:[v32i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2WZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16395,
      GIR_Done,
    // Label 783: @22085
    GIM_Reject,
    // Label 752: @22086
    GIM_Try, /*On fail goto*//*Label 784*/ 22109, // Rule ID 16392 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK64RegClassID,
      // (anyext:{ *:[v64i8] } VK64:{ *:[v64i1] }:$src)  =>  (VPMOVM2BZrr:{ *:[v64i8] } VK64:{ *:[v64i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2BZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16392,
      GIR_Done,
    // Label 784: @22109
    GIM_Reject,
    // Label 753: @22110
    GIM_Reject,
    // Label 10: @22111
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 22, /*)*//*default:*//*Label 793*/ 22541,
    /*GILLT_s8*//*Label 785*/ 22138,
    /*GILLT_s16*//*Label 786*/ 22217, 0, 0, 0, 0, 0, 0, 0,
    /*GILLT_v4s32*//*Label 787*/ 22255, 0, 0,
    /*GILLT_v8s16*//*Label 788*/ 22279,
    /*GILLT_v8s32*//*Label 789*/ 22326, 0, 0,
    /*GILLT_v16s8*//*Label 790*/ 22350,
    /*GILLT_v16s16*//*Label 791*/ 22493, 0, 0,
    /*GILLT_v32s8*//*Label 792*/ 22517,
    // Label 785: @22138
    GIM_Try, /*On fail goto*//*Label 794*/ 22177, // Rule ID 17271 //
      GIM_CheckFeatures, GIFBS_In64BitMode,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (trunc:{ *:[i8] } GR32:{ *:[i32] }:$src)  =>  (EXTRACT_SUBREG:{ *:[i8] } GR32:{ *:[i32] }:$src, sub_8bit:{ *:[i32] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopySubReg, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, /*SubRegIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR8*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/1, /*RC GR32*/27,
      // GIR_Coverage, 17271,
      GIR_Done,
    // Label 794: @22177
    GIM_Try, /*On fail goto*//*Label 795*/ 22216, // Rule ID 17272 //
      GIM_CheckFeatures, GIFBS_In64BitMode,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      // (trunc:{ *:[i8] } GR16:{ *:[i16] }:$src)  =>  (EXTRACT_SUBREG:{ *:[i8] } GR16:{ *:[i16] }:$src, sub_8bit:{ *:[i32] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopySubReg, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, /*SubRegIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR8*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/1, /*RC GR16*/6,
      // GIR_Coverage, 17272,
      GIR_Done,
    // Label 795: @22216
    GIM_Reject,
    // Label 786: @22217
    GIM_Try, /*On fail goto*//*Label 796*/ 22254, // Rule ID 17265 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (trunc:{ *:[i16] } GR32:{ *:[i32] }:$src)  =>  (EXTRACT_SUBREG:{ *:[i16] } GR32:{ *:[i32] }:$src, sub_16bit:{ *:[i32] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopySubReg, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, /*SubRegIdx*/4, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC GR16*/6,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/1, /*RC GR32*/27,
      // GIR_Coverage, 17265,
      GIR_Done,
    // Label 796: @22254
    GIM_Reject,
    // Label 787: @22255
    GIM_Try, /*On fail goto*//*Label 797*/ 22278, // Rule ID 9415 //
      GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (trunc:{ *:[v4i32] } VR256X:{ *:[v4i64] }:$src1)  =>  (VPMOVQDZ256rr:{ *:[v4i32] } VR256X:{ *:[v4i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVQDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9415,
      GIR_Done,
    // Label 797: @22278
    GIM_Reject,
    // Label 788: @22279
    GIM_Try, /*On fail goto*//*Label 798*/ 22302, // Rule ID 9391 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (trunc:{ *:[v8i16] } VR512:{ *:[v8i64] }:$src1)  =>  (VPMOVQWZrr:{ *:[v8i16] } VR512:{ *:[v8i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVQWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9391,
      GIR_Done,
    // Label 798: @22302
    GIM_Try, /*On fail goto*//*Label 799*/ 22325, // Rule ID 9469 //
      GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (trunc:{ *:[v8i16] } VR256X:{ *:[v8i32] }:$src1)  =>  (VPMOVDWZ256rr:{ *:[v8i16] } VR256X:{ *:[v8i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVDWZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9469,
      GIR_Done,
    // Label 799: @22325
    GIM_Reject,
    // Label 789: @22326
    GIM_Try, /*On fail goto*//*Label 800*/ 22349, // Rule ID 9418 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (trunc:{ *:[v8i32] } VR512:{ *:[v8i64] }:$src1)  =>  (VPMOVQDZrr:{ *:[v8i32] } VR512:{ *:[v8i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVQDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9418,
      GIR_Done,
    // Label 800: @22349
    GIM_Reject,
    // Label 790: @22350
    GIM_Try, /*On fail goto*//*Label 801*/ 22373, // Rule ID 9445 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (trunc:{ *:[v16i8] } VR512:{ *:[v16i32] }:$src1)  =>  (VPMOVDBZrr:{ *:[v16i8] } VR512:{ *:[v16i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVDBZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9445,
      GIR_Done,
    // Label 801: @22373
    GIM_Try, /*On fail goto*//*Label 802*/ 22396, // Rule ID 9496 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (trunc:{ *:[v16i8] } VR256X:{ *:[v16i16] }:$src1)  =>  (VPMOVWBZ256rr:{ *:[v16i8] } VR256X:{ *:[v16i16] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVWBZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9496,
      GIR_Done,
    // Label 802: @22396
    GIM_Try, /*On fail goto*//*Label 803*/ 22444, // Rule ID 16386 //
      GIM_CheckFeatures, GIFBS_HasAVX512_NoBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (trunc:{ *:[v16i8] } VR256X:{ *:[v16i16] }:$src)  =>  (VPMOVDBZrr:{ *:[v16i8] } (VPMOVZXWDZrr:{ *:[v16i32] } VR256X:{ *:[v16i16] }:$src))
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::VPMOVZXWDZrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMOVDBZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16386,
      GIR_Done,
    // Label 803: @22444
    GIM_Try, /*On fail goto*//*Label 804*/ 22492, // Rule ID 16389 //
      GIM_CheckFeatures, GIFBS_HasAVX512_NoBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (trunc:{ *:[v16i8] } VR256X:{ *:[v16i16] }:$src)  =>  (VPMOVDBZrr:{ *:[v16i8] } (VPMOVZXWDZrr:{ *:[v16i32] } VR256X:{ *:[v16i16] }:$src))
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::VPMOVZXWDZrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMOVDBZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16389,
      GIR_Done,
    // Label 804: @22492
    GIM_Reject,
    // Label 791: @22493
    GIM_Try, /*On fail goto*//*Label 805*/ 22516, // Rule ID 9472 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (trunc:{ *:[v16i16] } VR512:{ *:[v16i32] }:$src1)  =>  (VPMOVDWZrr:{ *:[v16i16] } VR512:{ *:[v16i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVDWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9472,
      GIR_Done,
    // Label 805: @22516
    GIM_Reject,
    // Label 792: @22517
    GIM_Try, /*On fail goto*//*Label 806*/ 22540, // Rule ID 9499 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (trunc:{ *:[v32i8] } VR512:{ *:[v32i16] }:$src1)  =>  (VPMOVWBZrr:{ *:[v32i8] } VR512:{ *:[v32i16] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVWBZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9499,
      GIR_Done,
    // Label 806: @22540
    GIM_Reject,
    // Label 793: @22541
    GIM_Reject,
    // Label 11: @22542
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 5, /*)*//*default:*//*Label 811*/ 22744,
    /*GILLT_s8*//*Label 807*/ 22552,
    /*GILLT_s16*//*Label 808*/ 22574,
    /*GILLT_s32*//*Label 809*/ 22596,
    /*GILLT_s64*//*Label 810*/ 22693,
    // Label 807: @22552
    GIM_Try, /*On fail goto*//*Label 812*/ 22573, // Rule ID 19 //
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (imm:{ *:[i8] }):$src  =>  (MOV8ri:{ *:[i8] } (imm:{ *:[i8] }):$src)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV8ri,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 19,
      GIR_Done,
    // Label 812: @22573
    GIM_Reject,
    // Label 808: @22574
    GIM_Try, /*On fail goto*//*Label 813*/ 22595, // Rule ID 20 //
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (imm:{ *:[i16] }):$src  =>  (MOV16ri:{ *:[i16] } (imm:{ *:[i16] }):$src)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV16ri,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 20,
      GIR_Done,
    // Label 813: @22595
    GIM_Reject,
    // Label 809: @22596
    GIM_Try, /*On fail goto*//*Label 814*/ 22618, // Rule ID 12089 //
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      // MIs[0] Operand 1
      GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, 0,
      // 0:{ *:[i32] }  =>  (MOV32r0:{ *:[i32] }:{ *:[i32] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV32r0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12089,
      GIR_Done,
    // Label 814: @22618
    GIM_Try, /*On fail goto*//*Label 815*/ 22642, // Rule ID 12090 //
      GIM_CheckFeatures, GIFBS_Not64BitMode_OptForSize,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      // MIs[0] Operand 1
      GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, 1,
      // 1:{ *:[i32] }  =>  (MOV32r1:{ *:[i32] }:{ *:[i32] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV32r1,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12090,
      GIR_Done,
    // Label 815: @22642
    GIM_Try, /*On fail goto*//*Label 816*/ 22666, // Rule ID 12091 //
      GIM_CheckFeatures, GIFBS_Not64BitMode_OptForSize,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      // MIs[0] Operand 1
      GIM_CheckLiteralInt, /*MI*/0, /*Op*/1, -1,
      // -1:{ *:[i32] }  =>  (MOV32r_1:{ *:[i32] }:{ *:[i32] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV32r_1,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12091,
      GIR_Done,
    // Label 816: @22666
    GIM_Try, /*On fail goto*//*Label 817*/ 22692, // Rule ID 12092 //
      GIM_CheckFeatures, GIFBS_NotWin64WithoutFP_OptForMinSize,
      GIM_CheckI64ImmPredicate, /*MI*/0, /*Predicate*/GIPFP_I64_Predicate_i32immSExt8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (imm:{ *:[i32] })<<P:Predicate_i32immSExt8>>:$src  =>  (MOV32ImmSExti8:{ *:[i32] } (imm:{ *:[i32] }):$src)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV32ImmSExti8,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12092,
      GIR_Done,
    // Label 817: @22692
    GIM_Reject,
    // Label 810: @22693
    GIM_Try, /*On fail goto*//*Label 818*/ 22719, // Rule ID 12093 //
      GIM_CheckFeatures, GIFBS_NotWin64WithoutFP_OptForMinSize,
      GIM_CheckI64ImmPredicate, /*MI*/0, /*Predicate*/GIPFP_I64_Predicate_i64immSExt8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (imm:{ *:[i64] })<<P:Predicate_i64immSExt8>>:$src  =>  (MOV64ImmSExti8:{ *:[i64] } (imm:{ *:[i64] }):$src)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV64ImmSExti8,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12093,
      GIR_Done,
    // Label 818: @22719
    GIM_Try, /*On fail goto*//*Label 819*/ 22743, // Rule ID 22 //
      GIM_CheckI64ImmPredicate, /*MI*/0, /*Predicate*/GIPFP_I64_Predicate_i64immSExt32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (imm:{ *:[i64] })<<P:Predicate_i64immSExt32>>:$src  =>  (MOV64ri32:{ *:[i64] } (imm:{ *:[i64] }):$src)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::MOV64ri32,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/0, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 22,
      GIR_Done,
    // Label 819: @22743
    GIM_Reject,
    // Label 811: @22744
    GIM_Reject,
    // Label 12: @22745
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 6, /*)*//*default:*//*Label 823*/ 22937,
    /*GILLT_s32*//*Label 820*/ 22754,
    /*GILLT_s64*//*Label 821*/ 22801,
    /*GILLT_s80*//*Label 822*/ 22894,
    // Label 820: @22754
    GIM_Try, /*On fail goto*//*Label 824*/ 22777, // Rule ID 887 //
      GIM_CheckFeatures, GIFBS_FPStackf32,
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm0,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f32] })<<P:Predicate_fpimm0>>  =>  (LD_Fp032:{ *:[f32] }:{ *:[i16] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LD_Fp032,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 887,
      GIR_Done,
    // Label 824: @22777
    GIM_Try, /*On fail goto*//*Label 825*/ 22800, // Rule ID 888 //
      GIM_CheckFeatures, GIFBS_FPStackf32,
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f32] })<<P:Predicate_fpimm1>>  =>  (LD_Fp132:{ *:[f32] }:{ *:[i16] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LD_Fp132,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 888,
      GIR_Done,
    // Label 825: @22800
    GIM_Reject,
    // Label 821: @22801
    GIM_Try, /*On fail goto*//*Label 826*/ 22824, // Rule ID 889 //
      GIM_CheckFeatures, GIFBS_FPStackf64,
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm0,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f64] })<<P:Predicate_fpimm0>>  =>  (LD_Fp064:{ *:[f64] }:{ *:[i16] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LD_Fp064,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 889,
      GIR_Done,
    // Label 826: @22824
    GIM_Try, /*On fail goto*//*Label 827*/ 22847, // Rule ID 890 //
      GIM_CheckFeatures, GIFBS_FPStackf64,
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f64] })<<P:Predicate_fpimm1>>  =>  (LD_Fp164:{ *:[f64] }:{ *:[i16] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LD_Fp164,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 890,
      GIR_Done,
    // Label 827: @22847
    GIM_Try, /*On fail goto*//*Label 828*/ 22870, // Rule ID 1277 //
      GIM_CheckFeatures, GIFBS_HasSSE2_NoAVX512,
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm0,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f64] })<<P:Predicate_fpimm0>>  =>  (FsFLD0SD:{ *:[f64] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::FsFLD0SD,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1277,
      GIR_Done,
    // Label 828: @22870
    GIM_Try, /*On fail goto*//*Label 829*/ 22893, // Rule ID 2780 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm0,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f64] })<<P:Predicate_fpimm0>>  =>  (AVX512_FsFLD0SD:{ *:[f64] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::AVX512_FsFLD0SD,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 2780,
      GIR_Done,
    // Label 829: @22893
    GIM_Reject,
    // Label 822: @22894
    GIM_Try, /*On fail goto*//*Label 830*/ 22915, // Rule ID 891 //
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm0,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f80] })<<P:Predicate_fpimm0>>  =>  (LD_Fp080:{ *:[f80] }:{ *:[i16] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LD_Fp080,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 891,
      GIR_Done,
    // Label 830: @22915
    GIM_Try, /*On fail goto*//*Label 831*/ 22936, // Rule ID 892 //
      GIM_CheckAPFloatImmPredicate, /*MI*/0, /*Predicate*/GIPFP_APFloat_Predicate_fpimm1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      // MIs[0] Operand 1
      // No operand predicates
      // (fpimm:{ *:[f80] })<<P:Predicate_fpimm1>>  =>  (LD_Fp180:{ *:[f80] }:{ *:[i16] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::LD_Fp180,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 892,
      GIR_Done,
    // Label 831: @22936
    GIM_Reject,
    // Label 823: @22937
    GIM_Reject,
    // Label 13: @22938
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 25, /*)*//*default:*//*Label 846*/ 23742,
    /*GILLT_s32*//*Label 832*/ 22966,
    /*GILLT_s64*//*Label 833*/ 23009, 0, 0, 0,
    /*GILLT_v2s64*//*Label 834*/ 23075, 0,
    /*GILLT_v4s32*//*Label 835*/ 23099,
    /*GILLT_v4s64*//*Label 836*/ 23123, 0,
    /*GILLT_v8s16*//*Label 837*/ 23193,
    /*GILLT_v8s32*//*Label 838*/ 23256,
    /*GILLT_v8s64*//*Label 839*/ 23326, 0,
    /*GILLT_v16s8*//*Label 840*/ 23396,
    /*GILLT_v16s16*//*Label 841*/ 23459,
    /*GILLT_v16s32*//*Label 842*/ 23577, 0,
    /*GILLT_v32s8*//*Label 843*/ 23647,
    /*GILLT_v32s16*//*Label 844*/ 23671, 0,
    /*GILLT_v64s8*//*Label 845*/ 23718,
    // Label 832: @22966
    GIM_Try, /*On fail goto*//*Label 847*/ 22987, // Rule ID 532 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      // (sext:{ *:[i32] } GR8:{ *:[i8] }:$src)  =>  (MOVSX32rr8:{ *:[i32] } GR8:{ *:[i8] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSX32rr8,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 532,
      GIR_Done,
    // Label 847: @22987
    GIM_Try, /*On fail goto*//*Label 848*/ 23008, // Rule ID 534 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      // (sext:{ *:[i32] } GR16:{ *:[i16] }:$src)  =>  (MOVSX32rr16:{ *:[i32] } GR16:{ *:[i16] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSX32rr16,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 534,
      GIR_Done,
    // Label 848: @23008
    GIM_Reject,
    // Label 833: @23009
    GIM_Try, /*On fail goto*//*Label 849*/ 23030, // Rule ID 540 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      // (sext:{ *:[i64] } GR8:{ *:[i8] }:$src)  =>  (MOVSX64rr8:{ *:[i64] } GR8:{ *:[i8] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSX64rr8,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 540,
      GIR_Done,
    // Label 849: @23030
    GIM_Try, /*On fail goto*//*Label 850*/ 23051, // Rule ID 542 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      // (sext:{ *:[i64] } GR16:{ *:[i16] }:$src)  =>  (MOVSX64rr16:{ *:[i64] } GR16:{ *:[i16] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSX64rr16,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 542,
      GIR_Done,
    // Label 850: @23051
    GIM_Try, /*On fail goto*//*Label 851*/ 23074, // Rule ID 544 //
      GIM_CheckFeatures, GIFBS_In64BitMode,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sext:{ *:[i64] } GR32:{ *:[i32] }:$src)  =>  (MOVSX64rr32:{ *:[i64] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVSX64rr32,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 544,
      GIR_Done,
    // Label 851: @23074
    GIM_Reject,
    // Label 834: @23075
    GIM_Try, /*On fail goto*//*Label 852*/ 23098, // Rule ID 9795 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK2RegClassID,
      // (sext:{ *:[v2i64] } VK2:{ *:[v2i1] }:$src)  =>  (VPMOVM2QZ128rr:{ *:[v2i64] } VK2:{ *:[v2i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2QZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9795,
      GIR_Done,
    // Label 852: @23098
    GIM_Reject,
    // Label 835: @23099
    GIM_Try, /*On fail goto*//*Label 853*/ 23122, // Rule ID 9792 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK4RegClassID,
      // (sext:{ *:[v4i32] } VK4:{ *:[v4i1] }:$src)  =>  (VPMOVM2DZ128rr:{ *:[v4i32] } VK4:{ *:[v4i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2DZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9792,
      GIR_Done,
    // Label 853: @23122
    GIM_Reject,
    // Label 836: @23123
    GIM_Try, /*On fail goto*//*Label 854*/ 23146, // Rule ID 9724 //
      GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (sext:{ *:[v4i64] } VR128X:{ *:[v4i32] }:$src)  =>  (VPMOVSXDQZ256rr:{ *:[v4i64] } VR128X:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVSXDQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9724,
      GIR_Done,
    // Label 854: @23146
    GIM_Try, /*On fail goto*//*Label 855*/ 23169, // Rule ID 9794 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK4RegClassID,
      // (sext:{ *:[v4i64] } VK4:{ *:[v4i1] }:$src)  =>  (VPMOVM2QZ256rr:{ *:[v4i64] } VK4:{ *:[v4i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2QZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9794,
      GIR_Done,
    // Label 855: @23169
    GIM_Try, /*On fail goto*//*Label 856*/ 23192, // Rule ID 12766 //
      GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (sext:{ *:[v4i64] } VR128:{ *:[v4i32] }:$src)  =>  (VPMOVSXDQYrr:{ *:[v4i64] } VR128:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVSXDQYrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12766,
      GIR_Done,
    // Label 856: @23192
    GIM_Reject,
    // Label 837: @23193
    GIM_Try, /*On fail goto*//*Label 857*/ 23255,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      GIM_Try, /*On fail goto*//*Label 858*/ 23218, // Rule ID 9789 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        // (sext:{ *:[v8i16] } VK8:{ *:[v8i1] }:$src)  =>  (VPMOVM2WZ128rr:{ *:[v8i16] } VK8:{ *:[v8i1] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2WZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 9789,
        GIR_Done,
      // Label 858: @23218
      GIM_Try, /*On fail goto*//*Label 859*/ 23254, // Rule ID 16416 //
        GIM_CheckFeatures, GIFBS_HasDQI_HasVLX_NoBWI,
        // (sext:{ *:[v8i16] } VK8:{ *:[v8i1] }:$src)  =>  (VPMOVDWZ256rr:{ *:[v8i16] } (VPMOVM2DZ256rr:{ *:[v8i32] } VK8:{ *:[v8i1] }:$src))
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v8s32,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::VPMOVM2DZ256rr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMOVDWZ256rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16416,
        GIR_Done,
      // Label 859: @23254
      GIM_Reject,
    // Label 857: @23255
    GIM_Reject,
    // Label 838: @23256
    GIM_Try, /*On fail goto*//*Label 860*/ 23279, // Rule ID 9688 //
      GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (sext:{ *:[v8i32] } VR128X:{ *:[v8i16] }:$src)  =>  (VPMOVSXWDZ256rr:{ *:[v8i32] } VR128X:{ *:[v8i16] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVSXWDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9688,
      GIR_Done,
    // Label 860: @23279
    GIM_Try, /*On fail goto*//*Label 861*/ 23302, // Rule ID 9791 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      // (sext:{ *:[v8i32] } VK8:{ *:[v8i1] }:$src)  =>  (VPMOVM2DZ256rr:{ *:[v8i32] } VK8:{ *:[v8i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2DZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9791,
      GIR_Done,
    // Label 861: @23302
    GIM_Try, /*On fail goto*//*Label 862*/ 23325, // Rule ID 12765 //
      GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (sext:{ *:[v8i32] } VR128:{ *:[v8i16] }:$src)  =>  (VPMOVSXWDYrr:{ *:[v8i32] } VR128:{ *:[v8i16] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVSXWDYrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12765,
      GIR_Done,
    // Label 862: @23325
    GIM_Reject,
    // Label 839: @23326
    GIM_Try, /*On fail goto*//*Label 863*/ 23349, // Rule ID 9712 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (sext:{ *:[v8i64] } VR128X:{ *:[v8i16] }:$src)  =>  (VPMOVSXWQZrr:{ *:[v8i64] } VR128X:{ *:[v8i16] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVSXWQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9712,
      GIR_Done,
    // Label 863: @23349
    GIM_Try, /*On fail goto*//*Label 864*/ 23372, // Rule ID 9730 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (sext:{ *:[v8i64] } VR256X:{ *:[v8i32] }:$src)  =>  (VPMOVSXDQZrr:{ *:[v8i64] } VR256X:{ *:[v8i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVSXDQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9730,
      GIR_Done,
    // Label 864: @23372
    GIM_Try, /*On fail goto*//*Label 865*/ 23395, // Rule ID 9793 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK8RegClassID,
      // (sext:{ *:[v8i64] } VK8:{ *:[v8i1] }:$src)  =>  (VPMOVM2QZrr:{ *:[v8i64] } VK8:{ *:[v8i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2QZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9793,
      GIR_Done,
    // Label 865: @23395
    GIM_Reject,
    // Label 840: @23396
    GIM_Try, /*On fail goto*//*Label 866*/ 23458,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      GIM_Try, /*On fail goto*//*Label 867*/ 23421, // Rule ID 9786 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        // (sext:{ *:[v16i8] } VK16:{ *:[v16i1] }:$src)  =>  (VPMOVM2BZ128rr:{ *:[v16i8] } VK16:{ *:[v16i1] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2BZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 9786,
        GIR_Done,
      // Label 867: @23421
      GIM_Try, /*On fail goto*//*Label 868*/ 23457, // Rule ID 16412 //
        GIM_CheckFeatures, GIFBS_HasDQI_NoBWI,
        // (sext:{ *:[v16i8] } VK16:{ *:[v16i1] }:$src)  =>  (VPMOVDBZrr:{ *:[v16i8] } (VPMOVM2DZrr:{ *:[v16i32] } VK16:{ *:[v16i1] }:$src))
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s32,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::VPMOVM2DZrr,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMOVDBZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16412,
        GIR_Done,
      // Label 868: @23457
      GIM_Reject,
    // Label 866: @23458
    GIM_Reject,
    // Label 841: @23459
    GIM_Try, /*On fail goto*//*Label 869*/ 23482, // Rule ID 9634 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (sext:{ *:[v16i16] } VR128X:{ *:[v16i8] }:$src)  =>  (VPMOVSXBWZ256rr:{ *:[v16i16] } VR128X:{ *:[v16i8] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVSXBWZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9634,
      GIR_Done,
    // Label 869: @23482
    GIM_Try, /*On fail goto*//*Label 870*/ 23505, // Rule ID 9788 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      // (sext:{ *:[v16i16] } VK16:{ *:[v16i1] }:$src)  =>  (VPMOVM2WZ256rr:{ *:[v16i16] } VK16:{ *:[v16i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2WZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9788,
      GIR_Done,
    // Label 870: @23505
    GIM_Try, /*On fail goto*//*Label 871*/ 23528, // Rule ID 12764 //
      GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX_Or_NoBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (sext:{ *:[v16i16] } VR128:{ *:[v16i8] }:$src)  =>  (VPMOVSXBWYrr:{ *:[v16i16] } VR128:{ *:[v16i8] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVSXBWYrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12764,
      GIR_Done,
    // Label 871: @23528
    GIM_Try, /*On fail goto*//*Label 872*/ 23576, // Rule ID 16413 //
      GIM_CheckFeatures, GIFBS_HasDQI_NoBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      // (sext:{ *:[v16i16] } VK16:{ *:[v16i1] }:$src)  =>  (VPMOVDWZrr:{ *:[v16i16] } (VPMOVM2DZrr:{ *:[v16i32] } VK16:{ *:[v16i1] }:$src))
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_v16s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/X86::VPMOVM2DZrr,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_Copy, /*NewInsnID*/1, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VPMOVDWZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16413,
      GIR_Done,
    // Label 872: @23576
    GIM_Reject,
    // Label 842: @23577
    GIM_Try, /*On fail goto*//*Label 873*/ 23600, // Rule ID 9658 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (sext:{ *:[v16i32] } VR128X:{ *:[v16i8] }:$src)  =>  (VPMOVSXBDZrr:{ *:[v16i32] } VR128X:{ *:[v16i8] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVSXBDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9658,
      GIR_Done,
    // Label 873: @23600
    GIM_Try, /*On fail goto*//*Label 874*/ 23623, // Rule ID 9694 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (sext:{ *:[v16i32] } VR256X:{ *:[v16i16] }:$src)  =>  (VPMOVSXWDZrr:{ *:[v16i32] } VR256X:{ *:[v16i16] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVSXWDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9694,
      GIR_Done,
    // Label 874: @23623
    GIM_Try, /*On fail goto*//*Label 875*/ 23646, // Rule ID 9790 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK16RegClassID,
      // (sext:{ *:[v16i32] } VK16:{ *:[v16i1] }:$src)  =>  (VPMOVM2DZrr:{ *:[v16i32] } VK16:{ *:[v16i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2DZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9790,
      GIR_Done,
    // Label 875: @23646
    GIM_Reject,
    // Label 843: @23647
    GIM_Try, /*On fail goto*//*Label 876*/ 23670, // Rule ID 9785 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK32RegClassID,
      // (sext:{ *:[v32i8] } VK32:{ *:[v32i1] }:$src)  =>  (VPMOVM2BZ256rr:{ *:[v32i8] } VK32:{ *:[v32i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2BZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9785,
      GIR_Done,
    // Label 876: @23670
    GIM_Reject,
    // Label 844: @23671
    GIM_Try, /*On fail goto*//*Label 877*/ 23694, // Rule ID 9640 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (sext:{ *:[v32i16] } VR256X:{ *:[v32i8] }:$src)  =>  (VPMOVSXBWZrr:{ *:[v32i16] } VR256X:{ *:[v32i8] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVSXBWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9640,
      GIR_Done,
    // Label 877: @23694
    GIM_Try, /*On fail goto*//*Label 878*/ 23717, // Rule ID 9787 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK32RegClassID,
      // (sext:{ *:[v32i16] } VK32:{ *:[v32i1] }:$src)  =>  (VPMOVM2WZrr:{ *:[v32i16] } VK32:{ *:[v32i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2WZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9787,
      GIR_Done,
    // Label 878: @23717
    GIM_Reject,
    // Label 845: @23718
    GIM_Try, /*On fail goto*//*Label 879*/ 23741, // Rule ID 9784 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s1,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VK64RegClassID,
      // (sext:{ *:[v64i8] } VK64:{ *:[v64i1] }:$src)  =>  (VPMOVM2BZrr:{ *:[v64i8] } VK64:{ *:[v64i1] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVM2BZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9784,
      GIR_Done,
    // Label 879: @23741
    GIM_Reject,
    // Label 846: @23742
    GIM_Reject,
    // Label 14: @23743
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 23, /*)*//*default:*//*Label 887*/ 24156,
    /*GILLT_s32*//*Label 880*/ 23769, 0, 0, 0, 0, 0, 0, 0,
    /*GILLT_v4s64*//*Label 881*/ 23900, 0, 0,
    /*GILLT_v8s32*//*Label 882*/ 23946,
    /*GILLT_v8s64*//*Label 883*/ 23992, 0, 0,
    /*GILLT_v16s16*//*Label 884*/ 24039,
    /*GILLT_v16s32*//*Label 885*/ 24085, 0, 0,
    /*GILLT_v32s16*//*Label 886*/ 24132,
    // Label 880: @23769
    GIM_Try, /*On fail goto*//*Label 888*/ 23812, // Rule ID 13841 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
      GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_BITCAST,
      GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v16s1,
      GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK16RegClassID,
      GIM_CheckIsSafeToFold, /*InsnID*/1,
      // (zext:{ *:[i32] } (bitconvert:{ *:[i16] } VK16:{ *:[v16i1] }:$src))  =>  (KMOVWrk:{ *:[i32] } VK16:{ *:[v16i1] }:$src)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::KMOVWrk,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 13841,
      GIR_Done,
    // Label 888: @23812
    GIM_Try, /*On fail goto*//*Label 889*/ 23857, // Rule ID 13843 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/1, // MIs[1]
      GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_BITCAST,
      GIM_CheckType, /*MI*/1, /*Op*/1, /*Type*/GILLT_v8s1,
      GIM_CheckRegBankForClass, /*MI*/1, /*Op*/1, /*RC*/X86::VK8RegClassID,
      GIM_CheckIsSafeToFold, /*InsnID*/1,
      // (zext:{ *:[i32] } (bitconvert:{ *:[i8] } VK8:{ *:[v8i1] }:$src))  =>  (KMOVBrk:{ *:[i32] } VK8:{ *:[v8i1] }:$src)
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::KMOVBrk,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/1, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 13843,
      GIR_Done,
    // Label 889: @23857
    GIM_Try, /*On fail goto*//*Label 890*/ 23878, // Rule ID 536 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      // (zext:{ *:[i32] } GR8:{ *:[i8] }:$src)  =>  (MOVZX32rr8:{ *:[i32] } GR8:{ *:[i8] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVZX32rr8,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 536,
      GIR_Done,
    // Label 890: @23878
    GIM_Try, /*On fail goto*//*Label 891*/ 23899, // Rule ID 538 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      // (zext:{ *:[i32] } GR16:{ *:[i16] }:$src)  =>  (MOVZX32rr16:{ *:[i32] } GR16:{ *:[i16] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MOVZX32rr16,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 538,
      GIR_Done,
    // Label 891: @23899
    GIM_Reject,
    // Label 881: @23900
    GIM_Try, /*On fail goto*//*Label 892*/ 23945,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 893*/ 23925, // Rule ID 9616 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        // (zext:{ *:[v4i64] } VR128X:{ *:[v4i32] }:$src)  =>  (VPMOVZXDQZ256rr:{ *:[v4i64] } VR128X:{ *:[v4i32] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXDQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 9616,
        GIR_Done,
      // Label 893: @23925
      GIM_Try, /*On fail goto*//*Label 894*/ 23944, // Rule ID 12799 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        // (zext:{ *:[v4i64] } VR128:{ *:[v4i32] }:$src)  =>  (VPMOVZXDQYrr:{ *:[v4i64] } VR128:{ *:[v4i32] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXDQYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12799,
        GIR_Done,
      // Label 894: @23944
      GIM_Reject,
    // Label 892: @23945
    GIM_Reject,
    // Label 882: @23946
    GIM_Try, /*On fail goto*//*Label 895*/ 23991,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_Try, /*On fail goto*//*Label 896*/ 23971, // Rule ID 9580 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        // (zext:{ *:[v8i32] } VR128X:{ *:[v8i16] }:$src)  =>  (VPMOVZXWDZ256rr:{ *:[v8i32] } VR128X:{ *:[v8i16] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXWDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 9580,
        GIR_Done,
      // Label 896: @23971
      GIM_Try, /*On fail goto*//*Label 897*/ 23990, // Rule ID 12798 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        // (zext:{ *:[v8i32] } VR128:{ *:[v8i16] }:$src)  =>  (VPMOVZXWDYrr:{ *:[v8i32] } VR128:{ *:[v8i16] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXWDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12798,
        GIR_Done,
      // Label 897: @23990
      GIM_Reject,
    // Label 895: @23991
    GIM_Reject,
    // Label 883: @23992
    GIM_Try, /*On fail goto*//*Label 898*/ 24015, // Rule ID 9604 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (zext:{ *:[v8i64] } VR128X:{ *:[v8i16] }:$src)  =>  (VPMOVZXWQZrr:{ *:[v8i64] } VR128X:{ *:[v8i16] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXWQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9604,
      GIR_Done,
    // Label 898: @24015
    GIM_Try, /*On fail goto*//*Label 899*/ 24038, // Rule ID 9622 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (zext:{ *:[v8i64] } VR256X:{ *:[v8i32] }:$src)  =>  (VPMOVZXDQZrr:{ *:[v8i64] } VR256X:{ *:[v8i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXDQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9622,
      GIR_Done,
    // Label 899: @24038
    GIM_Reject,
    // Label 884: @24039
    GIM_Try, /*On fail goto*//*Label 900*/ 24084,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_Try, /*On fail goto*//*Label 901*/ 24064, // Rule ID 9526 //
        GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        // (zext:{ *:[v16i16] } VR128X:{ *:[v16i8] }:$src)  =>  (VPMOVZXBWZ256rr:{ *:[v16i16] } VR128X:{ *:[v16i8] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXBWZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 9526,
        GIR_Done,
      // Label 901: @24064
      GIM_Try, /*On fail goto*//*Label 902*/ 24083, // Rule ID 12797 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX_Or_NoBWI,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        // (zext:{ *:[v16i16] } VR128:{ *:[v16i8] }:$src)  =>  (VPMOVZXBWYrr:{ *:[v16i16] } VR128:{ *:[v16i8] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXBWYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12797,
        GIR_Done,
      // Label 902: @24083
      GIM_Reject,
    // Label 900: @24084
    GIM_Reject,
    // Label 885: @24085
    GIM_Try, /*On fail goto*//*Label 903*/ 24108, // Rule ID 9550 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (zext:{ *:[v16i32] } VR128X:{ *:[v16i8] }:$src)  =>  (VPMOVZXBDZrr:{ *:[v16i32] } VR128X:{ *:[v16i8] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXBDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9550,
      GIR_Done,
    // Label 903: @24108
    GIM_Try, /*On fail goto*//*Label 904*/ 24131, // Rule ID 9586 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (zext:{ *:[v16i32] } VR256X:{ *:[v16i16] }:$src)  =>  (VPMOVZXWDZrr:{ *:[v16i32] } VR256X:{ *:[v16i16] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXWDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9586,
      GIR_Done,
    // Label 904: @24131
    GIM_Reject,
    // Label 886: @24132
    GIM_Try, /*On fail goto*//*Label 905*/ 24155, // Rule ID 9532 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (zext:{ *:[v32i16] } VR256X:{ *:[v32i8] }:$src)  =>  (VPMOVZXBWZrr:{ *:[v32i16] } VR256X:{ *:[v32i8] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPMOVZXBWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 9532,
      GIR_Done,
    // Label 905: @24155
    GIM_Reject,
    // Label 887: @24156
    GIM_Reject,
    // Label 15: @24157
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 23, /*)*//*default:*//*Label 919*/ 24881,
    /*GILLT_s8*//*Label 906*/ 24185,
    /*GILLT_s16*//*Label 907*/ 24261,
    /*GILLT_s32*//*Label 908*/ 24337,
    /*GILLT_s64*//*Label 909*/ 24413, 0, 0, 0,
    /*GILLT_v2s64*//*Label 910*/ 24489, 0,
    /*GILLT_v4s32*//*Label 911*/ 24547,
    /*GILLT_v4s64*//*Label 912*/ 24605, 0,
    /*GILLT_v8s16*//*Label 913*/ 24663,
    /*GILLT_v8s32*//*Label 914*/ 24695,
    /*GILLT_v8s64*//*Label 915*/ 24753, 0, 0,
    /*GILLT_v16s16*//*Label 916*/ 24785,
    /*GILLT_v16s32*//*Label 917*/ 24817, 0, 0,
    /*GILLT_v32s16*//*Label 918*/ 24849,
    // Label 906: @24185
    GIM_Try, /*On fail goto*//*Label 920*/ 24260,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 921*/ 24229, // Rule ID 17287 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (shl:{ *:[i8] } GR8:{ *:[i8] }:$src1, 1:{ *:[i8] })  =>  (ADD8rr:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, GR8:{ *:[i8] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD8rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17287,
        GIR_Done,
      // Label 921: @24229
      GIM_Try, /*On fail goto*//*Label 922*/ 24259, // Rule ID 595 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (shl:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHL8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHL8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 595,
        GIR_Done,
      // Label 922: @24259
      GIM_Reject,
    // Label 920: @24260
    GIM_Reject,
    // Label 907: @24261
    GIM_Try, /*On fail goto*//*Label 923*/ 24336,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 924*/ 24305, // Rule ID 17288 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (shl:{ *:[i16] } GR16:{ *:[i16] }:$src1, 1:{ *:[i8] })  =>  (ADD16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, GR16:{ *:[i16] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD16rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17288,
        GIR_Done,
      // Label 924: @24305
      GIM_Try, /*On fail goto*//*Label 925*/ 24335, // Rule ID 596 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (shl:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHL16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHL16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 596,
        GIR_Done,
      // Label 925: @24335
      GIM_Reject,
    // Label 923: @24336
    GIM_Reject,
    // Label 908: @24337
    GIM_Try, /*On fail goto*//*Label 926*/ 24412,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      GIM_Try, /*On fail goto*//*Label 927*/ 24381, // Rule ID 17289 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (shl:{ *:[i32] } GR32:{ *:[i32] }:$src1, 1:{ *:[i8] })  =>  (ADD32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, GR32:{ *:[i32] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD32rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17289,
        GIR_Done,
      // Label 927: @24381
      GIM_Try, /*On fail goto*//*Label 928*/ 24411, // Rule ID 597 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (shl:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHL32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHL32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 597,
        GIR_Done,
      // Label 928: @24411
      GIM_Reject,
    // Label 926: @24412
    GIM_Reject,
    // Label 909: @24413
    GIM_Try, /*On fail goto*//*Label 929*/ 24488,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      GIM_Try, /*On fail goto*//*Label 930*/ 24457, // Rule ID 17290 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (shl:{ *:[i64] } GR64:{ *:[i64] }:$src1, 1:{ *:[i8] })  =>  (ADD64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, GR64:{ *:[i64] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ADD64rr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 17290,
        GIR_Done,
      // Label 930: @24457
      GIM_Try, /*On fail goto*//*Label 931*/ 24487, // Rule ID 598 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (shl:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHL64ri:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHL64ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 598,
        GIR_Done,
      // Label 931: @24487
      GIM_Reject,
    // Label 929: @24488
    GIM_Reject,
    // Label 910: @24489
    GIM_Try, /*On fail goto*//*Label 932*/ 24546,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 933*/ 24522, // Rule ID 2723 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (shl:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPSLLVQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVQrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2723,
        GIR_Done,
      // Label 933: @24522
      GIM_Try, /*On fail goto*//*Label 934*/ 24545, // Rule ID 6421 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (shl:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPSLLVQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6421,
        GIR_Done,
      // Label 934: @24545
      GIM_Reject,
    // Label 932: @24546
    GIM_Reject,
    // Label 911: @24547
    GIM_Try, /*On fail goto*//*Label 935*/ 24604,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 936*/ 24580, // Rule ID 2719 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (shl:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPSLLVDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2719,
        GIR_Done,
      // Label 936: @24580
      GIM_Try, /*On fail goto*//*Label 937*/ 24603, // Rule ID 6394 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (shl:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPSLLVDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6394,
        GIR_Done,
      // Label 937: @24603
      GIM_Reject,
    // Label 935: @24604
    GIM_Reject,
    // Label 912: @24605
    GIM_Try, /*On fail goto*//*Label 938*/ 24662,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 939*/ 24638, // Rule ID 2725 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (shl:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPSLLVQYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVQYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2725,
        GIR_Done,
      // Label 939: @24638
      GIM_Try, /*On fail goto*//*Label 940*/ 24661, // Rule ID 6412 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (shl:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPSLLVQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6412,
        GIR_Done,
      // Label 940: @24661
      GIM_Reject,
    // Label 938: @24662
    GIM_Reject,
    // Label 913: @24663
    GIM_Try, /*On fail goto*//*Label 941*/ 24694, // Rule ID 6442 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
      // (shl:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPSLLVWZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVWZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6442,
      GIR_Done,
    // Label 941: @24694
    GIM_Reject,
    // Label 914: @24695
    GIM_Try, /*On fail goto*//*Label 942*/ 24752,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 943*/ 24728, // Rule ID 2721 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (shl:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPSLLVDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2721,
        GIR_Done,
      // Label 943: @24728
      GIM_Try, /*On fail goto*//*Label 944*/ 24751, // Rule ID 6385 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (shl:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPSLLVDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6385,
        GIR_Done,
      // Label 944: @24751
      GIM_Reject,
    // Label 942: @24752
    GIM_Reject,
    // Label 915: @24753
    GIM_Try, /*On fail goto*//*Label 945*/ 24784, // Rule ID 6403 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (shl:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPSLLVQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6403,
      GIR_Done,
    // Label 945: @24784
    GIM_Reject,
    // Label 916: @24785
    GIM_Try, /*On fail goto*//*Label 946*/ 24816, // Rule ID 6436 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
      // (shl:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPSLLVWZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVWZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6436,
      GIR_Done,
    // Label 946: @24816
    GIM_Reject,
    // Label 917: @24817
    GIM_Try, /*On fail goto*//*Label 947*/ 24848, // Rule ID 6376 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (shl:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPSLLVDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6376,
      GIR_Done,
    // Label 947: @24848
    GIM_Reject,
    // Label 918: @24849
    GIM_Try, /*On fail goto*//*Label 948*/ 24880, // Rule ID 6430 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (shl:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPSLLVWZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSLLVWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6430,
      GIR_Done,
    // Label 948: @24880
    GIM_Reject,
    // Label 919: @24881
    GIM_Reject,
    // Label 16: @24882
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 23, /*)*//*default:*//*Label 962*/ 25590,
    /*GILLT_s8*//*Label 949*/ 24910,
    /*GILLT_s16*//*Label 950*/ 24982,
    /*GILLT_s32*//*Label 951*/ 25054,
    /*GILLT_s64*//*Label 952*/ 25126, 0, 0, 0,
    /*GILLT_v2s64*//*Label 953*/ 25198, 0,
    /*GILLT_v4s32*//*Label 954*/ 25256,
    /*GILLT_v4s64*//*Label 955*/ 25314, 0,
    /*GILLT_v8s16*//*Label 956*/ 25372,
    /*GILLT_v8s32*//*Label 957*/ 25404,
    /*GILLT_v8s64*//*Label 958*/ 25462, 0, 0,
    /*GILLT_v16s16*//*Label 959*/ 25494,
    /*GILLT_v16s32*//*Label 960*/ 25526, 0, 0,
    /*GILLT_v32s16*//*Label 961*/ 25558,
    // Label 949: @24910
    GIM_Try, /*On fail goto*//*Label 963*/ 24981,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 964*/ 24950, // Rule ID 619 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (srl:{ *:[i8] } GR8:{ *:[i8] }:$src1, 1:{ *:[i8] })  =>  (SHR8r1:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR8r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 619,
        GIR_Done,
      // Label 964: @24950
      GIM_Try, /*On fail goto*//*Label 965*/ 24980, // Rule ID 615 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (srl:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHR8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 615,
        GIR_Done,
      // Label 965: @24980
      GIM_Reject,
    // Label 963: @24981
    GIM_Reject,
    // Label 950: @24982
    GIM_Try, /*On fail goto*//*Label 966*/ 25053,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 967*/ 25022, // Rule ID 620 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (srl:{ *:[i16] } GR16:{ *:[i16] }:$src1, 1:{ *:[i8] })  =>  (SHR16r1:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR16r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 620,
        GIR_Done,
      // Label 967: @25022
      GIM_Try, /*On fail goto*//*Label 968*/ 25052, // Rule ID 616 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (srl:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHR16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 616,
        GIR_Done,
      // Label 968: @25052
      GIM_Reject,
    // Label 966: @25053
    GIM_Reject,
    // Label 951: @25054
    GIM_Try, /*On fail goto*//*Label 969*/ 25125,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      GIM_Try, /*On fail goto*//*Label 970*/ 25094, // Rule ID 621 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (srl:{ *:[i32] } GR32:{ *:[i32] }:$src1, 1:{ *:[i8] })  =>  (SHR32r1:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR32r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 621,
        GIR_Done,
      // Label 970: @25094
      GIM_Try, /*On fail goto*//*Label 971*/ 25124, // Rule ID 617 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (srl:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHR32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 617,
        GIR_Done,
      // Label 971: @25124
      GIM_Reject,
    // Label 969: @25125
    GIM_Reject,
    // Label 952: @25126
    GIM_Try, /*On fail goto*//*Label 972*/ 25197,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      GIM_Try, /*On fail goto*//*Label 973*/ 25166, // Rule ID 622 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (srl:{ *:[i64] } GR64:{ *:[i64] }:$src1, 1:{ *:[i8] })  =>  (SHR64r1:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR64r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 622,
        GIR_Done,
      // Label 973: @25166
      GIM_Try, /*On fail goto*//*Label 974*/ 25196, // Rule ID 618 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (srl:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SHR64ri:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SHR64ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 618,
        GIR_Done,
      // Label 974: @25196
      GIM_Reject,
    // Label 972: @25197
    GIM_Reject,
    // Label 953: @25198
    GIM_Try, /*On fail goto*//*Label 975*/ 25255,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 976*/ 25231, // Rule ID 2731 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (srl:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)  =>  (VPSRLVQrr:{ *:[v2i64] } VR128:{ *:[v2i64] }:$src1, VR128:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVQrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2731,
        GIR_Done,
      // Label 976: @25231
      GIM_Try, /*On fail goto*//*Label 977*/ 25254, // Rule ID 6565 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (srl:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPSRLVQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVQZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6565,
        GIR_Done,
      // Label 977: @25254
      GIM_Reject,
    // Label 975: @25255
    GIM_Reject,
    // Label 954: @25256
    GIM_Try, /*On fail goto*//*Label 978*/ 25313,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 979*/ 25289, // Rule ID 2727 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (srl:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPSRLVDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2727,
        GIR_Done,
      // Label 979: @25289
      GIM_Try, /*On fail goto*//*Label 980*/ 25312, // Rule ID 6538 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (srl:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPSRLVDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6538,
        GIR_Done,
      // Label 980: @25312
      GIM_Reject,
    // Label 978: @25313
    GIM_Reject,
    // Label 955: @25314
    GIM_Try, /*On fail goto*//*Label 981*/ 25371,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 982*/ 25347, // Rule ID 2733 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (srl:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)  =>  (VPSRLVQYrr:{ *:[v4i64] } VR256:{ *:[v4i64] }:$src1, VR256:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVQYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2733,
        GIR_Done,
      // Label 982: @25347
      GIM_Try, /*On fail goto*//*Label 983*/ 25370, // Rule ID 6556 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (srl:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPSRLVQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVQZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6556,
        GIR_Done,
      // Label 983: @25370
      GIM_Reject,
    // Label 981: @25371
    GIM_Reject,
    // Label 956: @25372
    GIM_Try, /*On fail goto*//*Label 984*/ 25403, // Rule ID 6586 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
      // (srl:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPSRLVWZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVWZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6586,
      GIR_Done,
    // Label 984: @25403
    GIM_Reject,
    // Label 957: @25404
    GIM_Try, /*On fail goto*//*Label 985*/ 25461,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 986*/ 25437, // Rule ID 2729 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (srl:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPSRLVDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2729,
        GIR_Done,
      // Label 986: @25437
      GIM_Try, /*On fail goto*//*Label 987*/ 25460, // Rule ID 6529 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (srl:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPSRLVDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6529,
        GIR_Done,
      // Label 987: @25460
      GIM_Reject,
    // Label 985: @25461
    GIM_Reject,
    // Label 958: @25462
    GIM_Try, /*On fail goto*//*Label 988*/ 25493, // Rule ID 6547 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (srl:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPSRLVQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6547,
      GIR_Done,
    // Label 988: @25493
    GIM_Reject,
    // Label 959: @25494
    GIM_Try, /*On fail goto*//*Label 989*/ 25525, // Rule ID 6580 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
      // (srl:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPSRLVWZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVWZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6580,
      GIR_Done,
    // Label 989: @25525
    GIM_Reject,
    // Label 960: @25526
    GIM_Try, /*On fail goto*//*Label 990*/ 25557, // Rule ID 6520 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (srl:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPSRLVDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6520,
      GIR_Done,
    // Label 990: @25557
    GIM_Reject,
    // Label 961: @25558
    GIM_Try, /*On fail goto*//*Label 991*/ 25589, // Rule ID 6574 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (srl:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPSRLVWZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRLVWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6574,
      GIR_Done,
    // Label 991: @25589
    GIM_Reject,
    // Label 962: @25590
    GIM_Reject,
    // Label 17: @25591
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/1, 23, /*)*//*default:*//*Label 1005*/ 26247,
    /*GILLT_s8*//*Label 992*/ 25619,
    /*GILLT_s16*//*Label 993*/ 25691,
    /*GILLT_s32*//*Label 994*/ 25763,
    /*GILLT_s64*//*Label 995*/ 25835, 0, 0, 0,
    /*GILLT_v2s64*//*Label 996*/ 25907, 0,
    /*GILLT_v4s32*//*Label 997*/ 25939,
    /*GILLT_v4s64*//*Label 998*/ 25997, 0,
    /*GILLT_v8s16*//*Label 999*/ 26029,
    /*GILLT_v8s32*//*Label 1000*/ 26061,
    /*GILLT_v8s64*//*Label 1001*/ 26119, 0, 0,
    /*GILLT_v16s16*//*Label 1002*/ 26151,
    /*GILLT_v16s32*//*Label 1003*/ 26183, 0, 0,
    /*GILLT_v32s16*//*Label 1004*/ 26215,
    // Label 992: @25619
    GIM_Try, /*On fail goto*//*Label 1006*/ 25690,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s8,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR8RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR8RegClassID,
      GIM_Try, /*On fail goto*//*Label 1007*/ 25659, // Rule ID 643 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (sra:{ *:[i8] } GR8:{ *:[i8] }:$src1, 1:{ *:[i8] })  =>  (SAR8r1:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR8r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 643,
        GIR_Done,
      // Label 1007: @25659
      GIM_Try, /*On fail goto*//*Label 1008*/ 25689, // Rule ID 639 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sra:{ *:[i8] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SAR8ri:{ *:[i8] }:{ *:[i32] } GR8:{ *:[i8] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR8ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 639,
        GIR_Done,
      // Label 1008: @25689
      GIM_Reject,
    // Label 1006: @25690
    GIM_Reject,
    // Label 993: @25691
    GIM_Try, /*On fail goto*//*Label 1009*/ 25762,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      GIM_Try, /*On fail goto*//*Label 1010*/ 25731, // Rule ID 644 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (sra:{ *:[i16] } GR16:{ *:[i16] }:$src1, 1:{ *:[i8] })  =>  (SAR16r1:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR16r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 644,
        GIR_Done,
      // Label 1010: @25731
      GIM_Try, /*On fail goto*//*Label 1011*/ 25761, // Rule ID 640 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sra:{ *:[i16] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SAR16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR16ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 640,
        GIR_Done,
      // Label 1011: @25761
      GIM_Reject,
    // Label 1009: @25762
    GIM_Reject,
    // Label 994: @25763
    GIM_Try, /*On fail goto*//*Label 1012*/ 25834,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      GIM_Try, /*On fail goto*//*Label 1013*/ 25803, // Rule ID 645 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (sra:{ *:[i32] } GR32:{ *:[i32] }:$src1, 1:{ *:[i8] })  =>  (SAR32r1:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR32r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 645,
        GIR_Done,
      // Label 1013: @25803
      GIM_Try, /*On fail goto*//*Label 1014*/ 25833, // Rule ID 641 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sra:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SAR32ri:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR32ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 641,
        GIR_Done,
      // Label 1014: @25833
      GIM_Reject,
    // Label 1012: @25834
    GIM_Reject,
    // Label 995: @25835
    GIM_Try, /*On fail goto*//*Label 1015*/ 25906,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      GIM_Try, /*On fail goto*//*Label 1016*/ 25875, // Rule ID 646 //
        GIM_CheckConstantInt, /*MI*/0, /*Op*/2, 1,
        // (sra:{ *:[i64] } GR64:{ *:[i64] }:$src1, 1:{ *:[i8] })  =>  (SAR64r1:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR64r1,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 646,
        GIR_Done,
      // Label 1016: @25875
      GIM_Try, /*On fail goto*//*Label 1017*/ 25905, // Rule ID 642 //
        GIM_RecordInsn, /*DefineMI*/1, /*MI*/0, /*OpIdx*/2, // MIs[1]
        GIM_CheckOpcode, /*MI*/1, TargetOpcode::G_CONSTANT,
        // MIs[1] Operand 1
        // No operand predicates
        GIM_CheckIsSafeToFold, /*InsnID*/1,
        // (sra:{ *:[i64] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i8] }):$src2)  =>  (SAR64ri:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src1, (imm:{ *:[i8] }):$src2)
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::SAR64ri,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src1
        GIR_CopyConstantAsSImm, /*NewInsnID*/0, /*OldInsnID*/1, // src2
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 642,
        GIR_Done,
      // Label 1017: @25905
      GIM_Reject,
    // Label 1015: @25906
    GIM_Reject,
    // Label 996: @25907
    GIM_Try, /*On fail goto*//*Label 1018*/ 25938, // Rule ID 6493 //
      GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
      // (sra:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)  =>  (VPSRAVQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1, VR128X:{ *:[v2i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6493,
      GIR_Done,
    // Label 1018: @25938
    GIM_Reject,
    // Label 997: @25939
    GIM_Try, /*On fail goto*//*Label 1019*/ 25996,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 1020*/ 25972, // Rule ID 2735 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (sra:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)  =>  (VPSRAVDrr:{ *:[v4i32] } VR128:{ *:[v4i32] }:$src1, VR128:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2735,
        GIR_Done,
      // Label 1020: @25972
      GIM_Try, /*On fail goto*//*Label 1021*/ 25995, // Rule ID 6466 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (sra:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)  =>  (VPSRAVDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1, VR128X:{ *:[v4i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6466,
        GIR_Done,
      // Label 1021: @25995
      GIM_Reject,
    // Label 1019: @25996
    GIM_Reject,
    // Label 998: @25997
    GIM_Try, /*On fail goto*//*Label 1022*/ 26028, // Rule ID 6484 //
      GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
      // (sra:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)  =>  (VPSRAVQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1, VR256X:{ *:[v4i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6484,
      GIR_Done,
    // Label 1022: @26028
    GIM_Reject,
    // Label 999: @26029
    GIM_Try, /*On fail goto*//*Label 1023*/ 26060, // Rule ID 6514 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
      // (sra:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)  =>  (VPSRAVWZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1, VR128X:{ *:[v8i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVWZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6514,
      GIR_Done,
    // Label 1023: @26060
    GIM_Reject,
    // Label 1000: @26061
    GIM_Try, /*On fail goto*//*Label 1024*/ 26118,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 1025*/ 26094, // Rule ID 2737 //
        GIM_CheckFeatures, GIFBS_HasAVX2_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (sra:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)  =>  (VPSRAVDYrr:{ *:[v8i32] } VR256:{ *:[v8i32] }:$src1, VR256:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 2737,
        GIR_Done,
      // Label 1025: @26094
      GIM_Try, /*On fail goto*//*Label 1026*/ 26117, // Rule ID 6457 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (sra:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)  =>  (VPSRAVDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1, VR256X:{ *:[v8i32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 6457,
        GIR_Done,
      // Label 1026: @26117
      GIM_Reject,
    // Label 1024: @26118
    GIM_Reject,
    // Label 1001: @26119
    GIM_Try, /*On fail goto*//*Label 1027*/ 26150, // Rule ID 6475 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sra:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)  =>  (VPSRAVQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1, VR512:{ *:[v8i64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6475,
      GIR_Done,
    // Label 1027: @26150
    GIM_Reject,
    // Label 1002: @26151
    GIM_Try, /*On fail goto*//*Label 1028*/ 26182, // Rule ID 6508 //
      GIM_CheckFeatures, GIFBS_HasBWI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
      // (sra:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)  =>  (VPSRAVWZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1, VR256X:{ *:[v16i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVWZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6508,
      GIR_Done,
    // Label 1028: @26182
    GIM_Reject,
    // Label 1003: @26183
    GIM_Try, /*On fail goto*//*Label 1029*/ 26214, // Rule ID 6448 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sra:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)  =>  (VPSRAVDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1, VR512:{ *:[v16i32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6448,
      GIR_Done,
    // Label 1029: @26214
    GIM_Reject,
    // Label 1004: @26215
    GIM_Try, /*On fail goto*//*Label 1030*/ 26246, // Rule ID 6502 //
      GIM_CheckFeatures, GIFBS_HasBWI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (sra:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)  =>  (VPSRAVWZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1, VR512:{ *:[v32i16] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPSRAVWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6502,
      GIR_Done,
    // Label 1030: @26246
    GIM_Reject,
    // Label 1005: @26247
    GIM_Reject,
    // Label 18: @26248
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 1040*/ 26860,
    /*GILLT_s32*//*Label 1031*/ 26271,
    /*GILLT_s64*//*Label 1032*/ 26378,
    /*GILLT_s80*//*Label 1033*/ 26485, 0, 0,
    /*GILLT_v2s64*//*Label 1034*/ 26518, 0,
    /*GILLT_v4s32*//*Label 1035*/ 26599,
    /*GILLT_v4s64*//*Label 1036*/ 26680, 0, 0,
    /*GILLT_v8s32*//*Label 1037*/ 26738,
    /*GILLT_v8s64*//*Label 1038*/ 26796, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1039*/ 26828,
    // Label 1031: @26271
    GIM_Try, /*On fail goto*//*Label 1041*/ 26377,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 1042*/ 26307, // Rule ID 740 //
        GIM_CheckFeatures, GIFBS_FPStackf32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP32RegClassID,
        // (fadd:{ *:[f32] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)  =>  (ADD_Fp32:{ *:[f32] }:{ *:[i16] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD_Fp32,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 740,
        GIR_Done,
      // Label 1042: @26307
      GIM_Try, /*On fail goto*//*Label 1043*/ 26330, // Rule ID 1577 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fadd:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (VADDSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1577,
        GIR_Done,
      // Label 1043: @26330
      GIM_Try, /*On fail goto*//*Label 1044*/ 26353, // Rule ID 1581 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fadd:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (ADDSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADDSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1581,
        GIR_Done,
      // Label 1044: @26353
      GIM_Try, /*On fail goto*//*Label 1045*/ 26376, // Rule ID 5090 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32XRegClassID,
        // (fadd:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)  =>  (VADDSSZrr:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDSSZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5090,
        GIR_Done,
      // Label 1045: @26376
      GIM_Reject,
    // Label 1041: @26377
    GIM_Reject,
    // Label 1032: @26378
    GIM_Try, /*On fail goto*//*Label 1046*/ 26484,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 1047*/ 26414, // Rule ID 741 //
        GIM_CheckFeatures, GIFBS_FPStackf64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP64RegClassID,
        // (fadd:{ *:[f64] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)  =>  (ADD_Fp64:{ *:[f64] }:{ *:[i16] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD_Fp64,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 741,
        GIR_Done,
      // Label 1047: @26414
      GIM_Try, /*On fail goto*//*Label 1048*/ 26437, // Rule ID 1579 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fadd:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (VADDSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1579,
        GIR_Done,
      // Label 1048: @26437
      GIM_Try, /*On fail goto*//*Label 1049*/ 26460, // Rule ID 1583 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fadd:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (ADDSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADDSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1583,
        GIR_Done,
      // Label 1049: @26460
      GIM_Try, /*On fail goto*//*Label 1050*/ 26483, // Rule ID 5101 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64XRegClassID,
        // (fadd:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)  =>  (VADDSDZrr:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDSDZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5101,
        GIR_Done,
      // Label 1050: @26483
      GIM_Reject,
    // Label 1046: @26484
    GIM_Reject,
    // Label 1033: @26485
    GIM_Try, /*On fail goto*//*Label 1051*/ 26517, // Rule ID 742 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP80RegClassID,
      // (fadd:{ *:[f80] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)  =>  (ADD_Fp80:{ *:[f80] }:{ *:[i16] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADD_Fp80,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 742,
      GIR_Done,
    // Label 1051: @26517
    GIM_Reject,
    // Label 1034: @26518
    GIM_Try, /*On fail goto*//*Label 1052*/ 26598,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 1053*/ 26551, // Rule ID 1567 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fadd:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (VADDPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1567,
        GIR_Done,
      // Label 1053: @26551
      GIM_Try, /*On fail goto*//*Label 1054*/ 26574, // Rule ID 1575 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fadd:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (ADDPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADDPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1575,
        GIR_Done,
      // Label 1054: @26574
      GIM_Try, /*On fail goto*//*Label 1055*/ 26597, // Rule ID 5260 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fadd:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)  =>  (VADDPDZ128rr:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5260,
        GIR_Done,
      // Label 1055: @26597
      GIM_Reject,
    // Label 1052: @26598
    GIM_Reject,
    // Label 1035: @26599
    GIM_Try, /*On fail goto*//*Label 1056*/ 26679,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 1057*/ 26632, // Rule ID 1565 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fadd:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (VADDPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1565,
        GIR_Done,
      // Label 1057: @26632
      GIM_Try, /*On fail goto*//*Label 1058*/ 26655, // Rule ID 1573 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fadd:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (ADDPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::ADDPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1573,
        GIR_Done,
      // Label 1058: @26655
      GIM_Try, /*On fail goto*//*Label 1059*/ 26678, // Rule ID 5242 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fadd:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)  =>  (VADDPSZ128rr:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPSZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5242,
        GIR_Done,
      // Label 1059: @26678
      GIM_Reject,
    // Label 1056: @26679
    GIM_Reject,
    // Label 1036: @26680
    GIM_Try, /*On fail goto*//*Label 1060*/ 26737,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 1061*/ 26713, // Rule ID 1571 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fadd:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)  =>  (VADDPDYrr:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1571,
        GIR_Done,
      // Label 1061: @26713
      GIM_Try, /*On fail goto*//*Label 1062*/ 26736, // Rule ID 5269 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fadd:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)  =>  (VADDPDZ256rr:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5269,
        GIR_Done,
      // Label 1062: @26736
      GIM_Reject,
    // Label 1060: @26737
    GIM_Reject,
    // Label 1037: @26738
    GIM_Try, /*On fail goto*//*Label 1063*/ 26795,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 1064*/ 26771, // Rule ID 1569 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fadd:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)  =>  (VADDPSYrr:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1569,
        GIR_Done,
      // Label 1064: @26771
      GIM_Try, /*On fail goto*//*Label 1065*/ 26794, // Rule ID 5251 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fadd:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)  =>  (VADDPSZ256rr:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPSZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5251,
        GIR_Done,
      // Label 1065: @26794
      GIM_Reject,
    // Label 1063: @26795
    GIM_Reject,
    // Label 1038: @26796
    GIM_Try, /*On fail goto*//*Label 1066*/ 26827, // Rule ID 5233 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fadd:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)  =>  (VADDPDZrr:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5233,
      GIR_Done,
    // Label 1066: @26827
    GIM_Reject,
    // Label 1039: @26828
    GIM_Try, /*On fail goto*//*Label 1067*/ 26859, // Rule ID 5224 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fadd:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)  =>  (VADDPSZrr:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VADDPSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5224,
      GIR_Done,
    // Label 1067: @26859
    GIM_Reject,
    // Label 1040: @26860
    GIM_Reject,
    // Label 19: @26861
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 1077*/ 27473,
    /*GILLT_s32*//*Label 1068*/ 26884,
    /*GILLT_s64*//*Label 1069*/ 26991,
    /*GILLT_s80*//*Label 1070*/ 27098, 0, 0,
    /*GILLT_v2s64*//*Label 1071*/ 27131, 0,
    /*GILLT_v4s32*//*Label 1072*/ 27212,
    /*GILLT_v4s64*//*Label 1073*/ 27293, 0, 0,
    /*GILLT_v8s32*//*Label 1074*/ 27351,
    /*GILLT_v8s64*//*Label 1075*/ 27409, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1076*/ 27441,
    // Label 1068: @26884
    GIM_Try, /*On fail goto*//*Label 1078*/ 26990,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 1079*/ 26920, // Rule ID 743 //
        GIM_CheckFeatures, GIFBS_FPStackf32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP32RegClassID,
        // (fsub:{ *:[f32] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)  =>  (SUB_Fp32:{ *:[f32] }:{ *:[i16] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB_Fp32,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 743,
        GIR_Done,
      // Label 1079: @26920
      GIM_Try, /*On fail goto*//*Label 1080*/ 26943, // Rule ID 1617 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fsub:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (VSUBSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1617,
        GIR_Done,
      // Label 1080: @26943
      GIM_Try, /*On fail goto*//*Label 1081*/ 26966, // Rule ID 1621 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fsub:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (SUBSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUBSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1621,
        GIR_Done,
      // Label 1081: @26966
      GIM_Try, /*On fail goto*//*Label 1082*/ 26989, // Rule ID 5134 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32XRegClassID,
        // (fsub:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)  =>  (VSUBSSZrr:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBSSZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5134,
        GIR_Done,
      // Label 1082: @26989
      GIM_Reject,
    // Label 1078: @26990
    GIM_Reject,
    // Label 1069: @26991
    GIM_Try, /*On fail goto*//*Label 1083*/ 27097,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 1084*/ 27027, // Rule ID 744 //
        GIM_CheckFeatures, GIFBS_FPStackf64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP64RegClassID,
        // (fsub:{ *:[f64] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)  =>  (SUB_Fp64:{ *:[f64] }:{ *:[i16] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB_Fp64,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 744,
        GIR_Done,
      // Label 1084: @27027
      GIM_Try, /*On fail goto*//*Label 1085*/ 27050, // Rule ID 1619 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fsub:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (VSUBSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1619,
        GIR_Done,
      // Label 1085: @27050
      GIM_Try, /*On fail goto*//*Label 1086*/ 27073, // Rule ID 1623 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fsub:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (SUBSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUBSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1623,
        GIR_Done,
      // Label 1086: @27073
      GIM_Try, /*On fail goto*//*Label 1087*/ 27096, // Rule ID 5145 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64XRegClassID,
        // (fsub:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)  =>  (VSUBSDZrr:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBSDZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5145,
        GIR_Done,
      // Label 1087: @27096
      GIM_Reject,
    // Label 1083: @27097
    GIM_Reject,
    // Label 1070: @27098
    GIM_Try, /*On fail goto*//*Label 1088*/ 27130, // Rule ID 745 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP80RegClassID,
      // (fsub:{ *:[f80] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)  =>  (SUB_Fp80:{ *:[f80] }:{ *:[i16] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUB_Fp80,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 745,
      GIR_Done,
    // Label 1088: @27130
    GIM_Reject,
    // Label 1071: @27131
    GIM_Try, /*On fail goto*//*Label 1089*/ 27211,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 1090*/ 27164, // Rule ID 1607 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fsub:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (VSUBPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1607,
        GIR_Done,
      // Label 1090: @27164
      GIM_Try, /*On fail goto*//*Label 1091*/ 27187, // Rule ID 1615 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fsub:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (SUBPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUBPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1615,
        GIR_Done,
      // Label 1091: @27187
      GIM_Try, /*On fail goto*//*Label 1092*/ 27210, // Rule ID 5380 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fsub:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)  =>  (VSUBPDZ128rr:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5380,
        GIR_Done,
      // Label 1092: @27210
      GIM_Reject,
    // Label 1089: @27211
    GIM_Reject,
    // Label 1072: @27212
    GIM_Try, /*On fail goto*//*Label 1093*/ 27292,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 1094*/ 27245, // Rule ID 1605 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fsub:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (VSUBPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1605,
        GIR_Done,
      // Label 1094: @27245
      GIM_Try, /*On fail goto*//*Label 1095*/ 27268, // Rule ID 1613 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fsub:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (SUBPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::SUBPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1613,
        GIR_Done,
      // Label 1095: @27268
      GIM_Try, /*On fail goto*//*Label 1096*/ 27291, // Rule ID 5362 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fsub:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)  =>  (VSUBPSZ128rr:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPSZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5362,
        GIR_Done,
      // Label 1096: @27291
      GIM_Reject,
    // Label 1093: @27292
    GIM_Reject,
    // Label 1073: @27293
    GIM_Try, /*On fail goto*//*Label 1097*/ 27350,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 1098*/ 27326, // Rule ID 1611 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fsub:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)  =>  (VSUBPDYrr:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1611,
        GIR_Done,
      // Label 1098: @27326
      GIM_Try, /*On fail goto*//*Label 1099*/ 27349, // Rule ID 5389 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fsub:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)  =>  (VSUBPDZ256rr:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5389,
        GIR_Done,
      // Label 1099: @27349
      GIM_Reject,
    // Label 1097: @27350
    GIM_Reject,
    // Label 1074: @27351
    GIM_Try, /*On fail goto*//*Label 1100*/ 27408,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 1101*/ 27384, // Rule ID 1609 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fsub:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)  =>  (VSUBPSYrr:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1609,
        GIR_Done,
      // Label 1101: @27384
      GIM_Try, /*On fail goto*//*Label 1102*/ 27407, // Rule ID 5371 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fsub:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)  =>  (VSUBPSZ256rr:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPSZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5371,
        GIR_Done,
      // Label 1102: @27407
      GIM_Reject,
    // Label 1100: @27408
    GIM_Reject,
    // Label 1075: @27409
    GIM_Try, /*On fail goto*//*Label 1103*/ 27440, // Rule ID 5353 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fsub:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)  =>  (VSUBPDZrr:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5353,
      GIR_Done,
    // Label 1103: @27440
    GIM_Reject,
    // Label 1076: @27441
    GIM_Try, /*On fail goto*//*Label 1104*/ 27472, // Rule ID 5344 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fsub:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)  =>  (VSUBPSZrr:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VSUBPSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5344,
      GIR_Done,
    // Label 1104: @27472
    GIM_Reject,
    // Label 1077: @27473
    GIM_Reject,
    // Label 20: @27474
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 1114*/ 28086,
    /*GILLT_s32*//*Label 1105*/ 27497,
    /*GILLT_s64*//*Label 1106*/ 27604,
    /*GILLT_s80*//*Label 1107*/ 27711, 0, 0,
    /*GILLT_v2s64*//*Label 1108*/ 27744, 0,
    /*GILLT_v4s32*//*Label 1109*/ 27825,
    /*GILLT_v4s64*//*Label 1110*/ 27906, 0, 0,
    /*GILLT_v8s32*//*Label 1111*/ 27964,
    /*GILLT_v8s64*//*Label 1112*/ 28022, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1113*/ 28054,
    // Label 1105: @27497
    GIM_Try, /*On fail goto*//*Label 1115*/ 27603,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 1116*/ 27533, // Rule ID 746 //
        GIM_CheckFeatures, GIFBS_FPStackf32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP32RegClassID,
        // (fmul:{ *:[f32] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)  =>  (MUL_Fp32:{ *:[f32] }:{ *:[i16] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MUL_Fp32,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 746,
        GIR_Done,
      // Label 1116: @27533
      GIM_Try, /*On fail goto*//*Label 1117*/ 27556, // Rule ID 1597 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fmul:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (VMULSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1597,
        GIR_Done,
      // Label 1117: @27556
      GIM_Try, /*On fail goto*//*Label 1118*/ 27579, // Rule ID 1601 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fmul:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (MULSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MULSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1601,
        GIR_Done,
      // Label 1118: @27579
      GIM_Try, /*On fail goto*//*Label 1119*/ 27602, // Rule ID 5112 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32XRegClassID,
        // (fmul:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)  =>  (VMULSSZrr:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULSSZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5112,
        GIR_Done,
      // Label 1119: @27602
      GIM_Reject,
    // Label 1115: @27603
    GIM_Reject,
    // Label 1106: @27604
    GIM_Try, /*On fail goto*//*Label 1120*/ 27710,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 1121*/ 27640, // Rule ID 747 //
        GIM_CheckFeatures, GIFBS_FPStackf64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP64RegClassID,
        // (fmul:{ *:[f64] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)  =>  (MUL_Fp64:{ *:[f64] }:{ *:[i16] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MUL_Fp64,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 747,
        GIR_Done,
      // Label 1121: @27640
      GIM_Try, /*On fail goto*//*Label 1122*/ 27663, // Rule ID 1599 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fmul:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (VMULSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1599,
        GIR_Done,
      // Label 1122: @27663
      GIM_Try, /*On fail goto*//*Label 1123*/ 27686, // Rule ID 1603 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fmul:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (MULSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MULSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1603,
        GIR_Done,
      // Label 1123: @27686
      GIM_Try, /*On fail goto*//*Label 1124*/ 27709, // Rule ID 5123 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64XRegClassID,
        // (fmul:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)  =>  (VMULSDZrr:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULSDZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5123,
        GIR_Done,
      // Label 1124: @27709
      GIM_Reject,
    // Label 1120: @27710
    GIM_Reject,
    // Label 1107: @27711
    GIM_Try, /*On fail goto*//*Label 1125*/ 27743, // Rule ID 748 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP80RegClassID,
      // (fmul:{ *:[f80] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)  =>  (MUL_Fp80:{ *:[f80] }:{ *:[i16] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MUL_Fp80,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 748,
      GIR_Done,
    // Label 1125: @27743
    GIM_Reject,
    // Label 1108: @27744
    GIM_Try, /*On fail goto*//*Label 1126*/ 27824,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 1127*/ 27777, // Rule ID 1587 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fmul:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (VMULPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1587,
        GIR_Done,
      // Label 1127: @27777
      GIM_Try, /*On fail goto*//*Label 1128*/ 27800, // Rule ID 1595 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fmul:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (MULPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MULPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1595,
        GIR_Done,
      // Label 1128: @27800
      GIM_Try, /*On fail goto*//*Label 1129*/ 27823, // Rule ID 5320 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fmul:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)  =>  (VMULPDZ128rr:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5320,
        GIR_Done,
      // Label 1129: @27823
      GIM_Reject,
    // Label 1126: @27824
    GIM_Reject,
    // Label 1109: @27825
    GIM_Try, /*On fail goto*//*Label 1130*/ 27905,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 1131*/ 27858, // Rule ID 1585 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fmul:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (VMULPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1585,
        GIR_Done,
      // Label 1131: @27858
      GIM_Try, /*On fail goto*//*Label 1132*/ 27881, // Rule ID 1593 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fmul:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (MULPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::MULPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1593,
        GIR_Done,
      // Label 1132: @27881
      GIM_Try, /*On fail goto*//*Label 1133*/ 27904, // Rule ID 5302 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fmul:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)  =>  (VMULPSZ128rr:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPSZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5302,
        GIR_Done,
      // Label 1133: @27904
      GIM_Reject,
    // Label 1130: @27905
    GIM_Reject,
    // Label 1110: @27906
    GIM_Try, /*On fail goto*//*Label 1134*/ 27963,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 1135*/ 27939, // Rule ID 1591 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fmul:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)  =>  (VMULPDYrr:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1591,
        GIR_Done,
      // Label 1135: @27939
      GIM_Try, /*On fail goto*//*Label 1136*/ 27962, // Rule ID 5329 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fmul:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)  =>  (VMULPDZ256rr:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5329,
        GIR_Done,
      // Label 1136: @27962
      GIM_Reject,
    // Label 1134: @27963
    GIM_Reject,
    // Label 1111: @27964
    GIM_Try, /*On fail goto*//*Label 1137*/ 28021,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 1138*/ 27997, // Rule ID 1589 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fmul:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)  =>  (VMULPSYrr:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1589,
        GIR_Done,
      // Label 1138: @27997
      GIM_Try, /*On fail goto*//*Label 1139*/ 28020, // Rule ID 5311 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fmul:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)  =>  (VMULPSZ256rr:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPSZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5311,
        GIR_Done,
      // Label 1139: @28020
      GIM_Reject,
    // Label 1137: @28021
    GIM_Reject,
    // Label 1112: @28022
    GIM_Try, /*On fail goto*//*Label 1140*/ 28053, // Rule ID 5293 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fmul:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)  =>  (VMULPDZrr:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5293,
      GIR_Done,
    // Label 1140: @28053
    GIM_Reject,
    // Label 1113: @28054
    GIM_Try, /*On fail goto*//*Label 1141*/ 28085, // Rule ID 5284 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fmul:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)  =>  (VMULPSZrr:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VMULPSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5284,
      GIR_Done,
    // Label 1141: @28085
    GIM_Reject,
    // Label 1114: @28086
    GIM_Reject,
    // Label 21: @28087
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 1151*/ 28699,
    /*GILLT_s32*//*Label 1142*/ 28110,
    /*GILLT_s64*//*Label 1143*/ 28217,
    /*GILLT_s80*//*Label 1144*/ 28324, 0, 0,
    /*GILLT_v2s64*//*Label 1145*/ 28357, 0,
    /*GILLT_v4s32*//*Label 1146*/ 28438,
    /*GILLT_v4s64*//*Label 1147*/ 28519, 0, 0,
    /*GILLT_v8s32*//*Label 1148*/ 28577,
    /*GILLT_v8s64*//*Label 1149*/ 28635, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1150*/ 28667,
    // Label 1142: @28110
    GIM_Try, /*On fail goto*//*Label 1152*/ 28216,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 1153*/ 28146, // Rule ID 749 //
        GIM_CheckFeatures, GIFBS_FPStackf32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP32RegClassID,
        // (fdiv:{ *:[f32] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)  =>  (DIV_Fp32:{ *:[f32] }:{ *:[i16] } RFP32:{ *:[f32] }:$src1, RFP32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIV_Fp32,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 749,
        GIR_Done,
      // Label 1153: @28146
      GIM_Try, /*On fail goto*//*Label 1154*/ 28169, // Rule ID 1637 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fdiv:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (VDIVSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1637,
        GIR_Done,
      // Label 1154: @28169
      GIM_Try, /*On fail goto*//*Label 1155*/ 28192, // Rule ID 1641 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32RegClassID,
        // (fdiv:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)  =>  (DIVSSrr:{ *:[f32] } FR32:{ *:[f32] }:$src1, FR32:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIVSSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1641,
        GIR_Done,
      // Label 1155: @28192
      GIM_Try, /*On fail goto*//*Label 1156*/ 28215, // Rule ID 5156 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR32XRegClassID,
        // (fdiv:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)  =>  (VDIVSSZrr:{ *:[f32] } FR32X:{ *:[f32] }:$src1, FR32X:{ *:[f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVSSZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5156,
        GIR_Done,
      // Label 1156: @28215
      GIM_Reject,
    // Label 1152: @28216
    GIM_Reject,
    // Label 1143: @28217
    GIM_Try, /*On fail goto*//*Label 1157*/ 28323,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s64,
      GIM_Try, /*On fail goto*//*Label 1158*/ 28253, // Rule ID 750 //
        GIM_CheckFeatures, GIFBS_FPStackf64,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP64RegClassID,
        // (fdiv:{ *:[f64] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)  =>  (DIV_Fp64:{ *:[f64] }:{ *:[i16] } RFP64:{ *:[f64] }:$src1, RFP64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIV_Fp64,
        GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 750,
        GIR_Done,
      // Label 1158: @28253
      GIM_Try, /*On fail goto*//*Label 1159*/ 28276, // Rule ID 1639 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fdiv:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (VDIVSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1639,
        GIR_Done,
      // Label 1159: @28276
      GIM_Try, /*On fail goto*//*Label 1160*/ 28299, // Rule ID 1643 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64RegClassID,
        // (fdiv:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)  =>  (DIVSDrr:{ *:[f64] } FR64:{ *:[f64] }:$src1, FR64:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIVSDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1643,
        GIR_Done,
      // Label 1160: @28299
      GIM_Try, /*On fail goto*//*Label 1161*/ 28322, // Rule ID 5167 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::FR64XRegClassID,
        // (fdiv:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)  =>  (VDIVSDZrr:{ *:[f64] } FR64X:{ *:[f64] }:$src1, FR64X:{ *:[f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVSDZrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5167,
        GIR_Done,
      // Label 1161: @28322
      GIM_Reject,
    // Label 1157: @28323
    GIM_Reject,
    // Label 1144: @28324
    GIM_Try, /*On fail goto*//*Label 1162*/ 28356, // Rule ID 751 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::RFP80RegClassID,
      // (fdiv:{ *:[f80] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)  =>  (DIV_Fp80:{ *:[f80] }:{ *:[i16] } RFP80:{ *:[f80] }:$src1, RFP80:{ *:[f80] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIV_Fp80,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 751,
      GIR_Done,
    // Label 1162: @28356
    GIM_Reject,
    // Label 1145: @28357
    GIM_Try, /*On fail goto*//*Label 1163*/ 28437,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v2s64,
      GIM_Try, /*On fail goto*//*Label 1164*/ 28390, // Rule ID 1627 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fdiv:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (VDIVPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1627,
        GIR_Done,
      // Label 1164: @28390
      GIM_Try, /*On fail goto*//*Label 1165*/ 28413, // Rule ID 1635 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fdiv:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)  =>  (DIVPDrr:{ *:[v2f64] } VR128:{ *:[v2f64] }:$src1, VR128:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIVPDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1635,
        GIR_Done,
      // Label 1165: @28413
      GIM_Try, /*On fail goto*//*Label 1166*/ 28436, // Rule ID 5440 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fdiv:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)  =>  (VDIVPDZ128rr:{ *:[v2f64] } VR128X:{ *:[v2f64] }:$src1, VR128X:{ *:[v2f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPDZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5440,
        GIR_Done,
      // Label 1166: @28436
      GIM_Reject,
    // Label 1163: @28437
    GIM_Reject,
    // Label 1146: @28438
    GIM_Try, /*On fail goto*//*Label 1167*/ 28518,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 1168*/ 28471, // Rule ID 1625 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fdiv:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (VDIVPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1625,
        GIR_Done,
      // Label 1168: @28471
      GIM_Try, /*On fail goto*//*Label 1169*/ 28494, // Rule ID 1633 //
        GIM_CheckFeatures, GIFBS_UseSSE1,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128RegClassID,
        // (fdiv:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)  =>  (DIVPSrr:{ *:[v4f32] } VR128:{ *:[v4f32] }:$src1, VR128:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::DIVPSrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1633,
        GIR_Done,
      // Label 1169: @28494
      GIM_Try, /*On fail goto*//*Label 1170*/ 28517, // Rule ID 5422 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR128XRegClassID,
        // (fdiv:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)  =>  (VDIVPSZ128rr:{ *:[v4f32] } VR128X:{ *:[v4f32] }:$src1, VR128X:{ *:[v4f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPSZ128rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5422,
        GIR_Done,
      // Label 1170: @28517
      GIM_Reject,
    // Label 1167: @28518
    GIM_Reject,
    // Label 1147: @28519
    GIM_Try, /*On fail goto*//*Label 1171*/ 28576,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 1172*/ 28552, // Rule ID 1631 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fdiv:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)  =>  (VDIVPDYrr:{ *:[v4f64] } VR256:{ *:[v4f64] }:$src1, VR256:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1631,
        GIR_Done,
      // Label 1172: @28552
      GIM_Try, /*On fail goto*//*Label 1173*/ 28575, // Rule ID 5449 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fdiv:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)  =>  (VDIVPDZ256rr:{ *:[v4f64] } VR256X:{ *:[v4f64] }:$src1, VR256X:{ *:[v4f64] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5449,
        GIR_Done,
      // Label 1173: @28575
      GIM_Reject,
    // Label 1171: @28576
    GIM_Reject,
    // Label 1148: @28577
    GIM_Try, /*On fail goto*//*Label 1174*/ 28634,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s32,
      GIM_Try, /*On fail goto*//*Label 1175*/ 28610, // Rule ID 1629 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256RegClassID,
        // (fdiv:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)  =>  (VDIVPSYrr:{ *:[v8f32] } VR256:{ *:[v8f32] }:$src1, VR256:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1629,
        GIR_Done,
      // Label 1175: @28610
      GIM_Try, /*On fail goto*//*Label 1176*/ 28633, // Rule ID 5431 //
        GIM_CheckFeatures, GIFBS_HasAVX512_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR256XRegClassID,
        // (fdiv:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)  =>  (VDIVPSZ256rr:{ *:[v8f32] } VR256X:{ *:[v8f32] }:$src1, VR256X:{ *:[v8f32] }:$src2)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPSZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 5431,
        GIR_Done,
      // Label 1176: @28633
      GIM_Reject,
    // Label 1174: @28634
    GIM_Reject,
    // Label 1149: @28635
    GIM_Try, /*On fail goto*//*Label 1177*/ 28666, // Rule ID 5413 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fdiv:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)  =>  (VDIVPDZrr:{ *:[v8f64] } VR512:{ *:[v8f64] }:$src1, VR512:{ *:[v8f64] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5413,
      GIR_Done,
    // Label 1177: @28666
    GIM_Reject,
    // Label 1150: @28667
    GIM_Try, /*On fail goto*//*Label 1178*/ 28698, // Rule ID 5404 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckType, /*MI*/0, /*Op*/2, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/2, /*RC*/X86::VR512RegClassID,
      // (fdiv:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)  =>  (VDIVPSZrr:{ *:[v16f32] } VR512:{ *:[v16f32] }:$src1, VR512:{ *:[v16f32] }:$src2)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VDIVPSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5404,
      GIR_Done,
    // Label 1178: @28698
    GIM_Reject,
    // Label 1151: @28699
    GIM_Reject,
    // Label 22: @28700
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 6, /*)*//*default:*//*Label 1182*/ 28788,
    /*GILLT_s32*//*Label 1179*/ 28709,
    /*GILLT_s64*//*Label 1180*/ 28736,
    /*GILLT_s80*//*Label 1181*/ 28763,
    // Label 1179: @28709
    GIM_Try, /*On fail goto*//*Label 1183*/ 28735, // Rule ID 818 //
      GIM_CheckFeatures, GIFBS_FPStackf32,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
      // (fneg:{ *:[f32] } RFP32:{ *:[f32] }:$src)  =>  (CHS_Fp32:{ *:[f32] }:{ *:[i16] } RFP32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CHS_Fp32,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 818,
      GIR_Done,
    // Label 1183: @28735
    GIM_Reject,
    // Label 1180: @28736
    GIM_Try, /*On fail goto*//*Label 1184*/ 28762, // Rule ID 819 //
      GIM_CheckFeatures, GIFBS_FPStackf64,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
      // (fneg:{ *:[f64] } RFP64:{ *:[f64] }:$src)  =>  (CHS_Fp64:{ *:[f64] }:{ *:[i16] } RFP64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CHS_Fp64,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 819,
      GIR_Done,
    // Label 1184: @28762
    GIM_Reject,
    // Label 1181: @28763
    GIM_Try, /*On fail goto*//*Label 1185*/ 28787, // Rule ID 820 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      // (fneg:{ *:[f80] } RFP80:{ *:[f80] }:$src)  =>  (CHS_Fp80:{ *:[f80] }:{ *:[i16] } RFP80:{ *:[f80] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CHS_Fp80,
      GIR_AddImplicitDef, /*InsnID*/0, X86::FPSW,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 820,
      GIR_Done,
    // Label 1185: @28787
    GIM_Reject,
    // Label 1182: @28788
    GIM_Reject,
    // Label 23: @28789
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/4, 16, /*)*//*default:*//*Label 1190*/ 29064,
    /*GILLT_s64*//*Label 1186*/ 28807,
    /*GILLT_s80*//*Label 1187*/ 28943, 0, 0, 0, 0, 0,
    /*GILLT_v4s64*//*Label 1188*/ 28994, 0, 0, 0,
    /*GILLT_v8s64*//*Label 1189*/ 29040,
    // Label 1186: @28807
    GIM_Try, /*On fail goto*//*Label 1191*/ 28942,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_Try, /*On fail goto*//*Label 1192*/ 28832, // Rule ID 1403 //
        GIM_CheckFeatures, GIFBS_UseSSE2,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        // (fpextend:{ *:[f64] } FR32:{ *:[f32] }:$src)  =>  (CVTSS2SDrr:{ *:[f64] } FR32:{ *:[f32] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTSS2SDrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1403,
        GIR_Done,
      // Label 1192: @28832
      GIM_Try, /*On fail goto*//*Label 1193*/ 28853, // Rule ID 12290 //
        GIM_CheckFeatures, GIFBS_FPStackf32,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
        // (fpextend:{ *:[f64] } RFP32:{ *:[f32] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[f64] } RFP32:{ *:[f32] }:$src, RFP64:{ *:[i32] })
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
        GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC RFP64*/57,
        // GIR_Coverage, 12290,
        GIR_Done,
      // Label 1193: @28853
      GIM_Try, /*On fail goto*//*Label 1194*/ 28897, // Rule ID 12460 //
        GIM_CheckFeatures, GIFBS_UseAVX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
        // (fpextend:{ *:[f64] } FR32:{ *:[f32] }:$src)  =>  (VCVTSS2SDrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), FR32:{ *:[f32] }:$src)
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSS2SDrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 12460,
        GIR_Done,
      // Label 1194: @28897
      GIM_Try, /*On fail goto*//*Label 1195*/ 28941, // Rule ID 16012 //
        GIM_CheckFeatures, GIFBS_HasAVX512,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
        // (fpextend:{ *:[f64] } FR32X:{ *:[f32] }:$src)  =>  (VCVTSS2SDZrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), FR32X:{ *:[f32] }:$src)
        GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
        GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
        GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
        GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSS2SDZrr,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
        GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
        GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
        GIR_EraseFromParent, /*InsnID*/0,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 16012,
        GIR_Done,
      // Label 1195: @28941
      GIM_Reject,
    // Label 1191: @28942
    GIM_Reject,
    // Label 1187: @28943
    GIM_Try, /*On fail goto*//*Label 1196*/ 28968, // Rule ID 12291 //
      GIM_CheckFeatures, GIFBS_FPStackf32,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP32RegClassID,
      // (fpextend:{ *:[f80] } RFP32:{ *:[f32] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[f80] } RFP32:{ *:[f32] }:$src, RFP80:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC RFP80*/103,
      // GIR_Coverage, 12291,
      GIR_Done,
    // Label 1196: @28968
    GIM_Try, /*On fail goto*//*Label 1197*/ 28993, // Rule ID 12292 //
      GIM_CheckFeatures, GIFBS_FPStackf64,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP80RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
      // (fpextend:{ *:[f80] } RFP64:{ *:[f64] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[f80] } RFP64:{ *:[f64] }:$src, RFP80:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC RFP80*/103,
      // GIR_Coverage, 12292,
      GIR_Done,
    // Label 1197: @28993
    GIM_Reject,
    // Label 1188: @28994
    GIM_Try, /*On fail goto*//*Label 1198*/ 29039,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_Try, /*On fail goto*//*Label 1199*/ 29019, // Rule ID 1431 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
        // (fpextend:{ *:[v4f64] } VR128:{ *:[v4f32] }:$src)  =>  (VCVTPS2PDYrr:{ *:[v4f64] } VR128:{ *:[v4f32] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTPS2PDYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1431,
        GIR_Done,
      // Label 1199: @29019
      GIM_Try, /*On fail goto*//*Label 1200*/ 29038, // Rule ID 8179 //
        GIM_CheckFeatures, GIFBS_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
        // (fpextend:{ *:[v4f64] } VR128X:{ *:[v4f32] }:$src)  =>  (VCVTPS2PDZ256rr:{ *:[v4f64] } VR128X:{ *:[v4f32] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTPS2PDZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 8179,
        GIR_Done,
      // Label 1200: @29038
      GIM_Reject,
    // Label 1198: @29039
    GIM_Reject,
    // Label 1189: @29040
    GIM_Try, /*On fail goto*//*Label 1201*/ 29063, // Rule ID 8158 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fpextend:{ *:[v8f64] } VR256X:{ *:[v8f32] }:$src)  =>  (VCVTPS2PDZrr:{ *:[v8f64] } VR256X:{ *:[v8f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTPS2PDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8158,
      GIR_Done,
    // Label 1201: @29063
    GIM_Reject,
    // Label 1190: @29064
    GIM_Reject,
    // Label 24: @29065
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 15, /*)*//*default:*//*Label 1206*/ 29349,
    /*GILLT_s32*//*Label 1202*/ 29083,
    /*GILLT_s64*//*Label 1203*/ 29253, 0, 0, 0, 0, 0,
    /*GILLT_v4s32*//*Label 1204*/ 29279, 0, 0, 0,
    /*GILLT_v8s32*//*Label 1205*/ 29325,
    // Label 1202: @29083
    GIM_Try, /*On fail goto*//*Label 1207*/ 29106, // Rule ID 1397 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (fpround:{ *:[f32] } FR64:{ *:[f64] }:$src)  =>  (CVTSD2SSrr:{ *:[f32] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTSD2SSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1397,
      GIR_Done,
    // Label 1207: @29106
    GIM_Try, /*On fail goto*//*Label 1208*/ 29131, // Rule ID 12293 //
      GIM_CheckFeatures, GIFBS_FPStackf32,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP64RegClassID,
      // (fpround:{ *:[f32] } RFP64:{ *:[f64] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[f32] } RFP64:{ *:[f64] }:$src, RFP32:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC RFP32*/33,
      // GIR_Coverage, 12293,
      GIR_Done,
    // Label 1208: @29131
    GIM_Try, /*On fail goto*//*Label 1209*/ 29156, // Rule ID 12294 //
      GIM_CheckFeatures, GIFBS_FPStackf32,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      // (fpround:{ *:[f32] } RFP80:{ *:[f80] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[f32] } RFP80:{ *:[f80] }:$src, RFP32:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC RFP32*/33,
      // GIR_Coverage, 12294,
      GIR_Done,
    // Label 1209: @29156
    GIM_Try, /*On fail goto*//*Label 1210*/ 29204, // Rule ID 12459 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (fpround:{ *:[f32] } FR64:{ *:[f64] }:$src)  =>  (VCVTSD2SSrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), FR64:{ *:[f64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSD2SSrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12459,
      GIR_Done,
    // Label 1210: @29204
    GIM_Try, /*On fail goto*//*Label 1211*/ 29252, // Rule ID 16016 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
      // (fpround:{ *:[f32] } FR64X:{ *:[f64] }:$src)  =>  (VCVTSD2SSZrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), FR64X:{ *:[f64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSD2SSZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16016,
      GIR_Done,
    // Label 1211: @29252
    GIM_Reject,
    // Label 1203: @29253
    GIM_Try, /*On fail goto*//*Label 1212*/ 29278, // Rule ID 12295 //
      GIM_CheckFeatures, GIFBS_FPStackf64,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s80,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::RFP64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::RFP80RegClassID,
      // (fpround:{ *:[f64] } RFP80:{ *:[f80] }:$src)  =>  (COPY_TO_REGCLASS:{ *:[f64] } RFP80:{ *:[f80] }:$src, RFP64:{ *:[i32] })
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/TargetOpcode::COPY,
      GIR_ConstrainOperandRC, /*InsnID*/0, /*Op*/0, /*RC RFP64*/57,
      // GIR_Coverage, 12295,
      GIR_Done,
    // Label 1212: @29278
    GIM_Reject,
    // Label 1204: @29279
    GIM_Try, /*On fail goto*//*Label 1213*/ 29324,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_Try, /*On fail goto*//*Label 1214*/ 29304, // Rule ID 1443 //
        GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
        // (fpround:{ *:[v4f32] } VR256:{ *:[v4f64] }:$src)  =>  (VCVTPD2PSYrr:{ *:[v4f32] } VR256:{ *:[v4f64] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTPD2PSYrr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 1443,
        GIR_Done,
      // Label 1214: @29304
      GIM_Try, /*On fail goto*//*Label 1215*/ 29323, // Rule ID 8149 //
        GIM_CheckFeatures, GIFBS_HasVLX,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
        GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
        // (fpround:{ *:[v4f32] } VR256X:{ *:[v4f64] }:$src)  =>  (VCVTPD2PSZ256rr:{ *:[v4f32] } VR256X:{ *:[v4f64] }:$src)
        GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTPD2PSZ256rr,
        GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
        // GIR_Coverage, 8149,
        GIR_Done,
      // Label 1215: @29323
      GIM_Reject,
    // Label 1213: @29324
    GIM_Reject,
    // Label 1205: @29325
    GIM_Try, /*On fail goto*//*Label 1216*/ 29348, // Rule ID 8128 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fpround:{ *:[v8f32] } VR512:{ *:[v8f64] }:$src)  =>  (VCVTPD2PSZrr:{ *:[v8f32] } VR512:{ *:[v8f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTPD2PSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8128,
      GIR_Done,
    // Label 1216: @29348
    GIM_Reject,
    // Label 1206: @29349
    GIM_Reject,
    // Label 25: @29350
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 1225*/ 29979,
    /*GILLT_s32*//*Label 1217*/ 29373,
    /*GILLT_s64*//*Label 1218*/ 29512, 0, 0, 0,
    /*GILLT_v2s64*//*Label 1219*/ 29651, 0,
    /*GILLT_v4s32*//*Label 1220*/ 29675,
    /*GILLT_v4s64*//*Label 1221*/ 29791, 0, 0,
    /*GILLT_v8s32*//*Label 1222*/ 29838,
    /*GILLT_v8s64*//*Label 1223*/ 29908, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1224*/ 29955,
    // Label 1217: @29373
    GIM_Try, /*On fail goto*//*Label 1226*/ 29396, // Rule ID 1335 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
      // (fp_to_sint:{ *:[i32] } FR32:{ *:[f32] }:$src)  =>  (VCVTTSS2SIrr:{ *:[i32] } FR32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSS2SIrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1335,
      GIR_Done,
    // Label 1226: @29396
    GIM_Try, /*On fail goto*//*Label 1227*/ 29419, // Rule ID 1339 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (fp_to_sint:{ *:[i32] } FR64:{ *:[f64] }:$src)  =>  (VCVTTSD2SIrr:{ *:[i32] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSD2SIrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1339,
      GIR_Done,
    // Label 1227: @29419
    GIM_Try, /*On fail goto*//*Label 1228*/ 29442, // Rule ID 1343 //
      GIM_CheckFeatures, GIFBS_UseSSE1,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
      // (fp_to_sint:{ *:[i32] } FR32:{ *:[f32] }:$src)  =>  (CVTTSS2SIrr:{ *:[i32] } FR32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTTSS2SIrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1343,
      GIR_Done,
    // Label 1228: @29442
    GIM_Try, /*On fail goto*//*Label 1229*/ 29465, // Rule ID 1347 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (fp_to_sint:{ *:[i32] } FR64:{ *:[f64] }:$src)  =>  (CVTTSD2SIrr:{ *:[i32] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTTSD2SIrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1347,
      GIR_Done,
    // Label 1229: @29465
    GIM_Try, /*On fail goto*//*Label 1230*/ 29488, // Rule ID 8070 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
      // (fp_to_sint:{ *:[i32] } FR32X:{ *:[f32] }:$src)  =>  (VCVTTSS2SIZrr:{ *:[i32] } FR32X:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSS2SIZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8070,
      GIR_Done,
    // Label 1230: @29488
    GIM_Try, /*On fail goto*//*Label 1231*/ 29511, // Rule ID 8080 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
      // (fp_to_sint:{ *:[i32] } FR64X:{ *:[f64] }:$src)  =>  (VCVTTSD2SIZrr:{ *:[i32] } FR64X:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSD2SIZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8080,
      GIR_Done,
    // Label 1231: @29511
    GIM_Reject,
    // Label 1218: @29512
    GIM_Try, /*On fail goto*//*Label 1232*/ 29535, // Rule ID 1337 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
      // (fp_to_sint:{ *:[i64] } FR32:{ *:[f32] }:$src)  =>  (VCVTTSS2SI64rr:{ *:[i64] } FR32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSS2SI64rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1337,
      GIR_Done,
    // Label 1232: @29535
    GIM_Try, /*On fail goto*//*Label 1233*/ 29558, // Rule ID 1341 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (fp_to_sint:{ *:[i64] } FR64:{ *:[f64] }:$src)  =>  (VCVTTSD2SI64rr:{ *:[i64] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSD2SI64rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1341,
      GIR_Done,
    // Label 1233: @29558
    GIM_Try, /*On fail goto*//*Label 1234*/ 29581, // Rule ID 1345 //
      GIM_CheckFeatures, GIFBS_UseSSE1,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32RegClassID,
      // (fp_to_sint:{ *:[i64] } FR32:{ *:[f32] }:$src)  =>  (CVTTSS2SI64rr:{ *:[i64] } FR32:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTTSS2SI64rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1345,
      GIR_Done,
    // Label 1234: @29581
    GIM_Try, /*On fail goto*//*Label 1235*/ 29604, // Rule ID 1349 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64RegClassID,
      // (fp_to_sint:{ *:[i64] } FR64:{ *:[f64] }:$src)  =>  (CVTTSD2SI64rr:{ *:[i64] } FR64:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTTSD2SI64rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1349,
      GIR_Done,
    // Label 1235: @29604
    GIM_Try, /*On fail goto*//*Label 1236*/ 29627, // Rule ID 8075 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
      // (fp_to_sint:{ *:[i64] } FR32X:{ *:[f32] }:$src)  =>  (VCVTTSS2SI64Zrr:{ *:[i64] } FR32X:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSS2SI64Zrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8075,
      GIR_Done,
    // Label 1236: @29627
    GIM_Try, /*On fail goto*//*Label 1237*/ 29650, // Rule ID 8085 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
      // (fp_to_sint:{ *:[i64] } FR64X:{ *:[f64] }:$src)  =>  (VCVTTSD2SI64Zrr:{ *:[i64] } FR64X:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSD2SI64Zrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8085,
      GIR_Done,
    // Label 1237: @29650
    GIM_Reject,
    // Label 1219: @29651
    GIM_Try, /*On fail goto*//*Label 1238*/ 29674, // Rule ID 16056 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (fp_to_sint:{ *:[v2i64] } VR128X:{ *:[v2f64] }:$src)  =>  (VCVTTPD2QQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2QQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16056,
      GIR_Done,
    // Label 1238: @29674
    GIM_Reject,
    // Label 1220: @29675
    GIM_Try, /*On fail goto*//*Label 1239*/ 29698, // Rule ID 12486 //
      GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (fp_to_sint:{ *:[v4i32] } VR128:{ *:[v4f32] }:$src)  =>  (VCVTTPS2DQrr:{ *:[v4i32] } VR128:{ *:[v4f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2DQrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12486,
      GIR_Done,
    // Label 1239: @29698
    GIM_Try, /*On fail goto*//*Label 1240*/ 29721, // Rule ID 12490 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (fp_to_sint:{ *:[v4i32] } VR128:{ *:[v4f32] }:$src)  =>  (CVTTPS2DQrr:{ *:[v4i32] } VR128:{ *:[v4f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTTPS2DQrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12490,
      GIR_Done,
    // Label 1240: @29721
    GIM_Try, /*On fail goto*//*Label 1241*/ 29744, // Rule ID 12492 //
      GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (fp_to_sint:{ *:[v4i32] } VR256:{ *:[v4f64] }:$src)  =>  (VCVTTPD2DQYrr:{ *:[v4i32] } VR256:{ *:[v4f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2DQYrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12492,
      GIR_Done,
    // Label 1241: @29744
    GIM_Try, /*On fail goto*//*Label 1242*/ 29767, // Rule ID 16032 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (fp_to_sint:{ *:[v4i32] } VR128X:{ *:[v4f32] }:$src)  =>  (VCVTTPS2DQZ128rr:{ *:[v4i32] } VR128X:{ *:[v4f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2DQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16032,
      GIR_Done,
    // Label 1242: @29767
    GIM_Try, /*On fail goto*//*Label 1243*/ 29790, // Rule ID 16040 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_sint:{ *:[v4i32] } VR256X:{ *:[v4f64] }:$src)  =>  (VCVTTPD2DQZ256rr:{ *:[v4i32] } VR256X:{ *:[v4f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2DQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16040,
      GIR_Done,
    // Label 1243: @29790
    GIM_Reject,
    // Label 1221: @29791
    GIM_Try, /*On fail goto*//*Label 1244*/ 29814, // Rule ID 16052 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (fp_to_sint:{ *:[v4i64] } VR128X:{ *:[v4f32] }:$src)  =>  (VCVTTPS2QQZ256rr:{ *:[v4i64] } VR128X:{ *:[v4f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2QQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16052,
      GIR_Done,
    // Label 1244: @29814
    GIM_Try, /*On fail goto*//*Label 1245*/ 29837, // Rule ID 16060 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_sint:{ *:[v4i64] } VR256X:{ *:[v4f64] }:$src)  =>  (VCVTTPD2QQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2QQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16060,
      GIR_Done,
    // Label 1245: @29837
    GIM_Reject,
    // Label 1222: @29838
    GIM_Try, /*On fail goto*//*Label 1246*/ 29861, // Rule ID 12488 //
      GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (fp_to_sint:{ *:[v8i32] } VR256:{ *:[v8f32] }:$src)  =>  (VCVTTPS2DQYrr:{ *:[v8i32] } VR256:{ *:[v8f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2DQYrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12488,
      GIR_Done,
    // Label 1246: @29861
    GIM_Try, /*On fail goto*//*Label 1247*/ 29884, // Rule ID 16028 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fp_to_sint:{ *:[v8i32] } VR512:{ *:[v8f64] }:$src)  =>  (VCVTTPD2DQZrr:{ *:[v8i32] } VR512:{ *:[v8f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2DQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16028,
      GIR_Done,
    // Label 1247: @29884
    GIM_Try, /*On fail goto*//*Label 1248*/ 29907, // Rule ID 16036 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_sint:{ *:[v8i32] } VR256X:{ *:[v8f32] }:$src)  =>  (VCVTTPS2DQZ256rr:{ *:[v8i32] } VR256X:{ *:[v8f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2DQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16036,
      GIR_Done,
    // Label 1248: @29907
    GIM_Reject,
    // Label 1223: @29908
    GIM_Try, /*On fail goto*//*Label 1249*/ 29931, // Rule ID 16044 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_sint:{ *:[v8i64] } VR256X:{ *:[v8f32] }:$src)  =>  (VCVTTPS2QQZrr:{ *:[v8i64] } VR256X:{ *:[v8f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2QQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16044,
      GIR_Done,
    // Label 1249: @29931
    GIM_Try, /*On fail goto*//*Label 1250*/ 29954, // Rule ID 16048 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fp_to_sint:{ *:[v8i64] } VR512:{ *:[v8f64] }:$src)  =>  (VCVTTPD2QQZrr:{ *:[v8i64] } VR512:{ *:[v8f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2QQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16048,
      GIR_Done,
    // Label 1250: @29954
    GIM_Reject,
    // Label 1224: @29955
    GIM_Try, /*On fail goto*//*Label 1251*/ 29978, // Rule ID 16024 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fp_to_sint:{ *:[v16i32] } VR512:{ *:[v16f32] }:$src)  =>  (VCVTTPS2DQZrr:{ *:[v16i32] } VR512:{ *:[v16f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2DQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16024,
      GIR_Done,
    // Label 1251: @29978
    GIM_Reject,
    // Label 1225: @29979
    GIM_Reject,
    // Label 26: @29980
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 1260*/ 30333,
    /*GILLT_s32*//*Label 1252*/ 30003,
    /*GILLT_s64*//*Label 1253*/ 30050, 0, 0, 0,
    /*GILLT_v2s64*//*Label 1254*/ 30097, 0,
    /*GILLT_v4s32*//*Label 1255*/ 30121,
    /*GILLT_v4s64*//*Label 1256*/ 30168, 0, 0,
    /*GILLT_v8s32*//*Label 1257*/ 30215,
    /*GILLT_v8s64*//*Label 1258*/ 30262, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1259*/ 30309,
    // Label 1252: @30003
    GIM_Try, /*On fail goto*//*Label 1261*/ 30026, // Rule ID 8090 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
      // (fp_to_uint:{ *:[i32] } FR32X:{ *:[f32] }:$src)  =>  (VCVTTSS2USIZrr:{ *:[i32] } FR32X:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSS2USIZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8090,
      GIR_Done,
    // Label 1261: @30026
    GIM_Try, /*On fail goto*//*Label 1262*/ 30049, // Rule ID 8100 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
      // (fp_to_uint:{ *:[i32] } FR64X:{ *:[f64] }:$src)  =>  (VCVTTSD2USIZrr:{ *:[i32] } FR64X:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSD2USIZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8100,
      GIR_Done,
    // Label 1262: @30049
    GIM_Reject,
    // Label 1253: @30050
    GIM_Try, /*On fail goto*//*Label 1263*/ 30073, // Rule ID 8095 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR32XRegClassID,
      // (fp_to_uint:{ *:[i64] } FR32X:{ *:[f32] }:$src)  =>  (VCVTTSS2USI64Zrr:{ *:[i64] } FR32X:{ *:[f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSS2USI64Zrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8095,
      GIR_Done,
    // Label 1263: @30073
    GIM_Try, /*On fail goto*//*Label 1264*/ 30096, // Rule ID 8105 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::FR64XRegClassID,
      // (fp_to_uint:{ *:[i64] } FR64X:{ *:[f64] }:$src)  =>  (VCVTTSD2USI64Zrr:{ *:[i64] } FR64X:{ *:[f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTSD2USI64Zrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8105,
      GIR_Done,
    // Label 1264: @30096
    GIM_Reject,
    // Label 1254: @30097
    GIM_Try, /*On fail goto*//*Label 1265*/ 30120, // Rule ID 16058 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (fp_to_uint:{ *:[v2i64] } VR128X:{ *:[v2f64] }:$src)  =>  (VCVTTPD2UQQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2UQQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16058,
      GIR_Done,
    // Label 1265: @30120
    GIM_Reject,
    // Label 1255: @30121
    GIM_Try, /*On fail goto*//*Label 1266*/ 30144, // Rule ID 16034 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (fp_to_uint:{ *:[v4i32] } VR128X:{ *:[v4f32] }:$src)  =>  (VCVTTPS2UDQZ128rr:{ *:[v4i32] } VR128X:{ *:[v4f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2UDQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16034,
      GIR_Done,
    // Label 1266: @30144
    GIM_Try, /*On fail goto*//*Label 1267*/ 30167, // Rule ID 16042 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_uint:{ *:[v4i32] } VR256X:{ *:[v4f64] }:$src)  =>  (VCVTTPD2UDQZ256rr:{ *:[v4i32] } VR256X:{ *:[v4f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2UDQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16042,
      GIR_Done,
    // Label 1267: @30167
    GIM_Reject,
    // Label 1256: @30168
    GIM_Try, /*On fail goto*//*Label 1268*/ 30191, // Rule ID 16054 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (fp_to_uint:{ *:[v4i64] } VR128X:{ *:[v4f32] }:$src)  =>  (VCVTTPS2UQQZ256rr:{ *:[v4i64] } VR128X:{ *:[v4f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2UQQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16054,
      GIR_Done,
    // Label 1268: @30191
    GIM_Try, /*On fail goto*//*Label 1269*/ 30214, // Rule ID 16062 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_uint:{ *:[v4i64] } VR256X:{ *:[v4f64] }:$src)  =>  (VCVTTPD2UQQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2UQQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16062,
      GIR_Done,
    // Label 1269: @30214
    GIM_Reject,
    // Label 1257: @30215
    GIM_Try, /*On fail goto*//*Label 1270*/ 30238, // Rule ID 16030 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fp_to_uint:{ *:[v8i32] } VR512:{ *:[v8f64] }:$src)  =>  (VCVTTPD2UDQZrr:{ *:[v8i32] } VR512:{ *:[v8f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2UDQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16030,
      GIR_Done,
    // Label 1270: @30238
    GIM_Try, /*On fail goto*//*Label 1271*/ 30261, // Rule ID 16038 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_uint:{ *:[v8i32] } VR256X:{ *:[v8f32] }:$src)  =>  (VCVTTPS2UDQZ256rr:{ *:[v8i32] } VR256X:{ *:[v8f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2UDQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16038,
      GIR_Done,
    // Label 1271: @30261
    GIM_Reject,
    // Label 1258: @30262
    GIM_Try, /*On fail goto*//*Label 1272*/ 30285, // Rule ID 16046 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (fp_to_uint:{ *:[v8i64] } VR256X:{ *:[v8f32] }:$src)  =>  (VCVTTPS2UQQZrr:{ *:[v8i64] } VR256X:{ *:[v8f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2UQQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16046,
      GIR_Done,
    // Label 1272: @30285
    GIM_Try, /*On fail goto*//*Label 1273*/ 30308, // Rule ID 16050 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fp_to_uint:{ *:[v8i64] } VR512:{ *:[v8f64] }:$src)  =>  (VCVTTPD2UQQZrr:{ *:[v8i64] } VR512:{ *:[v8f64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPD2UQQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16050,
      GIR_Done,
    // Label 1273: @30308
    GIM_Reject,
    // Label 1259: @30309
    GIM_Try, /*On fail goto*//*Label 1274*/ 30332, // Rule ID 16026 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (fp_to_uint:{ *:[v16i32] } VR512:{ *:[v16f32] }:$src)  =>  (VCVTTPS2UDQZrr:{ *:[v16i32] } VR512:{ *:[v16f32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTTPS2UDQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 16026,
      GIR_Done,
    // Label 1274: @30332
    GIM_Reject,
    // Label 1260: @30333
    GIM_Reject,
    // Label 27: @30334
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 1283*/ 31163,
    /*GILLT_s32*//*Label 1275*/ 30357,
    /*GILLT_s64*//*Label 1276*/ 30596, 0, 0, 0,
    /*GILLT_v2s64*//*Label 1277*/ 30835, 0,
    /*GILLT_v4s32*//*Label 1278*/ 30859,
    /*GILLT_v4s64*//*Label 1279*/ 30952, 0, 0,
    /*GILLT_v8s32*//*Label 1280*/ 31022,
    /*GILLT_v8s64*//*Label 1281*/ 31092, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1282*/ 31139,
    // Label 1275: @30357
    GIM_Try, /*On fail goto*//*Label 1284*/ 30380, // Rule ID 1351 //
      GIM_CheckFeatures, GIFBS_UseSSE1,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sint_to_fp:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (CVTSI2SSrr:{ *:[f32] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTSI2SSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1351,
      GIR_Done,
    // Label 1284: @30380
    GIM_Try, /*On fail goto*//*Label 1285*/ 30403, // Rule ID 1353 //
      GIM_CheckFeatures, GIFBS_UseSSE1,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (sint_to_fp:{ *:[f32] } GR64:{ *:[i64] }:$src)  =>  (CVTSI642SSrr:{ *:[f32] } GR64:{ *:[i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTSI642SSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1353,
      GIR_Done,
    // Label 1285: @30403
    GIM_Try, /*On fail goto*//*Label 1286*/ 30451, // Rule ID 12455 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sint_to_fp:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (VCVTSI2SSrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), GR32:{ *:[i32] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI2SSrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12455,
      GIR_Done,
    // Label 1286: @30451
    GIM_Try, /*On fail goto*//*Label 1287*/ 30499, // Rule ID 12456 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (sint_to_fp:{ *:[f32] } GR64:{ *:[i64] }:$src)  =>  (VCVTSI642SSrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), GR64:{ *:[i64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI642SSrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12456,
      GIR_Done,
    // Label 1287: @30499
    GIM_Try, /*On fail goto*//*Label 1288*/ 30547, // Rule ID 15984 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sint_to_fp:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (VCVTSI2SSZrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), GR32:{ *:[i32] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI2SSZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 15984,
      GIR_Done,
    // Label 1288: @30547
    GIM_Try, /*On fail goto*//*Label 1289*/ 30595, // Rule ID 15985 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (sint_to_fp:{ *:[f32] } GR64:{ *:[i64] }:$src)  =>  (VCVTSI642SSZrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), GR64:{ *:[i64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI642SSZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 15985,
      GIR_Done,
    // Label 1289: @30595
    GIM_Reject,
    // Label 1276: @30596
    GIM_Try, /*On fail goto*//*Label 1290*/ 30619, // Rule ID 1355 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sint_to_fp:{ *:[f64] } GR32:{ *:[i32] }:$src)  =>  (CVTSI2SDrr:{ *:[f64] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTSI2SDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1355,
      GIR_Done,
    // Label 1290: @30619
    GIM_Try, /*On fail goto*//*Label 1291*/ 30642, // Rule ID 1357 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (sint_to_fp:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (CVTSI642SDrr:{ *:[f64] } GR64:{ *:[i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTSI642SDrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1357,
      GIR_Done,
    // Label 1291: @30642
    GIM_Try, /*On fail goto*//*Label 1292*/ 30690, // Rule ID 12457 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sint_to_fp:{ *:[f64] } GR32:{ *:[i32] }:$src)  =>  (VCVTSI2SDrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), GR32:{ *:[i32] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI2SDrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12457,
      GIR_Done,
    // Label 1292: @30690
    GIM_Try, /*On fail goto*//*Label 1293*/ 30738, // Rule ID 12458 //
      GIM_CheckFeatures, GIFBS_UseAVX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (sint_to_fp:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (VCVTSI642SDrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), GR64:{ *:[i64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI642SDrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 12458,
      GIR_Done,
    // Label 1293: @30738
    GIM_Try, /*On fail goto*//*Label 1294*/ 30786, // Rule ID 15986 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (sint_to_fp:{ *:[f64] } GR32:{ *:[i32] }:$src)  =>  (VCVTSI2SDZrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), GR32:{ *:[i32] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI2SDZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 15986,
      GIR_Done,
    // Label 1294: @30786
    GIM_Try, /*On fail goto*//*Label 1295*/ 30834, // Rule ID 15987 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (sint_to_fp:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (VCVTSI642SDZrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), GR64:{ *:[i64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTSI642SDZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 15987,
      GIR_Done,
    // Label 1295: @30834
    GIM_Reject,
    // Label 1277: @30835
    GIM_Try, /*On fail goto*//*Label 1296*/ 30858, // Rule ID 8794 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (sint_to_fp:{ *:[v2f64] } VR128X:{ *:[v2i64] }:$src)  =>  (VCVTQQ2PDZ128rr:{ *:[v2f64] } VR128X:{ *:[v2i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTQQ2PDZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8794,
      GIR_Done,
    // Label 1296: @30858
    GIM_Reject,
    // Label 1278: @30859
    GIM_Try, /*On fail goto*//*Label 1297*/ 30882, // Rule ID 1391 //
      GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (sint_to_fp:{ *:[v4f32] } VR128:{ *:[v4i32] }:$src)  =>  (VCVTDQ2PSrr:{ *:[v4f32] } VR128:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1391,
      GIR_Done,
    // Label 1297: @30882
    GIM_Try, /*On fail goto*//*Label 1298*/ 30905, // Rule ID 1395 //
      GIM_CheckFeatures, GIFBS_UseSSE2,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (sint_to_fp:{ *:[v4f32] } VR128:{ *:[v4i32] }:$src)  =>  (CVTDQ2PSrr:{ *:[v4f32] } VR128:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::CVTDQ2PSrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1395,
      GIR_Done,
    // Label 1298: @30905
    GIM_Try, /*On fail goto*//*Label 1299*/ 30928, // Rule ID 8227 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (sint_to_fp:{ *:[v4f32] } VR128X:{ *:[v4i32] }:$src)  =>  (VCVTDQ2PSZ128rr:{ *:[v4f32] } VR128X:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PSZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8227,
      GIR_Done,
    // Label 1299: @30928
    GIM_Try, /*On fail goto*//*Label 1300*/ 30951, // Rule ID 8863 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (sint_to_fp:{ *:[v4f32] } VR256X:{ *:[v4i64] }:$src)  =>  (VCVTQQ2PSZ256rr:{ *:[v4f32] } VR256X:{ *:[v4i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTQQ2PSZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8863,
      GIR_Done,
    // Label 1300: @30951
    GIM_Reject,
    // Label 1279: @30952
    GIM_Try, /*On fail goto*//*Label 1301*/ 30975, // Rule ID 1438 //
      GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128RegClassID,
      // (sint_to_fp:{ *:[v4f64] } VR128:{ *:[v4i32] }:$src)  =>  (VCVTDQ2PDYrr:{ *:[v4f64] } VR128:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PDYrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1438,
      GIR_Done,
    // Label 1301: @30975
    GIM_Try, /*On fail goto*//*Label 1302*/ 30998, // Rule ID 8206 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (sint_to_fp:{ *:[v4f64] } VR128X:{ *:[v4i32] }:$src)  =>  (VCVTDQ2PDZ256rr:{ *:[v4f64] } VR128X:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8206,
      GIR_Done,
    // Label 1302: @30998
    GIM_Try, /*On fail goto*//*Label 1303*/ 31021, // Rule ID 8803 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (sint_to_fp:{ *:[v4f64] } VR256X:{ *:[v4i64] }:$src)  =>  (VCVTQQ2PDZ256rr:{ *:[v4f64] } VR256X:{ *:[v4i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTQQ2PDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8803,
      GIR_Done,
    // Label 1303: @31021
    GIM_Reject,
    // Label 1280: @31022
    GIM_Try, /*On fail goto*//*Label 1304*/ 31045, // Rule ID 1393 //
      GIM_CheckFeatures, GIFBS_HasAVX_NoVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256RegClassID,
      // (sint_to_fp:{ *:[v8f32] } VR256:{ *:[v8i32] }:$src)  =>  (VCVTDQ2PSYrr:{ *:[v8f32] } VR256:{ *:[v8i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PSYrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 1393,
      GIR_Done,
    // Label 1304: @31045
    GIM_Try, /*On fail goto*//*Label 1305*/ 31068, // Rule ID 8236 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (sint_to_fp:{ *:[v8f32] } VR256X:{ *:[v8i32] }:$src)  =>  (VCVTDQ2PSZ256rr:{ *:[v8f32] } VR256X:{ *:[v8i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PSZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8236,
      GIR_Done,
    // Label 1305: @31068
    GIM_Try, /*On fail goto*//*Label 1306*/ 31091, // Rule ID 8842 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (sint_to_fp:{ *:[v8f32] } VR512:{ *:[v8i64] }:$src)  =>  (VCVTQQ2PSZrr:{ *:[v8f32] } VR512:{ *:[v8i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTQQ2PSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8842,
      GIR_Done,
    // Label 1306: @31091
    GIM_Reject,
    // Label 1281: @31092
    GIM_Try, /*On fail goto*//*Label 1307*/ 31115, // Rule ID 8188 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (sint_to_fp:{ *:[v8f64] } VR256X:{ *:[v8i32] }:$src)  =>  (VCVTDQ2PDZrr:{ *:[v8f64] } VR256X:{ *:[v8i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8188,
      GIR_Done,
    // Label 1307: @31115
    GIM_Try, /*On fail goto*//*Label 1308*/ 31138, // Rule ID 8782 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (sint_to_fp:{ *:[v8f64] } VR512:{ *:[v8i64] }:$src)  =>  (VCVTQQ2PDZrr:{ *:[v8f64] } VR512:{ *:[v8i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTQQ2PDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8782,
      GIR_Done,
    // Label 1308: @31138
    GIM_Reject,
    // Label 1282: @31139
    GIM_Try, /*On fail goto*//*Label 1309*/ 31162, // Rule ID 8215 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (sint_to_fp:{ *:[v16f32] } VR512:{ *:[v16i32] }:$src)  =>  (VCVTDQ2PSZrr:{ *:[v16f32] } VR512:{ *:[v16i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTDQ2PSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8215,
      GIR_Done,
    // Label 1309: @31162
    GIM_Reject,
    // Label 1283: @31163
    GIM_Reject,
    // Label 28: @31164
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/3, 20, /*)*//*default:*//*Label 1318*/ 31617,
    /*GILLT_s32*//*Label 1310*/ 31187,
    /*GILLT_s64*//*Label 1311*/ 31284, 0, 0, 0,
    /*GILLT_v2s64*//*Label 1312*/ 31381, 0,
    /*GILLT_v4s32*//*Label 1313*/ 31405,
    /*GILLT_v4s64*//*Label 1314*/ 31452, 0, 0,
    /*GILLT_v8s32*//*Label 1315*/ 31499,
    /*GILLT_v8s64*//*Label 1316*/ 31546, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1317*/ 31593,
    // Label 1310: @31187
    GIM_Try, /*On fail goto*//*Label 1319*/ 31235, // Rule ID 15992 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (uint_to_fp:{ *:[f32] } GR32:{ *:[i32] }:$src)  =>  (VCVTUSI2SSZrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), GR32:{ *:[i32] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTUSI2SSZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 15992,
      GIR_Done,
    // Label 1319: @31235
    GIM_Try, /*On fail goto*//*Label 1320*/ 31283, // Rule ID 15993 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR32XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (uint_to_fp:{ *:[f32] } GR64:{ *:[i64] }:$src)  =>  (VCVTUSI642SSZrr:{ *:[f32] } (IMPLICIT_DEF:{ *:[f32] }), GR64:{ *:[i64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s32,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTUSI642SSZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 15993,
      GIR_Done,
    // Label 1320: @31283
    GIM_Reject,
    // Label 1311: @31284
    GIM_Try, /*On fail goto*//*Label 1321*/ 31332, // Rule ID 15994 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (uint_to_fp:{ *:[f64] } GR32:{ *:[i32] }:$src)  =>  (VCVTUSI2SDZrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), GR32:{ *:[i32] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTUSI2SDZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 15994,
      GIR_Done,
    // Label 1321: @31332
    GIM_Try, /*On fail goto*//*Label 1322*/ 31380, // Rule ID 15995 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::FR64XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (uint_to_fp:{ *:[f64] } GR64:{ *:[i64] }:$src)  =>  (VCVTUSI642SDZrr:{ *:[f64] } (IMPLICIT_DEF:{ *:[f64] }), GR64:{ *:[i64] }:$src)
      GIR_MakeTempReg, /*TempRegID*/0, /*TypeID*/GILLT_s64,
      GIR_BuildMI, /*InsnID*/1, /*Opcode*/TargetOpcode::IMPLICIT_DEF,
      GIR_AddTempRegister, /*InsnID*/1, /*TempRegID*/0, /*TempRegFlags*/RegState::Define,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/1,
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::VCVTUSI642SDZrr,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_AddTempRegister, /*InsnID*/0, /*TempRegID*/0, /*TempRegFlags*/0,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 15995,
      GIR_Done,
    // Label 1322: @31380
    GIM_Reject,
    // Label 1312: @31381
    GIM_Try, /*On fail goto*//*Label 1323*/ 31404, // Rule ID 8824 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (uint_to_fp:{ *:[v2f64] } VR128X:{ *:[v2i64] }:$src)  =>  (VCVTUQQ2PDZ128rr:{ *:[v2f64] } VR128X:{ *:[v2i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUQQ2PDZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8824,
      GIR_Done,
    // Label 1323: @31404
    GIM_Reject,
    // Label 1313: @31405
    GIM_Try, /*On fail goto*//*Label 1324*/ 31428, // Rule ID 8404 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (uint_to_fp:{ *:[v4f32] } VR128X:{ *:[v4i32] }:$src)  =>  (VCVTUDQ2PSZ128rr:{ *:[v4f32] } VR128X:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUDQ2PSZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8404,
      GIR_Done,
    // Label 1324: @31428
    GIM_Try, /*On fail goto*//*Label 1325*/ 31451, // Rule ID 8893 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (uint_to_fp:{ *:[v4f32] } VR256X:{ *:[v4i64] }:$src)  =>  (VCVTUQQ2PSZ256rr:{ *:[v4f32] } VR256X:{ *:[v4i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUQQ2PSZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8893,
      GIR_Done,
    // Label 1325: @31451
    GIM_Reject,
    // Label 1314: @31452
    GIM_Try, /*On fail goto*//*Label 1326*/ 31475, // Rule ID 8383 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (uint_to_fp:{ *:[v4f64] } VR128X:{ *:[v4i32] }:$src)  =>  (VCVTUDQ2PDZ256rr:{ *:[v4f64] } VR128X:{ *:[v4i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUDQ2PDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8383,
      GIR_Done,
    // Label 1326: @31475
    GIM_Try, /*On fail goto*//*Label 1327*/ 31498, // Rule ID 8833 //
      GIM_CheckFeatures, GIFBS_HasDQI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (uint_to_fp:{ *:[v4f64] } VR256X:{ *:[v4i64] }:$src)  =>  (VCVTUQQ2PDZ256rr:{ *:[v4f64] } VR256X:{ *:[v4i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUQQ2PDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8833,
      GIR_Done,
    // Label 1327: @31498
    GIM_Reject,
    // Label 1315: @31499
    GIM_Try, /*On fail goto*//*Label 1328*/ 31522, // Rule ID 8413 //
      GIM_CheckFeatures, GIFBS_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (uint_to_fp:{ *:[v8f32] } VR256X:{ *:[v8i32] }:$src)  =>  (VCVTUDQ2PSZ256rr:{ *:[v8f32] } VR256X:{ *:[v8i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUDQ2PSZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8413,
      GIR_Done,
    // Label 1328: @31522
    GIM_Try, /*On fail goto*//*Label 1329*/ 31545, // Rule ID 8872 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (uint_to_fp:{ *:[v8f32] } VR512:{ *:[v8i64] }:$src)  =>  (VCVTUQQ2PSZrr:{ *:[v8f32] } VR512:{ *:[v8i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUQQ2PSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8872,
      GIR_Done,
    // Label 1329: @31545
    GIM_Reject,
    // Label 1316: @31546
    GIM_Try, /*On fail goto*//*Label 1330*/ 31569, // Rule ID 8365 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (uint_to_fp:{ *:[v8f64] } VR256X:{ *:[v8i32] }:$src)  =>  (VCVTUDQ2PDZrr:{ *:[v8f64] } VR256X:{ *:[v8i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUDQ2PDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8365,
      GIR_Done,
    // Label 1330: @31569
    GIM_Try, /*On fail goto*//*Label 1331*/ 31592, // Rule ID 8812 //
      GIM_CheckFeatures, GIFBS_HasDQI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (uint_to_fp:{ *:[v8f64] } VR512:{ *:[v8i64] }:$src)  =>  (VCVTUQQ2PDZrr:{ *:[v8f64] } VR512:{ *:[v8i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUQQ2PDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8812,
      GIR_Done,
    // Label 1331: @31592
    GIM_Reject,
    // Label 1317: @31593
    GIM_Try, /*On fail goto*//*Label 1332*/ 31616, // Rule ID 8392 //
      GIM_CheckFeatures, GIFBS_HasAVX512,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (uint_to_fp:{ *:[v16f32] } VR512:{ *:[v16i32] }:$src)  =>  (VCVTUDQ2PSZrr:{ *:[v16f32] } VR512:{ *:[v16i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VCVTUDQ2PSZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 8392,
      GIR_Done,
    // Label 1332: @31616
    GIM_Reject,
    // Label 1318: @31617
    GIM_Reject,
    // Label 29: @31618
    GIM_Try, /*On fail goto*//*Label 1333*/ 31630, // Rule ID 548 //
      // MIs[0] dst
      GIM_CheckIsMBB, /*MI*/0, /*Op*/0,
      // (br (bb:{ *:[Other] }):$dst)  =>  (JMP_1 (bb:{ *:[Other] }):$dst)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::JMP_1,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 548,
      GIR_Done,
    // Label 1333: @31630
    GIM_Reject,
    // Label 30: @31631
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/2, 5, /*)*//*default:*//*Label 1337*/ 31715,
    /*GILLT_s16*//*Label 1334*/ 31640,
    /*GILLT_s32*//*Label 1335*/ 31665,
    /*GILLT_s64*//*Label 1336*/ 31690,
    // Label 1334: @31640
    GIM_Try, /*On fail goto*//*Label 1338*/ 31664, // Rule ID 17477 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      // (cttz_zero_undef:{ *:[i16] } GR16:{ *:[i16] }:$src)  =>  (BSF16rr:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::BSF16rr,
      GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 17477,
      GIR_Done,
    // Label 1338: @31664
    GIM_Reject,
    // Label 1335: @31665
    GIM_Try, /*On fail goto*//*Label 1339*/ 31689, // Rule ID 17478 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (cttz_zero_undef:{ *:[i32] } GR32:{ *:[i32] }:$src)  =>  (BSF32rr:{ *:[i32] }:{ *:[i32] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::BSF32rr,
      GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 17478,
      GIR_Done,
    // Label 1339: @31689
    GIM_Reject,
    // Label 1336: @31690
    GIM_Try, /*On fail goto*//*Label 1340*/ 31714, // Rule ID 17479 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (cttz_zero_undef:{ *:[i64] } GR64:{ *:[i64] }:$src)  =>  (BSF64rr:{ *:[i64] }:{ *:[i32] } GR64:{ *:[i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::BSF64rr,
      GIR_AddImplicitDef, /*InsnID*/0, X86::EFLAGS,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 17479,
      GIR_Done,
    // Label 1340: @31714
    GIM_Reject,
    // Label 1337: @31715
    GIM_Reject,
    // Label 31: @31716
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/8, 20, /*)*//*default:*//*Label 1347*/ 31878,
    /*GILLT_v2s64*//*Label 1341*/ 31734, 0,
    /*GILLT_v4s32*//*Label 1342*/ 31758,
    /*GILLT_v4s64*//*Label 1343*/ 31782, 0, 0,
    /*GILLT_v8s32*//*Label 1344*/ 31806,
    /*GILLT_v8s64*//*Label 1345*/ 31830, 0, 0, 0,
    /*GILLT_v16s32*//*Label 1346*/ 31854,
    // Label 1341: @31734
    GIM_Try, /*On fail goto*//*Label 1348*/ 31757, // Rule ID 10480 //
      GIM_CheckFeatures, GIFBS_HasCDI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (ctlz:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1)  =>  (VPLZCNTQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPLZCNTQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10480,
      GIR_Done,
    // Label 1348: @31757
    GIM_Reject,
    // Label 1342: @31758
    GIM_Try, /*On fail goto*//*Label 1349*/ 31781, // Rule ID 10507 //
      GIM_CheckFeatures, GIFBS_HasCDI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (ctlz:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1)  =>  (VPLZCNTDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPLZCNTDZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10507,
      GIR_Done,
    // Label 1349: @31781
    GIM_Reject,
    // Label 1343: @31782
    GIM_Try, /*On fail goto*//*Label 1350*/ 31805, // Rule ID 10471 //
      GIM_CheckFeatures, GIFBS_HasCDI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (ctlz:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1)  =>  (VPLZCNTQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPLZCNTQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10471,
      GIR_Done,
    // Label 1350: @31805
    GIM_Reject,
    // Label 1344: @31806
    GIM_Try, /*On fail goto*//*Label 1351*/ 31829, // Rule ID 10498 //
      GIM_CheckFeatures, GIFBS_HasCDI_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (ctlz:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1)  =>  (VPLZCNTDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPLZCNTDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10498,
      GIR_Done,
    // Label 1351: @31829
    GIM_Reject,
    // Label 1345: @31830
    GIM_Try, /*On fail goto*//*Label 1352*/ 31853, // Rule ID 10462 //
      GIM_CheckFeatures, GIFBS_HasCDI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (ctlz:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1)  =>  (VPLZCNTQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPLZCNTQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10462,
      GIR_Done,
    // Label 1352: @31853
    GIM_Reject,
    // Label 1346: @31854
    GIM_Try, /*On fail goto*//*Label 1353*/ 31877, // Rule ID 10489 //
      GIM_CheckFeatures, GIFBS_HasCDI,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (ctlz:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1)  =>  (VPLZCNTDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPLZCNTDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10489,
      GIR_Done,
    // Label 1353: @31877
    GIM_Reject,
    // Label 1347: @31878
    GIM_Reject,
    // Label 32: @31879
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/8, 25, /*)*//*default:*//*Label 1366*/ 32190,
    /*GILLT_v2s64*//*Label 1354*/ 31902, 0,
    /*GILLT_v4s32*//*Label 1355*/ 31926,
    /*GILLT_v4s64*//*Label 1356*/ 31950, 0,
    /*GILLT_v8s16*//*Label 1357*/ 31974,
    /*GILLT_v8s32*//*Label 1358*/ 31998,
    /*GILLT_v8s64*//*Label 1359*/ 32022, 0,
    /*GILLT_v16s8*//*Label 1360*/ 32046,
    /*GILLT_v16s16*//*Label 1361*/ 32070,
    /*GILLT_v16s32*//*Label 1362*/ 32094, 0,
    /*GILLT_v32s8*//*Label 1363*/ 32118,
    /*GILLT_v32s16*//*Label 1364*/ 32142, 0,
    /*GILLT_v64s8*//*Label 1365*/ 32166,
    // Label 1354: @31902
    GIM_Try, /*On fail goto*//*Label 1367*/ 31925, // Rule ID 10588 //
      GIM_CheckFeatures, GIFBS_HasVLX_HasVPOPCNTDQ,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v2s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (ctpop:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1)  =>  (VPOPCNTQZ128rr:{ *:[v2i64] } VR128X:{ *:[v2i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTQZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10588,
      GIR_Done,
    // Label 1367: @31925
    GIM_Reject,
    // Label 1355: @31926
    GIM_Try, /*On fail goto*//*Label 1368*/ 31949, // Rule ID 10615 //
      GIM_CheckFeatures, GIFBS_HasVLX_HasVPOPCNTDQ,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (ctpop:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1)  =>  (VPOPCNTDZ128rr:{ *:[v4i32] } VR128X:{ *:[v4i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTDZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10615,
      GIR_Done,
    // Label 1368: @31949
    GIM_Reject,
    // Label 1356: @31950
    GIM_Try, /*On fail goto*//*Label 1369*/ 31973, // Rule ID 10579 //
      GIM_CheckFeatures, GIFBS_HasVLX_HasVPOPCNTDQ,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v4s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (ctpop:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1)  =>  (VPOPCNTQZ256rr:{ *:[v4i64] } VR256X:{ *:[v4i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTQZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10579,
      GIR_Done,
    // Label 1369: @31973
    GIM_Reject,
    // Label 1357: @31974
    GIM_Try, /*On fail goto*//*Label 1370*/ 31997, // Rule ID 11696 //
      GIM_CheckFeatures, GIFBS_HasBITALG_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (ctpop:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1)  =>  (VPOPCNTWZ128rr:{ *:[v8i16] } VR128X:{ *:[v8i16] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTWZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 11696,
      GIR_Done,
    // Label 1370: @31997
    GIM_Reject,
    // Label 1358: @31998
    GIM_Try, /*On fail goto*//*Label 1371*/ 32021, // Rule ID 10606 //
      GIM_CheckFeatures, GIFBS_HasVLX_HasVPOPCNTDQ,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (ctpop:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1)  =>  (VPOPCNTDZ256rr:{ *:[v8i32] } VR256X:{ *:[v8i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTDZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10606,
      GIR_Done,
    // Label 1371: @32021
    GIM_Reject,
    // Label 1359: @32022
    GIM_Try, /*On fail goto*//*Label 1372*/ 32045, // Rule ID 10570 //
      GIM_CheckFeatures, GIFBS_HasVPOPCNTDQ,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v8s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (ctpop:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1)  =>  (VPOPCNTQZrr:{ *:[v8i64] } VR512:{ *:[v8i64] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTQZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10570,
      GIR_Done,
    // Label 1372: @32045
    GIM_Reject,
    // Label 1360: @32046
    GIM_Try, /*On fail goto*//*Label 1373*/ 32069, // Rule ID 11678 //
      GIM_CheckFeatures, GIFBS_HasBITALG_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR128XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR128XRegClassID,
      // (ctpop:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1)  =>  (VPOPCNTBZ128rr:{ *:[v16i8] } VR128X:{ *:[v16i8] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTBZ128rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 11678,
      GIR_Done,
    // Label 1373: @32069
    GIM_Reject,
    // Label 1361: @32070
    GIM_Try, /*On fail goto*//*Label 1374*/ 32093, // Rule ID 11690 //
      GIM_CheckFeatures, GIFBS_HasBITALG_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (ctpop:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1)  =>  (VPOPCNTWZ256rr:{ *:[v16i16] } VR256X:{ *:[v16i16] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTWZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 11690,
      GIR_Done,
    // Label 1374: @32093
    GIM_Reject,
    // Label 1362: @32094
    GIM_Try, /*On fail goto*//*Label 1375*/ 32117, // Rule ID 10597 //
      GIM_CheckFeatures, GIFBS_HasVPOPCNTDQ,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v16s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (ctpop:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1)  =>  (VPOPCNTDZrr:{ *:[v16i32] } VR512:{ *:[v16i32] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTDZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 10597,
      GIR_Done,
    // Label 1375: @32117
    GIM_Reject,
    // Label 1363: @32118
    GIM_Try, /*On fail goto*//*Label 1376*/ 32141, // Rule ID 11672 //
      GIM_CheckFeatures, GIFBS_HasBITALG_HasVLX,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR256XRegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR256XRegClassID,
      // (ctpop:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1)  =>  (VPOPCNTBZ256rr:{ *:[v32i8] } VR256X:{ *:[v32i8] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTBZ256rr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 11672,
      GIR_Done,
    // Label 1376: @32141
    GIM_Reject,
    // Label 1364: @32142
    GIM_Try, /*On fail goto*//*Label 1377*/ 32165, // Rule ID 11684 //
      GIM_CheckFeatures, GIFBS_HasBITALG,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v32s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (ctpop:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1)  =>  (VPOPCNTWZrr:{ *:[v32i16] } VR512:{ *:[v32i16] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTWZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 11684,
      GIR_Done,
    // Label 1377: @32165
    GIM_Reject,
    // Label 1365: @32166
    GIM_Try, /*On fail goto*//*Label 1378*/ 32189, // Rule ID 11666 //
      GIM_CheckFeatures, GIFBS_HasBITALG,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_v64s8,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::VR512RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::VR512RegClassID,
      // (ctpop:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1)  =>  (VPOPCNTBZrr:{ *:[v64i8] } VR512:{ *:[v64i8] }:$src1)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::VPOPCNTBZrr,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 11666,
      GIR_Done,
    // Label 1378: @32189
    GIM_Reject,
    // Label 1366: @32190
    GIM_Reject,
    // Label 33: @32191
    GIM_SwitchType, /*MI*/0, /*Op*/0, /*[*/2, 5, /*)*//*default:*//*Label 1382*/ 32280,
    /*GILLT_s16*//*Label 1379*/ 32200,
    /*GILLT_s32*//*Label 1380*/ 32236,
    /*GILLT_s64*//*Label 1381*/ 32258,
    // Label 1379: @32200
    GIM_Try, /*On fail goto*//*Label 1383*/ 32235, // Rule ID 17483 //
      GIM_CheckFeatures, GIFBS_HasMOVBE,
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s16,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR16RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR16RegClassID,
      // (bswap:{ *:[i16] } GR16:{ *:[i16] }:$src)  =>  (ROL16ri:{ *:[i16] }:{ *:[i32] } GR16:{ *:[i16] }:$src, 8:{ *:[i8] })
      GIR_BuildMI, /*InsnID*/0, /*Opcode*/X86::ROL16ri,
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/0, // dst
      GIR_Copy, /*NewInsnID*/0, /*OldInsnID*/0, /*OpIdx*/1, // src
      GIR_AddImm, /*InsnID*/0, /*Imm*/8,
      GIR_EraseFromParent, /*InsnID*/0,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 17483,
      GIR_Done,
    // Label 1383: @32235
    GIM_Reject,
    // Label 1380: @32236
    GIM_Try, /*On fail goto*//*Label 1384*/ 32257, // Rule ID 5 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s32,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR32RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR32RegClassID,
      // (bswap:{ *:[i32] } GR32:{ *:[i32] }:$src)  =>  (BSWAP32r:{ *:[i32] } GR32:{ *:[i32] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::BSWAP32r,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 5,
      GIR_Done,
    // Label 1384: @32257
    GIM_Reject,
    // Label 1381: @32258
    GIM_Try, /*On fail goto*//*Label 1385*/ 32279, // Rule ID 6 //
      GIM_CheckType, /*MI*/0, /*Op*/1, /*Type*/GILLT_s64,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/0, /*RC*/X86::GR64RegClassID,
      GIM_CheckRegBankForClass, /*MI*/0, /*Op*/1, /*RC*/X86::GR64RegClassID,
      // (bswap:{ *:[i64] } GR64:{ *:[i64] }:$src)  =>  (BSWAP64r:{ *:[i64] } GR64:{ *:[i64] }:$src)
      GIR_MutateOpcode, /*InsnID*/0, /*RecycleInsnID*/0, /*Opcode*/X86::BSWAP64r,
      GIR_ConstrainSelectedInstOperands, /*InsnID*/0,
      // GIR_Coverage, 6,
      GIR_Done,
    // Label 1385: @32279
    GIM_Reject,
    // Label 1382: @32280
    GIM_Reject,
    // Label 34: @32281
    GIM_Reject,
    };
  return MatchTable0;
}
#endif // ifdef GET_GLOBALISEL_IMPL
#ifdef GET_GLOBALISEL_PREDICATES_DECL
PredicateBitset AvailableModuleFeatures;
mutable PredicateBitset AvailableFunctionFeatures;
PredicateBitset getAvailableFeatures() const {
  return AvailableModuleFeatures | AvailableFunctionFeatures;
}
PredicateBitset
computeAvailableModuleFeatures(const X86Subtarget *Subtarget) const;
PredicateBitset
computeAvailableFunctionFeatures(const X86Subtarget *Subtarget,
                                 const MachineFunction *MF) const;
#endif // ifdef GET_GLOBALISEL_PREDICATES_DECL
#ifdef GET_GLOBALISEL_PREDICATES_INIT
AvailableModuleFeatures(computeAvailableModuleFeatures(&STI)),
AvailableFunctionFeatures()
#endif // ifdef GET_GLOBALISEL_PREDICATES_INIT
