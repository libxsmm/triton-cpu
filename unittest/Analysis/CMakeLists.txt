add_triton_ut(
  NAME TestTritonAnalysis
  SRCS UtilityTest.cpp
  LIBS
    MLIRParser
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
)
