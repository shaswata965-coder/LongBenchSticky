// static_switch.h — Compile-time dtype dispatch macros
// Identical to DefensiveKV's implementation

#pragma once

// Dispatches between fp16 and bf16 at runtime using a compile-time template
#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = at::Half;            \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = at::BFloat16;        \
      return __VA_ARGS__();                  \
    }                                        \
  }()

// Extended dispatch: fp16 / bf16 / fp32
#define DTYPE_SWITCH(DTYPE, ...)                           \
  [&] {                                                    \
    if ((DTYPE) == torch::kFloat16) {                      \
      using elem_type = at::Half;                          \
      return __VA_ARGS__();                                \
    } else if ((DTYPE) == torch::kBFloat16) {              \
      using elem_type = at::BFloat16;                      \
      return __VA_ARGS__();                                \
    } else if ((DTYPE) == torch::kFloat32) {               \
      using elem_type = float;                             \
      return __VA_ARGS__();                                \
    } else {                                               \
      TORCH_CHECK(false, "Unsupported dtype for kernel");  \
    }                                                      \
  }()
