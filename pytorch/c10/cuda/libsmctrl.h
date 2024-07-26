/**
 * Copyright 2023 Joshua Bakita
 * Library to control TPC masks on CUDA launches. Co-opts preexisting debug
 * logic in the CUDA driver library, and thus requires a build with -lcuda.
 */


#include <stdint.h>
typedef unsigned __int128 uint128_t;
namespace c10::cuda {
/* PARTITIONING FUNCTIONS */

// Set global default TPC mask for all kernels, incl. CUDA-internal ones
// @param mask   A bitmask of enabled/disabled TPCs (see Notes on Bitmasks)
// Supported: CUDA 10.2, and CUDA 11.0 - CUDA 12.1
extern void libsmctrl_set_global_mask();
extern void libsmctrl_set_next_mask();
extern void libsmctrl_set_stream_mask(void* stream, uint64_t mask);
extern void libsmctrl_set_stream_mask_ext(void* stream, uint128_t mask);
}