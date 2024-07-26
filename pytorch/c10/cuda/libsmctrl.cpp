#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>
#include "libsmctrl.h"
#include <cuda.h>
#include <string>
#include <stdexcept>
#include <inttypes.h>
#define abort(ret, errno, ...) error_at_line(ret, errno, __FILE__, __LINE__, \
                                             __VA_ARGS__)
namespace c10::cuda {
static const CUuuid callback_funcs_id = {0x2c, (char)0x8e, 0x0a, (char)0xd8, 0x07, 0x10, (char)0xab, 0x4e, (char)0x90, (char)0xdd, 0x54, 0x71, (char)0x9f, (char)0xe5, (char)0xf7, 0x4b};

#define LAUNCH_DOMAIN 0x3
#define LAUNCH_PRE_UPLOAD 0x3

static uint64_t g_sm_mask = 0;
static thread_local uint64_t g_next_sm_mask = 0;
static char sm_control_setup_called = 0;

static void launchCallback(void* ukwn, int domain, int cbid, const void* in_params) {
    if (*(uint32_t*)in_params < 0x50) {
        abort(1, 0, "Unsupported CUDA version for callback-based SM masking. Aborting...");
    }

    void* tmd = **((uintptr_t***)in_params + 8);
    if (!tmd) {
        abort(1, 0, "TMD allocation appears NULL; likely forward-compatibility issue.\n");
    }
    uint32_t* lower_ptr = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(tmd) + 84);
    uint32_t* upper_ptr = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(tmd) + 88);

    if (g_next_sm_mask) {
        *lower_ptr = static_cast<uint32_t>(g_next_sm_mask);
        *upper_ptr = static_cast<uint32_t>(g_next_sm_mask >> 32);
        g_next_sm_mask = 0;
    } else if (!*lower_ptr && !*upper_ptr) {
        // printf("yes i am called!!\n  ");
        // printf("%" PRIu64 "\n", g_sm_mask);
        *lower_ptr = static_cast<uint32_t>(g_sm_mask);
        *upper_ptr = static_cast<uint32_t>(g_sm_mask >> 32);
    }
}
static void setup_sm_control_11() {
    using SubscribeFunc = int (*)(uint32_t*, void(*)(void*, int, int, const void*), void*);
    using EnableFunc = int (*)(uint32_t, uint32_t, int, int);
    const void* tbl_base;
    uint32_t my_hndl;
    // Avoid race conditions (setup can only be called once)
    if (__atomic_test_and_set(&sm_control_setup_called, __ATOMIC_SEQ_CST)) {
        return;
    }
    cuGetExportTable(&tbl_base, &callback_funcs_id);

    const uintptr_t* uintptr_tbl_base = static_cast<const uintptr_t*>(tbl_base);
    uintptr_t subscribe_func_addr = *(uintptr_tbl_base + 3);
    uintptr_t enable_func_addr = *(uintptr_tbl_base + 6);
    SubscribeFunc subscribe = reinterpret_cast<SubscribeFunc>(subscribe_func_addr);
    EnableFunc enable = reinterpret_cast<EnableFunc>(enable_func_addr);

    int res = 0;

    res = subscribe(&my_hndl, launchCallback, nullptr);
    printf("res: %d\n", res);
    if (res) {
        abort(1, 0, "Error subscribi ng to launch callback. CUDA returned error code %d.", res);
    }
    res = enable(1, my_hndl, LAUNCH_DOMAIN, LAUNCH_PRE_UPLOAD);
    printf("res ENABLE: %d\n", res);
    if (res) {
        abort(1, 0, "Error enabling launch callback. CUDA returned error code %d.", res);
    }
}


// Set default mask for all launches
void libsmctrl_set_global_mask() {
	printf("iam called global_mask\n");
	uint128_t mask = (1ULL << 0)
                    |(1ULL << 1)
					|(1ULL << 2)
                    |(1ULL << 3)
                    |(1ULL << 4)

                    |(1ULL << 5)
                    |(1ULL << 6)
                    |(1ULL << 7)
                    |(1ULL << 8)
                    |(1ULL << 9)


                    |(1ULL << 10)
                    |(1ULL << 11)
                    |(1ULL << 12)
                    |(1ULL << 13) 
                    |(1ULL << 14)


                    |(1ULL << 15)
					|(1ULL << 16)
                    |(1ULL << 17)
                    |(1ULL << 18) 
                    |(1ULL << 19)

					|(1ULL << 20)
                    |(1ULL << 21)
                    |(1ULL << 22)
                    |(1ULL << 23);
 	mask = ~mask;
	if (!sm_control_setup_called) {
        setup_sm_control_11();
	}
	g_sm_mask = mask;
}

// Set mask for next launch from this thread
void libsmctrl_set_next_mask() {
	// printf("iam called next_mask\n");
	// uint128_t mask = (1ULL << 1) ;
 	// mask = ~mask;
	if (!sm_control_setup_called) {
		setup_sm_control_11();
	}
	// g_next_sm_mask = mask;
}
/*** Per-Stream SM Mask (unlikely to be forward-compatible) ***/

// Offsets for the stream struct on x86_64
#define CU_8_0_MASK_OFF 0xec
#define CU_9_0_MASK_OFF 0x130
// CUDA 9.0 and 9.1 use the same offset
// 9.1 tested on 390.157
#define CU_9_2_MASK_OFF 0x140
#define CU_10_0_MASK_OFF 0x244
// CUDA 10.0, 10.1 and 10.2 use the same offset
// 10.1 tested on 418.113
// 10.2 tested on 440.100, 440.82, 440.64, and 440.36
#define CU_11_0_MASK_OFF 0x274
#define CU_11_1_MASK_OFF 0x2c4
#define CU_11_2_MASK_OFF 0x37c
// CUDA 11.2, 11.3, 11.4, and 11.5 use the same offset
// 11.4 tested on 470.223.02
#define CU_11_6_MASK_OFF 0x38c
#define CU_11_7_MASK_OFF 0x3c4
#define CU_11_8_MASK_OFF 0x47c
// 11.8 tested on 520.56.06
#define CU_12_0_MASK_OFF 0x4cc
// CUDA 12.0 and 12.1 use the same offset
// 12.0 tested on 525.147.05
#define CU_12_2_MASK_OFF 0x4e4
// 12.2 tested on 535.129.03

// Offsets for the stream struct on aarch64
// All tested on Nov 13th, 2023
#define CU_9_0_MASK_OFF_JETSON 0x128 // Tested on TX2
#define CU_10_2_MASK_OFF_JETSON 0x24c // Tested on TX2 and Jetson Xavier
#define CU_11_4_MASK_OFF_JETSON 0x394 // Tested on Jetson Orin

// Used up through CUDA 11.8 in the stream struct
struct stream_sm_mask {
	uint32_t upper;
	uint32_t lower;
};

// Used starting with CUDA 12.0 in the stream struct
struct stream_sm_mask_v2 {
	uint32_t enabled;
	uint32_t mask[4];
};

// Check if this system has a Parker SoC (TX2/PX2 chip)
// (CUDA 9.0 behaves slightly different on this platform.)
// @return 1 if detected, 0 if not, -cuda_err on error
#if __aarch64__
int detect_parker_soc() {
	int cap_major, cap_minor, err, dev_count;
	if (err = cuDeviceGetCount(&dev_count))
		return -err;
	// As CUDA devices are numbered by order of compute power, check every
	// device, in case a powerful discrete GPU is attached (such as on the
	// DRIVE PX2). We detect the Parker SoC via its unique CUDA compute
	// capability: 6.2.
	for (int i = 0; i < dev_count; i++) {
		if (err = cuDeviceGetAttribute(&cap_minor,
		                               CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
		                               i))
			return -err;
		if (err = cuDeviceGetAttribute(&cap_major,
		                               CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
		                               i))
			return -err;
		if (cap_major == 6 && cap_minor == 2)
			return 1;
	}
	return 0;
}
#endif // __aarch64__

// Should work for CUDA 8.0 through 12.2
// A cudaStream_t is a CUstream*. We use void* to avoid a cuda.h dependency in
// our header
void libsmctrl_set_stream_mask(void* stream, uint64_t mask) {
    uint128_t full_mask = static_cast<uint128_t>(-1);
    full_mask <<= 64;
    full_mask |= mask;
    libsmctrl_set_stream_mask_ext(stream, full_mask);
}

// Function to set extended stream mask
void libsmctrl_set_stream_mask_ext(void* stream, uint128_t mask) {
    char* stream_struct_base = *reinterpret_cast<char**>(stream);
    stream_sm_mask* hw_mask = nullptr;
    stream_sm_mask_v2* hw_mask_v2 = nullptr;
    int ver;
    cuDriverGetVersion(&ver);

    switch (ver) {
        case 8000:
            hw_mask = reinterpret_cast<stream_sm_mask*>(stream_struct_base + CU_8_0_MASK_OFF);
            break;
        case 9000:
        case 9010:
            hw_mask = reinterpret_cast<stream_sm_mask*>(stream_struct_base + CU_9_0_MASK_OFF);
            break;
        case 9020:
            hw_mask = reinterpret_cast<stream_sm_mask*>(stream_struct_base + CU_9_2_MASK_OFF);
            break;
        case 10000:
        case 10010:
        case 10020:
            hw_mask = reinterpret_cast<stream_sm_mask*>(stream_struct_base + CU_10_0_MASK_OFF);
            break;
        case 11000:
            hw_mask = reinterpret_cast<stream_sm_mask*>(stream_struct_base + CU_11_0_MASK_OFF);
            break;
        case 11010:
            hw_mask = reinterpret_cast<stream_sm_mask*>(stream_struct_base + CU_11_1_MASK_OFF);
            break;
        case 11020:
        case 11030:
        case 11040:
        case 11050:
            hw_mask = reinterpret_cast<stream_sm_mask*>(stream_struct_base + CU_11_2_MASK_OFF);
            break;
        case 11060:
            hw_mask = reinterpret_cast<stream_sm_mask*>(stream_struct_base + CU_11_6_MASK_OFF);
            break;
        case 11070:
            hw_mask = reinterpret_cast<stream_sm_mask*>(stream_struct_base + CU_11_7_MASK_OFF);
            break;
        case 11080:
            hw_mask = reinterpret_cast<stream_sm_mask*>(stream_struct_base + CU_11_8_MASK_OFF);
            break;
        case 12000:
        case 12010:
            hw_mask_v2 = reinterpret_cast<stream_sm_mask_v2*>(stream_struct_base + CU_12_0_MASK_OFF);
            break;
        case 12020:
            hw_mask_v2 = reinterpret_cast<stream_sm_mask_v2*>(stream_struct_base + CU_12_2_MASK_OFF);
            break;
        case 12040:
            hw_mask_v2 = reinterpret_cast<stream_sm_mask_v2*>(stream_struct_base + CU_11_8_MASK_OFF);
            break;
#if defined(__aarch64__)
        case 9000: {
            int is_parker = detect_parker_soc();
            if (is_parker < 0) {
                const char* err_str;
                cuGetErrorName(-is_parker, &err_str);
                throw std::runtime_error(std::string("CUDA call failed with error: ") + err_str);
            }
            if (!is_parker) {
                throw std::runtime_error("Not supported on non-Jetson aarch64.");
            }
            hw_mask = reinterpret_cast<stream_sm_mask*>(stream_struct_base + CU_9_0_MASK_OFF_JETSON);
            break;
        }
        case 10020:
            hw_mask = reinterpret_cast<stream_sm_mask*>(stream_struct_base + CU_10_2_MASK_OFF_JETSON);
            break;
        case 11040:
            hw_mask = reinterpret_cast<stream_sm_mask*>(stream_struct_base + CU_11_4_MASK_OFF_JETSON);
            break;
#endif
        default:
            break;
    }

    // Check MASK_OFF environment variable for experimental offsets
    char* mask_off_str = std::getenv("MASK_OFF");
    if (mask_off_str) {
        int off = std::atoi(mask_off_str);
        std::fprintf(stderr, "libsmctrl: Attempting offset %d on CUDA 12.2 base %#x (total off: %#x)\n", off, CU_12_2_MASK_OFF, CU_12_2_MASK_OFF + off);
        if (CU_12_2_MASK_OFF + off < 0) {
            throw std::runtime_error("Total offset cannot be less than 0! Aborting...");
        }
        hw_mask_v2 = reinterpret_cast<stream_sm_mask_v2*>(stream_struct_base + CU_12_2_MASK_OFF + off);
    }

    // Apply the mask
    if (hw_mask) {
        hw_mask->upper = static_cast<uint32_t>(mask >> 32);
        hw_mask->lower = static_cast<uint32_t>(mask);
    } else if (hw_mask_v2) {
        hw_mask_v2->enabled = 1;
        hw_mask_v2->mask[0] = static_cast<uint32_t>(mask);
        hw_mask_v2->mask[1] = static_cast<uint32_t>(mask >> 32);
        hw_mask_v2->mask[2] = static_cast<uint32_t>(mask >> 64);
        hw_mask_v2->mask[3] = static_cast<uint32_t>(mask >> 96);
    } else {
        throw std::runtime_error("Stream masking unsupported on this CUDA version (" + std::to_string(ver) + "), and no fallback MASK_OFF set!");
    }
}

/* INFORMATIONAL FUNCTIONS */

// Read an integer from a file in `/proc`
static int read_int_procfile(char* filename, uint64_t* out) {
	char f_data[18] = {0};
	int fd = open(filename, O_RDONLY);
	if (fd == -1)
		return errno;
	read(fd, f_data, 18);
	close(fd);
	*out = strtoll(f_data, NULL, 16);
	return 0;
}

// We support up to 12 GPCs per GPU, and up to 16 GPUs.
static uint64_t tpc_mask_per_gpc_per_dev[16][12];
// Output mask is vtpc-indexed (virtual TPC)
int libsmctrl_get_gpc_info(uint32_t* num_enabled_gpcs, uint64_t** tpcs_for_gpc, int dev) {
	uint32_t i, j, vtpc_idx = 0;
	uint64_t gpc_mask, num_tpc_per_gpc, max_gpcs, gpc_tpc_mask;
	int err;
	char filename[100];
	*num_enabled_gpcs = 0;
	// Maximum number of GPCs supported for this chip
	snprintf(filename, 100, "/proc/gpu%d/num_gpcs", dev);
	if (err = read_int_procfile(filename, &max_gpcs)) {
		fprintf(stderr, "libsmctrl: nvdebug module must be loaded into kernel before "
				"using libsmctrl_get_*_info() functions\n");
		return err;
	}
	// TODO: handle arbitrary-size GPUs
	if (dev > 16 || max_gpcs > 12) {
		fprintf(stderr, "libsmctrl: GPU possibly too large for preallocated map!\n");
		return ERANGE;
	}
	// Set bit = disabled GPC
	snprintf(filename, 100, "/proc/gpu%d/gpc_mask", dev);
	if (err = read_int_procfile(filename, &gpc_mask))
		return err;
	snprintf(filename, 100, "/proc/gpu%d/num_tpc_per_gpc", dev);
	if (err = read_int_procfile(filename, &num_tpc_per_gpc))
		return err;
	// For each enabled GPC
	for (i = 0; i < max_gpcs; i++) {
		// Skip this GPC if disabled
		if ((1 << i) & gpc_mask)
			continue;
		(*num_enabled_gpcs)++;
		// Get the bitstring of TPCs disabled for this GPC
		// Set bit = disabled TPC
		snprintf(filename, 100, "/proc/gpu%d/gpc%d_tpc_mask", dev, i);
		if (err = read_int_procfile(filename, &gpc_tpc_mask))
			return err;
		uint64_t* tpc_mask = &tpc_mask_per_gpc_per_dev[dev][*num_enabled_gpcs - 1];
		*tpc_mask = 0;
		for (j = 0; j < num_tpc_per_gpc; j++) {
				// Skip disabled TPCs
				if ((1 << j) & gpc_tpc_mask)
					continue;
				*tpc_mask |= (1ull << vtpc_idx);
				vtpc_idx++;
		}
	}
	*tpcs_for_gpc = tpc_mask_per_gpc_per_dev[dev];
	return 0;
}

int libsmctrl_get_tpc_info(uint32_t* num_tpcs, int dev) {
	uint32_t num_gpcs;
	uint64_t* tpcs_per_gpc;
	int res;
	if (res = libsmctrl_get_gpc_info(&num_gpcs, &tpcs_per_gpc, dev))
		return res;
	*num_tpcs = 0;
	for (int gpc = 0; gpc < num_gpcs; gpc++) {
		*num_tpcs += __builtin_popcountl(tpcs_per_gpc[gpc]);
	}
	return 0;
}

// @param dev Device index as understood by CUDA **can differ from nvdebug idx**
// This implementation is fragile, and could be incorrect for odd GPUs
// int libsmctrl_get_tpc_info_cuda(uint32_t* num_tpcs, int cuda_dev) {
//     int num_sms, major, minor;
//     CUresult res;
//     const char* err_str;

//     if ((res = cuInit(0)) != CUDA_SUCCESS)
//         goto abort_cuda;
//     if ((res = cuDeviceGetAttribute(&num_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuda_dev)) != CUDA_SUCCESS)
//         goto abort_cuda;
//     if ((res = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuda_dev)) != CUDA_SUCCESS)
//         goto abort_cuda;
//     if ((res = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuda_dev)) != CUDA_SUCCESS)
//         goto abort_cuda;

//     // SM masking only works on sm_35+
//     if (major < 3 || (major == 3 && minor < 5))
//         return ENOTSUP;

//     // Everything newer than Pascal (as of Hopper) has 2 SMs per TPC, as well
//     // as the P100, which is uniquely sm_60
//     int sms_per_tpc;
//     if (major > 6 || (major == 6 && minor == 0))
//         sms_per_tpc = 2;
//     else
//         sms_per_tpc = 1;

//     // It looks like there may be some upcoming weirdness (TPCs with only one SM?)
//     // with Hopper
//     if (major >= 9)
//         fprintf(stderr, "libsmctrl: WARNING, TPC masking is untested on Hopper,"
//                         " and will likely yield incorrect results! Proceed with caution.\n");

//     *num_tpcs = num_sms / sms_per_tpc;
//     return 0;

// abort_cuda:
//     cuGetErrorName(res, &err_str);
//     fprintf(stderr, "libsmctrl: CUDA call failed due to %s. Failing with EIO...\n", err_str);
//     return EIO;
// }


} // namespace c10::cuda