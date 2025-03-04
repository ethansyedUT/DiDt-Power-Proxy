#ifndef __SKIP_INTERNAL_FATBINARY_HEADERS
#include "fatbinary_section.h"
#endif
#define __CUDAFATBINSECTION  ".nvFatBinSegment"
#define __CUDAFATBINDATASECTION  ".nv_fatbin"
asm(
".section .nv_fatbin, \"a\"\n"
".align 8\n"
"fatbinData:\n"
".quad 0x00100001ba55ed50,0x00000000000003a8,0x0000004001010002,0x0000000000000368\n"
".quad 0x0000000000000000,0x0000003400010007,0x0000000000000000,0x0000000000000011\n"
".quad 0x0000000000000000,0x0000000000000000,0x33010102464c457f,0x0000000000000007\n"
".quad 0x0000008000be0002,0x0000000000000000,0x00000000000002f8,0x0000000000000178\n"
".quad 0x0038004000340534,0x0001000600400002,0x7472747368732e00,0x747274732e006261\n"
".quad 0x746d79732e006261,0x746d79732e006261,0x78646e68735f6261,0x666e692e766e2e00\n"
".quad 0x61632e766e2e006f,0x0068706172676c6c,0x746f72702e766e2e,0x6e2e00657079746f\n"
".quad 0x63612e6c65722e76,0x732e00006e6f6974,0x0062617472747368,0x006261747274732e\n"
".quad 0x006261746d79732e,0x5f6261746d79732e,0x6e2e0078646e6873,0x2e006f666e692e76\n"
".quad 0x676c6c61632e766e,0x766e2e0068706172,0x79746f746f72702e,0x722e766e2e006570\n"
".quad 0x6f697463612e6c65,0x000000000000006e,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0004000300000032,0x0000000000000000,0x0000000000000000\n"
".quad 0x000500030000004e,0x0000000000000000,0x0000000000000000,0xffffffff00000000\n"
".quad 0xfffffffe00000000,0xfffffffd00000000,0xfffffffc00000000,0x0000000000000073\n"
".quad 0x3605002511000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000000,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000300000001,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000040,0x000000000000005d,0x0000000000000000,0x0000000000000001\n"
".quad 0x0000000000000000,0x000000030000000b,0x0000000000000000,0x0000000000000000\n"
".quad 0x000000000000009d,0x000000000000005d,0x0000000000000000,0x0000000000000001\n"
".quad 0x0000000000000000,0x0000000200000013,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000100,0x0000000000000048,0x0000000300000002,0x0000000000000008\n"
".quad 0x0000000000000018,0x7000000100000032,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000148,0x0000000000000020,0x0000000000000003,0x0000000000000004\n"
".quad 0x0000000000000008,0x7000000b0000004e,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000168,0x0000000000000010,0x0000000000000000,0x0000000000000008\n"
".quad 0x0000000000000008,0x0000000500000006,0x00000000000002f8,0x0000000000000000\n"
".quad 0x0000000000000000,0x0000000000000070,0x0000000000000070,0x0000000000000008\n"
".quad 0x0000000500000001,0x00000000000002f8,0x0000000000000000,0x0000000000000000\n"
".quad 0x0000000000000070, 0x0000000000000070, 0x0000000000000008\n"
".text\n");
#ifdef __cplusplus
extern "C" {
#endif
extern const unsigned long long fatbinData[119];
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C" {
#endif
static const __fatBinC_Wrapper_t __fatDeviceText __attribute__ ((aligned (8))) __attribute__ ((section (__CUDAFATBINSECTION)))= 
	{ 0x466243b1, 2, fatbinData, (void**)__cudaPrelinkedFatbins };
#ifdef __cplusplus
}
#endif
