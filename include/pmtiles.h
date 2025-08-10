#pragma once
#include <stdint.h>

#if defined(_WIN32) && !defined(PMTILES_STATIC)
  #if defined(PMTILES_BUILDING_LIB)
    #define PMTILES_API __declspec(dllexport)
  #else
    #define PMTILES_API __declspec(dllimport)
  #endif
#else
  #define PMTILES_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct pm_tile_in {
    int z, x, y;
    const uint8_t* data;
    int len;
};

// Build to a malloc'd buffer. Return 1 on success; caller frees with pmtiles_free_buffer.
PMTILES_API int pmtiles_build_to_memory(const struct pm_tile_in* tiles, int count,
                                        uint8_t minZoom, uint8_t maxZoom,
                                        uint8_t** out_buf, int* out_len);

// Build straight to a file. Return 1 on success. Output file bytes are identical.
PMTILES_API int pmtiles_build_to_file(const struct pm_tile_in* tiles, int count,
                                      uint8_t minZoom, uint8_t maxZoom,
                                      const char* filename);

// Free memory returned by pmtiles_build_to_memory.
PMTILES_API void pmtiles_free_buffer(void* p);

#ifdef __cplusplus
}
#endif
