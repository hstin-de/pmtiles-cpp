# pmtiles-cpp

Tiny C++17 lib with a plain C API to write **PMTiles v3**. Internal dirs+metadata are gzip-compressed; tile payloads are written as-is. **Output bytes match the reference.**

**Disclaimer:** Independent project — not affiliated with, endorsed by, or sponsored by PMTiles or Protomaps.

## Build

Deps: **zlib**, **xxhash**
Arch: `sudo pacman -S base-devel cmake zlib xxhash`

```bash
cmake -S . -B build -DPMTILES_BUILD_SHARED=ON
cmake --build build --config Release
# optional
sudo cmake --install build
```

## API (see `include/pmtiles.h`)

```c
struct pm_tile_in { int z, x, y; const uint8_t* data; int len; };
int  pmtiles_build_to_file (const struct pm_tile_in*, int, uint8_t, uint8_t, const char*);
/* malloc'd buffer; free with pmtiles_free_buffer */
int  pmtiles_build_to_memory(const struct pm_tile_in*, int, uint8_t, uint8_t, uint8_t** , int*);
void pmtiles_free_buffer(void*);
```

## Minimal example

```c
struct pm_tile_in t[] = { {0,0,0, webp_bytes, webp_len} };
if (!pmtiles_build_to_file(t, 1, 0, 0, "out.pmtiles")) return 1;
```