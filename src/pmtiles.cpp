#include "pmtiles.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <chrono>
#include <ctime>

#include <zlib.h>
#include <xxhash.h>

namespace pmt {

enum class Compression : uint8_t {
    Unknown = 0,
    None    = 1,
    Gzip    = 2,
    Brotli  = 3,
    Zstd    = 4
};

enum class TileType : uint8_t {
    Unknown = 0,
    Mvt     = 1,
    Png     = 2,
    Jpeg    = 3,
    Webp    = 4,
    Avif    = 5
};

constexpr size_t HeaderV3LenBytes = 127;

struct HeaderV3 {
    uint64_t RootOffset          = 0;
    uint64_t RootLength          = 0;
    uint64_t MetadataOffset      = 0;
    uint64_t MetadataLength      = 0;
    uint64_t LeafDirectoryOffset = 0;
    uint64_t LeafDirectoryLength = 0;
    uint64_t TileDataOffset      = 0;
    uint64_t TileDataLength      = 0;
    uint64_t AddressedTilesCount = 0;
    uint64_t TileEntriesCount    = 0;
    uint64_t TileContentsCount   = 0;
    bool     Clustered           = true;
    Compression InternalCompression = Compression::Gzip;
    Compression TileCompression     = Compression::None;
    TileType TType = TileType::Webp;
    uint8_t  MinZoom = 0;
    uint8_t  MaxZoom = 0;
    int32_t  MinLonE7 = -1800000000;
    int32_t  MinLatE7 =  -850000000;
    int32_t  MaxLonE7 =  1800000000;
    int32_t  MaxLatE7 =   850000000;
    uint8_t  CenterZoom  = 0;
    int32_t  CenterLonE7 = 0;
    int32_t  CenterLatE7 = 0;
};

struct EntryV3 {
    uint64_t TileID;
    uint64_t Offset;
    uint32_t Length;
    uint32_t RunLength;
};

// Hilbert helpers (Z,X,Y <-> ID)
static inline void rotate(uint32_t n, uint32_t& x, uint32_t& y, uint32_t rx, uint32_t ry) {
    if (ry == 0) {
        if (rx != 0) {
            x = n - 1 - x;
            y = n - 1 - y;
        }
        uint32_t t = x; x = y; y = t;
    }
}
static inline uint64_t ZxyToID(uint8_t z, uint32_t x, uint32_t y) {
    uint64_t acc = ((((uint64_t)1) << (z * 2)) - 1) / 3;
    if (z == 0) return acc;
    uint32_t n = (uint32_t)z - 1;
    for (uint32_t s = (1u << n); s > 0; s >>= 1, n--) {
        uint32_t rx = x & s;
        uint32_t ry = y & s;
        acc += (uint64_t(((3 * rx) ^ ry)) << n);
        rotate(s, x, y, rx, ry);
    }
    return acc;
}

} // namespace pmt

struct OffsetLength { uint64_t Offset; uint32_t Length; };

struct Hash128 { uint64_t lo, hi; };
static inline bool operator==(const Hash128& a, const Hash128& b) noexcept {
    return a.lo == b.lo && a.hi == b.hi;
}
struct Hash128Hasher {
    size_t operator()(const Hash128& h) const noexcept {
        uint64_t x = h.lo ^ (h.hi + 0x9e3779b97f4a7c15ULL + (h.lo<<6) + (h.lo>>2));
#if INTPTR_MAX == INT64_MAX
        return (size_t)x;
#else
        return (size_t)(x ^ (x >> 32));
#endif
    }
};

static inline Hash128 xxh3_128(const uint8_t* p, size_t n) {
    XXH128_hash_t h = XXH3_128bits(p, n);
    return Hash128{h.low64, h.high64};
}

static inline void put64LE(std::vector<uint8_t>& b, size_t i, uint64_t v) {
    b[i+0]=(uint8_t)(v); b[i+1]=(uint8_t)(v>>8); b[i+2]=(uint8_t)(v>>16); b[i+3]=(uint8_t)(v>>24);
    b[i+4]=(uint8_t)(v>>32); b[i+5]=(uint8_t)(v>>40); b[i+6]=(uint8_t)(v>>48); b[i+7]=(uint8_t)(v>>56);
}
static inline void put32LE(std::vector<uint8_t>& b, size_t i, uint32_t v) {
    b[i+0]=(uint8_t)(v); b[i+1]=(uint8_t)(v>>8); b[i+2]=(uint8_t)(v>>16); b[i+3]=(uint8_t)(v>>24);
}

static inline void writeUVarint(std::vector<uint8_t>& out, uint64_t v) {
    while (v >= 0x80) { out.push_back(uint8_t(v) | 0x80); v >>= 7; }
    out.push_back(uint8_t(v));
}

static bool gzipCompress(const std::vector<uint8_t>& in, std::vector<uint8_t>& out, int level) {
    z_stream zs{};
    int init = deflateInit2(&zs, level, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY);
    if (init != Z_OK) { return false; }
    zs.next_in  = const_cast<Bytef*>(reinterpret_cast<const Bytef*>(in.data()));
    zs.avail_in = (uInt)in.size();

    std::vector<uint8_t> buf(std::max<size_t>(in.size()/2 + 64, 4096));
    int ret;
    do {
        zs.next_out = reinterpret_cast<Bytef*>(buf.data());
        zs.avail_out = (uInt)buf.size();
        ret = deflate(&zs, zs.avail_in ? Z_NO_FLUSH : Z_FINISH);
        size_t have = buf.size() - zs.avail_out;
        out.insert(out.end(), buf.data(), buf.data() + have);
    } while (ret == Z_OK);

    deflateEnd(&zs);
    return ret == Z_STREAM_END;
}

static std::vector<uint8_t> serializeEntries(
    const std::vector<pmt::EntryV3>& entries,
    bool compress /* gzip if true */)
{
    std::vector<uint8_t> raw;
    raw.reserve(entries.size() * 6);

    writeUVarint(raw, (uint64_t)entries.size());

    uint64_t lastID = 0;
    for (const auto& e : entries) {
        writeUVarint(raw, e.TileID - lastID);
        lastID = e.TileID;
    }
    for (const auto& e : entries) writeUVarint(raw, e.RunLength);
    for (const auto& e : entries) writeUVarint(raw, e.Length);

    for (size_t i = 0; i < entries.size(); ++i) {
        if (i > 0) {
            const auto& prev = entries[i-1];
            const auto& cur  = entries[i];
            if (cur.Offset == prev.Offset + prev.Length) { writeUVarint(raw, 0); continue; }
        }
        writeUVarint(raw, entries[i].Offset + 1);
    }

    if (!compress) return raw;

    std::vector<uint8_t> gz;
    gz.reserve(raw.size()/2 + 64);
    if (!gzipCompress(raw, gz, Z_BEST_COMPRESSION)) {
        return raw;
    }
    return gz;
}

static std::pair<std::vector<uint8_t>, std::vector<uint8_t>>
buildRootsLeaves(const std::vector<pmt::EntryV3>& entries,
                 int leafSize,
                 bool compress)
{
    if (leafSize <= 0) leafSize = 1;
    std::vector<pmt::EntryV3> rootEntries;
    rootEntries.reserve((entries.size() + size_t(leafSize) - 1) / size_t(leafSize));

    std::vector<uint8_t> leavesBytes;

    for (size_t idx = 0; idx < entries.size();) {
        const size_t span = std::min<size_t>(leafSize, entries.size() - idx);
        const size_t end  = idx + span;

        std::vector<pmt::EntryV3> chunk(entries.begin() + idx, entries.begin() + end);

        auto ser = serializeEntries(chunk, compress);

        pmt::EntryV3 re{};
        re.TileID    = chunk.front().TileID;
        re.Offset    = (uint64_t)leavesBytes.size();
        re.Length    = (uint32_t)ser.size();
        re.RunLength = 0;

        rootEntries.push_back(re);
        leavesBytes.insert(leavesBytes.end(), ser.begin(), ser.end());

        idx = end;
    }

    auto rootBytes = serializeEntries(rootEntries, compress);
    return {std::move(rootBytes), std::move(leavesBytes)};
}

static std::pair<std::vector<uint8_t>, std::vector<uint8_t>>
optimizeDirectories(const std::vector<pmt::EntryV3>& entries,
                    int targetRootLen,
                    bool compress)
{
    { // try single root first
        auto testRoot = serializeEntries(entries, compress);
        if (testRoot.size() <= (size_t)targetRootLen) {
            return {std::move(testRoot), {}};
        }
    }

    int leafSize = std::max<int>(4096, int(float(entries.size()) / 3500.f));
    if (leafSize <= 0) leafSize = 1;

    for (;;) {
        auto [rootBytes, leavesBytes] = buildRootsLeaves(entries, leafSize, compress);
        if (rootBytes.size() <= (size_t)targetRootLen) {
            return {std::move(rootBytes), std::move(leavesBytes)};
        }
        if (leafSize > int(entries.size())) {
            return {std::move(rootBytes), std::move(leavesBytes)};
        }
        leafSize = int(double(leafSize) * 1.2 + 1);
    }
}

static std::vector<uint8_t> serializeHeader(const pmt::HeaderV3& h) {
    using pmt::HeaderV3LenBytes;
    std::vector<uint8_t> b(HeaderV3LenBytes, 0);

    b[0]='P'; b[1]='M'; b[2]='T'; b[3]='i'; b[4]='l'; b[5]='e'; b[6]='s';
    b[7] = 3;

    auto P64=[&](size_t o, uint64_t v){ put64LE(b,o,v); };
    auto P32=[&](size_t o, uint32_t v){ put32LE(b,o,v); };

    P64( 8, h.RootOffset);
    P64(16, h.RootLength);
    P64(24, h.MetadataOffset);
    P64(32, h.MetadataLength);
    P64(40, h.LeafDirectoryOffset);
    P64(48, h.LeafDirectoryLength);
    P64(56, h.TileDataOffset);
    P64(64, h.TileDataLength);
    P64(72, h.AddressedTilesCount);
    P64(80, h.TileEntriesCount);
    P64(88, h.TileContentsCount);

    b[96]  = h.Clustered ? 0x1 : 0x0;
    b[97]  = (uint8_t)h.InternalCompression;
    b[98]  = (uint8_t)h.TileCompression;
    b[99]  = (uint8_t)h.TType;
    b[100] = h.MinZoom;
    b[101] = h.MaxZoom;

    P32(102, (uint32_t)h.MinLonE7);
    P32(106, (uint32_t)h.MinLatE7);
    P32(110, (uint32_t)h.MaxLonE7);
    P32(114, (uint32_t)h.MaxLatE7);

    b[118] = h.CenterZoom;
    P32(119, (uint32_t)h.CenterLonE7);
    P32(123, (uint32_t)h.CenterLatE7);

    return b;
}

static std::string buildMetadataJSON(uint8_t minZoom, uint8_t maxZoom) {
    using clock = std::chrono::system_clock;
    auto now = clock::now();
    std::time_t tt = clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    char when[32];
    std::strftime(when, sizeof(when), "%Y-%m-%d %H:%M", &tm);

    std::string s;
    s.reserve(256);
    s += "{";
    s += "\"format\":\"webp\",";
    s += "\"minzoom\":"; s += std::to_string((int)minZoom); s += ",";
    s += "\"maxzoom\":"; s += std::to_string((int)maxZoom); s += ",";
    s += "\"name\":\"Map Generated at "; s += when; s += "\",";
    s += "\"type\":\"overlay\",";
    s += "\"version\":\"1.0\",";
    s += "\"description\":\"Map Generated at "; s += when; s += "\"";
    s += "}";
    return s;
}

static std::vector<uint8_t> buildMetadataBytes(uint8_t minZoom, uint8_t maxZoom,
                                               pmt::Compression internal) {
    std::string json = buildMetadataJSON(minZoom, maxZoom);
    std::vector<uint8_t> in(json.begin(), json.end());
    if (internal == pmt::Compression::Gzip) {
        std::vector<uint8_t> gz;
        gz.reserve(in.size()/2 + 64);
        if (!gzipCompress(in, gz, Z_BEST_COMPRESSION)) {
            return in; // fallback to raw JSON if compression fails
        }
        return gz;
    } else {
        return in;
    }
}

struct TileRef {
    uint64_t id;
    const uint8_t* data;
    uint32_t len;
};
struct UniqueRef {
    const uint8_t* data;
    uint32_t len;
};

static inline uint64_t sum_run_lengths(const std::vector<pmt::EntryV3>& entries) {
    uint64_t s = 0;
    for (const auto& e : entries) s += e.RunLength;
    return s;
}

static void build_entries_and_uniques(const std::vector<TileRef>& tiles_sorted,
                                      std::vector<pmt::EntryV3>& entries_out,
                                      std::vector<UniqueRef>& uniques_out)
{
    entries_out.clear();
    uniques_out.clear();
    entries_out.reserve(tiles_sorted.size());

    std::unordered_map<Hash128, OffsetLength, Hash128Hasher> seen;
    seen.reserve(tiles_sorted.size());

    uint64_t offset = 0;

    for (const auto& t : tiles_sorted) {
        Hash128 h = xxh3_128(t.data, t.len);
        auto it = seen.find(h);

        uint64_t off;
        uint32_t len;
        if (it != seen.end()) {
            off = it->second.Offset;
            len = it->second.Length;
        } else {
            off = offset;
            len = t.len;
            seen.emplace(h, OffsetLength{off, len});
            uniques_out.push_back(UniqueRef{t.data, t.len});
            offset += (uint64_t)t.len;
        }

        if (!entries_out.empty()) {
            auto& last = entries_out.back();
            if (t.id == last.TileID + last.RunLength && last.Offset == off) {
                last.RunLength++;
                continue;
            }
        }
        entries_out.push_back(pmt::EntryV3{ t.id, off, len, 1u });
    }
}

static std::vector<uint8_t> assemble_buffer(uint8_t minZoom, uint8_t maxZoom,
                                            const std::vector<pmt::EntryV3>& entries,
                                            const std::vector<UniqueRef>& uniques,
                                            pmt::Compression internal_comp)
{
    const bool COMPRESS_INTERNAL = (internal_comp == pmt::Compression::Gzip);
    const int  headerRoom = int(16384 - pmt::HeaderV3LenBytes);

    auto [rootBytes, leavesBytes] = optimizeDirectories(entries, headerRoom, COMPRESS_INTERNAL);
    auto metadataBytes = buildMetadataBytes(minZoom, maxZoom, internal_comp);

    uint64_t tileDataLen = 0;
    for (const auto& u : uniques) tileDataLen += u.len;

    pmt::HeaderV3 hdr{};
    hdr.TType               = pmt::TileType::Webp;
    hdr.TileCompression     = pmt::Compression::None;
    hdr.InternalCompression = internal_comp;
    hdr.MinZoom             = minZoom;
    hdr.MaxZoom             = maxZoom;
    hdr.Clustered           = true;

    hdr.RootOffset          = pmt::HeaderV3LenBytes;
    hdr.RootLength          = (uint64_t)rootBytes.size();

    hdr.MetadataOffset      = hdr.RootOffset + hdr.RootLength;
    hdr.MetadataLength      = (uint64_t)metadataBytes.size();

    hdr.LeafDirectoryOffset = hdr.MetadataOffset + hdr.MetadataLength;
    hdr.LeafDirectoryLength = (uint64_t)leavesBytes.size();

    hdr.TileDataOffset      = hdr.LeafDirectoryOffset + hdr.LeafDirectoryLength;
    hdr.TileDataLength      = tileDataLen;

    hdr.TileEntriesCount    = (uint64_t)entries.size();
    hdr.TileContentsCount   = (uint64_t)uniques.size();
    hdr.AddressedTilesCount = sum_run_lengths(entries);

    std::vector<uint8_t> out;
    out.reserve((size_t)(hdr.TileDataOffset + hdr.TileDataLength));

    auto headerBytes = serializeHeader(hdr);
    out.insert(out.end(), headerBytes.begin(), headerBytes.end());
    out.insert(out.end(), rootBytes.begin(), rootBytes.end());
    out.insert(out.end(), metadataBytes.begin(), metadataBytes.end());
    out.insert(out.end(), leavesBytes.begin(), leavesBytes.end());

    for (const auto& u : uniques) {
        out.insert(out.end(), u.data, u.data + u.len);
    }

    return out;
}

static int assemble_file(uint8_t minZoom, uint8_t maxZoom,
                         const std::vector<pmt::EntryV3>& entries,
                         const std::vector<UniqueRef>& uniques,
                         pmt::Compression internal_comp,
                         const char* filename)
{
    if (!filename) return 0;

    const bool COMPRESS_INTERNAL = (internal_comp == pmt::Compression::Gzip);
    const int  headerRoom = int(16384 - pmt::HeaderV3LenBytes);

    auto [rootBytes, leavesBytes] = optimizeDirectories(entries, headerRoom, COMPRESS_INTERNAL);
    auto metadataBytes = buildMetadataBytes(minZoom, maxZoom, internal_comp);

    uint64_t tileDataLen = 0;
    for (const auto& u : uniques) tileDataLen += u.len;

    pmt::HeaderV3 hdr{};
    hdr.TType               = pmt::TileType::Webp;
    hdr.TileCompression     = pmt::Compression::None;
    hdr.InternalCompression = internal_comp;
    hdr.MinZoom             = minZoom;
    hdr.MaxZoom             = maxZoom;
    hdr.Clustered           = true;

    hdr.RootOffset          = pmt::HeaderV3LenBytes;
    hdr.RootLength          = (uint64_t)rootBytes.size();

    hdr.MetadataOffset      = hdr.RootOffset + hdr.RootLength;
    hdr.MetadataLength      = (uint64_t)metadataBytes.size();

    hdr.LeafDirectoryOffset = hdr.MetadataOffset + hdr.MetadataLength;
    hdr.LeafDirectoryLength = (uint64_t)leavesBytes.size();

    hdr.TileDataOffset      = hdr.LeafDirectoryOffset + hdr.LeafDirectoryLength;
    hdr.TileDataLength      = tileDataLen;

    hdr.TileEntriesCount    = (uint64_t)entries.size();
    hdr.TileContentsCount   = (uint64_t)uniques.size();
    hdr.AddressedTilesCount = sum_run_lengths(entries);

    std::fstream f(filename, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!f) return 0;

    // header placeholder
    {
        std::vector<uint8_t> zero(pmt::HeaderV3LenBytes, 0);
        f.write(reinterpret_cast<const char*>(zero.data()), (std::streamsize)zero.size());
        if (!f) return 0;
    }

    // directories + metadata
    f.write(reinterpret_cast<const char*>(rootBytes.data()),     (std::streamsize)rootBytes.size());
    f.write(reinterpret_cast<const char*>(metadataBytes.data()), (std::streamsize)metadataBytes.size());
    f.write(reinterpret_cast<const char*>(leavesBytes.data()),   (std::streamsize)leavesBytes.size());
    if (!f) return 0;

    // tiles
    for (const auto& u : uniques) {
        f.write(reinterpret_cast<const char*>(u.data), (std::streamsize)u.len);
        if (!f) return 0;
    }

    // backpatch header
    auto headerBytes = serializeHeader(hdr);
    f.seekp(0, std::ios::beg);
    f.write(reinterpret_cast<const char*>(headerBytes.data()), (std::streamsize)headerBytes.size());
    if (!f) return 0;

    f.flush();
    return f.good() ? 1 : 0;
}

static int prepare_entries(const pm_tile_in* tiles, int count,
                           uint8_t& minZoom, uint8_t& maxZoom,
                           std::vector<pmt::EntryV3>& entries,
                           std::vector<UniqueRef>& uniques)
{
    if (!tiles || count <= 0) return 0;

    std::vector<TileRef> refs;
    refs.reserve((size_t)count);

    uint8_t calcMinZ = 255, calcMaxZ = 0;

    for (int i = 0; i < count; ++i) {
        const pm_tile_in& t = tiles[i];
        if (!t.data || t.len <= 0) return 0;
        uint64_t id = pmt::ZxyToID((uint8_t)t.z, (uint32_t)t.x, (uint32_t)t.y);
        refs.push_back(TileRef{ id, t.data, (uint32_t)t.len });
        uint8_t z8 = (uint8_t)t.z;
        if (z8 < calcMinZ) calcMinZ = z8;
        if (z8 > calcMaxZ) calcMaxZ = z8;
    }

    if (minZoom > maxZoom) {
        minZoom = calcMinZ;
        maxZoom = calcMaxZ;
    }

    std::sort(refs.begin(), refs.end(),
              [](const TileRef& a, const TileRef& b){ return a.id < b.id; });

    build_entries_and_uniques(refs, entries, uniques);
    return 1;
}

// ---------- Public C API ----------
extern "C" PMTILES_API int pmtiles_build_to_memory(const pm_tile_in* tiles, int count,
                                                   uint8_t minZoom, uint8_t maxZoom,
                                                   uint8_t** out_buf, int* out_len)
{
    if (!out_buf || !out_len) return 0;

    std::vector<pmt::EntryV3> entries;
    std::vector<UniqueRef> uniques;
    if (!prepare_entries(tiles, count, minZoom, maxZoom, entries, uniques)) return 0;

    auto buf = assemble_buffer(minZoom, maxZoom, entries, uniques, pmt::Compression::Gzip);
    if (buf.empty()) return 0;

    void* mem = std::malloc(buf.size());
    if (!mem) return 0;
    std::memcpy(mem, buf.data(), buf.size());
    *out_buf = reinterpret_cast<uint8_t*>(mem);
    *out_len = (int)buf.size();
    return 1;
}

extern "C" PMTILES_API int pmtiles_build_to_file(const pm_tile_in* tiles, int count,
                                                 uint8_t minZoom, uint8_t maxZoom,
                                                 const char* filename)
{
    std::vector<pmt::EntryV3> entries;
    std::vector<UniqueRef> uniques;
    if (!prepare_entries(tiles, count, minZoom, maxZoom, entries, uniques)) return 0;

    return assemble_file(minZoom, maxZoom, entries, uniques,
                         pmt::Compression::Gzip, filename);
}

extern "C" PMTILES_API void pmtiles_free_buffer(void* p) {
    if (p) std::free(p);
}
