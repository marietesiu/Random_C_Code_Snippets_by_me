#include <stdint.h>
#include <string.h>
#include <stdio.h>

/* ------------------------------------------------------------------
 * fast_hash — non-cryptographic hash for fixed-size binary blocks.
 *
 * Built ONLY for two input sizes: 128 bytes and 4096 bytes.
 * Both are multiples of 32 bytes, so the main loop (4 words = 32
 * bytes per iteration) always divides evenly — no leftover "tail"
 * bytes to mask off, which most general-purpose hash functions have
 * to spend extra branches handling.
 *
 * Output is a single 64-bit integer: 16 hex characters, one CPU
 * compare instruction to check equality. Not for security use —
 * purely for speed (dedup, cache keys, fast equality checks).
 * ------------------------------------------------------------------ */

#define FH_K1 0x9E3779B97F4A7C15ULL
#define FH_K2 0xBF58476D1CE4E5B9ULL
#define FH_K3 0x94D049BB133111EBULL

/* Reads 8 bytes regardless of pointer alignment. memcpy of a
 * constant size compiles to a single load instruction under -O2 —
 * this costs nothing at runtime, it just avoids alignment faults /
 * strict-aliasing UB on platforms that care. */
static inline uint64_t fh_load64(const uint8_t *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;
}

/* Final avalanche mix (SplitMix64-style finalizer): spreads any
 * single-bit input change across all 64 output bits. */
static inline uint64_t fh_mix(uint64_t h) {
    h ^= h >> 33;
    h *= FH_K2;
    h ^= h >> 29;
    h *= FH_K3;
    h ^= h >> 32;
    return h;
}

/* len must be 128 or 4096 — nothing else is supported. */
uint64_t fast_hash(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    size_t words = len / 8;

    /* Four independent accumulators. Each one's multiply chain only
     * depends on itself, not the others — so the CPU's out-of-order
     * engine can run all four multiplies in parallel each iteration
     * instead of stalling on one long dependency chain. This is the
     * main speed trick (same idea xxHash/wyhash use). */
    uint64_t h0 = FH_K1, h1 = FH_K2, h2 = FH_K3, h3 = FH_K1 ^ FH_K2;

    for (size_t i = 0; i < words; i += 4) {
        h0 = (h0 ^ fh_load64(p + (i + 0) * 8)) * FH_K1;
        h1 = (h1 ^ fh_load64(p + (i + 1) * 8)) * FH_K2;
        h2 = (h2 ^ fh_load64(p + (i + 2) * 8)) * FH_K3;
        h3 = (h3 ^ fh_load64(p + (i + 3) * 8)) * FH_K1;
    }

    uint64_t h = h0;
    h = (h * FH_K1) ^ h1;
    h = (h * FH_K2) ^ h2;
    h = (h * FH_K3) ^ h3;
    return fh_mix(h ^ (uint64_t)len);
}

/* ---------------------------- demo ---------------------------- */
static void fill_pattern(uint8_t *buf, size_t n, uint8_t seed) {
    for (size_t i = 0; i < n; i++) buf[i] = (uint8_t)(seed + i * 31);
}

int main(void) {
    uint8_t block128[128];
    uint8_t block4k[4096];
    uint8_t block4k_b[4096];

    fill_pattern(block128, sizeof(block128), 0xA5);
    fill_pattern(block4k,   sizeof(block4k),   0x11);
    fill_pattern(block4k_b, sizeof(block4k_b), 0x11);
    block4k_b[2000] ^= 0x01;   /* flip ONE bit, rest identical */

    uint64_t h128  = fast_hash(block128, sizeof(block128));
    uint64_t h4k_a = fast_hash(block4k,   sizeof(block4k));
    uint64_t h4k_b = fast_hash(block4k_b, sizeof(block4k_b));

    printf("128B  hash : %016llx\n", (unsigned long long)h128);
    printf("4KB-a hash : %016llx\n", (unsigned long long)h4k_a);
    printf("4KB-b hash : %016llx  (one bit flipped at byte 2000)\n",
           (unsigned long long)h4k_b);

    printf("\nsame buffer rehashed twice match?   %s\n",
           (h4k_a == fast_hash(block4k, sizeof(block4k))) ? "yes" : "no");
    printf("one-bit-different buffers match?    %s (should be no)\n",
           (h4k_a == h4k_b) ? "yes" : "no");

    return 0;
}
