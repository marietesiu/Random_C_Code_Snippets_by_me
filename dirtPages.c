/* 
 *
 *  1. Hash-based change detection (random sampling):
 *       A random-position sampling hash is computed over a page at each scan.
 *       If the hash differs from the last stored value the page is considered
 *       dirty/volatile. The hash strength (number of sampled positions) is
 *       adaptively adjusted via a state machine so that the trade-off between
 *       scan speed and false-negative rate is continuously optimised.
 *
 *  2. PTE-level dirty-bit detection (write_protect_page):
 *       Before a page can be shared, its PTE is made read-only.  On the next
 *       scan UKSM checks pte_dirty()/pte_write(); if either is set the page
 *       was written (dirtied) since the last scan.
 *
 * Key entry points:
 *   page_hash()              - compute a (possibly sampled) hash for a page
 *   page_hash_max()          - compute the full-strength hash for a page
 *   check_collision()        - distinguish a real page change from a hash
 *                              collision (MERGE_ERR_CHANGED vs MERGE_ERR_COLLI)
 *   write_protect_page()     - write-protect a PTE and detect existing dirty
 *                              state via pte_dirty()/pte_write()
 *   rshash_adjust()          - adaptive hash-strength state machine; call once
 *                              per scan round; returns non-zero if hash_strength
 *                              changed (requiring stable-tree rehash)
 *   init_random_sampling()   - must be called once at module init
 *   init_zeropage_hash_table() - must be called once at module init
 */

#include <linux/errno.h>
#include <linux/mm.h>
#include <linux/fs.h>
#include <linux/mman.h>
#include <linux/sched.h>
#include <linux/sched/mm.h>
#include <linux/rwsem.h>
#include <linux/pagemap.h>
#include <linux/rmap.h>
#include <linux/spinlock.h>
#include <linux/delay.h>
#include <linux/slab.h>
#include <linux/rbtree.h>
#include <linux/mmu_notifier.h>
#include <linux/swap.h>
#include <linux/ksm.h>
#include <linux/random.h>
#include <linux/math64.h>
#include <linux/gcd.h>
#include <linux/sradix-tree.h>
#include <asm/tlbflush.h>
#include "internal.h"

/* =========================================================================
 * Section 1: Architecture-specific fast memory comparison helpers
 *
 * These replace the generic memcmp/is_full_zero with hand-written x86
 * assembly that exploits REPE CMPS / REPE SCAS for maximum throughput
 * when scanning 4 KB pages.
 * ========================================================================= */

#ifdef CONFIG_X86
#undef memcmp

#ifdef CONFIG_X86_32
#define memcmp memcmpx86_32

/* Compare two 4-byte-aligned buffers of length n bytes. */
int memcmpx86_32(void *s1, void *s2, size_t n)
{
	size_t num = n / 4;
	register int res;

	__asm__ __volatile__
	(
	 "testl %3,%3\n\t"
	 "repe; cmpsd\n\t"
	 "je        1f\n\t"
	 "sbbl      %0,%0\n\t"
	 "orl       $1,%0\n"
	 "1:"
	 : "=&a" (res), "+&S" (s1), "+&D" (s2), "+&c" (num)
	 : "0" (0)
	 : "cc");

	return res;
}

/* Return non-zero if all bytes in [s1, s1+len) are zero. */
static int is_full_zero(const void *s1, size_t len)
{
	unsigned char same;

	len /= 4;

	__asm__ __volatile__
	("repe; scasl;"
	 "sete %0"
	 : "=qm" (same), "+D" (s1), "+c" (len)
	 : "a" (0)
	 : "cc");

	return same;
}

#elif defined(CONFIG_X86_64)
#define memcmp memcmpx86_64

/* Compare two 8-byte-aligned buffers of length n bytes. */
int memcmpx86_64(void *s1, void *s2, size_t n)
{
	size_t num = n / 8;
	register int res;

	__asm__ __volatile__
	(
	 "testq %q3,%q3\n\t"
	 "repe; cmpsq\n\t"
	 "je        1f\n\t"
	 "sbbq      %q0,%q0\n\t"
	 "orq       $1,%q0\n"
	 "1:"
	 : "=&a" (res), "+&S" (s1), "+&D" (s2), "+&c" (num)
	 : "0" (0)
	 : "cc");

	return res;
}

/* Return non-zero if all bytes in [s1, s1+len) are zero. */
static int is_full_zero(const void *s1, size_t len)
{
	unsigned char same;

	len /= 8;

	__asm__ __volatile__
	("repe; scasq;"
	 "sete %0"
	 : "=qm" (same), "+D" (s1), "+c" (len)
	 : "a" (0)
	 : "cc");

	return same;
}

#endif  /* CONFIG_X86_32 / CONFIG_X86_64 */

#else   /* !CONFIG_X86 */

/* Generic fallback: scan one unsigned long at a time. */
static int is_full_zero(const void *s1, size_t len)
{
	unsigned long *src = (unsigned long *)s1;
	int i;

	len /= sizeof(*src);
	for (i = 0; i < len; i++) {
		if (src[i])
			return 0;
	}
	return 1;
}

#endif  /* CONFIG_X86 */


/* =========================================================================
 * Section 2: Hash constants and helper macros
 * ========================================================================= */

/*
 * HASH_STRENGTH_FULL: number of u32 words in a full PAGE_SIZE page.
 * This is the upper bound of the "normal" sampling range.
 */
#define HASH_STRENGTH_FULL      (PAGE_SIZE / sizeof(u32))

/*
 * HASH_STRENGTH_MAX: beyond FULL, the hash wraps around and re-samples
 * the beginning of the page a second time.  +10 extra positions are used
 * to create a distinguishable "loop-back" hash that still depends on the
 * full page content.
 */
#define HASH_STRENGTH_MAX       (HASH_STRENGTH_FULL + 10)

/*
 * Maximum exponent for the step size when adjusting hash_strength.
 * The actual step is (1 << hash_strength_delta).
 */
#define HASH_STRENGTH_DELTA_MAX 5

/*
 * Shift constants chosen so that 32/3 < shiftl,shiftr < 32/2:
 * they give good avalanche behaviour while being cheap on x86.
 */
#define shiftl  8
#define shiftr  12

/*
 * HASH_FROM_TO(from, to) - inner loop of the random-sampling hash.
 * Iterates positions [from, to) in the random_nums permutation table,
 * mixing key[pos] into `hash` with a multiply-xor step.
 *
 * Variables required in scope: index, pos, hash, key (u32 *).
 */
#define HASH_FROM_TO(from, to)                  \
for (index = from; index < to; index++) {       \
	pos = random_nums[index];               \
	hash += key[pos];                       \
	hash += (hash << shiftl);              \
	hash ^= (hash >> shiftr);              \
}

/*
 * HASH_FROM_DOWN_TO(from, to) - inverse of HASH_FROM_TO.
 * Used by delta_hash() to "undo" the contribution of positions [to, from)
 * when the hash strength is being decreased.
 */
#define HASH_FROM_DOWN_TO(from, to)             \
for (index = from - 1; index >= to; index--) { \
	hash ^= (hash >> shiftr);              \
	hash ^= (hash >> (shiftr * 2));        \
	hash -= (hash << shiftl);              \
	hash += (hash << (shiftl * 2));        \
	pos = random_nums[index];               \
	hash -= key[pos];                       \
}

/* Overflow guard: true if (x + delta) would wrap a u64. */
#define CAN_OVERFLOW_U64(x, delta) (U64_MAX - (x) < (delta))

/*
 * Return codes from check_collision() / write_protect_page().
 * Only MERGE_ERR_CHANGED and MERGE_ERR_COLLI are used by the dirty
 * detection path; the others are included for completeness.
 */
#define MERGE_ERR_PGERR         1   /* page is invalid, cannot continue */
#define MERGE_ERR_COLLI         2   /* hash collision, page NOT changed */
#define MERGE_ERR_COLLI_MAX     3   /* collision at HASH_STRENGTH_MAX   */
#define MERGE_ERR_CHANGED       4   /* page has changed since last hash  */

/* Flags stored in the low bits of rmap_item.address */
#define UNSTABLE_FLAG   0x1
#define STABLE_FLAG     0x2


/* =========================================================================
 * Section 3: Data structures
 * ========================================================================= */

/*
 * struct uksm_benefit - accumulated benefit of the random-sampling hash.
 *
 *tracks the ratio of "positive" work (time saved by not doing a full
 * memcmp when the hash proves pages differ) to "negative" work (time wasted
 * computing hashes when pages turn out to be identical, requiring a full
 * memcmp anyway).  This drives the hash_strength adaptation.
 *
 * pos     - total time units saved (hash detected difference → no memcmp)
 * neg     - total time units wasted (hash said same → memcmp confirmed same)
 * scanned - total pages scanned since last reset (normalisation denominator)
 * base    - right-shift applied to all three when overflow is near
 */
struct uksm_benefit {
	u64 pos;
	u64 neg;
	u64 scanned;
	unsigned long base;
};

/*
 * States of the hash-strength adaptation state machine (rshash_state).
 *
 * RSHASH_NEW       - freshly initialised; no history yet
 * RSHASH_STILL     - stable at current hash_strength; monitoring benefit
 * RSHASH_TRYDOWN   - probing whether decreasing strength improves benefit
 * RSHASH_TRYUP     - probing whether increasing strength improves benefit
 * RSHASH_PRE_STILL - transition state before returning to STILL
 */
enum rshash_states {
	RSHASH_STILL,
	RSHASH_TRYUP,
	RSHASH_TRYDOWN,
	RSHASH_NEW,
	RSHASH_PRE_STILL,
};

/* Direction decisions returned by judge_rshash_direction(). */
enum rshash_direct {
	GO_UP,
	GO_DOWN,
	OBSCURE,
	STILL,
};

/*
 * struct stable_node - a node in the stable (de-duplicated) rb-tree.
 *
 * Included here because stable_tree_delta_hash() — a direct dependency of
 * rshash_adjust() — iterates over stable nodes to recompute their hashes
 * whenever hash_strength changes.
 *
 * node       - link in the per-tree_node collision sub-tree
 * tree_node  - parent tree_node in the two-level stable tree
 * hlist      - head of node_vma items sharing this ksm page
 * kpfn       - PFN of the ksm page
 * hash_max   - hash at HASH_STRENGTH_MAX; 0 = not yet computed
 * all_list   - link in the global stable_node_list
 */
struct stable_node {
	struct rb_node      node;
	struct tree_node   *tree_node;
	struct hlist_head   hlist;
	unsigned long       kpfn;
	u32                 hash_max;
	struct list_head    all_list;
};

/*
 * struct tree_node - first-level node of the two-level stable/unstable tree.
 *
 * Each tree_node holds all stable_nodes (or rmap_items) that share the same
 * first-level hash value; collisions are resolved in sub_root.
 */
struct tree_node {
	struct rb_node   node;
	struct rb_root   sub_root;
	u32              hash;
	unsigned long    count;
	struct list_head all_list;
};

/*
 * struct node_vma - groups rmap_items that map the same ksm page within
 * the same VMA slot.  Needed by remove_node_from_stable_tree().
 */
struct node_vma {
	union {
		struct vma_slot  *slot;
		unsigned long     key;
	};
	struct hlist_node  hlist;
	struct hlist_head  rmap_hlist;
	struct stable_node *head;
};

/*
 * struct rmap_item - per-page reverse-mapping item.
 *
 * The dirty-detection path uses:
 *   page         - pointer to the physical page (cached address→page mapping)
 *   address      - virtual address + flags (STABLE_FLAG / UNSTABLE_FLAG)
 *   hash_round   - the uksm_hash_round when the unstable-tree hash was stored
 *   hash_max     - full-strength hash; 0 = not yet computed
 *   anon_vma     - (stable path) anon_vma for rmap unlink on dirty detection
 */
struct rmap_item {
	struct vma_slot  *slot;
	struct page      *page;
	unsigned long     address;
	unsigned long     hash_round;
	unsigned long     entry_index;
	union {
		struct {                    /* when in unstable tree */
			struct rb_node   node;
			struct tree_node *tree_node;
			u32              hash_max;
		};
		struct {                    /* when in stable tree */
			struct node_vma  *head;
			struct hlist_node hlist;
			struct anon_vma  *anon_vma;
		};
	};
} __attribute__((aligned(4)));


/* =========================================================================
 * Section 4: Global variables
 * ========================================================================= */

/* --- kmem caches (initialised in uksm_slab_init) --- */
static struct kmem_cache *stable_node_cache;
static struct kmem_cache *node_vma_cache;
static struct kmem_cache *tree_node_cache;

/* --- Random sampling permutation table ---
 *
 * random_nums[i] is a random permutation of [0, HASH_STRENGTH_FULL).
 * HASH_FROM_TO uses it to select which u32 words of a page to hash, so
 * that "hash_strength N" hashes a consistent, pseudo-random subset of
 * N page words without repeating positions.
 */
static u32 *random_nums;

/* --- Hash strength ---
 *
 * Current number of page-word positions sampled per hash.  Ranges from 1
 * (minimum, fastest scan) to HASH_STRENGTH_MAX (full page + loop-back).
 * Adapted dynamically by rshash_adjust().
 */
static unsigned long hash_strength = HASH_STRENGTH_FULL >> 4;

/* Step size (as a power-of-two exponent) for the next hash_strength change */
static unsigned long hash_strength_delta;

/* --- Benefit tracking counters ---
 *
 * rshash_pos accumulates the time "saved" in the current sub-round:
 *   each scanned page contributes (HASH_STRENGTH_FULL - hash_strength)
 *   to represent the memcmp work avoided when the hash detects a diff.
 *
 * rshash_neg accumulates the time "wasted":
 *   each memcmp forced by a hash match contributes memcmp_cost +
 *   (HASH_STRENGTH_MAX - hash_strength) for the full-hash computation.
 *
 * Both are reset by encode_benefit() and kept scaled via benefit.base.
 */
static u64 rshash_pos;
static u64 rshash_neg;

/* Encoded long-term benefit (accumulated across encode_benefit() calls) */
static struct uksm_benefit benefit;

/*
 * memcmp_cost - relative cost of one full-page memcmp expressed in units of
 * "one random-sample hash position."  Measured at init time by
 * cal_positive_negative_costs().
 */
static unsigned long memcmp_cost;

/* Consecutive rounds where rshash_neg_ratio was zero (drives GO_DOWN). */
static unsigned long rshash_neg_cont_zero;

/* Consecutive rounds where benefit looked "obscure" (non-monotone). */
static unsigned long rshash_cont_obscure;

/* --- Hash-strength adaptation state machine --- */
static struct {
	enum rshash_states state;
	enum rshash_direct pre_direct;
	u8  below_count;
	u8  lookup_window_index;
	u64 stable_benefit;
	unsigned long turn_point_down;
	unsigned long turn_benefit_down;
	unsigned long turn_point_up;
	unsigned long turn_benefit_up;
	unsigned long stable_point;
} rshash_state;

/* --- Page-scan counters (used by encode_benefit) --- */

/*
 * Total pages scanned since startup.  Wraps to 0 when U64_MAX is reached
 * (handled by inc_uksm_pages_scanned).
 */
static u64 uksm_pages_scanned;

/* Value of uksm_pages_scanned at the last encode_benefit() call */
static u64 uksm_pages_scanned_last;

/*
 * When uksm_pages_scanned wraps, the high-order portion is stored here
 * (right-shifted by pages_scanned_base) so statistics remain meaningful.
 */
static u64  pages_scanned_stored;
static unsigned long pages_scanned_base;

/* How many complete system-wide scan rounds have finished */
static unsigned long long fully_scanned_round = 1;

/* Incremented each time the unstable tree is rebuilt */
static unsigned long uksm_hash_round = 1;

/* --- Zero-page hash lookup table ---
 *
 * zero_hash_table[i] is the hash of an all-zero page at hash_strength i.
 * Used by find_zero_page_hash() to quickly test whether a page is zero
 * without doing a full is_full_zero() scan.
 */
static u32 *zero_hash_table;

/* --- Stable-tree state (needed by stable_tree_delta_hash) --- */

/* All known stable nodes, in insertion order */
static struct list_head stable_node_list = LIST_HEAD_INIT(stable_node_list);

/*
 * Two alternating rb_root / tree_node_list pairs for the stable tree.
 * When hash_strength changes, stable_tree_delta_hash() rebuilds the tree
 * into the inactive pair and then swaps the pointer.
 */
static struct list_head
	stable_tree_node_list[2] = { LIST_HEAD_INIT(stable_tree_node_list[0]),
				     LIST_HEAD_INIT(stable_tree_node_list[1]) };
static struct list_head *stable_tree_node_listp = &stable_tree_node_list[0];
static struct rb_root    root_stable_tree[2]     = { RB_ROOT, RB_ROOT };
static struct rb_root   *root_stable_treep       = &root_stable_tree[0];
static unsigned long     stable_tree_index;         /* 0 or 1 */

/* Counters updated when pages enter/leave the stable tree. */
static unsigned long uksm_pages_shared;
static unsigned long uksm_pages_sharing;

/* Zero-page PFNs (defined in uksm.c, referenced by write_protect_page). */
extern unsigned long zero_pfn __read_mostly;
extern unsigned long uksm_zero_pfn __read_mostly;


/* =========================================================================
 * Section 5: kmem-cache allocator helpers
 *
 * Thin wrappers around kmem_cache_{alloc,free} for the three object types
 * that the dirty-detection dependency chain touches (stable_node, tree_node,
 * node_vma).  The caches are initialised during uksm_slab_init().
 * ========================================================================= */

static inline struct stable_node *alloc_stable_node(void)
{
	struct stable_node *node;

	node = kmem_cache_alloc(stable_node_cache,
				GFP_KERNEL | __GFP_NORETRY | __GFP_NOWARN);
	if (!node)
		return NULL;

	INIT_HLIST_HEAD(&node->hlist);
	list_add(&node->all_list, &stable_node_list);
	return node;
}

static inline void free_stable_node(struct stable_node *stable_node)
{
	list_del(&stable_node->all_list);
	kmem_cache_free(stable_node_cache, stable_node);
}

static inline struct tree_node *alloc_tree_node(struct list_head *list)
{
	struct tree_node *node;

	node = kmem_cache_zalloc(tree_node_cache,
				 GFP_KERNEL | __GFP_NORETRY | __GFP_NOWARN);
	if (!node)
		return NULL;

	list_add(&node->all_list, list);
	return node;
}

static inline void free_tree_node(struct tree_node *node)
{
	list_del(&node->all_list);
	kmem_cache_free(tree_node_cache, node);
}

static inline void free_node_vma(struct node_vma *node_vma)
{
	kmem_cache_free(node_vma_cache, node_vma);
}


/* =========================================================================
 * Section 6: Core random-sampling hash functions
 *
 * random_sample_hash()  - hash a page using `hash_strength` random positions.
 * delta_hash()          - incrementally update a hash when hash_strength
 *                         changes, without re-hashing the whole page.
 * ========================================================================= */

/*
 * random_sample_hash - compute a rolling hash over `hash_strength` randomly
 * selected u32 words of the page at `addr`.
 *
 * When hash_strength <= HASH_STRENGTH_FULL the positions are chosen from the
 * random_nums permutation (each position is unique in the page).  When
 * hash_strength > HASH_STRENGTH_FULL the first HASH_STRENGTH_FULL positions
 * cover the whole page and the remaining (hash_strength - HASH_STRENGTH_FULL)
 * positions loop back to the start of random_nums, giving a stronger hash at
 * the cost of redundant reads.
 *
 * The mixing step (hash += key[pos]; hash += hash << shiftl;
 * hash ^= hash >> shiftr) is a simplified Jenkins-style integer hash.
 */
static u32 random_sample_hash(void *addr, u32 hash_strength)
{
	u32 hash = 0xdeadbeef;
	int index, pos, loop = hash_strength;
	u32 *key = (u32 *)addr;

	if (loop > HASH_STRENGTH_FULL)
		loop = HASH_STRENGTH_FULL;

	HASH_FROM_TO(0, loop);

	if (hash_strength > HASH_STRENGTH_FULL) {
		loop = hash_strength - HASH_STRENGTH_FULL;
		HASH_FROM_TO(0, loop);
	}

	return hash;
}

/*
 * delta_hash - update hash `hash` (computed at strength `from`) to strength
 * `to` without rehashing from scratch.
 *
 * When strength increases, HASH_FROM_TO appends the contribution of the new
 * positions.  When strength decreases, HASH_FROM_DOWN_TO reverses (un-mixes)
 * those positions using the algebraic inverse of the mixing step.
 *
 * Called by:
 *   stable_tree_delta_hash() - rehash all stable nodes after hash_strength
 *                              changes.
 *   page_hash_max()          - extend a partial hash to HASH_STRENGTH_MAX
 *                              without a full re-scan.
 */
static u32 delta_hash(void *addr, int from, int to, u32 hash)
{
	u32 *key = (u32 *)addr;
	int index, pos;   /* must be signed int for HASH_FROM_DOWN_TO */

	if (to > from) {
		if (from >= HASH_STRENGTH_FULL) {
			from -= HASH_STRENGTH_FULL;
			to   -= HASH_STRENGTH_FULL;
			HASH_FROM_TO(from, to);
		} else if (to <= HASH_STRENGTH_FULL) {
			HASH_FROM_TO(from, to);
		} else {
			HASH_FROM_TO(from, HASH_STRENGTH_FULL);
			HASH_FROM_TO(0, to - HASH_STRENGTH_FULL);
		}
	} else {
		if (from <= HASH_STRENGTH_FULL) {
			HASH_FROM_DOWN_TO(from, to);
		} else if (to >= HASH_STRENGTH_FULL) {
			from -= HASH_STRENGTH_FULL;
			to   -= HASH_STRENGTH_FULL;
			HASH_FROM_DOWN_TO(from, to);
		} else {
			HASH_FROM_DOWN_TO(from - HASH_STRENGTH_FULL, 0);
			HASH_FROM_DOWN_TO(HASH_STRENGTH_FULL, to);
		}
	}

	return hash;
}


/* =========================================================================
 * Section 7: Benefit tracking
 *
 * The adaptive hash-strength mechanism works by comparing the time saved by
 * the hash (rshash_pos) to the time wasted on false-positives (rshash_neg).
 * encode_benefit() collapses the running counters into the long-term
 * `benefit` struct; rshash_adjust() consults benefit to decide whether to
 * raise or lower hash_strength.
 * ========================================================================= */

/*
 * encode_benefit - transfer rshash_pos/rshash_neg into the scaled `benefit`
 * struct and reset the running counters.
 *
 * Called automatically by inc_rshash_pos/neg() before overflow, and
 * explicitly by rshash_adjust() at the end of each scan round.
 *
 * Returns 0 if no pages were scanned since the last call (nothing to encode),
 * 1 otherwise.
 */
static inline int encode_benefit(void)
{
	u64 scanned_delta, pos_delta, neg_delta;
	unsigned long base = benefit.base;

	scanned_delta = uksm_pages_scanned - uksm_pages_scanned_last;

	if (!scanned_delta)
		return 0;

	scanned_delta >>= base;
	pos_delta      = rshash_pos >> base;
	neg_delta      = rshash_neg >> base;

	if (CAN_OVERFLOW_U64(benefit.pos, pos_delta)     ||
	    CAN_OVERFLOW_U64(benefit.neg, neg_delta)      ||
	    CAN_OVERFLOW_U64(benefit.scanned, scanned_delta)) {
		benefit.scanned >>= 1;
		benefit.neg     >>= 1;
		benefit.pos     >>= 1;
		benefit.base++;
		scanned_delta >>= 1;
		pos_delta     >>= 1;
		neg_delta     >>= 1;
	}

	benefit.pos     += pos_delta;
	benefit.neg     += neg_delta;
	benefit.scanned += scanned_delta;

	BUG_ON(!benefit.scanned);

	rshash_pos = rshash_neg = 0;
	uksm_pages_scanned_last = uksm_pages_scanned;

	return 1;
}

/* reset_benefit - zero all benefit counters (called at the start of a new
 * hash-strength trial period by rshash_adjust). */
static inline void reset_benefit(void)
{
	benefit.pos     = 0;
	benefit.neg     = 0;
	benefit.base    = 0;
	benefit.scanned = 0;
}

/*
 * inc_rshash_pos - account for `delta` units of positive benefit.
 *
 * Called by page_hash() for every page that was hashed with a partial
 * (strength < FULL) hash.  The delta is (HASH_STRENGTH_FULL - hash_strength),
 * representing the memcmp work that will be avoided if the hash detects a
 * difference.
 */
static inline void inc_rshash_pos(unsigned long delta)
{
	if (CAN_OVERFLOW_U64(rshash_pos, delta))
		encode_benefit();

	rshash_pos += delta;
}

/*
 * inc_rshash_neg - account for `delta` units of negative benefit.
 *
 * Called when a memcmp is needed despite (or because of) the hash result,
 * and when page_hash_max() computes the full-strength hash to resolve a
 * collision.  The delta is memcmp_cost or (HASH_STRENGTH_MAX - hash_strength).
 */
static inline void inc_rshash_neg(unsigned long delta)
{
	if (CAN_OVERFLOW_U64(rshash_neg, delta))
		encode_benefit();

	rshash_neg += delta;
}


/* =========================================================================
 * Section 8: Per-page hash computation (public API for dirty detection)
 * ========================================================================= */

/*
 * page_hash - compute the random-sampling hash for `page` at the current
 * `hash_strength`, optionally recording cost in the benefit counters.
 *
 * @page            The page to hash (temporarily kmap'd).
 * @hash_strength   Number of positions to sample (usually the global value).
 * @cost_accounting If non-zero, call inc_rshash_pos() to credit the time
 *                  saved by sampling fewer positions.
 *
 * Returns the 32-bit hash value.
 *
 * Dirty detection usage:
 *   A rmap_item's stored hash is compared against the result of page_hash()
 *   on the next scan.  A mismatch means the page changed (is dirty).
 */
static inline u32 page_hash(struct page *page, unsigned long hash_strength,
			    int cost_accounting)
{
	u32 val;
	unsigned long delta;
	void *addr = kmap_atomic(page);

	val = random_sample_hash(addr, hash_strength);
	kunmap_atomic(addr);

	if (cost_accounting) {
		if (hash_strength < HASH_STRENGTH_FULL)
			delta = HASH_STRENGTH_FULL - hash_strength;
		else
			delta = 0;

		inc_rshash_pos(delta);
	}

	return val;
}

/*
 * page_hash_max - compute the full-strength hash (at HASH_STRENGTH_MAX) for
 * `page`, given that `hash_old` was computed at the current `hash_strength`.
 *
 * Uses delta_hash() to extend the partial hash incrementally rather than
 * re-reading the whole page from scratch.  Records the extra cost in
 * rshash_neg so that hash-strength adaptation accounts for this overhead.
 *
 * Returns a non-zero u32 (zero is reserved as "not yet computed" sentinel).
 *
 * Used by check_collision() to determine whether a hash mismatch is a real
 * page change (hash_max changed → MERGE_ERR_CHANGED) or merely a hash
 * collision at the current lower hash_strength (MERGE_ERR_COLLI).
 */
static inline u32 page_hash_max(struct page *page, u32 hash_old)
{
	u32 hash_max = 0;
	void *addr;

	addr = kmap_atomic(page);
	hash_max = delta_hash(addr, hash_strength,
			      HASH_STRENGTH_MAX, hash_old);
	kunmap_atomic(addr);

	if (!hash_max)
		hash_max = 1;   /* 0 is the "not computed" sentinel */

	inc_rshash_neg(HASH_STRENGTH_MAX - hash_strength);
	return hash_max;
}

/*
 * memcmp_pages_with_cost - compare two pages byte-by-byte.
 *
 * If cost_accounting is set, charges memcmp_cost units to rshash_neg to
 * represent the CPU time consumed relative to a hash-only check.
 *
 * Returns 0 if the pages are identical, non-zero otherwise.
 */
static int memcmp_pages_with_cost(struct page *page1, struct page *page2,
				  int cost_accounting)
{
	char *addr1, *addr2;
	int ret;

	addr1 = kmap_atomic(page1);
	addr2 = kmap_atomic(page2);
	ret   = memcmp(addr1, addr2, PAGE_SIZE);
	kunmap_atomic(addr2);
	kunmap_atomic(addr1);

	if (cost_accounting)
		inc_rshash_neg(memcmp_cost);

	return ret;
}

/* pages_identical_with_cost - returns 1 if the two pages are byte-identical */
static inline int pages_identical_with_cost(struct page *page1,
					    struct page *page2)
{
	return !memcmp_pages_with_cost(page1, page2, 0);
}


/* =========================================================================
 * Section 9: Page-scan counter
 * ========================================================================= */

/*
 * inc_uksm_pages_scanned - safely increment the total pages-scanned counter.
 *
 * Handles U64 wrap-around by encoding the high-order bits into
 * pages_scanned_stored (right-shifted by pages_scanned_base) before
 * resetting uksm_pages_scanned to zero.  This preserves statistical
 * validity across wrap-arounds without losing information.
 *
 * Must be called once for every page examined by the scanner.
 */
static inline void inc_uksm_pages_scanned(void)
{
	u64 delta;

	if (uksm_pages_scanned == U64_MAX) {
		encode_benefit();

		delta = uksm_pages_scanned >> pages_scanned_base;

		if (CAN_OVERFLOW_U64(pages_scanned_stored, delta)) {
			pages_scanned_stored >>= 1;
			delta >>= 1;
			pages_scanned_base++;
		}

		pages_scanned_stored += delta;

		uksm_pages_scanned = uksm_pages_scanned_last = 0;
	}

	uksm_pages_scanned++;
}


/* =========================================================================
 * Section 10: Zero-page detection helpers
 * ========================================================================= */

/*
 * is_page_full_zero - return non-zero if every byte of `page` is zero.
 *
 * Used to distinguish pages that should be merged with the UKSM zero page
 * rather than going through the normal hash-comparison path.
 */
static inline int is_page_full_zero(struct page *page)
{
	char *addr;
	int ret;

	addr = kmap_atomic(page);
	ret  = is_full_zero(addr, PAGE_SIZE);
	kunmap_atomic(addr);

	return ret;
}

/*
 * find_zero_page_hash - return non-zero if `hash` (computed at `strength`)
 * matches the precomputed hash of an all-zero page.
 *
 * A quick early-exit test: if the hash matches, call is_page_full_zero()
 * to confirm before treating the page as a zero-page candidate.
 */
static inline int find_zero_page_hash(int strength, u32 hash)
{
	return (zero_hash_table[strength] == hash);
}


/* =========================================================================
 * Section 11: PTE-level dirty detection — write_protect_page()
 *
 * This function write-protects a page's PTE so that any future write by
 * the owning process will trigger a copy-on-write fault.  Crucially, it
 * also inspects the current PTE *before* clearing the write-enable bit:
 *
 *   pte_write(*pvmw.pte)  - page is writable → has or could have new data
 *   pte_dirty(*pvmw.pte)  - page was written since the last TLB flush
 *
 * If either condition holds, the PTE is cleared and the hardware dirty bit
 * is propagated to the struct page via set_page_dirty(), so subsequent
 * callers (e.g. the page-reclaim path) know the page has been modified.
 *
 * Returns 0 on success, -EFAULT if the page could not be found in the VMA.
 * On success, *orig_pte receives the write-protected PTE value that must be
 * passed to replace_page() for a merge, or compared on re-scan to detect
 * further writes.
 * ========================================================================= */
static int write_protect_page(struct vm_area_struct *vma, struct page *page,
			      pte_t *orig_pte, pte_t *old_pte)
{
	struct mm_struct *mm = vma->vm_mm;
	struct page_vma_mapped_walk pvmw = {
		.page = page,
		.vma  = vma,
	};
	struct mmu_notifier_range range;
	int swapped;
	int err = -EFAULT;

	pvmw.address = page_address_in_vma(page, vma);
	if (pvmw.address == -EFAULT)
		goto out;

	BUG_ON(PageTransCompound(page));

	mmu_notifier_range_init(&range, MMU_NOTIFY_CLEAR, 0, vma, mm,
				pvmw.address, pvmw.address + PAGE_SIZE);
	mmu_notifier_invalidate_range_start(&range);

	if (!page_vma_mapped_walk(&pvmw))
		goto out_mn;
	if (WARN_ONCE(!pvmw.pte, "Unexpected PMD mapping?"))
		goto out_unlock;

	if (old_pte)
		*old_pte = *pvmw.pte;

	/*
	 * Dirty-detection point:
	 *   pte_write() - the mapping is still writable (page may be dirty)
	 *   pte_dirty() - the processor has set the hardware dirty bit
	 *   pte_savedwrite() - software-emulated write permission
	 *   mm_tlb_flush_pending() - a concurrent flush may have hidden a write
	 *
	 * Any of these conditions means we must make the page read-only now and
	 * transfer the hardware dirty state to the struct page dirty flag.
	 */
	if (pte_write(*pvmw.pte) || pte_dirty(*pvmw.pte) ||
	    (pte_protnone(*pvmw.pte) && pte_savedwrite(*pvmw.pte)) ||
	    mm_tlb_flush_pending(mm)) {
		pte_t entry;

		swapped = PageSwapCache(page);
		flush_cache_page(vma, pvmw.address, page_to_pfn(page));

		/*
		 * Atomically clear the PTE and flush the TLB so that no
		 * O_DIRECT / DMA I/O can race between the mapcount check
		 * and the write-protection.
		 */
		entry = ptep_clear_flush_notify(vma, pvmw.address, pvmw.pte);

		/* Verify no concurrent O_DIRECT is using the page. */
		if (page_mapcount(page) + 1 + swapped != page_count(page)) {
			set_pte_at(mm, pvmw.address, pvmw.pte, entry);
			goto out_unlock;
		}

		/* Transfer hardware dirty bit → struct page. */
		if (pte_dirty(entry))
			set_page_dirty(page);

		/* Re-insert as clean, read-only. */
		if (pte_protnone(entry))
			entry = pte_mkclean(pte_clear_savedwrite(entry));
		else
			entry = pte_mkclean(pte_wrprotect(entry));

		set_pte_at_notify(mm, pvmw.address, pvmw.pte, entry);
	}

	*orig_pte = *pvmw.pte;
	err = 0;

out_unlock:
	page_vma_mapped_walk_done(&pvmw);
out_mn:
	mmu_notifier_invalidate_range_end(&range);
out:
	return err;
}


/* =========================================================================
 * Section 12: Hash-collision vs. page-change discrimination
 *
 * check_collision() is the key function that answers: "did the page really
 * change, or did we just get unlucky with a hash collision?"
 *
 * Strategy:
 *   If the rmap_item already has a hash_max (full-strength hash), compute
 *   the current hash_max for the page and compare.  Different → changed.
 *   Same → genuine collision at the max-strength level.
 *
 *   If no hash_max exists yet, re-hash at the current hash_strength.
 *   Different → changed.  Same → collision at the current level.
 * ========================================================================= */

/*
 * hash_cmp - three-way comparison for u32 hash values (used by the rb-tree
 * search routines inside stable_tree_delta_hash / stable_node_reinsert).
 */
static inline int hash_cmp(u32 new_val, u32 node_val)
{
	if (new_val > node_val)       return  1;
	else if (new_val < node_val)  return -1;
	else                          return  0;
}

/*
 * stable_node_hash_max - lazily compute and cache the HASH_STRENGTH_MAX hash
 * for a stable_node's page.  Called during stable-tree rebuilds.
 */
static inline void stable_node_hash_max(struct stable_node *node,
					struct page *page, u32 hash)
{
	u32 hash_max = node->hash_max;

	if (!hash_max) {
		hash_max = page_hash_max(page, hash);
		node->hash_max = hash_max;
	}
}

/*
 * rmap_item_hash_max - lazily compute and cache the HASH_STRENGTH_MAX hash
 * for an rmap_item's page.  Caches the result in item->hash_max.
 */
static inline u32 rmap_item_hash_max(struct rmap_item *item, u32 hash)
{
	u32 hash_max = item->hash_max;

	if (!hash_max) {
		hash_max = page_hash_max(item->page, hash);
		item->hash_max = hash_max;
	}

	return hash_max;
}

/*
 * check_collision - determine whether a hash mismatch is a page change or
 * a hash collision, and return the appropriate MERGE_ERR_* code.
 *
 * @rmap_item  The rmap_item whose stored hash disagrees with the tree entry.
 * @hash       The hash value that caused the tree lookup mismatch.
 *
 * Returns:
 *   MERGE_ERR_CHANGED  - the page was modified since the last scan (dirty)
 *   MERGE_ERR_COLLI    - genuine hash collision; page content is unchanged
 */
static inline int check_collision(struct rmap_item *rmap_item, u32 hash)
{
	int err;
	struct page *page = rmap_item->page;

	if (rmap_item->hash_max) {
		/*
		 * A full-strength hash is already cached.  Recompute it and
		 * compare: if different, the page changed; if same, collision.
		 */
		inc_rshash_neg(memcmp_cost);
		inc_rshash_neg(HASH_STRENGTH_MAX - hash_strength);

		if (rmap_item->hash_max == page_hash_max(page, hash))
			err = MERGE_ERR_COLLI;
		else
			err = MERGE_ERR_CHANGED;
	} else {
		/*
		 * No cached full-strength hash yet.  Re-hash at the current
		 * hash_strength.
		 */
		inc_rshash_neg(memcmp_cost + hash_strength);

		if (page_hash(page, hash_strength, 0) == hash)
			err = MERGE_ERR_COLLI;
		else
			err = MERGE_ERR_CHANGED;
	}

	return err;
}


/* =========================================================================
 * Section 13: Hash-strength adaptation state machine
 *
 * These functions implement the feedback loop that continuously tunes
 * hash_strength to maximise scan throughput while keeping false-negative
 * (missed dirty page) rates acceptably low.
 *
 * Call sequence (once per scan round):
 *   1. inc_uksm_pages_scanned() for every page examined.
 *   2. inc_rshash_pos() / inc_rshash_neg() as pages are hashed/compared.
 *   3. rshash_adjust()  at the end of the round.
 * ========================================================================= */

static inline void inc_hash_strength(unsigned long delta)
{
	hash_strength += 1 << delta;
	if (hash_strength > HASH_STRENGTH_MAX)
		hash_strength = HASH_STRENGTH_MAX;
}

static inline void dec_hash_strength(unsigned long delta)
{
	unsigned long change = 1 << delta;

	if (hash_strength <= change + 1)
		hash_strength = 1;
	else
		hash_strength -= change;
}

static inline void inc_hash_strength_delta(void)
{
	hash_strength_delta++;
	if (hash_strength_delta > HASH_STRENGTH_DELTA_MAX)
		hash_strength_delta = HASH_STRENGTH_DELTA_MAX;
}

/*
 * get_current_neg_ratio - return the ratio of negative to total benefit
 * as a percentage [0, 100].
 *
 * A high ratio means hashing is costing more than it saves → increase
 * hash_strength to catch real page changes faster (fewer false same-hash
 * results that force a full memcmp).
 */
static inline unsigned long get_current_neg_ratio(void)
{
	u64 pos = benefit.pos;
	u64 neg = benefit.neg;

	if (!neg)
		return 0;
	if (!pos || neg > pos)
		return 100;

	if (neg > div64_u64(U64_MAX, 100))
		pos = div64_u64(pos, 100);
	else
		neg *= 100;

	return div64_u64(neg, pos);
}

/*
 * get_current_benefit - return the net benefit per page scanned.
 *
 * net_benefit = (pos - neg) / scanned
 *
 * A higher value means the current hash_strength is providing a better
 * speed/accuracy trade-off.
 */
static inline unsigned long get_current_benefit(void)
{
	u64 pos     = benefit.pos;
	u64 neg     = benefit.neg;
	u64 scanned = benefit.scanned;

	if (neg > pos)
		return 0;

	return div64_u64((pos - neg), scanned);
}

/*
 * judge_rshash_direction - decide whether hash_strength should increase,
 * decrease, stay the same, or be re-probed from scratch (OBSCURE).
 *
 * The decision is driven by:
 *  - get_current_neg_ratio():  high ratio → hash is unreliable → GO_UP
 *  - get_current_benefit():    compare against the last stable baseline
 *    to detect whether benefit is improving or degrading.
 *
 * Returns one of: GO_UP, GO_DOWN, STILL, OBSCURE.
 */
static inline int judge_rshash_direction(void)
{
	u64 current_neg_ratio, stable_benefit;
	u64 current_benefit, delta = 0;
	int ret = STILL;

	/*
	 * Periodically inject an OBSCURE decision so the state machine
	 * re-explores the benefit landscape after long stable periods.
	 */
	if ((fully_scanned_round & 0xFFULL) == 10) {
		ret = OBSCURE;
		goto out;
	}

	current_neg_ratio = get_current_neg_ratio();

	if (current_neg_ratio == 0) {
		/* Hash never wrong → maybe we can afford fewer samples. */
		rshash_neg_cont_zero++;
		if (rshash_neg_cont_zero > 2)
			return GO_DOWN;
		else
			return STILL;
	}
	rshash_neg_cont_zero = 0;

	if (current_neg_ratio > 90) {
		/* Hash almost always wrong → need stronger hash urgently. */
		ret = GO_UP;
		goto out;
	}

	current_benefit = get_current_benefit();
	stable_benefit  = rshash_state.stable_benefit;

	if (!stable_benefit) {
		ret = OBSCURE;
		goto out;
	}

	if (current_benefit > stable_benefit)
		delta = current_benefit - stable_benefit;
	else if (current_benefit < stable_benefit)
		delta = stable_benefit - current_benefit;

	delta = div64_u64(100 * delta, stable_benefit);

	if (delta > 50) {
		rshash_cont_obscure++;
		if (rshash_cont_obscure > 2)
			return OBSCURE;
		else
			return STILL;
	}

out:
	rshash_cont_obscure = 0;
	return ret;
}

/*
 * uksm_drop_anon_vma - release the anon_vma reference held by an rmap_item
 * that is leaving the stable tree.  Needed by remove_node_from_stable_tree().
 */
static void uksm_drop_anon_vma(struct rmap_item *rmap_item)
{
	struct anon_vma *anon_vma = rmap_item->anon_vma;

	put_anon_vma(anon_vma);
}

/*
 * remove_node_from_stable_tree - remove a stable_node from the stable tree.
 *
 * Dependency of get_uksm_page() which is in turn a dependency of
 * stable_tree_delta_hash().  When a ksm page is found to be stale (its
 * page->mapping no longer points back to this node), this function removes
 * the node and releases all associated rmap_items.
 *
 * @unlink_rb        if true, unlink the node's rb_node from its sub-tree.
 * @remove_tree_node if true and the parent tree_node becomes empty, free it.
 */
static void remove_node_from_stable_tree(struct stable_node *stable_node,
					 int unlink_rb, int remove_tree_node)
{
	struct node_vma  *node_vma;
	struct rmap_item *rmap_item;
	struct hlist_node *n;

	if (!hlist_empty(&stable_node->hlist)) {
		hlist_for_each_entry_safe(node_vma, n,
					  &stable_node->hlist, hlist) {
			hlist_for_each_entry(rmap_item,
					     &node_vma->rmap_hlist, hlist) {
				uksm_pages_sharing--;
				uksm_drop_anon_vma(rmap_item);
				rmap_item->address &= PAGE_MASK;
			}
			free_node_vma(node_vma);
			cond_resched();
		}

		/* the last sharing entry is counted as the shared page itself */
		uksm_pages_shared--;
		uksm_pages_sharing++;
	}

	if (stable_node->tree_node && unlink_rb) {
		rb_erase(&stable_node->node,
			 &stable_node->tree_node->sub_root);

		if (RB_EMPTY_ROOT(&stable_node->tree_node->sub_root) &&
		    remove_tree_node) {
			rb_erase(&stable_node->tree_node->node,
				 root_stable_treep);
			free_tree_node(stable_node->tree_node);
		} else {
			stable_node->tree_node->count--;
		}
	}

	free_stable_node(stable_node);
}

/*
 * get_uksm_page - validate and get a reference to the ksm page that a
 * stable_node points to.
 *
 * Uses a "keyhole reference" pattern: instead of holding a page reference
 * permanently (which would block swapping), it reads page->mapping and
 * verifies it still points back to this stable_node.  If the page has been
 * freed or zapped, the node is removed from the stable tree.
 *
 * Dependency of stable_tree_delta_hash(), which calls this for every node
 * when rehashing the stable tree after a hash_strength change.
 *
 * @unlink_rb        passed to remove_node_from_stable_tree.
 * @remove_tree_node passed to remove_node_from_stable_tree.
 *
 * Returns a locked-and-referenced struct page, or NULL if the node is stale.
 */
static struct page *get_uksm_page(struct stable_node *stable_node,
				  int unlink_rb, int remove_tree_node)
{
	struct page *page;
	void *expected_mapping;
	unsigned long kpfn;

	expected_mapping = (void *)((unsigned long)stable_node |
				    PAGE_MAPPING_KSM);
again:
	kpfn = READ_ONCE(stable_node->kpfn);
	page = pfn_to_page(kpfn);

	/*
	 * On Alpha we need an explicit barrier to ensure page->mapping is
	 * read after kpfn.
	 */
	smp_read_barrier_depends();

	if (READ_ONCE(page->mapping) != expected_mapping)
		goto stale;

	/*
	 * Get a reference; if the page is at refcount 0 it may be being freed.
	 * We use PageSwapCache as a secondary check (see comment in ksm.c).
	 */
	while (!get_page_unless_zero(page)) {
		if (!PageSwapCache(page))
			goto stale;
		cpu_relax();
	}

	if (READ_ONCE(page->mapping) != expected_mapping) {
		put_page(page);
		goto stale;
	}

	lock_page(page);
	if (READ_ONCE(page->mapping) != expected_mapping) {
		unlock_page(page);
		put_page(page);
		goto stale;
	}
	unlock_page(page);
	return page;

stale:
	/*
	 * Re-check kpfn in case a concurrent ksm_migrate_page() changed it;
	 * if so, retry rather than incorrectly removing the node.
	 */
	smp_rmb();
	if (stable_node->kpfn != kpfn)
		goto again;

	remove_node_from_stable_tree(stable_node, unlink_rb, remove_tree_node);
	return NULL;
}

/*
 * free_all_tree_nodes - free every tree_node on `list`.
 *
 * Called by stable_tree_delta_hash() to discard the old tree_node set after
 * the stable tree has been rebuilt with the new hash_strength.
 */
static inline void free_all_tree_nodes(struct list_head *list)
{
	struct tree_node *node, *tmp;

	list_for_each_entry_safe(node, tmp, list, all_list)
		free_tree_node(node);
}

/*
 * stable_node_reinsert - reinsert a stable_node into a newly-built rb-tree
 * using a freshly computed hash.
 *
 * Part of the stable-tree rebuild triggered whenever hash_strength changes.
 * Resolves collisions in the two-level tree: first-level by `hash`, second
 * level by hash_max (computed at HASH_STRENGTH_MAX).
 *
 * @new_node       The node to insert.
 * @page           The ksm page backing this node (already referenced).
 * @root_treep     The new rb_root to insert into.
 * @tree_node_listp The list to allocate new tree_nodes from.
 * @hash           The new first-level hash for this node.
 */
static inline void stable_node_reinsert(struct stable_node *new_node,
					struct page *page,
					struct rb_root *root_treep,
					struct list_head *tree_node_listp,
					u32 hash)
{
	struct rb_node   **new    = &root_treep->rb_node;
	struct rb_node    *parent = NULL;
	struct stable_node *stable_node;
	struct tree_node  *tree_node;
	struct page       *tree_page;
	int cmp;

	while (*new) {
		int cmp;

		tree_node = rb_entry(*new, struct tree_node, node);
		cmp = hash_cmp(hash, tree_node->hash);

		if (cmp < 0) {
			parent = *new;
			new    = &parent->rb_left;
		} else if (cmp > 0) {
			parent = *new;
			new    = &parent->rb_right;
		} else {
			break;
		}
	}

	if (*new) {
		/* A tree_node with this first-level hash already exists. */
		stable_node_hash_max(new_node, page, hash);
		if (tree_node->count == 1) {
			stable_node = rb_entry(tree_node->sub_root.rb_node,
					       struct stable_node, node);
			tree_page = get_uksm_page(stable_node, 1, 0);
			if (tree_page) {
				stable_node_hash_max(stable_node,
						     tree_page, hash);
				put_page(tree_page);

				cmp = hash_cmp(new_node->hash_max,
					       stable_node->hash_max);
				parent = &stable_node->node;
				if (cmp < 0)
					new = &parent->rb_left;
				else if (cmp > 0)
					new = &parent->rb_right;
				else
					goto failed;

				goto add_node;
			} else {
				goto tree_node_reuse;
			}
		}

		/* Search the collision sub-tree by hash_max. */
		new    = &tree_node->sub_root.rb_node;
		parent = NULL;
		BUG_ON(!*new);
		while (*new) {
			int cmp;

			stable_node = rb_entry(*new, struct stable_node, node);
			cmp = hash_cmp(new_node->hash_max,
				       stable_node->hash_max);

			if (cmp < 0) {
				parent = *new;
				new    = &parent->rb_left;
			} else if (cmp > 0) {
				parent = *new;
				new    = &parent->rb_right;
			} else {
				goto failed;   /* double collision */
			}
		}
		goto add_node;
	}

	/* No matching tree_node: allocate a fresh one. */
	tree_node = alloc_tree_node(tree_node_listp);
	if (!tree_node) {
		pr_err("UKSM: memory allocation error!\n");
		goto failed;
	} else {
		tree_node->hash = hash;
		rb_link_node(&tree_node->node, parent, new);
		rb_insert_color(&tree_node->node, root_treep);

tree_node_reuse:
		parent = NULL;
		new    = &tree_node->sub_root.rb_node;
	}

add_node:
	rb_link_node(&new_node->node, parent, new);
	rb_insert_color(&new_node->node, &tree_node->sub_root);
	new_node->tree_node = tree_node;
	tree_node->count++;
	return;

failed:
	/* Two-level collision: leave the node unlinked. */
	new_node->tree_node = NULL;
}

/*
 * stable_tree_delta_hash - recompute every stable node's hash for the new
 * hash_strength and rebuild the entire stable tree from scratch.
 *
 * Called by rshash_adjust() whenever hash_strength changes.  The old tree is
 * discarded and a fresh tree is built using the inactive rb_root / list pair
 * so that the switch can be made atomically (from uksmd's perspective).
 *
 * @prev_hash_strength  The hash_strength value that was active when each
 *                      node's tree_node->hash was computed; needed by
 *                      delta_hash() to extend the hash incrementally.
 *
 * NOTE: This function is a dependency of rshash_adjust() and is required for
 * correct dirty-detection behaviour after a hash_strength change.  However it
 * also maintains the deduplication tree structure as a side effect.
 */
static inline void stable_tree_delta_hash(u32 prev_hash_strength)
{
	struct stable_node *node, *tmp;
	struct rb_root      *root_new_treep;
	struct list_head    *new_tree_node_listp;

	stable_tree_index    = (stable_tree_index + 1) % 2;
	root_new_treep       = &root_stable_tree[stable_tree_index];
	new_tree_node_listp  = &stable_tree_node_list[stable_tree_index];
	*root_new_treep      = RB_ROOT;
	BUG_ON(!list_empty(new_tree_node_listp));

	list_for_each_entry_safe(node, tmp, &stable_node_list, all_list) {
		void   *addr;
		struct page *node_page;
		u32 hash;

		node_page = get_uksm_page(node, 0, 0);
		if (!node_page)
			continue;

		if (node->tree_node) {
			hash = node->tree_node->hash;

			addr = kmap_atomic(node_page);
			hash = delta_hash(addr, prev_hash_strength,
					  hash_strength, hash);
			kunmap_atomic(addr);
		} else {
			/*
			 * Node was not inserted in the rb-tree last round
			 * (double collision); full rehash needed.
			 */
			hash = page_hash(node_page, hash_strength, 0);
		}

		stable_node_reinsert(node, node_page, root_new_treep,
				     new_tree_node_listp, hash);
		put_page(node_page);
	}

	root_stable_treep = root_new_treep;
	free_all_tree_nodes(stable_tree_node_listp);
	BUG_ON(!list_empty(stable_tree_node_listp));
	stable_tree_node_listp = new_tree_node_listp;
}

/*
 * rshash_adjust - the main adaptive hash-strength state machine.
 *
 * Must be called once per scan round (after all pages in the round have been
 * hashed).  Drives the five-state machine (RSHASH_NEW → RSHASH_STILL →
 * RSHASH_TRYDOWN / RSHASH_TRYUP → RSHASH_PRE_STILL → RSHASH_STILL) to find
 * the hash_strength value that maximises (pos - neg) / scanned.
 *
 * Returns non-zero if hash_strength was changed.  In that case,
 * stable_tree_delta_hash() is called automatically to keep the stable tree
 * consistent.
 *
 * Dirty-detection relevance:
 *   A higher hash_strength catches more page changes per scan (fewer
 *   false-same results), at the cost of slower scanning.  The state machine
 *   continuously seeks the optimal point on this curve so that dirty pages
 *   are detected as quickly as possible without wasting CPU on unnecessary
 *   work.
 */
static inline int rshash_adjust(void)
{
	unsigned long prev_hash_strength = hash_strength;

	if (!encode_benefit())
		return 0;

	switch (rshash_state.state) {
	case RSHASH_STILL:
		switch (judge_rshash_direction()) {
		case GO_UP:
			if (rshash_state.pre_direct == GO_DOWN)
				hash_strength_delta = 0;

			inc_hash_strength(hash_strength_delta);
			inc_hash_strength_delta();
			rshash_state.stable_benefit = get_current_benefit();
			rshash_state.pre_direct = GO_UP;
			break;

		case GO_DOWN:
			if (rshash_state.pre_direct == GO_UP)
				hash_strength_delta = 0;

			dec_hash_strength(hash_strength_delta);
			inc_hash_strength_delta();
			rshash_state.stable_benefit = get_current_benefit();
			rshash_state.pre_direct = GO_DOWN;
			break;

		case OBSCURE:
			rshash_state.stable_point      = hash_strength;
			rshash_state.turn_point_down   = hash_strength;
			rshash_state.turn_point_up     = hash_strength;
			rshash_state.turn_benefit_down = get_current_benefit();
			rshash_state.turn_benefit_up   = get_current_benefit();
			rshash_state.lookup_window_index = 0;
			rshash_state.state = RSHASH_TRYDOWN;
			dec_hash_strength(hash_strength_delta);
			inc_hash_strength_delta();
			break;

		case STILL:
			break;
		default:
			BUG();
		}
		break;

	case RSHASH_TRYDOWN:
		if (rshash_state.lookup_window_index++ % 5 == 0)
			rshash_state.below_count = 0;

		if (get_current_benefit() < rshash_state.stable_benefit)
			rshash_state.below_count++;
		else if (get_current_benefit() > rshash_state.turn_benefit_down) {
			rshash_state.turn_point_down   = hash_strength;
			rshash_state.turn_benefit_down = get_current_benefit();
		}

		if (rshash_state.below_count >= 3 ||
		    judge_rshash_direction() == GO_UP ||
		    hash_strength == 1) {
			hash_strength       = rshash_state.stable_point;
			hash_strength_delta = 0;
			inc_hash_strength(hash_strength_delta);
			inc_hash_strength_delta();
			rshash_state.lookup_window_index = 0;
			rshash_state.state  = RSHASH_TRYUP;
			hash_strength_delta = 0;
		} else {
			dec_hash_strength(hash_strength_delta);
			inc_hash_strength_delta();
		}
		break;

	case RSHASH_TRYUP:
		if (rshash_state.lookup_window_index++ % 5 == 0)
			rshash_state.below_count = 0;

		if (get_current_benefit() < rshash_state.turn_benefit_down)
			rshash_state.below_count++;
		else if (get_current_benefit() > rshash_state.turn_benefit_up) {
			rshash_state.turn_point_up   = hash_strength;
			rshash_state.turn_benefit_up = get_current_benefit();
		}

		if (rshash_state.below_count >= 3 ||
		    judge_rshash_direction() == GO_DOWN ||
		    hash_strength == HASH_STRENGTH_MAX) {
			hash_strength = rshash_state.turn_benefit_up >
					rshash_state.turn_benefit_down ?
					rshash_state.turn_point_up :
					rshash_state.turn_point_down;

			rshash_state.state = RSHASH_PRE_STILL;
		} else {
			inc_hash_strength(hash_strength_delta);
			inc_hash_strength_delta();
		}
		break;

	case RSHASH_NEW:
	case RSHASH_PRE_STILL:
		rshash_state.stable_benefit = get_current_benefit();
		rshash_state.state          = RSHASH_STILL;
		hash_strength_delta         = 0;
		break;
	default:
		BUG();
	}

	reset_benefit();

	if (prev_hash_strength != hash_strength)
		stable_tree_delta_hash(prev_hash_strength);

	return prev_hash_strength != hash_strength;
}


/* =========================================================================
 * Section 14: Initialisation
 *
 * These functions must be called once during UKSM module init before the
 * scanner thread starts.
 * ========================================================================= */

/*
 * cal_positive_negative_costs - measure the relative cost of a full-page
 * memcmp versus a single hash-position sample on this CPU.
 *
 * Allocates two pages, fills them with identical random data (differing only
 * in the last byte so memcmp always returns non-zero), then times:
 *   - N iterations of page_hash(p1, HASH_STRENGTH_FULL, 0)
 *   - N iterations of pages_identical_with_cost(p1, p2)
 *
 * The ratio is stored in `memcmp_cost` in units of "hash positions per
 * memcmp", which is used by inc_rshash_neg() to correctly weight the cost
 * of comparisons in the benefit calculation.
 *
 * Returns 0 on success, -ENOMEM on allocation failure.
 */
static inline int cal_positive_negative_costs(void)
{
	struct page *p1, *p2;
	unsigned char *addr1, *addr2;
	unsigned long i, time_start, hash_cost;
	unsigned long loopnum = 0;

	/* volatile prevents the compiler from optimising away the loops. */
	volatile u32 hash;
	volatile int ret;

	p1 = alloc_page(GFP_KERNEL);
	if (!p1)
		return -ENOMEM;

	p2 = alloc_page(GFP_KERNEL);
	if (!p2) {
		__free_page(p1);
		return -ENOMEM;
	}

	addr1 = kmap_atomic(p1);
	addr2 = kmap_atomic(p2);
	memset(addr1, prandom_u32(), PAGE_SIZE);
	memcpy(addr2, addr1, PAGE_SIZE);
	addr2[PAGE_SIZE - 1] = ~addr2[PAGE_SIZE - 1];  /* ensure pages differ */
	kunmap_atomic(addr2);
	kunmap_atomic(addr1);

	time_start = jiffies;
	while (jiffies - time_start < 100) {
		for (i = 0; i < 100; i++)
			hash = page_hash(p1, HASH_STRENGTH_FULL, 0);
		loopnum += 100;
	}
	hash_cost = jiffies - time_start;

	time_start = jiffies;
	for (i = 0; i < loopnum; i++)
		ret = pages_identical_with_cost(p1, p2);
	memcmp_cost = HASH_STRENGTH_FULL * (jiffies - time_start);
	memcmp_cost /= hash_cost;

	pr_info("UKSM: relative memcmp_cost = %lu hash=%u cmp_ret=%d.\n",
		memcmp_cost, hash, ret);

	__free_page(p1);
	__free_page(p2);
	return 0;
}

/*
 * init_zeropage_hash_table - precompute the hash of an all-zero page at every
 * possible hash_strength value [0, HASH_STRENGTH_MAX).
 *
 * Stored in zero_hash_table[] so that find_zero_page_hash() can detect a
 * zero page in O(1) without a full is_full_zero() scan.
 *
 * Returns 0 on success, -ENOMEM on allocation failure.
 */
static int init_zeropage_hash_table(void)
{
	struct page *page;
	char *addr;
	int i;

	page = alloc_page(GFP_KERNEL);
	if (!page)
		return -ENOMEM;

	addr = kmap_atomic(page);
	memset(addr, 0, PAGE_SIZE);
	kunmap_atomic(addr);

	zero_hash_table = kmalloc_array(HASH_STRENGTH_MAX, sizeof(u32),
					GFP_KERNEL);
	if (!zero_hash_table) {
		__free_page(page);
		return -ENOMEM;
	}

	for (i = 0; i < HASH_STRENGTH_MAX; i++)
		zero_hash_table[i] = page_hash(page, i, 0);

	__free_page(page);
	return 0;
}

/*
 * init_random_sampling - initialise the random-position permutation table
 * and the hash-strength adaptation state machine.
 *
 * Uses a Fisher-Yates shuffle to generate a uniform random permutation of
 * [0, HASH_STRENGTH_FULL) so that each hash_strength value N samples a
 * consistent, non-repeating set of N positions across all calls.
 *
 * Also calls cal_positive_negative_costs() to calibrate memcmp_cost.
 *
 * Returns 0 on success, -ENOMEM on allocation failure, or the error from
 * cal_positive_negative_costs().
 */
static inline int init_random_sampling(void)
{
	unsigned long i;

	random_nums = kmalloc(PAGE_SIZE, GFP_KERNEL);
	if (!random_nums)
		return -ENOMEM;

	/* Initialise identity permutation then shuffle. */
	for (i = 0; i < HASH_STRENGTH_FULL; i++)
		random_nums[i] = i;

	for (i = 0; i < HASH_STRENGTH_FULL; i++) {
		unsigned long rand_range, swap_index, tmp;

		rand_range = HASH_STRENGTH_FULL - i;
		swap_index = i + prandom_u32() % rand_range;
		tmp              = random_nums[i];
		random_nums[i]   = random_nums[swap_index];
		random_nums[swap_index] = tmp;
	}

	rshash_state.state               = RSHASH_NEW;
	rshash_state.below_count         = 0;
	rshash_state.lookup_window_index = 0;

	return cal_positive_negative_costs();
}
