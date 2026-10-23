// The Metal conversation cache's two extra tiers (--serve): a pinned root slot (the shared system prompt + tools,
// reserved at start-up, never evicted by other requests) and parked conversations on the SSD (displaced by another
// chat or a side request, restored on the next request that continues them).  Both hold the exact bytes of the
// decode window's positional K/V and running state, so a restore gives the state the conversation had.
// The window's buffers come in as plain pointers, so the copies and the file format are testable on synthetic
// buffers (metal/tests/conversation_ssd_test.cpp).
#pragma once

#include "strata/core/conversation_cache.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core::kvstore {

/// One attention layer's K/V parts in the window: k, v, k_scale, v_scale, pooled indexer keys.
using Parts = std::array<void*, 5>;

/// The byte layout of a layer's parts: bytes per cell of k, v, k_scale, v_scale (cells rounded up to `page`), and
/// bytes per pooled row (`tokens / block + 1` rows: the completed blocks and the moving one).
struct Layout {
    std::array<uint64_t, 4> cell{};
    uint64_t pooled_row = 0;
    int64_t page = 4, block = 4;
    std::array<uint64_t, 5> sizes(int64_t tokens) const;
    bool operator==(const Layout&) const = default;
};

/// The running state's five parts as checkpoints hold them: gdn, ple, tails, dead, block_pos (the draft's tails).
using StateSizes = std::array<uint64_t, 5>;
/// A running state as segments in memory, per part in checkpoint byte order: a checkpoint's vectors (one segment a
/// part) or the window's own buffers (one segment per layer), so neither side needs a copy of the other's layout.
using Seg = std::pair<void*, size_t>;
using StateView = std::array<std::vector<Seg>, 5>;
StateView view_of(ConversationCheckpoint& c);
StateView view_of(const ConversationCheckpoint& c);
bool view_matches(const StateView& v, const StateSizes& s);
bool state_matches(const ConversationCheckpoint& c, const StateSizes& s);

/// A streaming 64-bit hash (four lanes over 8-byte words): the same value however the bytes are split.
class Hasher {
public:
    explicit Hasher(uint64_t seed);
    void update(const void* p, size_t n);
    uint64_t finish() const;

private:
    uint64_t h_[4];
    uint8_t buf_[32];
    size_t have_ = 0;
    uint64_t total_ = 0;
};
uint64_t hash_bytes(const void* p, size_t n, uint64_t seed);

/// The pinned root: host memory reserved once for up to `cap` tokens of every layer's K/V and one running state.
class RootSlot {
public:
    bool reserve(int64_t cap, size_t layers, const Layout& layout, const StateSizes& state, std::string& err);
    bool enabled() const { return cap_ > 0; }
    int64_t cap() const { return cap_; }
    const std::vector<int32_t>& ids() const { return ids_; }
    const ConversationCheckpoint& state() const { return state_; }
    uint64_t reserved_bytes() const { return reserved_; }
    /// Copies the first ids.size() tokens of each layer's parts and the running state at that point.  A refused
    /// root leaves the slot as it was; a failure after the copy began leaves it empty.
    bool fill(const std::vector<int32_t>& ids, const StateView& state, const std::vector<Parts>& src, std::string& err);
    /// Copies the slot's K/V into the first ids().size() positions of `dst` (the state is the caller's).
    bool restore_kv(const std::vector<Parts>& dst, std::string& err) const;
    /// The slot's bytes hashed again against the hash taken when it was filled: false (and why) when they changed.
    bool verify(std::string& err) const;
    uint64_t digest() const { return digest_; }
    void clear() { ids_.clear(); }

private:
    int64_t cap_ = 0;
    uint64_t reserved_ = 0;
    Layout layout_;
    StateSizes state_sizes_{};
    std::vector<int32_t> ids_;
    std::vector<std::array<std::vector<uint8_t>, 5>> kv_;
    ConversationCheckpoint state_;
    uint64_t digest_ = 0;
    uint64_t compute_digest() const;
};

/// One conversation file's identity: what the engine must be for its bytes to mean the same.
struct DiskIdentity {
    uint64_t engine = 0;                     ///< model files, settings, build, kernel sources, STRATA_* environment
    std::array<int64_t, 18> geometry{};
    uint32_t layers = 0;                     ///< K/V layers (the draft layer included when it is loaded)
    Layout layout;
    StateSizes state{};
};

struct DiskStats {
    uint64_t bytes = 0, files = 0, written = 0, read = 0, evicted = 0, dead_tmp = 0;
};

/// Parked conversations on the SSD: one file each under `dir` (0700, locked by one engine), written uncached
/// (F_NOCACHE), fsynced and renamed into place, LRU-evicted under `cap` bytes - every *.skv in the dir counts,
/// another engine's first.  Files of the same identity survive restarts (their headers are read at open).
class DiskStore {
public:
    struct Match { uint64_t id = 0; int64_t tokens = 0; bool live = false, root = false; };
    struct Options {
        uint64_t cap = 0;
        uint64_t min_free = 10ull << 30;     ///< free disk space kept, on the cache's volume and the boot volume
        int64_t stale_tmp_s = 600;           ///< unused: under the directory lock every *.skv.tmp is dead
    };
    ~DiskStore();
    bool open(const std::string& dir, const Options& opt, const DiskIdentity& id, std::string& err);
    bool enabled() const { return opt_.cap > 0; }
    const std::string& dir() const { return dir_; }
    /// The deepest saved point (the live end or a checkpoint) the prompt starts with, at most prompt.size() - 1.
    Match best(const std::vector<int64_t>& prompt) const;
    /// A pinned entry is never evicted, superseded or overwritten until unpin().
    void pin(uint64_t id);
    void unpin();
    /// Writes a conversation: `ids` the tokens its state holds, `live` the state after them, `checks` its
    /// checkpoints (prefixes of ids), `src` the K/V (at least ids.size() cells).  `root`: the pinned root's file.
    bool write(const std::vector<int32_t>& ids, const StateView& live, const std::vector<ConversationCheckpoint>& checks,
               bool cvec, bool root, const std::vector<Parts>& src, std::string& err, uint64_t* bytes = nullptr);
    /// What reading entry `id` back for a match at `upto` tokens needs in RAM: the checkpoints it loads, the deepest
    /// at most `upto` and the shallowest (the root) - the others stay in the file only.
    uint64_t read_ram(uint64_t id, int64_t upto) const;
    /// Reads entry `id` back for a prompt that matched it at `upto` tokens.  The file's tokens are read and checked
    /// against prompt[0, upto) BEFORE anything is written to `live` or `dst`.  `live` (the live state's
    /// destination) is read only when non-null; `checks` gets the deepest and the shallowest checkpoint of at most
    /// `upto` tokens (and a pinned one); `dst` the
    /// K/V.  Every section read is checked against its hash; on a failure after the check `dst` / `live` may be
    /// partly written (`touched` says so).
    /// `dst_cells`: the window's cells (a longer file is refused).  `damaged`: the file itself is wrong (header,
    /// hashes, tokens, a short file) - the only failures that should delete it.
    bool read(uint64_t id, const std::vector<int64_t>& prompt, int64_t upto, const StateView* live,
              std::vector<ConversationCheckpoint>& checks, bool& cvec, bool& root, const std::vector<Parts>& dst,
              int64_t dst_cells, std::string& err, bool& touched, bool& damaged, uint64_t* bytes = nullptr);
    void remove(uint64_t id);
    void remove_all();
    /// Removes the files a conversation's newer state supersedes: the same conversation a turn back (its deepest
    /// checkpoint is one of `checks` or `ids`).  Root and pinned files stay.  Returns how many went.
    /// Only when the file's tail past that checkpoint holds at most one `turn_token` (a reply, not a whole chat).
    size_t drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationCheckpoint>& checks,
                           int64_t turn_token);
    bool has(const std::vector<int32_t>& ids) const;
    const DiskStats& stats() const { return stats_; }
    size_t files() const;   ///< this identity's files

private:
    struct Entry {
        uint64_t id = 0;
        std::string path;
        std::vector<int32_t> ids;
        std::vector<int64_t> checks;
        uint64_t bytes = 0, used = 0;
        bool root = false, foreign = false;
    };
    bool make_room(uint64_t need, std::string& err);
    Entry* find(uint64_t id);
    const Entry* find(uint64_t id) const;
    void erase(size_t k);
    std::string dir_;
    Options opt_;
    int lock_fd_ = -1;
    uint64_t clock_ = 0, next_id_ = 0, pinned_ = 0;
    DiskIdentity id_;
    std::vector<Entry> entries_;
    DiskStats stats_;
};

}  // namespace strata::core::kvstore

namespace strata::core {
struct ModelGeometry;
/// Metal builds only (conversation_state.cpp): the decode window's K/V parts per layer (the QSA layers, then the
/// draft layer when loaded), its cell count, the parts' layout and the running state's sizes.
bool metal_kv_window(std::vector<kvstore::Parts>& parts, int64_t& cells, kvstore::Layout& layout,
                     kvstore::StateSizes& state, const ModelGeometry& g, std::string& error);
/// Metal builds only: the window's running state as segments of its own buffers.  `restore`: about to be written
/// (any pending fold is dropped), else about to be read (the pending fold runs first).
bool metal_state_view(kvstore::StateView& view, bool restore, std::string& error);
}  // namespace strata::core
