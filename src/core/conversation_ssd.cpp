#include "strata/core/conversation_ssd.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fcntl.h>
#include <random>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/time.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <stdio.h>   // renamex_np
#endif

namespace strata::core::kvstore {
namespace fs = std::filesystem;

namespace {
constexpr char kMagic[8] = {'S', 'T', 'R', 'A', 'T', 'K', 'V', '2'};
constexpr uint32_t kVersion = 2;
constexpr uint64_t kChunk = 64ull << 20;   // one read or write call
constexpr uint32_t kMaxChecks = 1024;

struct FileHeader {
    char magic[8];
    uint32_t version, kind;   // kind 1: the pinned root's file
    uint64_t engine;
    int64_t geometry[18];
    uint32_t layers, n_checks;
    int64_t tokens;
    uint64_t layout[6];       // cell[4], pooled_row, page | block << 32
    uint64_t state[5];
    uint64_t cvec;
    uint64_t table_hash;      // the section hash table that follows the header
    uint64_t header_hash;
};

uint64_t mix(uint64_t h) {
    h ^= h >> 33; h *= 0xff51afd7ed558ccdull;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ull;
    return h ^ (h >> 33);
}

uint64_t header_hash(const FileHeader& h) { return hash_bytes(&h, offsetof(FileHeader, header_hash), 0x5354524154ull); }

void fill_header(FileHeader& h, const DiskIdentity& id, bool root, uint32_t n_checks, int64_t tokens, bool cvec) {
    std::memset(&h, 0, sizeof h);
    std::memcpy(h.magic, kMagic, sizeof kMagic);
    h.version = kVersion;
    h.kind = root ? 1 : 0;
    h.engine = id.engine;
    std::copy(id.geometry.begin(), id.geometry.end(), h.geometry);
    h.layers = id.layers;
    h.n_checks = n_checks;
    h.tokens = tokens;
    for (int i = 0; i < 4; ++i) h.layout[i] = id.layout.cell[(size_t) i];
    h.layout[4] = id.layout.pooled_row;
    h.layout[5] = (uint64_t) id.layout.page | ((uint64_t) id.layout.block << 32);
    std::copy(id.state.begin(), id.state.end(), h.state);
    h.cvec = cvec ? 1 : 0;
}

bool header_valid(const FileHeader& h) {
    return std::memcmp(h.magic, kMagic, sizeof kMagic) == 0 && h.version == kVersion && h.kind <= 1 &&
           h.header_hash == header_hash(h) && h.tokens > 0 && h.tokens < (int64_t) 1 << 31 && h.n_checks <= kMaxChecks;
}

bool header_matches(const FileHeader& h, const DiskIdentity& id) {
    FileHeader want;
    fill_header(want, id, h.kind == 1, h.n_checks, h.tokens, h.cvec != 0);
    return header_valid(h) && h.engine == want.engine && std::memcmp(h.geometry, want.geometry, sizeof h.geometry) == 0 &&
           h.layers == want.layers && std::memcmp(h.layout, want.layout, sizeof h.layout) == 0 &&
           std::memcmp(h.state, want.state, sizeof h.state) == 0;
}

// the sections in file order: ids, the checkpoint table, the live state's 5 parts, each checkpoint's 5, the K/V
struct Plan {
    std::vector<uint64_t> size, offset;
    uint64_t end = 0;
    size_t live0 = 2, checks0 = 7, kv0 = 0;
};
Plan plan(const DiskIdentity& id, int64_t tokens, uint32_t n_checks) {
    Plan p;
    p.size.push_back((uint64_t) tokens * 4);
    p.size.push_back((uint64_t) n_checks * 16);
    for (int r = 0; r < 1 + (int) n_checks; ++r)
        for (uint64_t s : id.state) p.size.push_back(s);
    p.kv0 = p.size.size();
    const auto sz = id.layout.sizes(tokens);
    for (uint32_t j = 0; j < id.layers; ++j)
        for (uint64_t s : sz) p.size.push_back(s);
    uint64_t at = sizeof(FileHeader) + p.size.size() * 8;
    for (uint64_t s : p.size) { p.offset.push_back(at); at += s; }
    p.end = at;
    return p;
}

bool write_all(int fd, uint64_t at, const void* p, uint64_t n, std::string& err) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    while (n > 0) {
        const size_t k = (size_t) std::min(n, kChunk);
        const ssize_t w = ::pwrite(fd, b, k, (off_t) at);
        if (w <= 0) {
            if (w < 0 && errno == EINTR) continue;
            err = std::string("ssd cache: write failed: ") + std::strerror(errno);
            return false;
        }
        b += w; at += (uint64_t) w; n -= (uint64_t) w;
        progress_beat();
    }
    return true;
}

bool read_all(int fd, uint64_t at, void* p, uint64_t n, std::string& err) {
    uint8_t* b = static_cast<uint8_t*>(p);
    while (n > 0) {
        const size_t k = (size_t) std::min(n, kChunk);
        const ssize_t r = ::pread(fd, b, k, (off_t) at);
        if (r <= 0) {
            if (r < 0 && errno == EINTR) continue;
            err = r == 0 ? "ssd cache: the file is shorter than its header says"
                         : std::string("ssd cache: read failed: ") + std::strerror(errno);
            return false;
        }
        b += r; at += (uint64_t) r; n -= (uint64_t) r;
        progress_beat();
    }
    return true;
}

void uncached(int fd) {
#if defined(F_NOCACHE)
    (void) ::fcntl(fd, F_NOCACHE, 1);   // the pages are not kept in the file cache: the model's pages stay
#else
    (void) fd;
#endif
}

struct Fd {
    int fd = -1;
    ~Fd() { if (fd >= 0) ::close(fd); }
};

bool ends_with(const std::string& s, const char* t) {
    const size_t n = std::strlen(t);
    return s.size() >= n && s.compare(s.size() - n, n, t) == 0;
}

uint64_t free_bytes(const std::string& path) {
    struct statvfs sv{};
    if (::statvfs(path.c_str(), &sv) != 0) return UINT64_MAX;   // unknown: not the reason to refuse
    return (uint64_t) sv.f_bavail * (uint64_t) sv.f_frsize;
}
}  // namespace

// ---- hashing
Hasher::Hasher(uint64_t seed)
    : h_{seed ^ 0x9e3779b97f4a7c15ull, seed + 0x632be59bd9b4e019ull, ~seed, seed * 0x94d049bb133111ebull + 1} {}

void Hasher::update(const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    total_ += n;
    auto block = [&](const uint8_t* q) {
        for (int l = 0; l < 4; ++l) {
            uint64_t w;
            std::memcpy(&w, q + 8 * l, 8);
            h_[l] = (h_[l] ^ w) * 0x100000001b3ull + (h_[l] >> 29);
        }
    };
    if (have_ > 0) {
        const size_t k = std::min(n, 32 - have_);
        std::memcpy(buf_ + have_, b, k);
        have_ += k; b += k; n -= k;
        if (have_ < 32) return;
        block(buf_);
        have_ = 0;
    }
    for (; n >= 32; b += 32, n -= 32) block(b);
    std::memcpy(buf_, b, n);
    have_ = n;
}

uint64_t Hasher::finish() const {
    uint64_t t = total_;
    for (size_t i = 0; i < have_; ++i) t = (t ^ buf_[i]) * 0x100000001b3ull;
    return mix(mix(h_[0]) ^ mix(h_[1] + 1) ^ mix(h_[2] + 2) ^ mix(h_[3] + 3) ^ mix(t));
}

uint64_t hash_bytes(const void* p, size_t n, uint64_t seed) {
    Hasher h(seed);
    h.update(p, n);
    return h.finish();
}

// ---- layouts and state views
std::array<uint64_t, 5> Layout::sizes(int64_t tokens) const {
    const uint64_t cells = (uint64_t) ((tokens + page - 1) / page * page);
    return {cells * cell[0], cells * cell[1], cells * cell[2], cells * cell[3],
            (uint64_t) (tokens / block + 1) * pooled_row};
}

StateView view_of(ConversationCheckpoint& c) {
    StateView v;
    std::vector<uint8_t>* p[5] = {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos};
    for (size_t i = 0; i < 5; ++i) if (!p[i]->empty()) v[i].push_back({p[i]->data(), p[i]->size()});
    return v;
}
StateView view_of(const ConversationCheckpoint& c) { return view_of(const_cast<ConversationCheckpoint&>(c)); }

bool view_matches(const StateView& v, const StateSizes& s) {
    for (size_t i = 0; i < 5; ++i) {
        uint64_t n = 0;
        for (const Seg& g : v[i]) {
            if (g.first == nullptr && g.second > 0) return false;
            n += g.second;
        }
        if (n != s[i]) return false;
    }
    return true;
}

bool state_matches(const ConversationCheckpoint& c, const StateSizes& s) {
    return view_matches(view_of(c), s) && c.stage_parts.empty() && c.imgs.empty();
}

// ---- the root slot
bool RootSlot::reserve(int64_t cap, size_t layers, const Layout& layout, const StateSizes& state, std::string& err) {
    cap_ = 0;
    reserved_ = 0;
    ids_.clear();
    kv_.clear();
    if (cap <= 0) return true;
    try {
        const auto sz = layout.sizes(cap);
        kv_.resize(layers);
        for (auto& layer : kv_)
            for (size_t i = 0; i < 5; ++i) {
                layer[i].resize((size_t) sz[i]);   // zero-filled: the pages are taken now, not when a root arrives
                reserved_ += sz[i];
            }
        std::vector<uint8_t>* p[5] = {&state_.gdn, &state_.ple, &state_.tails, &state_.dead, &state_.block_pos};
        for (size_t i = 0; i < 5; ++i) {
            p[i]->resize((size_t) state[i]);
            reserved_ += state[i];
        }
        ids_.reserve((size_t) cap);
        state_.ids.reserve((size_t) cap);
    } catch (const std::bad_alloc&) {
        kv_.clear();
        err = "root slot: not enough memory to reserve it";
        return false;
    }
    layout_ = layout;
    state_sizes_ = state;
    cap_ = cap;
    return true;
}

bool RootSlot::fill(const std::vector<int32_t>& ids, const StateView& state, const std::vector<Parts>& src,
                    std::string& err) {
    if (!enabled()) { err = "root slot: off"; return false; }
    if (ids.empty() || (int64_t) ids.size() > cap_) { err = "root slot: the root is longer than the slot"; return false; }
    if (src.size() != kv_.size()) { err = "root slot: K/V layer count differs"; return false; }
    if (!view_matches(state, state_sizes_)) { err = "root slot: the state is not this window's"; return false; }
    for (const auto& layer : src)
        for (void* p : layer)
            if (p == nullptr) { err = "root slot: missing K/V buffer"; return false; }
    ids_.clear();   // from here on a failure leaves the slot empty, never half old and half new
    const auto sz = layout_.sizes((int64_t) ids.size());
    for (size_t j = 0; j < kv_.size(); ++j)
        for (size_t i = 0; i < 5; ++i) std::memcpy(kv_[j][i].data(), src[j][i], (size_t) sz[i]);
    std::vector<uint8_t>* to[5] = {&state_.gdn, &state_.ple, &state_.tails, &state_.dead, &state_.block_pos};
    for (size_t i = 0; i < 5; ++i) {
        size_t at = 0;
        for (const Seg& g : state[i]) {   // within the reserve: no allocation
            std::memcpy(to[i]->data() + at, g.first, g.second);
            at += g.second;
        }
    }
    state_.ids.assign(ids.begin(), ids.end());
    ids_.assign(ids.begin(), ids.end());
    digest_ = compute_digest();
    return true;
}

uint64_t RootSlot::compute_digest() const {
    Hasher h(0x726f6f74ull);
    h.update(ids_.data(), ids_.size() * 4);
    const auto sz = layout_.sizes((int64_t) ids_.size());
    for (const auto& layer : kv_)
        for (size_t i = 0; i < 5; ++i) h.update(layer[i].data(), (size_t) sz[i]);
    for (const auto* p : {&state_.gdn, &state_.ple, &state_.tails, &state_.dead, &state_.block_pos}) h.update(p->data(), p->size());
    return h.finish();
}

bool RootSlot::verify(std::string& err) const {
    if (ids_.empty()) { err = "root slot: empty"; return false; }
    if (compute_digest() != digest_) { err = "root slot: its bytes changed since it was filled"; return false; }
    return true;
}

bool RootSlot::restore_kv(const std::vector<Parts>& dst, std::string& err) const {
    if (ids_.empty()) { err = "root slot: empty"; return false; }
    if (dst.size() != kv_.size()) { err = "root slot: K/V layer count differs"; return false; }
    for (const auto& layer : dst)
        for (void* p : layer)
            if (p == nullptr) { err = "root slot: missing K/V buffer"; return false; }
    const auto sz = layout_.sizes((int64_t) ids_.size());
    for (size_t j = 0; j < kv_.size(); ++j)
        for (size_t i = 0; i < 5; ++i) std::memcpy(dst[j][i], kv_[j][i].data(), (size_t) sz[i]);
    return true;
}

// ---- the SSD store
DiskStore::~DiskStore() {
    if (lock_fd_ >= 0) ::close(lock_fd_);   // the lock goes with it
}

DiskStore::Entry* DiskStore::find(uint64_t id) {
    for (Entry& e : entries_) if (e.id == id) return &e;
    return nullptr;
}
const DiskStore::Entry* DiskStore::find(uint64_t id) const {
    for (const Entry& e : entries_) if (e.id == id) return &e;
    return nullptr;
}

size_t DiskStore::files() const {
    size_t n = 0;
    for (const Entry& e : entries_) n += e.foreign ? 0 : 1;
    return n;
}

bool DiskStore::open(const std::string& dir, const Options& opt, const DiskIdentity& id, std::string& err) {
    opt_ = {};
    entries_.clear();
    stats_ = {};
    if (lock_fd_ >= 0) { ::close(lock_fd_); lock_fd_ = -1; }
    if (opt.cap == 0) return true;
    struct stat st{};
    if (::lstat(dir.c_str(), &st) != 0) {
        if (::mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST) {
            err = "ssd cache: cannot make " + dir + ": " + std::strerror(errno);
            return false;
        }
        if (::lstat(dir.c_str(), &st) != 0) { err = "ssd cache: cannot read " + dir; return false; }
    }
    if (S_ISLNK(st.st_mode) || !S_ISDIR(st.st_mode)) { err = "ssd cache: " + dir + " is not a plain directory"; return false; }
    if (st.st_uid != ::getuid()) { err = "ssd cache: " + dir + " belongs to another user"; return false; }
    (void) ::chmod(dir.c_str(), 0700);
    // one engine per directory: two would evict and supersede each other's files
    const std::string lock = dir + "/.lock";
    lock_fd_ = ::open(lock.c_str(), O_CREAT | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (lock_fd_ < 0) { err = "ssd cache: cannot open " + lock + ": " + std::strerror(errno); return false; }
    if (::flock(lock_fd_, LOCK_EX | LOCK_NB) != 0) {
        const int why = errno;
        ::close(lock_fd_);
        lock_fd_ = -1;
        err = why == EWOULDBLOCK ? "ssd cache: another engine uses " + dir
                                 : "ssd cache: cannot lock " + lock + ": " + std::strerror(why);
        return false;
    }
    dir_ = dir;
    id_ = id;
    std::error_code ec;
    std::vector<std::pair<int64_t, Entry>> found;
    const int64_t now = (int64_t) std::time(nullptr);
    for (const auto& de : fs::directory_iterator(dir, ec)) {
        const std::string p = de.path().string();
        struct stat fst{};
        if (::lstat(p.c_str(), &fst) != 0 || !S_ISREG(fst.st_mode)) continue;
        if (ends_with(p, ".skv.tmp")) {
            // a write that never finished: the lock is held, so no live engine is writing it, whatever its age
            // (`stale_tmp_s` only covers a file system where the lock cannot be trusted - none is accepted here)
            (void) now;
            fs::remove(de.path(), ec);
            ++stats_.dead_tmp;
            continue;
        }
        if (!ends_with(p, ".skv")) continue;
        Entry en;
        en.id = ++next_id_;
        en.path = p;
        en.bytes = (uint64_t) fst.st_size;
        en.foreign = true;   // every file counts toward the cap; only this identity's can be restored
        Fd f;
        f.fd = ::open(p.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
        FileHeader h;
        std::string e;
        if (f.fd >= 0 && read_all(f.fd, 0, &h, sizeof h, e) && header_matches(h, id)) {
            const Plan pl = plan(id, h.tokens, h.n_checks);
            std::vector<int64_t> table((size_t) h.n_checks * 2);
            en.ids.resize((size_t) h.tokens);
            if (pl.end == en.bytes && read_all(f.fd, pl.offset[0], en.ids.data(), pl.size[0], e) &&
                (table.empty() || read_all(f.fd, pl.offset[1], table.data(), pl.size[1], e))) {
                bool ok = true;
                for (size_t c = 0; c < h.n_checks; ++c) {
                    ok = ok && table[2 * c] >= 1 && table[2 * c] <= h.tokens;
                    en.checks.push_back(table[2 * c]);
                }
                if (ok) { en.foreign = false; en.root = h.kind == 1; }
            }
        }
        if (en.foreign) { en.ids.clear(); en.checks.clear(); }
        found.emplace_back((int64_t) fst.st_mtime, std::move(en));
    }
    std::stable_sort(found.begin(), found.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    for (auto& f : found) {
        f.second.used = ++clock_;
        stats_.bytes += f.second.bytes;
        entries_.push_back(std::move(f.second));
    }
    stats_.files = files();
    opt_ = opt;
    std::string e;
    make_room(0, e);   // a smaller cap than last time, or another engine's files: they go now
    return true;
}

DiskStore::Match DiskStore::best(const std::vector<int64_t>& prompt) const {
    Match m;
    uint64_t m_bytes = 0;
    for (const Entry& e : entries_) {
        if (e.foreign || prompt.empty()) continue;
        // the longest prefix of the entry's tokens the prompt shares, then the deepest saved point within it
        const size_t lim = std::min(e.ids.size(), prompt.size() - 1);
        size_t same = 0;
        while (same < lim && (int64_t) e.ids[same] == prompt[same]) ++same;
        int64_t at = 0;
        bool live = false;
        if (same == e.ids.size()) { at = (int64_t) e.ids.size(); live = true; }
        for (int64_t c : e.checks)
            if (c <= (int64_t) same && c > at) { at = c; live = false; }
        // a tie goes to the smaller file (a root's): less to read, and a conversation's file is not used up for it
        if (at > m.tokens || (at > 0 && at == m.tokens && e.bytes < m_bytes)) {
            m = {e.id, at, live, e.root};
            m_bytes = e.bytes;
        }
    }
    return m;
}

void DiskStore::pin(uint64_t id) {
    pinned_ = id;
    if (Entry* e = find(id)) e->used = ++clock_;
}
void DiskStore::unpin() { pinned_ = 0; }

bool DiskStore::has(const std::vector<int32_t>& ids) const {
    for (const Entry& e : entries_) if (!e.foreign && e.ids == ids) return true;
    return false;
}

void DiskStore::erase(size_t k) {
    std::error_code ec;
    fs::remove(entries_[k].path, ec);
    // a file that could not be removed still takes its bytes: it leaves the index (never matched or picked again)
    // but keeps counting toward the cap
    if (!ec || !fs::exists(entries_[k].path, ec)) stats_.bytes -= std::min(stats_.bytes, entries_[k].bytes);
    entries_.erase(entries_.begin() + (std::ptrdiff_t) k);
    stats_.files = files();
}

void DiskStore::remove(uint64_t id) {
    for (size_t k = 0; k < entries_.size(); ++k)
        if (entries_[k].id == id) { erase(k); return; }
}

void DiskStore::remove_all() {
    for (size_t k = entries_.size(); k-- > 0;)
        if (!entries_[k].foreign) erase(k);
}

size_t DiskStore::drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationCheckpoint>& checks,
                                  int64_t turn_token) {
    auto held = [&](const std::vector<int32_t>& e, int64_t d) {
        if (d <= 0 || (size_t) d > ids.size() || !std::equal(e.begin(), e.begin() + d, ids.begin())) return false;
        if ((int64_t) ids.size() == d) return true;
        for (const auto& c : checks) if ((int64_t) c.ids.size() == d) return true;
        return false;
    };
    size_t n = 0;
    for (size_t k = entries_.size(); k-- > 0;) {
        const Entry& e = entries_[k];
        if (e.foreign || e.root || e.checks.empty() || e.id == pinned_ || e.ids == ids) continue;
        const int64_t d = *std::max_element(e.checks.begin(), e.checks.end());
        // only a reply's tail after that checkpoint (at most one turn): a whole chat past a shared root is not
        if (!held(e.ids, d) || std::count(e.ids.begin() + d, e.ids.end(), (int32_t) turn_token) > 1) continue;
        erase(k);
        ++n;
    }
    return n;
}

bool DiskStore::make_room(uint64_t need, std::string& err) {
    if (need > opt_.cap) { err = "ssd cache: the conversation is larger than the cache"; return false; }
    // another engine's or unreadable files first, then conversations, then roots; least recently used first
    auto victim = [&]() -> long {
        long v = -1;
        auto rank = [](const Entry& e) { return e.foreign ? 0 : e.root ? 2 : 1; };
        for (size_t k = 0; k < entries_.size(); ++k) {
            const Entry& e = entries_[k];
            if (e.id == pinned_) continue;
            if (v < 0 || rank(e) < rank(entries_[(size_t) v]) ||
                (rank(e) == rank(entries_[(size_t) v]) && e.used < entries_[(size_t) v].used))
                v = (long) k;
        }
        return v;
    };
    // the cap, then the disk: both volumes (the cache's and the boot volume, where swap lives) keep min_free; own
    // files go until both hold, and only when none is left to go is the write refused
    auto disk_short = [&]() -> std::string {
        if (need == 0) return {};
        for (const std::string& vol : {dir_, std::string("/")})
            if (free_bytes(vol) < need + opt_.min_free) return vol;
        return {};
    };
    std::string vol;
    while (stats_.bytes + need > opt_.cap || !(vol = disk_short()).empty()) {
        const long v = victim();
        if (v < 0) {
            err = !vol.empty() ? "ssd cache: less than the conversation plus " + std::to_string(opt_.min_free >> 30) +
                                     " GiB is free on " + vol
                               : "ssd cache: no room (the rest is pinned)";
            return false;
        }
        erase((size_t) v);
        ++stats_.evicted;
    }
    return true;
}

bool DiskStore::write(const std::vector<int32_t>& ids, const StateView& live,
                      const std::vector<ConversationCheckpoint>& checks, bool cvec, bool root,
                      const std::vector<Parts>& src, std::string& err, uint64_t* bytes) {
    if (!enabled()) { err = "ssd cache: off"; return false; }
    if (ids.empty() || src.size() != id_.layers || !view_matches(live, id_.state) || checks.size() > kMaxChecks) {
        err = "ssd cache: the conversation is not this engine's";
        return false;
    }
    for (const auto& c : checks)
        if (c.ids.empty() || c.ids.size() > ids.size() || !std::equal(c.ids.begin(), c.ids.end(), ids.begin()) ||
            !state_matches(c, id_.state)) {
            err = "ssd cache: a checkpoint is not a prefix of the conversation";
            return false;
        }
    for (const auto& layer : src)
        for (void* p : layer)
            if (p == nullptr) { err = "ssd cache: missing K/V buffer"; return false; }
    const Plan pl = plan(id_, (int64_t) ids.size(), (uint32_t) checks.size());
    if (!make_room(pl.end, err)) return false;
    std::random_device rd;
    char name[96];
    std::snprintf(name, sizeof name, "%016llx-%d-%08x%08x",
                  (unsigned long long) hash_bytes(ids.data(), ids.size() * 4, root ? 1 : 0), (int) ::getpid(), rd(), rd());
    const std::string path = dir_ + "/" + name + ".skv", tmp = path + ".tmp";
    {
        Fd f;
        f.fd = ::open(tmp.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_NOFOLLOW | O_CLOEXEC, 0600);
        if (f.fd < 0) { err = "ssd cache: cannot create " + tmp + ": " + std::strerror(errno); return false; }
        uncached(f.fd);
        std::vector<uint64_t> table(pl.size.size());
        size_t s = 0;
        bool ok = true;
        auto put = [&](const std::vector<Seg>& segs) {   // one section from its segments
            Hasher h(pl.size[s]);
            uint64_t at = pl.offset[s];
            for (const Seg& g : segs) {
                if (!ok) break;
                h.update(g.first, g.second);
                ok = write_all(f.fd, at, g.first, g.second, err);
                at += g.second;
            }
            table[s++] = h.finish();
        };
        std::vector<int64_t> ct;
        for (const auto& c : checks) { ct.push_back((int64_t) c.ids.size()); ct.push_back(c.pinned ? 1 : 0); }
        put({{const_cast<int32_t*>(ids.data()), ids.size() * 4}});
        put({{ct.data(), ct.size() * 8}});
        for (const auto& part : live) put(part);
        for (const auto& c : checks)
            for (const auto& part : view_of(c)) put(part);
        const auto sz = id_.layout.sizes((int64_t) ids.size());
        for (const auto& layer : src)
            for (size_t i = 0; i < 5; ++i) put({{layer[i], (size_t) sz[i]}});
        FileHeader h;
        fill_header(h, id_, root, (uint32_t) checks.size(), (int64_t) ids.size(), cvec);
        h.table_hash = hash_bytes(table.data(), table.size() * 8, 0x7461626c65ull);
        h.header_hash = header_hash(h);
        if (ok) ok = write_all(f.fd, sizeof h, table.data(), table.size() * 8, err) && write_all(f.fd, 0, &h, sizeof h, err);
        if (ok && ::fsync(f.fd) != 0) { err = std::string("ssd cache: fsync failed: ") + std::strerror(errno); ok = false; }
        if (!ok) { std::error_code ec; fs::remove(tmp, ec); return false; }
    }
#if defined(__APPLE__)
    const int rn = ::renamex_np(tmp.c_str(), path.c_str(), RENAME_EXCL);   // never over another file
#else
    const int rn = ::link(tmp.c_str(), path.c_str()) == 0 ? ::unlink(tmp.c_str()) : -1;
#endif
    if (rn != 0) {
        err = "ssd cache: rename failed: " + std::string(std::strerror(errno));
        std::error_code ec;
        fs::remove(tmp, ec);
        return false;
    }
    Entry e;
    e.id = ++next_id_;
    e.path = path;
    e.ids = ids;
    for (const auto& c : checks) e.checks.push_back((int64_t) c.ids.size());
    e.bytes = pl.end;
    e.used = ++clock_;
    e.root = root;
    entries_.push_back(std::move(e));
    stats_.bytes += pl.end;
    stats_.files = files();
    stats_.written += pl.end;
    if (bytes) *bytes = pl.end;
    return true;
}

namespace {
// the checkpoints a restore loads (RAM): the deepest one at most `upto` (where the request resumes, or the newest a
// later turn falls back to) and the shallowest (the shared root a branch goes back to), plus a pinned one
bool keep_check(const std::vector<int64_t>& lens, size_t c, int64_t upto, bool pinned) {
    if (lens[c] > upto) return false;
    int64_t lo = INT64_MAX, hi = 0;
    for (int64_t l : lens) if (l <= upto) { lo = std::min(lo, l); hi = std::max(hi, l); }
    return pinned || lens[c] == lo || lens[c] == hi;
}
}  // namespace

uint64_t DiskStore::read_ram(uint64_t id, int64_t upto) const {
    const Entry* e = find(id);
    if (e == nullptr) return 0;
    uint64_t per = 0, n = 0;
    for (uint64_t s : id_.state) per += s;
    for (size_t c = 0; c < e->checks.size(); ++c) n += keep_check(e->checks, c, upto, false) ? 1 : 0;
    return n * per + e->ids.size() * 4;
}

bool DiskStore::read(uint64_t id, const std::vector<int64_t>& prompt, int64_t upto, const StateView* live,
                     std::vector<ConversationCheckpoint>& checks, bool& cvec, bool& root, const std::vector<Parts>& dst,
                     int64_t dst_cells, std::string& err, bool& touched, bool& damaged, uint64_t* bytes) {
    touched = false;
    damaged = false;   // set for what the file itself got wrong (header, hashes, tokens): only then is it deleted
    Entry* e = find(id);
    if (e == nullptr || e->foreign) { err = "ssd cache: no such entry"; return false; }
    Fd f;
    f.fd = ::open(e->path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (f.fd < 0) { err = "ssd cache: cannot open " + e->path; return false; }
    uncached(f.fd);
    FileHeader h;
    if (!read_all(f.fd, 0, &h, sizeof h, err)) { damaged = err.find("shorter") != std::string::npos; return false; }
    if (!header_matches(h, id_) || (size_t) h.tokens != e->ids.size()) {
        err = "ssd cache: the file's header changed";
        damaged = true;
        return false;
    }
    if (h.tokens > dst_cells - 8) { err = "ssd cache: the conversation is longer than the window holds"; return false; }
    const Plan pl = plan(id_, h.tokens, h.n_checks);
    std::vector<uint64_t> table(pl.size.size());
    if (!read_all(f.fd, sizeof h, table.data(), table.size() * 8, err)) { damaged = err.find("shorter") != std::string::npos; return false; }
    if (hash_bytes(table.data(), table.size() * 8, 0x7461626c65ull) != h.table_hash) {
        err = "ssd cache: the file's section table does not match its hash";
        damaged = true;
        return false;
    }
    uint64_t moved = sizeof h + table.size() * 8;
    auto get = [&](size_t s, const std::vector<Seg>& segs) {   // one section into its segments, then its hash
        Hasher hh(pl.size[s]);
        uint64_t at = pl.offset[s];
        for (const Seg& g : segs) {
            if (!read_all(f.fd, at, g.first, g.second, err)) { damaged = err.find("shorter") != std::string::npos; return false; }
            hh.update(g.first, g.second);
            at += g.second;
        }
        moved += pl.size[s];
        if (hh.finish() != table[s]) { err = "ssd cache: the file's bytes do not match their hash"; damaged = true; return false; }
        return true;
    };
    // the tokens first: nothing is written to the window unless the file holds this prompt's first `upto` tokens
    std::vector<int32_t> fids((size_t) h.tokens);
    std::vector<int64_t> ct((size_t) h.n_checks * 2);
    if (!get(0, {{fids.data(), fids.size() * 4}}) || !get(1, {{ct.data(), ct.size() * 8}})) return false;
    if (upto < 1 || upto > h.tokens || (size_t) upto >= prompt.size() ||
        !std::equal(fids.begin(), fids.begin() + upto, prompt.begin(), [](int32_t a, int64_t b) { return (int64_t) a == b; })) {
        err = "ssd cache: the file does not hold this prompt's first tokens";
        damaged = true;   // its index entry promised them
        return false;
    }
    if (live != nullptr && upto != h.tokens) { err = "ssd cache: a live restore must match the whole file"; return false; }
    bool at_check = live != nullptr;
    for (size_t c = 0; c < h.n_checks; ++c) {
        if (ct[2 * c] < 1 || ct[2 * c] > h.tokens) { err = "ssd cache: a checkpoint is out of range"; damaged = true; return false; }
        at_check = at_check || ct[2 * c] == upto;
    }
    if (!at_check) { err = "ssd cache: no saved state at the matched point"; damaged = true; return false; }
    // the checkpoints a restore keeps (keep_check: the matched/deepest and the root; deeper ones would go at once)
    std::vector<ConversationCheckpoint> fchecks;
    std::vector<int64_t> lens;
    for (size_t c = 0; c < h.n_checks; ++c) lens.push_back(ct[2 * c]);
    try {
        for (size_t c = 0; c < h.n_checks; ++c) {
            if (!keep_check(lens, c, upto, ct[2 * c + 1] != 0)) continue;
            ConversationCheckpoint k;
            std::vector<uint8_t>* p[5] = {&k.gdn, &k.ple, &k.tails, &k.dead, &k.block_pos};
            for (size_t i = 0; i < 5; ++i) {
                p[i]->resize((size_t) id_.state[i]);
                if (!get(pl.checks0 + 5 * c + i, {{p[i]->data(), p[i]->size()}})) return false;
            }
            k.ids.assign(fids.begin(), fids.begin() + ct[2 * c]);
            k.pinned = ct[2 * c + 1] != 0;
            fchecks.push_back(std::move(k));
        }
    } catch (const std::bad_alloc&) {
        err = "ssd cache: not enough memory for the saved running states";
        return false;
    }
    if (live != nullptr && !view_matches(*live, id_.state)) { err = "ssd cache: the live state's target differs"; return false; }
    for (const auto& layer : dst)
        for (void* p : layer)
            if (p == nullptr) { err = "ssd cache: missing K/V buffer"; return false; }
    if (dst.size() != id_.layers) { err = "ssd cache: K/V layer count differs"; return false; }
    touched = true;
    if (live != nullptr)
        for (size_t i = 0; i < 5; ++i)
            if (!get(pl.live0 + i, (*live)[i])) return false;
    const auto sz = id_.layout.sizes(h.tokens);
    for (size_t j = 0; j < dst.size(); ++j)
        for (size_t i = 0; i < 5; ++i)
            if (!get(pl.kv0 + 5 * j + i, {{dst[j][i], (size_t) sz[i]}})) return false;
    (void) ::futimes(f.fd, nullptr);   // used now: the LRU order survives a restart
    checks = std::move(fchecks);
    cvec = h.cvec != 0;
    root = h.kind == 1;
    e->used = ++clock_;
    stats_.read += moved;
    if (bytes) *bytes = moved;
    return true;
}

}  // namespace strata::core::kvstore
