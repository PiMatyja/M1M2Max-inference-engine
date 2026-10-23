// conversation_ssd_test: the root slot and the SSD store round-trip window K/V and running state byte for byte
// (synthetic buffers; no device), and refuse what they must.  Exit 0 = every check passed.
#include "strata/core/conversation_ssd.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sys/time.h>
#include <unistd.h>

using namespace strata::core;
using namespace strata::core::kvstore;
namespace fs = std::filesystem;

static int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

static void fill_random(std::vector<uint8_t>& v, uint32_t seed) {
    std::mt19937 r(seed);
    for (auto& b : v) b = (uint8_t) r();
}

struct Window {   // a fake decode window: `layers` layers of five K/V parts, and a running state split per layer
    Layout layout;
    std::vector<std::array<std::vector<uint8_t>, 5>> buf;
    std::array<std::vector<std::vector<uint8_t>>, 5> state;   // part i = segs[i] concatenated
    Window(size_t layers, int64_t cells, const Layout& l, const StateSizes& s, uint32_t seed) : layout(l), buf(layers) {
        const auto sz = l.sizes(cells);
        uint32_t k = seed;
        for (auto& layer : buf)
            for (size_t i = 0; i < 5; ++i) { layer[i].resize(sz[i]); fill_random(layer[i], ++k); }
        for (size_t i = 0; i < 5; ++i) {   // three uneven segments a part, as the window's per-layer buffers
            const size_t a = s[i] / 3, b = s[i] / 2;
            for (size_t n : {a, b - a, (size_t) s[i] - b}) {
                state[i].emplace_back(n);
                fill_random(state[i].back(), ++k);
            }
        }
    }
    std::vector<Parts> parts() {
        std::vector<Parts> p;
        for (auto& layer : buf) p.push_back({layer[0].data(), layer[1].data(), layer[2].data(), layer[3].data(), layer[4].data()});
        return p;
    }
    StateView view() {
        StateView v;
        for (size_t i = 0; i < 5; ++i) for (auto& g : state[i]) v[i].push_back({g.data(), g.size()});
        return v;
    }
    bool same_prefix(const Window& o, int64_t tokens) const {
        const auto sz = layout.sizes(tokens);
        for (size_t j = 0; j < buf.size(); ++j)
            for (size_t i = 0; i < 5; ++i)
                if (std::memcmp(buf[j][i].data(), o.buf[j][i].data(), sz[i]) != 0) return false;
        return true;
    }
    bool same_state(const Window& o) const { return state == o.state; }
    bool same_all(const Window& o) const { return buf == o.buf && state == o.state; }
    // the state as one checkpoint (what checkpoint_save would give)
    ConversationCheckpoint checkpoint(const std::vector<int32_t>& ids) const {
        ConversationCheckpoint c;
        c.ids = ids;
        std::vector<uint8_t>* p[5] = {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos};
        for (size_t i = 0; i < 5; ++i) for (const auto& g : state[i]) p[i]->insert(p[i]->end(), g.begin(), g.end());
        return c;
    }
};

static ConversationCheckpoint state_of(const std::vector<int32_t>& ids, const StateSizes& s, uint32_t seed) {
    ConversationCheckpoint c;
    c.ids = ids;
    std::vector<uint8_t>* p[5] = {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos};
    for (size_t i = 0; i < 5; ++i) { p[i]->resize(s[i]); fill_random(*p[i], seed * 7 + (uint32_t) i); }
    return c;
}

static bool same_checkpoint(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    return a.ids == b.ids && a.gdn == b.gdn && a.ple == b.ple && a.tails == b.tails && a.dead == b.dead &&
           a.block_pos == b.block_pos && a.pinned == b.pinned;
}

static std::vector<int64_t> prompt_of(const std::vector<int32_t>& ids, size_t n, int64_t next) {
    std::vector<int64_t> p(ids.begin(), ids.begin() + (std::ptrdiff_t) n);
    p.push_back(next);
    return p;
}

static void age(const fs::path& p, int seconds) {
    struct timeval tv[2];
    gettimeofday(&tv[0], nullptr);
    tv[0].tv_sec -= seconds;
    tv[1] = tv[0];
    utimes(p.c_str(), tv);
}

int main() {
    Layout L;
    L.cell = {64, 64, 8, 8};
    L.pooled_row = 32;
    const StateSizes S = {4096, 512, 96, 32, 40};
    const size_t layers = 3;
    const int64_t cells = 8192;
    std::vector<int32_t> ids(3001);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = (int32_t) ((i * 2654435761u) % 151000);

    // ---- the hash: streaming over any split equals one shot; one bit changes it
    {
        std::vector<uint8_t> v(1000003);
        fill_random(v, 1);
        const uint64_t h = hash_bytes(v.data(), v.size(), 1);
        std::mt19937 r(2);
        for (int t = 0; t < 20; ++t) {
            Hasher hh(1);
            size_t at = 0;
            while (at < v.size()) {
                const size_t k = std::min(v.size() - at, (size_t) (r() % 100));
                hh.update(v.data() + at, k);
                at += k;
            }
            CHECK(hh.finish() == h);
        }
        v[500001] ^= 1;
        CHECK(h != hash_bytes(v.data(), v.size(), 1));
        CHECK(hash_bytes(v.data(), v.size(), 1) != hash_bytes(v.data(), v.size(), 2));
    }

    // ---- the root slot: from a segmented (window) state and from a checkpoint; restore writes exactly the prefix
    {
        RootSlot slot;
        std::string err;
        CHECK(slot.reserve(2048, layers, L, S, err));
        CHECK(slot.reserved_bytes() > 0);
        Window w(layers, cells, L, S, 1), dst(layers, cells, L, S, 2);
        const Window before(dst);
        std::vector<int32_t> root(ids.begin(), ids.begin() + 1999);
        CHECK(slot.fill(root, w.view(), w.parts(), err));
        CHECK(slot.ids() == root && same_checkpoint(slot.state(), w.checkpoint(root)));
        CHECK(slot.verify(err) && slot.digest() != 0);
        CHECK(slot.restore_kv(dst.parts(), err));
        CHECK(dst.same_prefix(w, 1999));
        const auto sz = L.sizes(1999);
        for (size_t j = 0; j < layers; ++j)
            for (size_t i = 0; i < 5; ++i)
                CHECK(std::memcmp(dst.buf[j][i].data() + sz[i], before.buf[j][i].data() + sz[i],
                                  dst.buf[j][i].size() - sz[i]) == 0);
        std::vector<int32_t> too_long(ids.begin(), ids.begin() + 2049);
        CHECK(!slot.fill(too_long, w.view(), w.parts(), err));
        CHECK(slot.ids() == root);   // a refused root leaves the slot as it was
        std::vector<int32_t> other(ids.begin(), ids.begin() + 700);
        const ConversationCheckpoint oc = state_of(other, S, 5);
        CHECK(slot.fill(other, view_of(oc), w.parts(), err));
        CHECK(slot.ids() == other && same_checkpoint(slot.state(), oc) && slot.verify(err));
        RootSlot empty;
        CHECK(!empty.verify(err));
        StateView short_view = w.view();
        short_view[0].pop_back();
        CHECK(!slot.fill(other, short_view, w.parts(), err));   // a state of the wrong size is refused
    }

    const fs::path dir = fs::temp_directory_path() / ("strata-ssd-test-" + std::to_string(getpid()));
    fs::remove_all(dir);
    DiskIdentity id;
    id.engine = 0x1234;
    id.geometry[0] = 2560;
    id.layers = (uint32_t) layers;
    id.layout = L;
    id.state = S;
    DiskStore::Options big;
    big.cap = 64ull << 20;
    big.min_free = 0;

    // ---- write a conversation from the window, read it back into another: every byte; refusals before any write
    {
        DiskStore store;
        std::string err;
        CHECK(store.open(dir.string(), big, id, err));
        {
            DiskStore second;   // one engine per directory
            CHECK(!second.open(dir.string(), big, id, err) && err.find("another engine") != std::string::npos);
        }
        Window w(layers, cells, L, S, 10);
        std::vector<ConversationCheckpoint> checks = {state_of({ids.begin(), ids.begin() + 1000}, S, 12),
                                                      state_of({ids.begin(), ids.begin() + 2500}, S, 13)};
        checks[0].pinned = true;
        uint64_t bytes = 0;
        CHECK(store.write(ids, w.view(), checks, true, false, w.parts(), err, &bytes));
        CHECK(bytes > 0 && store.files() == 1);
        const auto prompt = prompt_of(ids, ids.size(), 5);
        DiskStore::Match m = store.best(prompt);
        CHECK(m.id != 0 && m.tokens == 3001 && m.live);
        auto branch = prompt_of(ids, 2700, -7);
        CHECK(store.best(branch).tokens == 2500 && !store.best(branch).live);
        CHECK(store.best(std::vector<int64_t>(ids.begin(), ids.end())).tokens == 2500);   // never the whole prompt
        // live: K/V and the live state straight into the (segmented) window
        Window dst(layers, cells, L, S, 14);
        StateView dv = dst.view();
        std::vector<ConversationCheckpoint> rchecks;
        bool cvec = false, root = true, touched = false, damaged = false;
        CHECK(store.read(m.id, prompt, 3001, &dv, rchecks, cvec, root, dst.parts(), cells, err, touched, damaged));
        CHECK(touched && cvec && !root);
        CHECK(dst.same_prefix(w, 3001) && dst.same_state(w));
        CHECK(rchecks.size() == 2 && same_checkpoint(rchecks[0], checks[0]) && same_checkpoint(rchecks[1], checks[1]));
        // a checkpoint match: the live state is not read, deeper checkpoints are not loaded
        Window dst2(layers, cells, L, S, 15);
        const Window keep2(dst2);
        CHECK(store.read(m.id, branch, 2500, nullptr, rchecks, cvec, root, dst2.parts(), cells, err, touched, damaged));
        CHECK(dst2.same_state(keep2) && dst2.same_prefix(w, 3001));
        CHECK(rchecks.size() == 2);   // 1000 and 2500, both <= 2500
        CHECK(store.read(m.id, prompt_of(ids, 1500, -1), 1000, nullptr, rchecks, cvec, root, dst2.parts(), cells, err, touched, damaged));
        CHECK(rchecks.size() == 1 && rchecks[0].ids.size() == 1000);
        // a prompt that is not the file's: refused before anything is written (the stale-index case)
        Window dst3(layers, cells, L, S, 16);
        const Window keep3(dst3);
        StateView dv3 = dst3.view();
        auto wrong = prompt;
        wrong[17] ^= 1;
        CHECK(!store.read(m.id, wrong, 3001, &dv3, rchecks, cvec, root, dst3.parts(), cells, err, touched, damaged));
        CHECK(!touched && damaged && dst3.same_all(keep3) && err.find("first tokens") != std::string::npos);
        // a window too small for the file: refused, not damaged, nothing written
        CHECK(!store.read(m.id, prompt, 3001, &dv3, rchecks, cvec, root, dst3.parts(), 3005, err, touched, damaged));
        CHECK(!touched && !damaged && dst3.same_all(keep3));
        CHECK(!store.read(m.id, branch, 2400, nullptr, rchecks, cvec, root, dst3.parts(), cells, err, touched, damaged));   // no state at 2400
        CHECK(!touched && dst3.same_all(keep3));
        // a root file, kept for restarts; a match on it says so
        std::vector<int32_t> rootids(ids.begin(), ids.begin() + 900);
        CHECK(store.write(rootids, view_of(state_of(rootids, S, 15)), {}, true, true, w.parts(), err));
        CHECK(store.has(rootids));
        const auto mr = store.best(prompt_of(ids, 950, 3));
        CHECK(mr.tokens == 900 && mr.root && mr.live);
        CHECK(!store.best(prompt).root);
    }
    {
        // after a restart: the files are found again (LRU from their times); one flipped byte is refused
        DiskStore store;
        std::string err;
        CHECK(store.open(dir.string(), big, id, err));
        CHECK(store.files() == 2);
        const auto prompt = prompt_of(ids, ids.size(), 5);
        const DiskStore::Match m = store.best(prompt);
        CHECK(m.tokens == 3001 && m.live);
        Window w(layers, cells, L, S, 10), dst(layers, cells, L, S, 20);
        StateView dv = dst.view();
        std::vector<ConversationCheckpoint> rchecks;
        bool cvec = false, root = false, touched = false, damaged = false;
        CHECK(store.read(m.id, prompt, 3001, &dv, rchecks, cvec, root, dst.parts(), cells, err, touched, damaged));
        CHECK(dst.same_prefix(w, 3001));
        std::string path;
        for (const auto& de : fs::directory_iterator(dir))
            if (de.path().extension() == ".skv" && fs::file_size(de.path()) > 400000) path = de.path().string();
        CHECK(!path.empty());
        {
            std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
            f.seekg(-1000, std::ios::end);
            char c = 0;
            f.read(&c, 1);
            c ^= 0x10;
            f.seekp(-1000, std::ios::end);
            f.write(&c, 1);
        }
        err.clear();
        CHECK(!store.read(m.id, prompt, 3001, &dv, rchecks, cvec, root, dst.parts(), cells, err, touched, damaged));
        CHECK(touched && damaged && err.find("hash") != std::string::npos);
    }
    {
        // another engine's identity: its files are not restored, but they count and go first
        DiskIdentity other = id;
        other.engine = 0x9999;
        DiskStore foreign;
        std::string err;
        CHECK(foreign.open(dir.string(), big, other, err));
        CHECK(foreign.files() == 0 && foreign.stats().bytes > 0);
        CHECK(foreign.best(prompt_of(ids, ids.size(), 5)).tokens == 0);
    }
    {
        // stale temporary files go, fresh ones (another write in progress) stay; zero-header junk counts and goes first
        fs::remove_all(dir);
        fs::create_directories(dir);
        { std::ofstream(dir / "old.skv.tmp") << "x"; std::ofstream(dir / "new.skv.tmp") << "x"; }
        age(dir / "old.skv.tmp", 3600);
        { std::ofstream junk(dir / "junk.skv", std::ios::binary); std::vector<char> z(300000, 0); junk.write(z.data(), (std::streamsize) z.size()); }
        std::string err;
        Window w(layers, cells, L, S, 40);
        std::vector<int32_t> a(ids.begin(), ids.begin() + 1000);
        uint64_t one = 0;
        {
            DiskStore store;
            CHECK(store.open(dir.string(), big, id, err));
            // under the directory lock no engine is writing either: both go, whatever their age
            CHECK(!fs::exists(dir / "old.skv.tmp") && !fs::exists(dir / "new.skv.tmp") && store.stats().dead_tmp == 2);
            CHECK(store.files() == 0 && store.stats().bytes == 300000);
            CHECK(store.write(a, w.view(), {}, true, false, w.parts(), err, &one));
        }
        {   // a cap just over one file: the junk goes before the conversation
            DiskStore::Options tight = big;
            tight.cap = one + one / 2;
            DiskStore t;
            CHECK(t.open(dir.string(), tight, id, err));
            CHECK(!fs::exists(dir / "junk.skv") && t.files() == 1);
        }
        {   // larger than the whole cache: refused before anything is evicted
            DiskStore::Options tiny = big;
            tiny.cap = one * 3;
            DiskStore t;
            CHECK(t.open(dir.string(), tiny, id, err));
            std::vector<int32_t> huge(3000 * 4);
            for (size_t i = 0; i < huge.size(); ++i) huge[i] = (int32_t) i;
            Window wide(layers, 16384, L, S, 41);
            CHECK(!t.write(huge, wide.view(), {}, true, false, wide.parts(), err) && err.find("larger") != std::string::npos);
            CHECK(t.files() == 1 && t.stats().evicted == 0);
        }
        {   // the disk's free-space floor: this engine's own files go first, then the write is refused
            DiskStore::Options floor = big;
            floor.min_free = 1ull << 60;
            DiskStore t;
            CHECK(t.open(dir.string(), floor, id, err));
            std::vector<int32_t> b2(a);
            b2[5] ^= 1;
            CHECK(!t.write(b2, w.view(), {}, true, false, w.parts(), err) && err.find("free") != std::string::npos);
            CHECK(t.files() == 0 && t.stats().evicted == 1);
        }
    }
    {
        // LRU under the cap: a used file outlives an older one; a pinned file is never the victim; roots go last
        fs::remove_all(dir);
        std::string err;
        Window w(layers, cells, L, S, 50);
        std::vector<int32_t> a(ids.begin(), ids.begin() + 1000), b(ids.rbegin(), ids.rbegin() + 1000), c(a), r(a.begin(), a.begin() + 300);
        b[0] ^= 1;
        c[3] ^= 1;
        uint64_t one = 0, rb = 0;
        {
            DiskStore probe;
            CHECK(probe.open(dir.string(), big, id, err));
            CHECK(probe.write(a, w.view(), {}, true, false, w.parts(), err, &one));
            CHECK(probe.write(r, w.view(), {}, true, true, w.parts(), err, &rb));   // a root file, older than b
        }
        DiskStore small;
        DiskStore::Options o2 = big;
        o2.cap = one * 2 + rb + one / 4;
        CHECK(small.open(dir.string(), o2, id, err));
        CHECK(small.write(b, w.view(), {}, true, false, w.parts(), err));
        CHECK(small.files() == 3);
        const auto pa = prompt_of(a, a.size(), 1), pb = prompt_of(b, b.size(), 1);
        // a is pinned (a restore of it is under way): the next write evicts b, not a, and not the older root
        const auto ma = small.best(pa);
        small.pin(ma.id);
        CHECK(small.write(c, w.view(), {}, true, false, w.parts(), err));
        CHECK(small.best(pb).tokens == 0 && small.best(pa).tokens == 1000 && small.has(r));
        CHECK(small.stats().evicted == 1);
        // superseded files go only by the rule (never the pinned one)
        CHECK(small.drop_superseded(a, {}, 7) == 0);
        small.unpin();
    }
    {
        // superseded: the same chat a turn later holds the file's deepest checkpoint and the file's tail past it is
        // one turn; a file whose tail past a shared checkpoint holds several turns (a whole other chat) stays
        fs::remove_all(dir);
        std::string err;
        Window w(layers, cells, L, S, 60);
        const int32_t turn = 7;
        std::vector<int32_t> root(ids.begin(), ids.begin() + 500);
        std::vector<int32_t> chat1 = root, chat2 = root;
        for (int t = 0; t < 3; ++t) { chat1.push_back(turn); for (int k = 0; k < 100; ++k) chat1.push_back(1000 + t * 100 + k); }
        chat2.push_back(turn);
        for (int k = 0; k < 100; ++k) chat2.push_back(5000 + k);
        const std::vector<ConversationCheckpoint> at_root = {state_of(root, S, 61)};
        DiskStore store;
        CHECK(store.open(dir.string(), big, id, err));
        CHECK(store.write(chat1, w.view(), at_root, true, false, w.parts(), err));   // three turns past the root
        std::vector<int32_t> one_turn(root);
        one_turn.push_back(turn);
        for (int k = 0; k < 50; ++k) one_turn.push_back(9000 + k);
        CHECK(store.write(one_turn, w.view(), at_root, true, false, w.parts(), err));   // one reply past the root
        // another chat from the same root parks: chat1 stays (a whole chat), the one-reply file goes
        CHECK(store.drop_superseded(chat2, at_root, turn) == 1);
        CHECK(store.has(chat1) && !store.has(one_turn));
    }
    {
        // a restore loads the matched (deepest) checkpoint and the shallowest one only; read_ram says the same
        fs::remove_all(dir);
        std::string err;
        Window w(layers, cells, L, S, 70);
        std::vector<ConversationCheckpoint> ck = {state_of({ids.begin(), ids.begin() + 500}, S, 71),
                                                  state_of({ids.begin(), ids.begin() + 1000}, S, 72),
                                                  state_of({ids.begin(), ids.begin() + 2500}, S, 73)};
        DiskStore store;
        CHECK(store.open(dir.string(), big, id, err));
        CHECK(store.write(ids, w.view(), ck, true, false, w.parts(), err));
        const auto prompt = prompt_of(ids, ids.size(), 5);
        const auto m = store.best(prompt);
        uint64_t per = 0;
        for (uint64_t x : S) per += x;
        CHECK(store.read_ram(m.id, 3001) == 2 * per + ids.size() * 4);
        Window dst(layers, cells, L, S, 74);
        StateView dv = dst.view();
        std::vector<ConversationCheckpoint> got;
        bool cvec = false, root = false, touched = false, damaged = false;
        CHECK(store.read(m.id, prompt, 3001, &dv, got, cvec, root, dst.parts(), cells, err, touched, damaged));
        CHECK(got.size() == 2 && same_checkpoint(got[0], ck[0]) && same_checkpoint(got[1], ck[2]));
        CHECK(store.read(m.id, prompt_of(ids, 1200, -1), 1000, nullptr, got, cvec, root, dst.parts(), cells, err, touched, damaged));
        CHECK(got.size() == 2 && same_checkpoint(got[0], ck[0]) && same_checkpoint(got[1], ck[1]));
    }
    fs::remove_all(dir);
    std::printf("conversation_ssd_test: %s (%d failures)\n", failures ? "FAILED" : "ok", failures);
    return failures ? 1 : 0;
}
