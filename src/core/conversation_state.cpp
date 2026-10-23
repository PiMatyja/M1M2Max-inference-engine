#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_ssd.hpp"
#include "conversation_checked.hpp"
#include "conversation_copy.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace strata::core {
namespace {
using conversation_detail::add;
using conversation_detail::product;

std::array<int64_t, 18> geometry_key(const ModelGeometry& g) {
    return {g.n_embd, g.n_layers, g.qsa_interval, g.ssm_state_size, g.ssm_k_heads,
            g.ssm_v_heads, g.ssm_d_conv, g.ssm_conv_channels, g.ssm_value_dim,
            g.n_head, g.n_head_kv, g.head_dim, g.idx_q_heads, g.idx_key_dim,
            g.hc, g.hc_lr, g.n_expert, g.n_ff};
}

bool fail(std::string& error, const char* message) {
    error = std::string("conversation snapshot: ") + message;
    return false;
}

bool sync(std::string& error) {
    const auto status = cudaDeviceSynchronize();
    if (status == cudaSuccess) return true;
    error = std::string("conversation snapshot synchronize: ") + cudaGetErrorString(status);
    return false;
}

bool copy(void* dst, const void* src, size_t bytes, std::string& error) {
    if (!bytes) return true;
    if (!dst || !src) return fail(error, "missing running-state buffer");
    const auto status = conversation_detail::copy_nonblocking(dst, src, bytes);
    if (status == cudaSuccess) return true;
    error = std::string("conversation snapshot running-state copy: ") + cudaGetErrorString(status);
    return false;
}

// The session's carve (#216): `qsa_states` keeps global ordinals and holds the owned ones
// [qsa_ord0, qsa_ord0 + qsa_alloc); `gdn_state` holds `gdn_alloc` rows. A whole-model session owns every layer.
size_t owned_qsa(const SessionState& ss) { return (size_t) std::max<int64_t>(ss.qsa_alloc, 0); }
const QsaState& owned(const SessionState& ss, size_t j) { return ss.qsa_states[(size_t) ss.qsa_ord0 + j]; }

bool image_keys(const std::vector<ConversationImageKey>& images, size_t tokens) {
    int64_t previous = -1;
    for (const auto& image : images) {
        if (image.start <= previous || image.start < 0 || (uint64_t) image.start >= tokens) return false;
        previous = image.start;
    }
    return true;
}

bool checkpoint_targets(const SessionState& ss, const ModelGeometry& g, size_t tokens,
                        ConversationStateSizes& z, std::string& error) {
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (ss.max_cells < 0 || tokens > (uint64_t) ss.max_cells || (z.gdn && !ss.gdn_state) ||
        (owned_qsa(ss) && !ss.qsa_states)) return fail(error, "invalid session running-state targets");
    for (size_t j = 0; j < owned_qsa(ss); ++j) {
        const auto& st = owned(ss, j);
        const auto block = strata::kernels::qsa_real_shapes().idx_block;
        size_t pooled_bytes = 0;
        if (!st.idx_tail || !st.idx_dead || !st.idx_block_pos || !st.idx_pooled ||
            st.max_cells < 0 || tokens > (uint64_t) st.max_cells ||
            (tokens && tokens / (uint64_t) block >= (uint64_t) std::max<int64_t>(0, st.idx_pooled_rows)) ||
            !product(pooled_bytes, {tokens / (uint64_t) block + 1, (uint64_t) g.idx_key_dim, sizeof(float)}))
            return fail(error, "invalid indexer running-state target");
    }
    return true;
}

bool view_validate(const ConversationView& view, const SessionState& ss,
                   const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, view.ids.size(), z, error)) return false;
    if (view.ids.empty() || !image_keys(view.images, view.ids.size()) ||
        std::any_of(view.ids.begin(), view.ids.end(), [](int32_t id) { return id < 0; }))
        return fail(error, "invalid live token/image metadata");
    for (const auto& c : view.checkpoints) {
        if (!c.stage_parts.empty() || c.ids.empty() || c.ids.size() > view.ids.size() ||
            !std::equal(c.ids.begin(), c.ids.end(), view.ids.begin()))
            return fail(error, "checkpoint is not a live token prefix");
        if (!conversation_checkpoint_validate(c, ss, g, error)) return false;
        size_t image = 0;
        for (const auto& key : view.images) {
            if ((uint64_t) key.start >= c.ids.size()) break;
            if (image >= c.imgs.size() || !(c.imgs[image++] == key))
                return fail(error, "checkpoint image identity differs");
        }
        if (image != c.imgs.size()) return fail(error, "checkpoint image prefix differs");
    }
    return true;
}

bool metadata_bytes(const ConversationCheckpoint& c, size_t& total) {
    size_t ids = 0, images = 0;
    if (!product(ids, {c.ids.size(), sizeof(int32_t)}) ||
        !product(images, {c.imgs.size(), sizeof(ConversationImageKey)})) return false;
    for (size_t n : {ids, images, c.gdn.size(), c.ple.size(), c.tails.size(), c.dead.size(), c.block_pos.size()})
        if (!add(total, n)) return false;
    return true;
}
} // namespace

bool conversation_state_sizes(const ModelGeometry& g, ConversationStateSizes& z, std::string& error) {
    z = {};
    const auto key = geometry_key(g);
    for (size_t i = 0; i < key.size(); ++i)
        if (key[i] < 0 || (i != 1 && key[i] == 0)) return fail(error, "invalid model geometry");
    size_t recurrence = 0, convolution = 0;
    if (!product(recurrence, {(uint64_t) g.ssm_state_size, (uint64_t) g.ssm_v_heads, (uint64_t) g.ssm_state_size}) ||
        !product(convolution, {(uint64_t) g.ssm_conv_channels, (uint64_t) (g.ssm_d_conv - 1)}) ||
        !add(recurrence, convolution) ||
        !product(z.gdn, {(uint64_t) g.n_gdn_layers(), recurrence, sizeof(float)}) ||
        !product(z.ple, {strata::kernels::NG_HIST, strata::kernels::NG_HC_DIM, sizeof(float)}) ||
        !product(z.tail, {(uint64_t) (strata::kernels::qsa_real_shapes().idx_block - 1),
                          (uint64_t) g.idx_key_dim, sizeof(float)}) ||
        !product(z.dead, {(uint64_t) g.idx_key_dim, sizeof(float)}))
        return fail(error, "running-state byte count overflow");
    z.block_pos = sizeof(int32_t);
    size_t total = 0;
    if (!product(total, {(uint64_t) g.n_qsa_layers(), z.tail}) ||
        !product(total, {(uint64_t) g.n_qsa_layers(), z.dead}) ||
        !product(total, {(uint64_t) g.n_qsa_layers(), z.block_pos}))
        return fail(error, "indexer byte count overflow");
    return true;
}

bool conversation_session_sizes(const ModelGeometry& g, const SessionState& ss, ConversationStateSizes& z,
                                std::string& error) {
    if (!conversation_state_sizes(g, z, error)) return false;
    const int64_t n_gdn = g.n_gdn_layers(), n_qsa = g.n_qsa_layers();
    if (ss.gdn_alloc < 0 || ss.gdn_alloc > n_gdn || ss.qsa_ord0 < 0 || ss.qsa_alloc < 0 ||
        ss.qsa_ord0 > n_qsa || ss.qsa_alloc > n_qsa - ss.qsa_ord0)
        return fail(error, "invalid session layer carve");
    // z.gdn is n_gdn whole rows; the session holds gdn_alloc of them (all of them for the default full range)
    z.gdn = n_gdn > 0 ? z.gdn / (size_t) n_gdn * (size_t) ss.gdn_alloc : 0;
    return true;
}

bool conversation_checkpoint_validate(const ConversationCheckpoint& c, const SessionState& ss,
                                      const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, c.ids.size(), z, error)) return false;
    const size_t layers = owned_qsa(ss);
    if (c.gdn.size() != z.gdn || c.ple.size() != (ss.ple_hist ? z.ple : 0) ||
        c.tails.size() != layers * z.tail || c.dead.size() != layers * z.dead ||
        c.block_pos.size() != layers * z.block_pos || !image_keys(c.imgs, c.ids.size()))
        return fail(error, "invalid checkpoint running-state payload");
    return true;
}

#if defined(STRATA_GPU_METAL)
}  // namespace strata::core
namespace strata::metal_engine {
// the running state lives in the Metal decode window (metal/src/engine/window.cpp), not in the session's buffers
bool window_checkpoint_save(strata::core::ConversationCheckpoint& c, std::string& error);
bool window_checkpoint_restore(const strata::core::ConversationCheckpoint& c, std::string& error);
bool window_kv_parts(std::vector<std::array<void*, 5>>& parts, int64_t& cells, std::string& error);
bool window_state_segments(std::array<std::vector<std::pair<void*, size_t>>, 5>& v, bool restore, std::string& error);
bool window_state_sizes(size_t& gdn, size_t& ple, size_t& tails, size_t& dead, size_t& draft, std::string& error);
}  // namespace strata::metal_engine
namespace strata::core {
#endif

bool conversation_checkpoint_save(ConversationCheckpoint& c, const SessionState& ss,
                                  const ModelGeometry& g, std::string& error) {
#if defined(STRATA_GPU_METAL)
    (void) ss; (void) g;
    return strata::metal_engine::window_checkpoint_save(c, error);
#endif
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, c.ids.size(), z, error)) return false;
    const size_t layers = owned_qsa(ss);
    c.gdn.resize(z.gdn); c.ple.resize(ss.ple_hist ? z.ple : 0);
    c.tails.resize(layers * z.tail); c.dead.resize(layers * z.dead); c.block_pos.resize(layers * z.block_pos);
    if (!copy(c.gdn.data(), ss.gdn_state, c.gdn.size(), error) ||
        !copy(c.ple.data(), ss.ple_hist, c.ple.size(), error)) return false;
    for (size_t j = 0; j < layers; ++j) {
        const auto& st = owned(ss, j);
        if (!copy(c.tails.data() + j * z.tail, st.idx_tail, z.tail, error) ||
            !copy(c.dead.data() + j * z.dead, st.idx_dead, z.dead, error) ||
            !copy(c.block_pos.data() + j * z.block_pos, st.idx_block_pos, z.block_pos, error)) return false;
    }
    return true;
}

bool conversation_checkpoint_restore(const ConversationCheckpoint& c, SessionState& ss,
                                     const ModelGeometry& g, std::string& error) {
#if defined(STRATA_GPU_METAL)
    (void) g;
    if (!strata::metal_engine::window_checkpoint_restore(c, error)) return false;
    ss.ple_prev[0] = c.ids.size() >= 2 ? c.ids[c.ids.size() - 2] : -1;
    ss.ple_prev[1] = c.ids.size() >= 1 ? c.ids[c.ids.size() - 1] : -1;
    return true;
#endif
    if (!conversation_checkpoint_validate(c, ss, g, error)) return false;
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (!sync(error)) return false;   // the copies below no longer ride the legacy stream's implicit ordering
    if (!copy(ss.gdn_state, c.gdn.data(), c.gdn.size(), error) ||
        !copy(ss.ple_hist, c.ple.data(), c.ple.size(), error)) return false;
    for (size_t j = 0; j < owned_qsa(ss); ++j) {
        const auto& st = owned(ss, j);
        if (!copy(st.idx_tail, c.tails.data() + j * z.tail, z.tail, error) ||
            !copy(st.idx_dead, c.dead.data() + j * z.dead, z.dead, error) ||
            !copy(st.idx_block_pos, c.block_pos.data() + j * z.block_pos, z.block_pos, error)) return false;
        if (!c.ids.empty()) {
            const size_t row = c.ids.size() / strata::kernels::qsa_real_shapes().idx_block;
            if (!copy(st.idx_pooled + row * g.idx_key_dim, c.dead.data() + j * z.dead, z.dead, error)) return false;
        }
    }
    const size_t tokens = c.ids.size();
    ss.ple_prev[0] = tokens >= 2 ? c.ids[tokens - 2] : -1;
    ss.ple_prev[1] = tokens >= 1 ? c.ids[tokens - 1] : -1;
    return sync(error);
}

#if defined(STRATA_GPU_METAL)
// Parked conversations on Metal: the decode window keeps its own positional K/V (SessionState's QSA pools are not
// used there), so an image holds the window's bytes - every QSA layer, then the draft layer when it is loaded - and
// the window's running state; restoring them puts back exactly the bytes the conversation left.
namespace {
constexpr int kMetalKvFormat = 0x4D;   // the window's INT8 K/V with half scales; restored only into a window

struct MetalKvSizes {
    std::array<size_t, 5> part{};   // k, v, k_scale, v_scale, pooled
    int64_t cells = 0, rows = 0;
};

bool metal_kv_sizes(const ModelGeometry& g, int64_t upto, int64_t window_cells, MetalKvSizes& s, std::string& error) {
    const auto sh = strata::kernels::qsa_real_shapes();
    if (g.n_head_kv != 2 || g.head_dim != 256 || g.idx_key_dim != 128 || sh.page_size != 4 || sh.idx_block != 4)
        return fail(error, "the Metal window's K/V layout does not match this model");
    if (upto < 1 || upto > window_cells - 8) return fail(error, "invalid K/V extent");
    s.cells = (upto + 3) / 4 * 4;
    s.rows = upto / 4 + 1;   // the completed blocks and the moving one
    const size_t kv = (size_t) s.cells * (size_t) (g.n_head_kv * g.head_dim);   // INT8
    const size_t sc = (size_t) s.cells * (size_t) (g.n_head_kv * (g.head_dim / 64) * 2);   // half per 64
    s.part = {kv, kv, sc, sc, (size_t) s.rows * (size_t) g.idx_key_dim * sizeof(float)};
    return true;
}

bool metal_window(std::vector<std::array<void*, 5>>& parts, int64_t& cells, size_t state[5], std::string& error) {
    return strata::metal_engine::window_kv_parts(parts, cells, error) &&
           strata::metal_engine::window_state_sizes(state[0], state[1], state[2], state[3], state[4], error);
}

bool metal_state_ok(const ConversationCheckpoint& c, const size_t state[5]) {
    return c.gdn.size() == state[0] && c.ple.size() == state[1] && c.tails.size() == state[2] &&
           c.dead.size() == state[3] && c.block_pos.size() == state[4] && c.stage_parts.empty() && image_keys(c.imgs, c.ids.size());
}

bool metal_view_validate(const ConversationView& view, int64_t cells, const size_t state[5], std::string& error) {
    if (view.ids.empty() || (int64_t) view.ids.size() > cells - 8 || !image_keys(view.images, view.ids.size()) ||
        std::any_of(view.ids.begin(), view.ids.end(), [](int32_t id) { return id < 0; }))
        return fail(error, "invalid live token/image metadata");
    for (const auto& c : view.checkpoints) {
        if (c.ids.empty() || c.ids.size() > view.ids.size() || !std::equal(c.ids.begin(), c.ids.end(), view.ids.begin()))
            return fail(error, "checkpoint is not a live token prefix");
        if (!metal_state_ok(c, state)) return fail(error, "checkpoint is not this window's running state");
        size_t image = 0;
        for (const auto& key : view.images) {
            if ((uint64_t) key.start >= c.ids.size()) break;
            if (image >= c.imgs.size() || !(c.imgs[image++] == key))
                return fail(error, "checkpoint image identity differs");
        }
        if (image != c.imgs.size()) return fail(error, "checkpoint image prefix differs");
    }
    return true;
}

std::array<ConversationBuffer*, 5> metal_buffers(ConversationKv& kv) { return {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled}; }
std::array<const ConversationBuffer*, 5> metal_buffers(const ConversationKv& kv) {
    return {&kv.k, &kv.v, &kv.k_scale, &kv.v_scale, &kv.pooled};
}

bool metal_kv_validate(const ConversationKv& kv, const ModelGeometry& g, int64_t upto, int64_t cells, std::string& error) {
    MetalKvSizes s;
    if (!metal_kv_sizes(g, upto, cells, s, error)) return false;
    if (kv.format != kMetalKvFormat || kv.cells != s.cells || kv.heads != g.n_head_kv || kv.head_dim != g.head_dim ||
        kv.page_size != 4 || kv.pooled_rows != s.rows || kv.idx_dim != g.idx_key_dim)
        return fail(error, "incompatible K/V geometry");
    const auto b = metal_buffers(kv);
    for (size_t i = 0; i < b.size(); ++i)
        if (b[i]->size() != s.part[i]) return fail(error, "invalid K/V payload size");
    return true;
}

// `unchanged`: the retained image's cells below it still hold this session's bytes (as conversation_kv_save)
bool metal_kv_save(ConversationKv& kv, const std::array<void*, 5>& src, const ModelGeometry& g, int64_t upto,
                   int64_t cells, int64_t unchanged, size_t* reused, std::string& error) {
    MetalKvSizes s;
    if (!metal_kv_sizes(g, upto, cells, s, error)) return false;
    if (unchanged < 0 || unchanged > upto || unchanged > kv.cells) return fail(error, "invalid unchanged prefix");
    const int64_t whole = unchanged / 4 * 4;
    kv.format = kMetalKvFormat;
    kv.cells = s.cells; kv.heads = g.n_head_kv; kv.head_dim = g.head_dim; kv.page_size = 4;
    kv.pooled_rows = s.rows; kv.idx_dim = g.idx_key_dim;
    const auto dst = metal_buffers(kv);
    for (size_t i = 0; i < dst.size(); ++i) {
        if (!src[i]) return fail(error, "missing window K/V buffer");
        const size_t keep = i == 4 ? (size_t) (unchanged / 4) * (size_t) g.idx_key_dim * sizeof(float)
                                   : s.part[i] / (size_t) s.cells * (size_t) whole;
        if (keep > dst[i]->size()) return fail(error, "missing reusable prefix");
        dst[i]->resize(s.part[i]);
        dst[i]->visit(keep, s.part[i] - keep, [&](uint8_t* p, size_t n, size_t at) {
            std::memcpy(p, static_cast<const uint8_t*>(src[i]) + at, n);
            progress_beat();
            return true;
        });
        if (reused) *reused += keep;
    }
    return true;
}

bool metal_kv_restore(const ConversationKv& kv, const std::array<void*, 5>& dst) {
    const auto src = metal_buffers(kv);
    for (size_t i = 0; i < src.size(); ++i)
        src[i]->visit(0, src[i]->size(), [&](const uint8_t* p, size_t n, size_t at) {
            std::memcpy(static_cast<uint8_t*>(dst[i]) + at, p, n);
            progress_beat();
            return true;
        });
    return true;
}

bool metal_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view, const ModelGeometry& g,
                                  size_t& bytes, std::string& error) {
    bytes = 0;
    std::vector<std::array<void*, 5>> parts;
    int64_t cells = 0;
    size_t state[5] = {};
    if (!metal_window(parts, cells, state, error) || !metal_view_validate(view, cells, state, error)) return false;
    const int64_t upto = (int64_t) view.ids.size();
    MetalKvSizes s;
    if (!metal_kv_sizes(g, upto, cells, s, error)) return false;
    const bool reusing = !reuse.kv.empty();
    if (reusing && (reuse.kv.size() != parts.size() || reuse.unchanged_tokens < 0 ||
                    reuse.unchanged_tokens > reuse.captured_tokens || reuse.unchanged_tokens > upto))
        return fail(error, "invalid retained K/V prefix");
    size_t ids = 0, images = 0, checkpoints = 0, layers = 0;
    if (!product(ids, {view.ids.size(), sizeof(int32_t)}) ||
        !product(images, {view.images.size(), sizeof(ConversationImageKey)}) ||
        !product(checkpoints, {view.checkpoints.size(), sizeof(ConversationCheckpoint)}) ||
        !product(layers, {std::max(parts.size(), reusing ? reuse.kv.capacity() : 0), sizeof(ConversationKv)}))
        return fail(error, "snapshot metadata byte count overflow");
    for (size_t n : {ids, images, checkpoints, layers, state[0], state[1], state[2], state[3], state[4]})
        if (!add(bytes, n)) return fail(error, "snapshot byte count overflow");
    for (const auto& c : view.checkpoints)
        if (!metadata_bytes(c, bytes)) return fail(error, "checkpoint byte count overflow");
    for (size_t j = 0; j < parts.size(); ++j) {
        const ConversationKv empty;
        const auto b = metal_buffers(reusing ? reuse.kv[j] : empty);
        for (size_t i = 0; i < b.size(); ++i) {
            const size_t n = b[i]->allocation_peak(s.part[i]);
            if (n == SIZE_MAX || !add(bytes, n)) return fail(error, "K/V allocation overflow");
        }
    }
    return true;
}

bool metal_snapshot_save(SavedConversation& image, const ConversationView& view, const SessionState& ss,
                         const ModelGeometry& g, std::string& error, ConversationKvReuse reuse, size_t* reused_bytes) {
    std::vector<std::array<void*, 5>> parts;
    int64_t cells = 0;
    size_t state[5] = {};
    if (!metal_window(parts, cells, state, error) || !metal_view_validate(view, cells, state, error) || !sync(error))
        return false;
    if (!reuse.kv.empty() && reuse.kv.size() != parts.size()) reuse = {};
    SavedConversation captured;
    captured.geometry = geometry_key(g);
    captured.layer_lo = ss.layer_lo; captured.layer_hi = ss.layer_hi;
    captured.live.ids = view.ids; captured.live.imgs = view.images;
    captured.cvec = view.cvec; captured.checkpoints = conversation_detail::copy_checkpoints_beating(view.checkpoints);
    const int64_t upto = (int64_t) view.ids.size();
    const int64_t unchanged = reuse.kv.empty() ? 0 : std::min<int64_t>(reuse.unchanged_tokens, upto);
    captured.kv = std::move(reuse.kv);
    captured.kv.resize(parts.size());
    if (!conversation_checkpoint_save(captured.live, ss, g, error)) return false;
    for (size_t j = 0; j < parts.size(); ++j)
        if (!metal_kv_save(captured.kv[j], parts[j], g, upto, cells, unchanged, reused_bytes, error)) return false;
    image = std::move(captured);
    return true;
}

bool metal_snapshot_validate(const SavedConversation& image, const SessionState& ss, const ModelGeometry& g,
                             std::string& error) {
    if (!image.live.stage_parts.empty() || !image.stage_images.empty())
        return fail(error, "a layer split's image restored into the Metal window");
    if (image.geometry != geometry_key(g)) return fail(error, "incompatible runtime geometry");
    if (image.layer_lo != ss.layer_lo || image.layer_hi != ss.layer_hi)
        return fail(error, "snapshot from another session layer range");
    std::vector<std::array<void*, 5>> parts;
    int64_t cells = 0;
    size_t state[5] = {};
    const ConversationView view{image.live.ids, image.live.imgs, image.checkpoints, image.cvec};
    if (!metal_window(parts, cells, state, error) || !metal_view_validate(view, cells, state, error)) return false;
    if (!metal_state_ok(image.live, state)) return fail(error, "the live state is not this window's running state");
    if (image.kv.size() != parts.size()) return fail(error, "invalid K/V layer count (the draft layer differs?)");
    for (const auto& kv : image.kv)
        if (!metal_kv_validate(kv, g, (int64_t) image.live.ids.size(), cells, error)) return false;
    return true;
}

ConversationRestore metal_snapshot_restore(const SavedConversation& image, SessionState& ss, const ModelGeometry& g,
                                           std::string& error) {
    if (!metal_snapshot_validate(image, ss, g, error)) return ConversationRestore::invalid;
    if (!sync(error)) return ConversationRestore::transfer_failed;
    std::vector<std::array<void*, 5>> parts;
    int64_t cells = 0;
    if (!strata::metal_engine::window_kv_parts(parts, cells, error)) return ConversationRestore::transfer_failed;
    for (size_t j = 0; j < parts.size(); ++j) metal_kv_restore(image.kv[j], parts[j]);
    if (!conversation_checkpoint_restore(image.live, ss, g, error)) return ConversationRestore::transfer_failed;
    return ConversationRestore::restored;
}
}  // namespace
#endif

#if defined(STRATA_GPU_METAL)
bool metal_kv_window(std::vector<kvstore::Parts>& parts, int64_t& cells, kvstore::Layout& layout, kvstore::StateSizes& state,
                     const ModelGeometry& g, std::string& error) {
    size_t st[5] = {};
    if (!metal_window(parts, cells, st, error)) return false;
    MetalKvSizes one;
    if (!metal_kv_sizes(g, 1, cells, one, error)) return false;   // the layout check
    layout.cell = {(uint64_t) (g.n_head_kv * g.head_dim), (uint64_t) (g.n_head_kv * g.head_dim),
                   (uint64_t) (g.n_head_kv * (g.head_dim / 64) * 2), (uint64_t) (g.n_head_kv * (g.head_dim / 64) * 2)};
    layout.pooled_row = (uint64_t) g.idx_key_dim * sizeof(float);
    layout.page = 4;
    layout.block = 4;
    for (size_t i = 0; i < 5; ++i) state[i] = st[i];
    return true;
}
#endif

#if defined(STRATA_GPU_METAL)
bool metal_state_view(kvstore::StateView& view, bool restore, std::string& error) {
    return strata::metal_engine::window_state_segments(view, restore, error);
}
#endif

bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& ss,
                                 const ModelGeometry& g, const QsaState* draft, size_t& bytes, std::string& error) {
#if defined(STRATA_GPU_METAL)
    (void) ss; (void) draft;
    return metal_snapshot_capture_bytes({}, view, g, bytes, error);
#endif
    bytes = 0;
    if (!view_validate(view, ss, g, error)) return false;
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    size_t ids = 0, images = 0, checkpoints = 0, layers = 0, tails = 0, dead = 0, positions = 0;
    const auto qsa = (uint64_t) owned_qsa(ss);
    if (!product(ids, {view.ids.size(), sizeof(int32_t)}) ||
        !product(images, {view.images.size(), sizeof(ConversationImageKey)}) ||
        !product(checkpoints, {view.checkpoints.size(), sizeof(ConversationCheckpoint)}) ||
        !product(layers, {qsa + (draft ? 1 : 0), sizeof(ConversationKv)}) ||
        !product(tails, {qsa, z.tail}) || !product(dead, {qsa, z.dead}) ||
        !product(positions, {qsa, z.block_pos})) return fail(error, "snapshot metadata byte count overflow");
    for (size_t n : {ids, images, checkpoints, layers, tails, dead, positions, z.gdn, ss.ple_hist ? z.ple : 0})
        if (!add(bytes, n)) return fail(error, "snapshot byte count overflow");
    for (const auto& c : view.checkpoints)
        if (!metadata_bytes(c, bytes)) return fail(error, "checkpoint byte count overflow");
    const int64_t upto = (int64_t) view.ids.size(); // view_validate bounds this by signed max_cells
    for (uint64_t i = 0; i < qsa + (draft ? 1 : 0); ++i) {
        const auto& st = i == qsa ? *draft : owned(ss, (size_t) i);
        const size_t n = conversation_kv_bytes(st, g, upto, i != qsa);
        if (!n || !add(bytes, n)) return fail(error, "invalid or overflowing K/V byte estimate");
    }
    return true;
}

bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& ss, const ModelGeometry& g,
                                         const QsaState* draft, size_t& bytes, std::string& error) {
#if defined(STRATA_GPU_METAL)
    (void) ss; (void) draft;
    return metal_snapshot_capture_bytes(reuse, view, g, bytes, error);
#endif
    if (!conversation_snapshot_bytes(view, ss, g, draft, bytes, error)) return false;
    if (reuse.kv.empty()) return true;
    const size_t layers = owned_qsa(ss) + (draft ? 1 : 0);
    if (reuse.kv.size() != layers || reuse.unchanged_tokens < 0 ||
        reuse.unchanged_tokens > reuse.captured_tokens || reuse.unchanged_tokens > int64_t(view.ids.size()))
        return fail(error, "invalid retained K/V prefix");
    for (size_t i = 0; i < layers; ++i) {
        const bool index = i < owned_qsa(ss);
        const auto& st = index ? owned(ss, i) : *draft;
        if (!conversation_kv_validate(reuse.kv[i], st, g, reuse.captured_tokens, index, error)) return false;
        const size_t fresh = conversation_kv_bytes(st, g, int64_t(view.ids.size()), index);
        size_t retained = 0;
        if (!conversation_kv_capture_bytes(reuse.kv[i], st, g, int64_t(view.ids.size()), index, retained, error)) return false;
        bytes -= fresh;
        if (!add(bytes, retained)) return fail(error, "retained K/V allocation overflow");
    }
    size_t directory = 0;
    if (!product(directory, {reuse.kv.capacity() - layers, sizeof(ConversationKv)}) || !add(bytes, directory))
        return fail(error, "retained K/V directory overflow");
    return true;
}

bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& ss, const ModelGeometry& g,
                                const QsaState* draft, std::string& error,
                                ConversationKvReuse reuse, size_t* reused_bytes) {
#if defined(STRATA_GPU_METAL)
    (void) draft;
    return metal_snapshot_save(image, view, ss, g, error, std::move(reuse), reused_bytes);
#endif
    size_t estimate = 0;
    if (!conversation_snapshot_capture_bytes(reuse, view, ss, g, draft, estimate, error) || !sync(error)) return false;
    // Build into a new object so a failure cannot publish a partial snapshot.
    SavedConversation captured;
    captured.geometry = geometry_key(g);
    captured.layer_lo = ss.layer_lo; captured.layer_hi = ss.layer_hi;
    captured.live.ids = view.ids; captured.live.imgs = view.images;
    captured.cvec = view.cvec; captured.checkpoints = conversation_detail::copy_checkpoints_beating(view.checkpoints);
    const int64_t unchanged = reuse.kv.empty() ? 0 : reuse.unchanged_tokens;
    captured.kv = std::move(reuse.kv);
    const size_t layers = owned_qsa(ss);
    captured.kv.resize(layers + (draft ? 1 : 0));
    if (!conversation_checkpoint_save(captured.live, ss, g, error)) return false;
    const int64_t upto = (int64_t) view.ids.size();
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_save(captured.kv[j], owned(ss, j), g, upto, true, error,
                                  unchanged, reused_bytes)) return false;
    // The draft's final cell may not have been computed when the output cap was
    // reached. Refresh that page even when the main prefix continued unchanged.
    if (draft && !conversation_kv_save(captured.kv.back(), *draft, g, upto, false, error,
                                       std::max<int64_t>(0, unchanged - 1), reused_bytes)) return false;
    image = std::move(captured);
    return true;
}

bool conversation_snapshot_sources(SavedConversation& meta, std::vector<SessionKvSource>& sources,
                                   const ConversationView& view, const SessionState& ss, const ModelGeometry& g,
                                   const QsaState& draft, std::string& error) {
#if defined(STRATA_GPU_METAL)
    // session files read the session's QSA pools, which the Metal window does not use: refused, never garbage
    (void) meta; (void) sources; (void) view; (void) ss; (void) g; (void) draft;
    return fail(error, "session files are not supported on Metal yet");
#endif
    if (!view_validate(view, ss, g, error) || !sync(error)) return false;
    SavedConversation captured;
    captured.geometry = geometry_key(g);
    captured.layer_lo = ss.layer_lo; captured.layer_hi = ss.layer_hi;
    captured.live.ids = view.ids; captured.live.imgs = view.images;
    captured.cvec = view.cvec; captured.checkpoints = conversation_detail::copy_checkpoints_beating(view.checkpoints);
    if (!conversation_checkpoint_save(captured.live, ss, g, error)) return false;
    const int64_t upto = (int64_t) view.ids.size();
    const size_t layers = owned_qsa(ss);
    std::vector<SessionKvSource> out(layers + 1);
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_source(out[j], owned(ss, j), g, upto, true, error)) return false;
    if (!conversation_kv_source(out.back(), draft, g, upto, false, error)) return false;
    meta = std::move(captured);
    sources = std::move(out);
    return true;
}

bool conversation_session_read_limits(SessionReadLimits& limits, const SessionState& ss, const ModelGeometry& g,
                                      const QsaState& draft, uint64_t max_tokens, uint64_t max_checkpoints,
                                      std::string& error) {
#if defined(STRATA_GPU_METAL)
    (void) limits; (void) ss; (void) g; (void) draft; (void) max_tokens; (void) max_checkpoints;
    return fail(error, "session files are not supported on Metal yet");
#endif
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (ss.max_cells < 0) return fail(error, "invalid session cells");
    const uint64_t tokens = std::min<uint64_t>(max_tokens, (uint64_t) ss.max_cells);
    const size_t layers = owned_qsa(ss);
    if ((owned_qsa(ss) && !ss.qsa_states)) return fail(error, "invalid session running-state targets");
    size_t tails = 0, dead = 0, block_pos = 0;
    if (!product(tails, {layers, z.tail}) || !product(dead, {layers, z.dead}) ||
        !product(block_pos, {layers, z.block_pos})) return fail(error, "indexer byte count overflow");
    SessionReadLimits l = limits;   // keeps the caller's admit / progress / max_file_bytes
    l.max_tokens = tokens;
    l.max_checkpoints = max_checkpoints;
    l.max_kv_layers = layers + 1;
    l.geometry = geometry_key(g);
    l.layer_range = std::make_pair(ss.layer_lo, ss.layer_hi);
    l.max_state_bytes = {z.gdn, ss.ple_hist ? z.ple : 0, tails, dead, block_pos};
    l.max_kv_bytes.assign(layers + 1, {});
    for (size_t j = 0; j < layers; ++j) {
        const auto& st = owned(ss, j);
        const int64_t upto = (int64_t) std::min<uint64_t>(tokens, (uint64_t) std::max<int64_t>(st.max_cells, 0));
        if (!conversation_kv_part_sizes(st, g, upto, true, l.max_kv_bytes[j], error)) return false;
    }
    const int64_t dupto = (int64_t) std::min<uint64_t>(tokens, (uint64_t) std::max<int64_t>(draft.max_cells, 0));
    if (!conversation_kv_part_sizes(draft, g, dupto, false, l.max_kv_bytes.back(), error)) return false;
    limits = std::move(l);
    return true;
}

bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& ss,
                                    const ModelGeometry& g, const QsaState* draft, std::string& error) {
#if defined(STRATA_GPU_METAL)
    (void) draft;
    return metal_snapshot_validate(image, ss, g, error);
#endif
    if (!image.live.stage_parts.empty())
        return fail(error, "a running checkpoint's stage parts belong in the stage images, not in live.stage_parts");
    if (image.geometry != geometry_key(g)) return fail(error, "incompatible runtime geometry");
    // an image holds exactly one carve's running state and K/V: same layer range or nothing
    if (image.layer_lo != ss.layer_lo || image.layer_hi != ss.layer_hi)
        return fail(error, "snapshot from another session layer range");
    const ConversationView view{image.live.ids, image.live.imgs, image.checkpoints, image.cvec};
    if (!view_validate(view, ss, g, error) || !conversation_checkpoint_validate(image.live, ss, g, error)) return false;
    const size_t layers = owned_qsa(ss);
    if (image.kv.size() != layers + (draft ? 1 : 0)) return fail(error, "invalid K/V layer count");
    const int64_t upto = (int64_t) image.live.ids.size();
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_validate(image.kv[j], owned(ss, j), g, upto, true, error)) return false;
    return !draft || conversation_kv_validate(image.kv.back(), *draft, g, upto, false, error);
}

ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& ss,
                                                   const ModelGeometry& g, const QsaState* draft, std::string& error) {
#if defined(STRATA_GPU_METAL)
    (void) draft;
    return metal_snapshot_restore(image, ss, g, error);
#endif
    if (!conversation_snapshot_validate(image, ss, g, draft, error)) return ConversationRestore::invalid;
    if (!sync(error)) return ConversationRestore::transfer_failed;
    const int64_t upto = (int64_t) image.live.ids.size();
    for (size_t j = 0; j < owned_qsa(ss); ++j)
        if (!conversation_kv_restore(image.kv[j], owned(ss, j), g, upto, true, error))
            return ConversationRestore::transfer_failed;
    if ((draft && !conversation_kv_restore(image.kv.back(), *draft, g, upto, false, error)) ||
        !conversation_checkpoint_restore(image.live, ss, g, error)) return ConversationRestore::transfer_failed;
    return ConversationRestore::restored;
}

// the draft layer's K/V included (the stage that owns the draft head, or no layer split)
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& ss, const ModelGeometry& g,
                                 const QsaState& draft, size_t& bytes, std::string& error) {
    return conversation_snapshot_bytes(view, ss, g, &draft, bytes, error);
}
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& ss, const ModelGeometry& g, const QsaState& draft,
                                         size_t& bytes, std::string& error) {
    return conversation_snapshot_capture_bytes(reuse, view, ss, g, &draft, bytes, error);
}
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view, const SessionState& ss,
                                const ModelGeometry& g, const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse, size_t* reused_bytes) {
    return conversation_snapshot_save(image, view, ss, g, &draft, error, std::move(reuse), reused_bytes);
}
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& ss, const ModelGeometry& g,
                                    const QsaState& draft, std::string& error) {
    // the whole-session form: a layer split's image (stage images) is restored stage by stage, never through here
    if (!image.stage_images.empty()) return fail(error, "a layer split's image restored as a single session");
    return conversation_snapshot_validate(image, ss, g, &draft, error);
}
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& ss,
                                                   const ModelGeometry& g, const QsaState& draft, std::string& error) {
    return conversation_snapshot_restore(image, ss, g, &draft, error);
}
} // namespace strata::core
