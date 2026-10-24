// metal/src/gguf_lite.cpp - see gguf_lite.h.
#include "gguf_lite.h"

#include "ref_quant.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace gguf {

namespace {
struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;
    template <class T> T get() {
        T v{};
        if ((size_t)(end - p) < sizeof(T)) { ok = false; return v; }
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        const uint64_t n = get<uint64_t>();
        if (!ok || n > (uint64_t)(end - p)) { ok = false; return {}; }
        std::string s((const char*)p, n);
        p += n;
        return s;
    }
};
// value types: 0 u8 1 i8 2 u16 3 i16 4 u32 5 i32 6 f32 7 bool 8 string 9 array 10 u64 11 i64 12 f64
size_t scalar_size(uint32_t t) {
    switch (t) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
    }
    return 0;
}
double scalar_num(Reader& r, uint32_t t) {
    switch (t) {
        case 0: return r.get<uint8_t>(); case 1: return r.get<int8_t>(); case 2: return r.get<uint16_t>();
        case 3: return r.get<int16_t>(); case 4: return r.get<uint32_t>(); case 5: return r.get<int32_t>();
        case 6: return r.get<float>(); case 7: return r.get<uint8_t>(); case 10: return (double)r.get<uint64_t>();
        case 11: return (double)r.get<int64_t>(); case 12: return r.get<double>();
    }
    r.ok = false;
    return 0;
}
}  // namespace

bool File::open(const std::string& p, std::string& err) {
    path = p;
    const int fd = ::open(p.c_str(), O_RDONLY);
    if (fd < 0) { err = "cannot open " + p; return false; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { ::close(fd); err = "cannot stat " + p; return false; }
    file_bytes = (size_t)st.st_size;
    const size_t page = (size_t)getpagesize();
    map_bytes = (file_bytes + page - 1) / page * page;
    map = mmap(nullptr, map_bytes, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (map == MAP_FAILED) { map = nullptr; err = "mmap failed: " + p; return false; }
    Reader r{(const uint8_t*)map, (const uint8_t*)map + file_bytes};
    if (r.get<uint32_t>() != 0x46554747u) { err = "not a GGUF file"; return false; }
    if (r.get<uint32_t>() != 3) { err = "GGUF version is not 3"; return false; }
    const uint64_t n_t = r.get<uint64_t>(), n_kv = r.get<uint64_t>();
    uint64_t align = 32;
    for (uint64_t i = 0; i < n_kv && r.ok; ++i) {
        const std::string key = r.str();
        const uint32_t t = r.get<uint32_t>();
        if (t == 8) meta_str[key] = r.str();
        else if (t == 9) {
            const uint32_t et = r.get<uint32_t>();
            const uint64_t n = r.get<uint64_t>();
            if (et == 8) for (uint64_t k = 0; k < n && r.ok; ++k) r.str();
            else {
                const size_t sz = scalar_size(et);   // a nested array or an unknown type is refused, not skipped blind
                if (sz == 0 || n > (uint64_t)(r.end - r.p) / sz) { r.ok = false; break; }
                r.p += n * sz;
            }
        } else {
            const double v = scalar_num(r, t);
            meta_num[key] = v;
            if (key == "general.alignment") align = (v >= 1 && v <= 65536) ? (uint64_t)v : 0;   // 0 is refused below
        }
    }
    std::vector<Tensor> ts;
    for (uint64_t i = 0; i < n_t && r.ok; ++i) {
        Tensor t;
        t.name = r.str();
        const uint32_t nd = r.get<uint32_t>();
        if (nd == 0 || nd > 4) { r.ok = false; break; }   // GGML_MAX_DIMS
        for (uint32_t k = 0; k < nd; ++k) t.shape.push_back(r.get<uint64_t>());
        t.type = (int)r.get<uint32_t>();
        t.offset = r.get<uint64_t>();
        ts.push_back(t);
    }
    if (!r.ok) { err = "truncated GGUF header"; return false; }
    if (align == 0 || (align & (align - 1)) != 0) { err = "bad GGUF alignment"; return false; }
    const uint64_t pos = (uint64_t)(r.p - (const uint8_t*)map);
    const uint64_t data = (pos + align - 1) / align * align;
    for (Tensor& t : ts) {
        t.offset += data;
        uint64_t n = 1;
        for (uint64_t d : t.shape)
            if (__builtin_mul_overflow(n, d, &n)) { err = "tensor shape overflows: " + t.name; return false; }
        if (!ref::known_type(t.type)) { err = "unknown tensor type " + std::to_string(t.type) + ": " + t.name; return false; }
        const uint64_t be = ref::block_elems(t.type);
        const uint64_t bb = (uint64_t)ref::block_bytes(t.type);
        if (n / be > UINT64_MAX / bb) { err = "tensor size overflows: " + t.name; return false; }
        for (uint64_t d : t.shape)
            if (d == 0 || d > INT32_MAX) { err = "invalid tensor dimension: " + t.name; return false; }
        if (t.shape[0] % be != 0) { err = "incomplete tensor row block: " + t.name; return false; }
        t.bytes = n / be * bb;
        if (t.offset < data || t.offset > file_bytes || t.bytes > file_bytes - t.offset) {
            err = "tensor lies outside the file: " + t.name;
            return false;
        }
        tensors[t.name] = t;
    }
    return true;
}

bool validate_window_tensors(const File& file, bool ple, std::string& err) {
    const std::vector<int> gemv_types = {ref::F32, ref::BF16, ref::Q8_0, ref::Q4_0, ref::Q5_0, ref::IQ4_NL,
                                         ref::IQ4_XS, ref::Q2_0, ref::IQ3_S, ref::Q3_K, ref::Q4_K, ref::Q5_K, ref::Q6_K};
    const std::vector<int> gu_types = {ref::Q2_0, ref::IQ3_XXS, ref::IQ2_XXS, ref::IQ2_XS, ref::IQ2_S,
                                      ref::IQ3_S, ref::IQ4_XS, ref::Q4_K, ref::Q5_K, ref::Q6_K, ref::Q8_0};
    const std::vector<int> dn_types = {ref::Q5_1, ref::Q8_0, ref::Q2_0, ref::IQ4_NL};
    auto need = [&](const std::string& name, std::vector<uint64_t> shape, std::vector<int> types = {}) {
        const Tensor* t = file.find(name);
        if (!t) { err = "metal window: missing tensor: " + name; return false; }
        if (t->shape != shape || (!types.empty() && std::find(types.begin(), types.end(), t->type) == types.end())) {
            err = "metal window: unsupported tensor shape or type: " + name;
            return false;
        }
        return true;
    };
    const Tensor* head = file.find("output.weight");
    if (!head || head->shape.size() != 2 || head->shape[0] != 2560 || head->shape[1] == 0 || head->shape[1] > 248320) {
        err = "metal window: invalid or missing output.weight matrix";
        return false;
    }
    if (!need("output.weight", head->shape, gemv_types) ||
        !need("token_embd.weight", {2560, head->shape[1]})) return false;
    auto norm = [&](const std::string& name, uint64_t n) {
        const Tensor* t = file.find(name);
        // Norms may be stored flat or with their stream axis intact.
        uint64_t count = 1;
        if (t) for (uint64_t d : t->shape) count *= d;
        if (!t || t->type != ref::F32 || count != n) {
            err = "metal window: invalid or missing F32 tensor: " + name;
            return false;
        }
        return true;
    };
    auto hc = [&](const std::string& prefix, bool final) {
        if (!norm(prefix + "_norm.weight", 10240) ||
            !need(prefix + "_down.weight", {10240, 320}, {ref::BF16, ref::Q8_0}) ||
            !need(prefix + "_up.weight", {320, 10240}, {ref::BF16, ref::Q8_0})) return false;
        const int type = file.find(prefix + "_down.weight")->type;
        if (file.find(prefix + "_up.weight")->type != type) {
            err = "metal window: mixed HC matrix types: " + prefix;
            return false;
        }
        return final || need(prefix + "_inject.weight", {10240, 4},
                            type == ref::Q8_0 ? std::vector<int>{ref::Q8_0, ref::F32} : std::vector<int>{ref::BF16});
    };
    if (!hc("output_hc", true)) return false;
    for (int l = 0; l < 48; ++l) {
        const std::string b = "blk." + std::to_string(l) + ".";
        auto matrix = [&](const char* suffix, uint64_t in, uint64_t out) { return need(b + suffix, {in, out}, gemv_types); };
        if (!hc(b + "hc_attn", false) || !hc(b + "hc_ffn", false) ||
            !matrix("ffn_gate_inp.weight", 2560, 512) ||
            !matrix("ffn_gate_shexp.weight", 2560, 640) || !matrix("ffn_up_shexp.weight", 2560, 640) ||
            !matrix("ffn_down_shexp.weight", 640, 2560)) return false;
        const Tensor* gate = file.find(b + "ffn_gate_inp_shexp.weight");
        if (!gate || (gate->shape != std::vector<uint64_t>{2560} && gate->shape != std::vector<uint64_t>{2560, 1}) ||
            std::find(gemv_types.begin(), gemv_types.end(), gate->type) == gemv_types.end()) {
            err = "metal window: invalid or missing shared gate: " + b + "ffn_gate_inp_shexp.weight";
            return false;
        }
        if (!need(b + "ffn_gate_exps.weight", {2560, 640, 512}, gu_types) ||
            !need(b + "ffn_up_exps.weight", {2560, 640, 512}, gu_types) ||
            !need(b + "ffn_down_exps.weight", {640, 2560, 512}, dn_types)) return false;
        if (file.find(b + "ffn_gate_exps.weight")->type != file.find(b + "ffn_up_exps.weight")->type) {
            err = "metal window: mixed expert gate/up types: " + b;
            return false;
        }
        if (l % 4 != 3) {
            if (!matrix("attn_qkv.weight", 2560, 10240) || !matrix("attn_gate.weight", 2560, 6144) ||
                !matrix("ssm_alpha.weight", 2560, 48) || !matrix("ssm_beta.weight", 2560, 48) ||
                !matrix("ssm_out.weight", 6144, 2560) || !need(b + "ssm_conv1d.weight", {4, 10240}, {ref::F32}) ||
                !norm(b + "ssm_a", 48) || !norm(b + "ssm_dt.bias", 48) || !norm(b + "ssm_norm.weight", 128)) return false;
        } else {
            if (!matrix("attn_q.weight", 2560, 12288) || !matrix("attn_k.weight", 2560, 512) ||
                !matrix("attn_v.weight", 2560, 512) || !matrix("attn_output.weight", 6144, 2560) ||
                !matrix("indexer.q_proj.weight", 2560, 512) || !matrix("indexer.k_proj.weight", 2560, 128) ||
                !norm(b + "attn_q_norm.weight", 256) || !norm(b + "attn_k_norm.weight", 256) ||
                !norm(b + "indexer.q_norm.weight", 128) || !norm(b + "indexer.k_norm.weight", 128)) return false;
        }
    }
    if (ple) {
        if (!need("blk.1.ple_key.weight", {2560, 10240}, gemv_types) || !need("blk.1.ple_value.weight", {2560, 2560}, gemv_types) ||
            !norm("blk.1.ple_norm_key.weight", 10240) || !norm("blk.1.ple_norm_query.weight", 10240) ||
            !norm("blk.1.ple_norm_conv.weight", 10240) ||
            !need("blk.1.ple_conv1d.weight", {4, 10240}, {ref::F16, ref::F32})) return false;
    }
    return true;
}

File::~File() {
    if (map) munmap(map, map_bytes);
}

}  // namespace gguf
