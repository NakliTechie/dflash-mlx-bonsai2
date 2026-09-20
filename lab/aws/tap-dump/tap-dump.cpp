// tap-dump: teacher-force a corpus through a GGUF model and dump, per token, the residual stream at chosen
// layers (l_out-N, int8 per-row quantized + fp16 scale) and the model's own top-8 next-token ids/logprobs.
// Output: raw shard files + shard_XXXX.json headers (feat int8 [n,taps,n_embd], scale f16 [n,taps],
// ids i32 [n], doc i32 [n], pos i32 [n], top_ids i32 [n,8], top_lp f16 [n,8]).
//   tap-dump -m model.gguf --corpus mix.jsonl --out DIR --taps 5,19,33,47,61 --budget 4000000 -c 2048 -b 512 -ub 512 -ngl 99 -fa on
// mix.jsonl: one JSON string per line ({"kind":"code","text":"..."}) — same content as the Mac corpus.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

static uint16_t f32_to_f16(float f) { uint32_t x; memcpy(&x, &f, 4); uint32_t sign = (x >> 16) & 0x8000; int32_t exp = ((x >> 23) & 0xff) - 127 + 15; uint32_t mant = x & 0x7fffff;
    if (exp <= 0) { if (exp < -10) return sign; mant |= 0x800000; uint32_t t = 14 - exp; uint32_t a = (1u << (t - 1)) - 1, b = (mant >> t) & 1; return sign | ((mant + a + b) >> t); }
    if (exp >= 0x1f) return sign | 0x7c00; return sign | (exp << 10) | ((mant + 0xfff + ((mant >> 13) & 1)) >> 13); }

struct tap_state {
    std::vector<int> taps;                 // layer ids
    std::map<int, int> tap_index;          // layer id -> slot
    int n_embd = 0;
    std::vector<float> buf;                // [n_tokens_ubatch, taps, n_embd] for the current ubatch
    int cur_tokens = 0;
    std::vector<float> host;
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (tap_state *) user_data;
    const char * name = t->name;
    if (strncmp(name, "l_out-", 6) != 0) return false;
    int il = atoi(name + 6);
    auto it = st->tap_index.find(il);
    if (it == st->tap_index.end()) return false;
    if (ask) return true;                  // yes, we want this tensor's data
    const int n_embd = (int) t->ne[0], n_tok = (int) t->ne[1];
    if (st->n_embd == 0) st->n_embd = n_embd;
    if (st->cur_tokens != n_tok) { st->cur_tokens = n_tok; st->buf.assign((size_t) n_tok * st->taps.size() * n_embd, 0.0f); }
    st->host.resize((size_t) ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, st->host.data(), 0, ggml_nbytes(t));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<uint16_t> h16(ggml_nelements(t)); ggml_backend_tensor_get(t, h16.data(), 0, ggml_nbytes(t));
        for (size_t i = 0; i < h16.size(); ++i) st->host[i] = ggml_fp16_to_fp32(h16[i]);
    } else { LOG_ERR("unexpected tensor type for %s\n", name); return true; }
    const int slot = it->second, T = (int) st->taps.size();
    for (int tk = 0; tk < n_tok; ++tk) memcpy(&st->buf[((size_t) tk * T + slot) * n_embd], &st->host[(size_t) tk * n_embd], sizeof(float) * n_embd);
    return true;
}

struct shard_writer {
    std::string dir; int idx = 0, per_shard = 20000, n = 0, taps = 5, n_embd = 0;
    std::vector<int8_t> feat; std::vector<uint16_t> scale; std::vector<int32_t> ids, doc, pos, top_ids; std::vector<uint16_t> top_lp;
    long total = 0;
    void add_token(const float * f, int32_t id, int32_t d, int32_t p, const int32_t * tid, const float * tlp) {
        for (int s = 0; s < taps; ++s) { const float * row = f + (size_t) s * n_embd; float amax = 1e-6f; for (int i = 0; i < n_embd; ++i) amax = std::max(amax, std::fabs(row[i]));
            float sc = amax / 127.0f; scale.push_back(f32_to_f16(sc)); for (int i = 0; i < n_embd; ++i) feat.push_back((int8_t) lrintf(row[i] / sc)); }
        ids.push_back(id); doc.push_back(d); pos.push_back(p); for (int k = 0; k < 8; ++k) { top_ids.push_back(tid[k]); top_lp.push_back(f32_to_f16(tlp[k])); }
        ++n; ++total; if (n >= per_shard) flush();
    }
    template <class V> void put(const std::string & path, const V & v) { std::ofstream o(path, std::ios::binary); o.write((const char *) v.data(), (std::streamsize) (v.size() * sizeof(v[0]))); }
    void flush() {
        if (n == 0) return; char base[512]; snprintf(base, sizeof base, "%s/shard_%04d", dir.c_str(), idx);
        put(std::string(base) + ".feat.i8", feat); put(std::string(base) + ".scale.f16", scale); put(std::string(base) + ".ids.i32", ids); put(std::string(base) + ".doc.i32", doc);
        put(std::string(base) + ".pos.i32", pos); put(std::string(base) + ".top_ids.i32", top_ids); put(std::string(base) + ".top_lp.f16", top_lp);
        std::ofstream j(std::string(base) + ".json"); j << "{\"n\":" << n << ",\"taps\":" << taps << ",\"n_embd\":" << n_embd << ",\"topk\":8}\n";
        LOG_INF("shard %d written: %d tokens (total %ld)\n", idx, n, total);
        feat.clear(); scale.clear(); ids.clear(); doc.clear(); pos.clear(); top_ids.clear(); top_lp.clear(); n = 0; ++idx;
    }
};

static std::vector<std::string> read_corpus(const std::string & path) {   // JSONL with "text" fields (minimal parser: takes the value of the last "text" key)
    std::vector<std::string> docs; std::ifstream in(path); std::string line;
    while (std::getline(in, line)) { size_t k = line.rfind("\"text\":"); if (k == std::string::npos) continue; size_t q = line.find('"', k + 7); if (q == std::string::npos) continue;
        std::string s; for (size_t i = q + 1; i < line.size(); ++i) { char c = line[i]; if (c == '\\' && i + 1 < line.size()) { char e = line[++i]; if (e == 'n') s += '\n'; else if (e == 't') s += '\t'; else if (e == 'u' && i + 4 < line.size()) { unsigned cp = std::stoul(line.substr(i + 1, 4), nullptr, 16); i += 4; if (cp < 0x80) s += (char) cp; else if (cp < 0x800) { s += (char) (0xC0 | (cp >> 6)); s += (char) (0x80 | (cp & 0x3F)); } else { s += (char) (0xE0 | (cp >> 12)); s += (char) (0x80 | ((cp >> 6) & 0x3F)); s += (char) (0x80 | (cp & 0x3F)); } } else s += e; } else if (c == '"') break; else s += c; }
        if (!s.empty()) docs.push_back(s); }
    return docs;
}

int main(int argc, char ** argv) {
    // our own flags first, then hand the rest to common
    std::string corpus, out = "tap-shards"; std::vector<int> taps = {5, 19, 33, 47, 61}; long budget = 600000; int doc_max = 1536, per_shard = 20000;
    std::vector<char *> rest; rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) { std::string a = argv[i];
        if (a == "--corpus" && i + 1 < argc) corpus = argv[++i]; else if (a == "--out" && i + 1 < argc) out = argv[++i]; else if (a == "--budget" && i + 1 < argc) budget = atol(argv[++i]);
        else if (a == "--doc-max" && i + 1 < argc) doc_max = atoi(argv[++i]); else if (a == "--per-shard" && i + 1 < argc) per_shard = atoi(argv[++i]);
        else if (a == "--taps" && i + 1 < argc) { taps.clear(); std::stringstream ss(argv[++i]); std::string x; while (std::getline(ss, x, ',')) taps.push_back(atoi(x.c_str())); }
        else rest.push_back(argv[i]); }
    common_params params; common_init();
    if (!common_params_parse((int) rest.size(), rest.data(), params, LLAMA_EXAMPLE_COMMON)) return 1;
    if (corpus.empty()) { LOG_ERR("--corpus is required\n"); return 1; }
    tap_state st; st.taps = taps; for (size_t i = 0; i < taps.size(); ++i) st.tap_index[taps[i]] = (int) i;
    llama_backend_init(); llama_numa_init(params.numa);
    params.cb_eval = cb_eval; params.cb_eval_user_data = &st; params.warmup = false;
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model(); auto * ctx = llama_init->context();
    if (!model || !ctx) { LOG_ERR("init failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model); const int n_vocab = llama_vocab_n_tokens(vocab);
    auto docs = read_corpus(corpus); LOG_INF("corpus: %zu docs; taps:", docs.size()); for (int t : taps) LOG_INF(" %d", t); LOG_INF("; budget %ld tokens\n", budget);
    shard_writer w; w.dir = out; w.per_shard = per_shard; w.taps = (int) taps.size(); std::string mk = "mkdir -p " + out; if (system(mk.c_str()) != 0) return 1;
    const int n_batch = params.n_batch;
    std::vector<int32_t> tid(8); std::vector<float> tlp(8); std::vector<std::pair<float,int>> cand;
    for (size_t d = 0; d < docs.size() && w.total < budget; ++d) {
        std::vector<llama_token> toks = common_tokenize(ctx, docs[d], false, true); if ((int) toks.size() > doc_max) toks.resize(doc_max); if (toks.size() < 64) continue;
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int p0 = 0; p0 < (int) toks.size(); p0 += n_batch) {
            int n = std::min(n_batch, (int) toks.size() - p0);
            llama_batch batch = llama_batch_init(n, 0, 1);
            for (int i = 0; i < n; ++i) { batch.token[i] = toks[p0 + i]; batch.pos[i] = p0 + i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = true; } batch.n_tokens = n;
            st.cur_tokens = 0;
            if (llama_decode(ctx, batch) != 0) { LOG_ERR("decode failed at doc %zu pos %d\n", d, p0); llama_batch_free(batch); return 1; }
            if (st.cur_tokens != n) { LOG_ERR("callback saw %d tokens, batch had %d (set -ub = -b)\n", st.cur_tokens, n); llama_batch_free(batch); return 1; }
            if (w.n_embd == 0) w.n_embd = st.n_embd;
            for (int i = 0; i < n; ++i) {
                const float * lg = llama_get_logits_ith(ctx, i);
                float mx = -1e30f; for (int v = 0; v < n_vocab; ++v) mx = std::max(mx, lg[v]);
                double sum = 0; for (int v = 0; v < n_vocab; ++v) sum += std::exp((double) (lg[v] - mx)); const float lse = mx + (float) std::log(sum);
                cand.clear(); cand.reserve(n_vocab); for (int v = 0; v < n_vocab; ++v) cand.emplace_back(lg[v], v);
                std::partial_sort(cand.begin(), cand.begin() + 8, cand.end(), [](auto & a, auto & b) { return a.first > b.first; });
                for (int k = 0; k < 8; ++k) { tid[k] = cand[k].second; tlp[k] = cand[k].first - lse; }
                w.add_token(&st.buf[(size_t) i * taps.size() * st.n_embd], toks[p0 + i], (int32_t) d, p0 + i, tid.data(), tlp.data());
            }
            llama_batch_free(batch);
            if (w.total >= budget) break;
        }
        if (d % 20 == 0) LOG_INF("doc %zu/%zu, %ld tokens\n", d, docs.size(), w.total);
    }
    w.flush(); LOG_INF("done: %ld tokens in %d shards\n", w.total, w.idx);
    llama_perf_context_print(ctx); llama_backend_free(); return 0;
}
