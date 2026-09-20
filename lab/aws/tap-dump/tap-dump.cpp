// tap-dump: teacher-force a corpus through a GGUF model and dump, per token, the residual stream at chosen
// layers (l_out-N, int8 per-row quantized + fp16 scale) and the model's own top-8 next-token ids/logprobs.
// Output: raw shard files + shard_XXXX.json headers (feat int8 [n,taps,n_embd], scale f16 [n,taps],
// ids i32 [n], doc i32 [n], pos i32 [n], top_ids i32 [n,8], top_lp f16 [n,8]).
//   tap-dump -m model.gguf --corpus mix.jsonl --out DIR --taps 5,19,33,47,61 --budget 4000000 -c 2048 -b 512 -ub 512 -ngl 99 -fa on
// mix.jsonl: one JSON string per line ({"kind":"code","text":"..."}) — same content as the Mac corpus.
// Generate mode (--gen N --seqs K): each corpus doc is a PROMPT; the model prefills it (rows flagged gen=0) and then
// greedy-decodes up to N tokens (gen=1), K prompts in flight as K sequences of one llama_batch. Row order inside a
// multi-sequence ubatch is recovered by matching the graph's result_output rows to llama_get_logits_ith. Pass
// -np K and -c (prompt_max + N) * K (the KV cache is split per sequence). Extra shard file: gen.i8 [n].
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
    std::vector<float> buf;                // [rows, taps, n_embd] accumulated over the ubatches of one llama_decode
    int cur_tokens = 0;                    // rows seen so far in this decode
    int ub_off = 0;                        // row offset of the current ubatch
    std::vector<float> host;
    std::vector<float> fp;                 // [rows, 8] first 8 logits of each result_output row (ubatch order)
    void reset() { cur_tokens = 0; ub_off = 0; fp.clear(); }
};

static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (tap_state *) user_data;
    const char * name = t->name;
    if (strcmp(name, "result_output") == 0) {          // logits rows of this ubatch: keep 8 values per row as a fingerprint
        if (ask) return true;
        const int nv = (int) t->ne[0], nr = (int) t->ne[1]; float v[8];
        for (int r = 0; r < nr; ++r) { ggml_backend_tensor_get(t, v, (size_t) r * nv * sizeof(float), sizeof v); st->fp.insert(st->fp.end(), v, v + 8); }
        return true;
    }
    if (strncmp(name, "l_out-", 6) != 0) return false;
    int il = atoi(name + 6);
    auto it = st->tap_index.find(il);
    if (it == st->tap_index.end()) return false;
    if (ask) return true;                  // yes, we want this tensor's data
    const int n_embd = (int) t->ne[0], n_tok = (int) t->ne[1];
    if (st->n_embd == 0) st->n_embd = n_embd;
    if (it->second == 0) { st->ub_off = st->cur_tokens; st->cur_tokens += n_tok; st->buf.resize((size_t) st->cur_tokens * st->taps.size() * n_embd, 0.0f); }
    st->host.resize((size_t) ggml_nelements(t));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, st->host.data(), 0, ggml_nbytes(t));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<uint16_t> h16(ggml_nelements(t)); ggml_backend_tensor_get(t, h16.data(), 0, ggml_nbytes(t));
        for (size_t i = 0; i < h16.size(); ++i) st->host[i] = ggml_fp16_to_fp32(h16[i]);
    } else { LOG_ERR("unexpected tensor type for %s\n", name); return true; }
    const int slot = it->second, T = (int) st->taps.size();
    for (int tk = 0; tk < n_tok; ++tk) memcpy(&st->buf[((size_t) (st->ub_off + tk) * T + slot) * n_embd], &st->host[(size_t) tk * n_embd], sizeof(float) * n_embd);
    return true;
}

struct shard_writer {
    std::string dir; int idx = 0, per_shard = 20000, n = 0, taps = 5, n_embd = 0;
    std::vector<int8_t> feat; std::vector<uint16_t> scale; std::vector<int32_t> ids, doc, pos, top_ids; std::vector<uint16_t> top_lp; std::vector<int8_t> gen;
    long total = 0;
    void add_token(const float * f, int32_t id, int32_t d, int32_t p, const int32_t * tid, const float * tlp, int8_t g = 0) {
        for (int s = 0; s < taps; ++s) { const float * row = f + (size_t) s * n_embd; float amax = 1e-6f; for (int i = 0; i < n_embd; ++i) amax = std::max(amax, std::fabs(row[i]));
            float sc = amax / 127.0f; scale.push_back(f32_to_f16(sc)); for (int i = 0; i < n_embd; ++i) feat.push_back((int8_t) lrintf(row[i] / sc)); }
        ids.push_back(id); doc.push_back(d); pos.push_back(p); gen.push_back(g); for (int k = 0; k < 8; ++k) { top_ids.push_back(tid[k]); top_lp.push_back(f32_to_f16(tlp[k])); }
        ++n; ++total; if (n >= per_shard) flush();
    }
    template <class V> void put(const std::string & path, const V & v) { std::ofstream o(path, std::ios::binary); o.write((const char *) v.data(), (std::streamsize) (v.size() * sizeof(v[0]))); }
    void flush() {
        if (n == 0) return; char base[512]; snprintf(base, sizeof base, "%s/shard_%04d", dir.c_str(), idx);
        put(std::string(base) + ".feat.i8", feat); put(std::string(base) + ".scale.f16", scale); put(std::string(base) + ".ids.i32", ids); put(std::string(base) + ".doc.i32", doc);
        put(std::string(base) + ".pos.i32", pos); put(std::string(base) + ".top_ids.i32", top_ids); put(std::string(base) + ".top_lp.f16", top_lp); put(std::string(base) + ".gen.i8", gen);
        std::ofstream j(std::string(base) + ".json"); j << "{\"n\":" << n << ",\"taps\":" << taps << ",\"n_embd\":" << n_embd << ",\"topk\":8}\n";
        LOG_INF("shard %d written: %d tokens (total %ld)\n", idx, n, total);
        feat.clear(); scale.clear(); ids.clear(); doc.clear(); pos.clear(); top_ids.clear(); top_lp.clear(); gen.clear(); n = 0; ++idx;
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
    std::string corpus, out = "tap-shards"; std::vector<int> taps = {5, 19, 33, 47, 61}; long budget = 600000; int doc_max = 1536, per_shard = 20000; int start_doc = 0, start_shard = 0; int gen_max = 0, n_seqs = 8, prompt_max = 1024;
    std::vector<char *> rest; rest.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) { std::string a = argv[i];
        if (a == "--corpus" && i + 1 < argc) corpus = argv[++i]; else if (a == "--out" && i + 1 < argc) out = argv[++i]; else if (a == "--budget" && i + 1 < argc) budget = atol(argv[++i]);
        else if (a == "--doc-max" && i + 1 < argc) doc_max = atoi(argv[++i]); else if (a == "--start-doc" && i + 1 < argc) start_doc = atoi(argv[++i]); else if (a == "--start-shard" && i + 1 < argc) start_shard = atoi(argv[++i]); else if (a == "--per-shard" && i + 1 < argc) per_shard = atoi(argv[++i]);
        else if (a == "--gen" && i + 1 < argc) gen_max = atoi(argv[++i]); else if (a == "--seqs" && i + 1 < argc) n_seqs = atoi(argv[++i]); else if (a == "--prompt-max" && i + 1 < argc) prompt_max = atoi(argv[++i]);
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
    shard_writer w; w.dir = out; w.per_shard = per_shard; w.idx = start_shard; w.total = (long) start_shard * per_shard; w.taps = (int) taps.size(); std::string mk = "mkdir -p " + out; if (system(mk.c_str()) != 0) return 1;
    const int n_batch = params.n_batch;
    std::vector<int32_t> tid(8); std::vector<float> tlp(8); std::vector<std::pair<float,int>> cand;
    auto top8 = [&](const float * lg) {
        float mx = -1e30f; for (int v = 0; v < n_vocab; ++v) mx = std::max(mx, lg[v]);
        double sum = 0; for (int v = 0; v < n_vocab; ++v) sum += std::exp((double) (lg[v] - mx)); const float lse = mx + (float) std::log(sum);
        cand.clear(); cand.reserve(n_vocab); for (int v = 0; v < n_vocab; ++v) cand.emplace_back(lg[v], v);
        std::partial_sort(cand.begin(), cand.begin() + 8, cand.end(), [](auto & a, auto & b) { return a.first > b.first; });
        for (int k = 0; k < 8; ++k) { tid[k] = cand[k].second; tlp[k] = cand[k].first - lse; } };
    // ubatch row -> batch row, by matching the result_output fingerprint against llama_get_logits_ith (exact float equality)
    auto row_perm = [&](int n) {
        std::vector<int> perm(n, -1); std::vector<char> used(n, 0);
        if ((int) st.fp.size() != 8 * n) { LOG_ERR("fingerprint rows %zu != batch %d\n", st.fp.size() / 8, n); return std::vector<int>(); }
        for (int r = 0; r < n; ++r) for (int i = 0; i < n; ++i) if (!used[i] && memcmp(&st.fp[8 * r], llama_get_logits_ith(ctx, i), 32) == 0) { perm[r] = i; used[i] = 1; break; }
        for (int r = 0; r < n; ++r) if (perm[r] < 0) { LOG_ERR("could not align ubatch row %d to a batch row\n", r); return std::vector<int>(); }
        return perm; };
    if (gen_max > 0) {
        auto * mem = llama_get_memory(ctx); llama_memory_clear(mem, true);
        struct slot { int prompt = -1; std::vector<llama_token> toks; llama_token next = 0; int n_gen = 0; bool active = false; };
        std::vector<slot> slots(n_seqs); size_t next_prompt = (size_t) start_doc; long n_prompts = 0, n_gen_tokens = 0;
        auto prefill = [&](int si) -> bool {           // pull the next usable prompt into slot si; false when the corpus is exhausted
            while (next_prompt < docs.size()) {
                size_t d = next_prompt++; std::vector<llama_token> toks = common_tokenize(ctx, docs[d], false, true);
                if ((int) toks.size() > prompt_max) toks.resize(prompt_max); if (toks.size() < 8) continue;
                llama_memory_seq_rm(mem, si, -1, -1); slot & S = slots[si]; S = slot(); S.prompt = (int) d; S.toks = toks; S.active = true;
                for (int p0 = 0; p0 < (int) toks.size(); p0 += n_batch) {
                    int n = std::min(n_batch, (int) toks.size() - p0); llama_batch batch = llama_batch_init(n, 0, 1);
                    for (int i = 0; i < n; ++i) { batch.token[i] = toks[p0 + i]; batch.pos[i] = p0 + i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = si; batch.logits[i] = true; } batch.n_tokens = n;
                    st.reset(); if (llama_decode(ctx, batch) != 0) { LOG_ERR("prefill failed at prompt %zu pos %d\n", d, p0); llama_batch_free(batch); return false; }
                    if (st.cur_tokens != n) { LOG_ERR("callback saw %d tokens, batch had %d\n", st.cur_tokens, n); llama_batch_free(batch); return false; }
                    if (w.n_embd == 0) w.n_embd = st.n_embd;
                    for (int i = 0; i < n; ++i) { top8(llama_get_logits_ith(ctx, i)); w.add_token(&st.buf[(size_t) i * taps.size() * st.n_embd], toks[p0 + i], (int32_t) d, p0 + i, tid.data(), tlp.data(), 0); S.next = tid[0]; }
                    llama_batch_free(batch); }
                ++n_prompts; return true; }
            return false; };
        for (int si = 0; si < n_seqs; ++si) if (!prefill(si)) break;
        llama_batch batch = llama_batch_init(n_seqs, 0, 1);
        while (w.total < budget) {
            std::vector<int> rows; batch.n_tokens = 0;
            for (int si = 0; si < n_seqs; ++si) { slot & S = slots[si]; if (!S.active) continue; int i = batch.n_tokens++;
                batch.token[i] = S.next; batch.pos[i] = (int) S.toks.size(); batch.n_seq_id[i] = 1; batch.seq_id[i][0] = si; batch.logits[i] = true; rows.push_back(si); }
            if (batch.n_tokens == 0) break;
            st.reset(); if (llama_decode(ctx, batch) != 0) { LOG_ERR("decode step failed\n"); return 1; }
            if (st.cur_tokens != batch.n_tokens) { LOG_ERR("callback saw %d rows, batch had %d\n", st.cur_tokens, batch.n_tokens); return 1; }
            auto perm = row_perm(batch.n_tokens); if (perm.empty()) return 1;
            for (int r = 0; r < batch.n_tokens; ++r) { int i = perm[r]; slot & S = slots[rows[i]];
                top8(llama_get_logits_ith(ctx, i)); w.add_token(&st.buf[(size_t) r * taps.size() * st.n_embd], S.next, S.prompt, (int32_t) S.toks.size(), tid.data(), tlp.data(), 1);
                S.toks.push_back(S.next); S.next = tid[0]; ++S.n_gen; ++n_gen_tokens;
                if (llama_vocab_is_eog(vocab, S.next) || S.n_gen >= gen_max) { S.active = false; } }
            for (int si = 0; si < n_seqs; ++si) if (!slots[si].active && next_prompt < docs.size()) prefill(si);
            if (n_prompts % 8 == 0 && n_prompts > 0) { static long last = -1; if (last != n_prompts) { last = n_prompts;
                { std::ofstream pj(out + "/progress.json"); pj << "{\"next_doc\":" << next_prompt << ",\"next_shard\":" << w.idx << ",\"tokens\":" << w.total << "}\n"; }
                LOG_INF("prompt %ld (%zu/%zu), %ld tokens (%ld generated)\n", n_prompts, next_prompt, docs.size(), w.total, n_gen_tokens); } }
        }
        llama_batch_free(batch);
        w.flush(); { std::ofstream pj(out + "/progress.json"); pj << "{\"next_doc\":" << next_prompt << ",\"next_shard\":" << w.idx << ",\"tokens\":" << w.total << ",\"generated\":" << n_gen_tokens << (next_prompt >= docs.size() ? ",\"done\":true" : "") << "}\n"; }
        LOG_INF("done: %ld tokens (%ld generated) in %d shards\n", w.total, n_gen_tokens, w.idx);
        llama_perf_context_print(ctx); llama_backend_free(); return 0;
    }
    for (size_t d = (size_t) start_doc; d < docs.size() && w.total < budget; ++d) {
        std::vector<llama_token> toks = common_tokenize(ctx, docs[d], false, true); if ((int) toks.size() > doc_max) toks.resize(doc_max); if (toks.size() < 64) continue;
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int p0 = 0; p0 < (int) toks.size(); p0 += n_batch) {
            int n = std::min(n_batch, (int) toks.size() - p0);
            llama_batch batch = llama_batch_init(n, 0, 1);
            for (int i = 0; i < n; ++i) { batch.token[i] = toks[p0 + i]; batch.pos[i] = p0 + i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = true; } batch.n_tokens = n;
            st.reset();
            if (llama_decode(ctx, batch) != 0) { LOG_ERR("decode failed at doc %zu pos %d\n", d, p0); llama_batch_free(batch); return 1; }
            if (st.cur_tokens != n) { LOG_ERR("callback saw %d tokens, batch had %d (set -ub = -b)\n", st.cur_tokens, n); llama_batch_free(batch); return 1; }
            if (w.n_embd == 0) w.n_embd = st.n_embd;
            for (int i = 0; i < n; ++i) {
                top8(llama_get_logits_ith(ctx, i));
                w.add_token(&st.buf[(size_t) i * taps.size() * st.n_embd], toks[p0 + i], (int32_t) d, p0 + i, tid.data(), tlp.data());
            }
            llama_batch_free(batch);
            if (w.total >= budget) break;
        }
        { std::ofstream pj(out + "/progress.json"); pj << "{\"next_doc\":" << (d + 1) << ",\"next_shard\":" << w.idx << ",\"tokens\":" << w.total << "}\n"; }
        if (d % 20 == 0) LOG_INF("doc %zu/%zu, %ld tokens\n", d, docs.size(), w.total);
    }
    w.flush(); { std::ofstream pj(out + "/progress.json"); pj << "{\"next_doc\":" << docs.size() << ",\"next_shard\":" << w.idx << ",\"tokens\":" << w.total << ",\"done\":true}\n"; }
    LOG_INF("done: %ld tokens in %d shards\n", w.total, w.idx);
    llama_perf_context_print(ctx); llama_backend_free(); return 0;
}
