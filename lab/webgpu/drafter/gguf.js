// Minimal GGUF v3 reader + K-quant dequantizers (Q4_K, Q6_K, F16, F32) for the DFlash2 drafter port.
// Runs in the browser (fetch + Range) and in Node (fs) through a pluggable byte source: source(offset, length) -> Uint8Array.
// Layouts follow ggml-quants.c (block_q4_K: d f16, dmin f16, scales[12], qs[128]; block_q6_K: ql[128], qh[64], scales[16] i8, d f16).
export const GGML = { F32: 0, F16: 1, Q4_K: 12, Q6_K: 14 };
const TYPE_NAME = { 0: 'F32', 1: 'F16', 12: 'Q4_K', 14: 'Q6_K' };
const BLOCK = { 12: [256, 144], 14: [256, 210], 0: [1, 4], 1: [1, 2] };   // type -> [elements per block, bytes per block]
export function tensorBytes(type, n) { const [be, bb] = BLOCK[type]; if (n % be) throw new Error('n % block'); return (n / be) * bb; }
export function typeName(t) { return TYPE_NAME[t] ?? String(t); }

const f16buf = new ArrayBuffer(4), f16u = new Uint32Array(f16buf), f16f = new Float32Array(f16buf);
export function f16ToF32(h) {
  const s = (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if (e === 0) { if (m === 0) { f16u[0] = s; return f16f[0]; } let mm = m, ee = 113; while (!(mm & 0x400)) { mm <<= 1; ee--; } f16u[0] = s | (ee << 23) | ((mm & 0x3ff) << 13); return f16f[0]; }
  if (e === 31) { f16u[0] = s | 0x7f800000 | (m << 13); return f16f[0]; }
  f16u[0] = s | ((e + 112) << 23) | (m << 13); return f16f[0];
}
// f32 -> f16 bits (round to nearest even). Used when Float16Array is unavailable.
export function f32ToF16(v) {
  f16f[0] = v; const x = f16u[0]; const s = (x >>> 16) & 0x8000; let e = (x >>> 23) & 0xff; let m = x & 0x7fffff;
  if (e === 0xff) return s | 0x7c00 | (m ? 0x200 : 0);
  e = e - 127 + 15;
  if (e >= 0x1f) return s | 0x7c00;
  if (e <= 0) { if (e < -10) return s; m = (m | 0x800000) >> (1 - e); if (m & 0x1000) m += 0x2000; return s | (m >> 13); }
  let h = s | (e << 10) | (m >> 13); if (m & 0x1000) { if ((m & 0x2fff) !== 0x1000 || (h & 1)) h += 1; }   // RNE
  return h;
}
export function f32ArrayToF16(f32) {
  if (typeof Float16Array !== 'undefined') { const out = new Float16Array(f32.length); out.set(f32); return new Uint16Array(out.buffer); }
  const out = new Uint16Array(f32.length); for (let i = 0; i < f32.length; ++i) out[i] = f32ToF16(f32[i]); return out;
}

// ---- header ----
const VT = { UINT8: 0, INT8: 1, UINT16: 2, INT16: 3, UINT32: 4, INT32: 5, FLOAT32: 6, BOOL: 7, STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11, FLOAT64: 12 };
class Cursor {
  constructor(source) { this.source = source; this.pos = 0; this.buf = null; this.bufStart = 0; }
  async ensure(n) { if (this.buf && this.pos + n <= this.bufStart + this.buf.byteLength) return; const len = Math.max(n, 1 << 20); this.buf = await this.source(this.pos, len); this.bufStart = this.pos; this.dv = new DataView(this.buf.buffer, this.buf.byteOffset, this.buf.byteLength); }
  async u8() { await this.ensure(1); const v = this.dv.getUint8(this.pos - this.bufStart); this.pos += 1; return v; }
  async u16() { await this.ensure(2); const v = this.dv.getUint16(this.pos - this.bufStart, true); this.pos += 2; return v; }
  async u32() { await this.ensure(4); const v = this.dv.getUint32(this.pos - this.bufStart, true); this.pos += 4; return v; }
  async i32() { await this.ensure(4); const v = this.dv.getInt32(this.pos - this.bufStart, true); this.pos += 4; return v; }
  async f32() { await this.ensure(4); const v = this.dv.getFloat32(this.pos - this.bufStart, true); this.pos += 4; return v; }
  async u64() { await this.ensure(8); const v = Number(this.dv.getBigUint64(this.pos - this.bufStart, true)); this.pos += 8; return v; }
  async i64() { await this.ensure(8); const v = Number(this.dv.getBigInt64(this.pos - this.bufStart, true)); this.pos += 8; return v; }
  async f64() { await this.ensure(8); const v = this.dv.getFloat64(this.pos - this.bufStart, true); this.pos += 8; return v; }
  async str() { const n = await this.u64(); await this.ensure(n); const b = this.buf.subarray(this.pos - this.bufStart, this.pos - this.bufStart + n); this.pos += n; return new TextDecoder().decode(b); }
  async value(t, skipStrings) {
    switch (t) {
      case VT.UINT8: case VT.BOOL: return this.u8(); case VT.INT8: { const v = await this.u8(); return v > 127 ? v - 256 : v; }
      case VT.UINT16: return this.u16(); case VT.INT16: { const v = await this.u16(); return v > 32767 ? v - 65536 : v; }
      case VT.UINT32: return this.u32(); case VT.INT32: return this.i32(); case VT.FLOAT32: return this.f32();
      case VT.UINT64: return this.u64(); case VT.INT64: return this.i64(); case VT.FLOAT64: return this.f64();
      case VT.STRING: return this.str();
      case VT.ARRAY: { const st = await this.u32(); const n = await this.u64(); const out = []; for (let i = 0; i < n; ++i) { const v = await this.value(st, skipStrings); if (!(skipStrings && st === VT.STRING) && n <= 4096) out.push(v); } return n > 4096 ? { array_length: n, elem_type: st } : out; }
      default: throw new Error('gguf: bad value type ' + t);
    }
  }
}
export async function readGGUF(source) {
  const c = new Cursor(source);
  const magic = await c.u32(); if (magic !== 0x46554747) throw new Error('not GGUF');
  const version = await c.u32(); if (version !== 3 && version !== 2) throw new Error('gguf version ' + version);
  const nTensors = await c.u64(), nKv = await c.u64();
  const kv = {};
  for (let i = 0; i < nKv; ++i) { const k = await c.str(); const t = await c.u32(); kv[k] = await c.value(t, true); }
  const tensors = [];
  for (let i = 0; i < nTensors; ++i) {
    const name = await c.str(); const nd = await c.u32(); const ne = []; for (let d = 0; d < nd; ++d) ne.push(await c.u64());
    const type = await c.u32(); const offset = await c.u64();
    const n = ne.reduce((a, b) => a * b, 1);
    tensors.push({ name, ne, type, typeName: typeName(type), offset, n, bytes: tensorBytes(type, n) });
  }
  const alignment = kv['general.alignment'] ?? 32;
  const dataStart = Math.ceil(c.pos / alignment) * alignment;
  for (const t of tensors) t.absOffset = dataStart + t.offset;
  const byName = Object.fromEntries(tensors.map(t => [t.name, t]));
  return { version, kv, tensors, byName, dataStart };
}

// ---- dequantizers: raw block bytes -> Float32Array (n elements) ----
function scaleMinK4(j, sc) {   // get_scale_min_k4
  if (j < 4) return [sc[j] & 63, sc[j + 4] & 63];
  return [(sc[j + 4] & 0xf) | ((sc[j - 4] >> 6) << 4), (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)];
}
export function dequantQ4K(raw, n, out, outOff = 0) {
  const nb = n / 256; const sc = new Uint8Array(8);   // scratch not needed; index raw directly
  for (let b = 0; b < nb; ++b) {
    const p = b * 144; const d = f16ToF32(raw[p] | (raw[p + 1] << 8)), dmin = f16ToF32(raw[p + 2] | (raw[p + 3] << 8));
    const scales = raw.subarray(p + 4, p + 16); let q = p + 16; let y = outOff + b * 256; let is = 0;
    for (let j = 0; j < 256; j += 64) {
      const [s1, m1] = scaleMinK4(is, scales), [s2, m2] = scaleMinK4(is + 1, scales);
      const d1 = d * s1, mm1 = dmin * m1, d2 = d * s2, mm2 = dmin * m2;
      for (let l = 0; l < 32; ++l) out[y + l] = d1 * (raw[q + l] & 0xf) - mm1;
      for (let l = 0; l < 32; ++l) out[y + 32 + l] = d2 * (raw[q + l] >> 4) - mm2;
      y += 64; q += 32; is += 2;
    }
  }
  return out;
}
export function dequantQ6K(raw, n, out, outOff = 0) {
  const nb = n / 256;
  for (let b = 0; b < nb; ++b) {
    const p = b * 210; const d = f16ToF32(raw[p + 208] | (raw[p + 209] << 8));
    let ql = p, qh = p + 128, sc = p + 192, y = outOff + b * 256;
    for (let half = 0; half < 2; ++half) {
      for (let l = 0; l < 32; ++l) {
        const is = (l / 16) | 0;
        const s0 = (raw[sc + is] << 24) >> 24, s2 = (raw[sc + is + 2] << 24) >> 24, s4 = (raw[sc + is + 4] << 24) >> 24, s6 = (raw[sc + is + 6] << 24) >> 24;
        const q1 = ((raw[ql + l] & 0xf) | (((raw[qh + l] >> 0) & 3) << 4)) - 32;
        const q2 = ((raw[ql + l + 32] & 0xf) | (((raw[qh + l] >> 2) & 3) << 4)) - 32;
        const q3 = ((raw[ql + l] >> 4) | (((raw[qh + l] >> 4) & 3) << 4)) - 32;
        const q4 = ((raw[ql + l + 32] >> 4) | (((raw[qh + l] >> 6) & 3) << 4)) - 32;
        out[y + l] = d * s0 * q1; out[y + l + 32] = d * s2 * q2; out[y + l + 64] = d * s4 * q3; out[y + l + 96] = d * s6 * q4;
      }
      y += 128; ql += 64; qh += 32; sc += 8;
    }
  }
  return out;
}
export function dequantF16(raw, n, out, outOff = 0) { for (let i = 0; i < n; ++i) out[outOff + i] = f16ToF32(raw[2 * i] | (raw[2 * i + 1] << 8)); return out; }
export function dequantF32(raw, n, out, outOff = 0) { out.set(new Float32Array(raw.buffer, raw.byteOffset, n), outOff); return out; }
export function dequantize(type, raw, n, out = new Float32Array(n), outOff = 0) {
  switch (type) { case GGML.Q4_K: return dequantQ4K(raw, n, out, outOff); case GGML.Q6_K: return dequantQ6K(raw, n, out, outOff); case GGML.F16: return dequantF16(raw, n, out, outOff); case GGML.F32: return dequantF32(raw, n, out, outOff); default: throw new Error('dequantize: unsupported type ' + typeName(type)); }
}
// Dequantize one row of a 2-D tensor whose row length is a multiple of the block size (used for codebook row lookups).
export function dequantRow(type, rawTensorBytes, rowLen, row, out = new Float32Array(rowLen)) {
  const [be, bb] = BLOCK[type]; const blocksPerRow = rowLen / be; const p = row * blocksPerRow * bb;
  return dequantize(type, rawTensorBytes.subarray(p, p + blocksPerRow * bb), rowLen, out, 0);
}
// Byte sources
export function fsSource(fd, fs) { return (off, len) => { const b = new Uint8Array(len); const n = fs.readSync(fd, b, 0, len, off); return Promise.resolve(n === len ? b : b.subarray(0, n)); }; }
export function fetchSource(url) { return async (off, len) => { const r = await fetch(url, { headers: { Range: `bytes=${off}-${off + len - 1}` } }); if (r.status !== 206) throw new Error(`range fetch ${url} -> ${r.status} (server must support Range)`); return new Uint8Array(await r.arrayBuffer()); }; }
