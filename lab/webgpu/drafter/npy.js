// Tiny .npy loader (v1/v2 headers, little-endian f4/i4/i8/f8, C order) for the oracle dumps.
export function parseNpy(ab) {
  const u8 = new Uint8Array(ab); if (String.fromCharCode(...u8.subarray(1, 6)) !== 'NUMPY') throw new Error('not npy');
  const major = u8[6]; const dv = new DataView(ab); const hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true); const hstart = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(u8.subarray(hstart, hstart + hlen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)[1]; const fortran = /'fortran_order':\s*(True|False)/.exec(header)[1] === 'True';
  const shape = (/'shape':\s*\(([^)]*)\)/.exec(header)[1]).split(',').map(s => s.trim()).filter(Boolean).map(Number);
  if (fortran) throw new Error('fortran order unsupported'); const off = hstart + hlen; const n = shape.reduce((a, b) => a * b, 1);
  let data;
  switch (descr) { case '<f4': data = new Float32Array(ab.slice(off, off + 4 * n)); break; case '<i4': data = new Int32Array(ab.slice(off, off + 4 * n)); break; case '<i8': { const b = new BigInt64Array(ab.slice(off, off + 8 * n)); data = Int32Array.from(b, Number); break; } case '<f8': { const b = new Float64Array(ab.slice(off, off + 8 * n)); data = Float32Array.from(b); break; } default: throw new Error('npy dtype ' + descr); }
  return { shape, data, descr };
}
export async function fetchNpy(url) { const r = await fetch(url); if (!r.ok) throw new Error(`${url} -> ${r.status}`); return parseNpy(await r.arrayBuffer()); }
