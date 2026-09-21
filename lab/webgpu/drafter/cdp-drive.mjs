// Drive harness.html in headless Chrome over CDP (Node 22 WebSocket). Usage: node cdp-drive.mjs <url> [timeoutSec]
// Pattern: lab/webgpu/cdp-drive.mjs. Prints window.__dr state transitions, then RESULTS json.
import { spawn } from 'node:child_process';
const [url, tmo = '1800'] = process.argv.slice(2);
const CH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const PORT = 9337;
const chrome = spawn(CH, ['--headless=new', '--no-first-run', '--no-default-browser-check', `--user-data-dir=${process.cwd()}/chrome-profile-drafter`, '--enable-unsafe-webgpu', '--enable-features=WebGPU', `--remote-debugging-port=${PORT}`, '--window-size=1280,800', 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const kill = () => { try { chrome.kill('SIGTERM'); } catch {} };
for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) process.on(sig, () => { kill(); process.exit(130); });
process.on('exit', kill);
let ws; try {
  for (let i = 0; i < 50; ++i) { try { await fetch(`http://127.0.0.1:${PORT}/json/version`); break; } catch { await sleep(200); } }
  const targets = await (await fetch(`http://127.0.0.1:${PORT}/json`)).json(); const page = targets.find(t => t.type === 'page');
  ws = new WebSocket(page.webSocketDebuggerUrl); await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let id = 0; const pending = new Map();
  ws.onmessage = (ev) => { const m = JSON.parse(ev.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } else if (m.method === 'Runtime.consoleAPICalled') { const a = m.params.args.map(x => x.value ?? x.description).join(' '); console.log('[console]', a.slice(0, 400)); } else if (m.method === 'Runtime.exceptionThrown') { console.log('[exception]', JSON.stringify(m.params.exceptionDetails).slice(0, 600)); } };
  const send = (method, params = {}) => new Promise(res => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
  await send('Runtime.enable'); await send('Page.enable');
  await send('Page.navigate', { url }); await sleep(1500);
  const evalJs = async (expr) => { const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true }); return r.result?.result?.value ?? r.result?.exceptionDetails?.text; };
  let last = ''; const t0 = Date.now();
  while ((Date.now() - t0) / 1000 < Number(tmo)) {
    const st = await evalJs('JSON.stringify({state: window.__dr && window.__dr.state, prog: window.__dr && window.__dr.prog && {done: window.__dr.prog.done, total: window.__dr.prog.total, ms: Math.round(window.__dr.prog.ms)}})');
    if (st !== last) { console.log(new Date().toTimeString().slice(0, 8), st); last = st; }
    const s = JSON.parse(st || '{}'); if (s.state === 'done' || s.state === 'error') break;
    await sleep(3000);
  }
  console.log('RESULTS', await evalJs('JSON.stringify({gpu: window.__dr.gpu, oracle: window.__dr.oracle, weights: window.__dr.weights, timing: window.__dr.timing, results: window.__dr.results, error: window.__dr.error})'));
} finally { try { ws?.close(); } catch {} chrome.kill('SIGTERM'); }
