// Drive the harness in headless Chrome over CDP (Node 22 built-in WebSocket). Usage: node cdp-drive.mjs <url> [timeoutSec]
import { spawn } from 'node:child_process';
const [url, tmo = '1800'] = process.argv.slice(2);
const CH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const chrome = spawn(CH, ['--headless=new', '--no-first-run', '--no-default-browser-check', `--user-data-dir=${process.cwd()}/chrome-profile`, '--enable-unsafe-webgpu', '--enable-features=WebGPU', '--remote-debugging-port=9333', '--window-size=1280,800', 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
let ws; try {
  for (let i = 0; i < 50; ++i) { try { await fetch('http://127.0.0.1:9333/json/version'); break; } catch { await sleep(200); } }
  const targets = await (await fetch('http://127.0.0.1:9333/json')).json(); const page = targets.find(t => t.type === 'page');
  ws = new WebSocket(page.webSocketDebuggerUrl); await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let id = 0; const pending = new Map();
  ws.onmessage = (ev) => { const m = JSON.parse(ev.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } else if (m.method === 'Runtime.consoleAPICalled') { const a = m.params.args.map(x => x.value ?? x.description).join(' '); if (/error|warn|fail|\[vs\]/i.test(a)) console.log('[console]', a.slice(0, 400)); } };
  const send = (method, params = {}) => new Promise(res => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
  await send('Runtime.enable'); await send('Page.enable');
  await send('Page.navigate', { url }); await sleep(1500);
  const evalJs = async (expr) => { const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true }); return r.result?.result?.value ?? r.result?.exceptionDetails?.text; };
  let last = ''; const t0 = Date.now();
  while ((Date.now() - t0) / 1000 < Number(tmo)) {
    const st = await evalJs('JSON.stringify({state: window.__vs && window.__vs.state, prog: window.__vs && window.__vs.prog, buildMs: window.__vs && window.__vs.buildMs})');
    if (st !== last) { console.log(new Date().toTimeString().slice(0, 8), st); last = st; }
    const s = JSON.parse(st || '{}'); if (s.state && (s.state.startsWith('done') || s.state === 'error')) break;
    await sleep(5000);
  }
  console.log('RESULTS', await evalJs('JSON.stringify({results: window.__vs.results, error: window.__vs.error && String(window.__vs.error).slice(0, 1200), log: window.__vs.log.map(l => l[1])}, null, 1)'));
} finally { try { ws?.close(); } catch {} chrome.kill('SIGTERM'); }
