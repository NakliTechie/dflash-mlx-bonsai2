// Headless-Chrome CDP driver for the spike pages. Usage: node cdp-drive.mjs <url> [timeoutSec] [global=__gs]
// Polls window.<global>.state, prints changes, dumps window.<global>.results when state starts with "done" or is "error".
import { spawn } from 'node:child_process';
const [url, tmo = '1800', G = '__gs'] = process.argv.slice(2);
const CH = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
// Reuse the engine dir's chrome-profile (its IndexedDB holds the 5.6 GB weight cache); override with SPIKE_PROFILE.
const profile = process.env.SPIKE_PROFILE || '/private/tmp/claude-501/-Users-chiragpatnaik-Code-naklios-universe-LocalMind/4a85d733-7706-45bd-bd5e-98f4de49b3f7/scratchpad/engine/chrome-profile';
const port = 9340 + Math.floor(Math.random() * 50);
const chrome = spawn(CH, ['--headless=new', '--no-first-run', '--no-default-browser-check', `--user-data-dir=${profile}`, '--enable-unsafe-webgpu', '--enable-features=WebGPU', '--enable-dawn-features=allow_unsafe_apis', `--remote-debugging-port=${port}`, '--window-size=1280,800', 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
let ws; try {
  for (let i = 0; i < 50; ++i) { try { await fetch(`http://127.0.0.1:${port}/json/version`); break; } catch { await sleep(200); } }
  const targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); const page = targets.find(t => t.type === 'page');
  ws = new WebSocket(page.webSocketDebuggerUrl); await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
  let id = 0; const pending = new Map();
  ws.onmessage = (ev) => { const m = JSON.parse(ev.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } else if (m.method === 'Runtime.consoleAPICalled') { const a = m.params.args.map(x => x.value ?? x.description).join(' '); console.log('[console]', a.slice(0, 600)); } else if (m.method === 'Runtime.exceptionThrown') { pageThrew = true; console.log('[exception]', JSON.stringify(m.params.exceptionDetails).slice(0, 600)); } };
  const send = (method, params = {}) => new Promise(res => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
  await send('Runtime.enable'); await send('Page.enable');
  await send('Page.navigate', { url }); await sleep(1500);
  const evalJs = async (expr) => { const r = await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true }); return r.result?.result?.value ?? JSON.stringify(r.result?.exceptionDetails); };
  let last = ''; const t0 = Date.now(); let pageThrew = false;
  while ((Date.now() - t0) / 1000 < Number(tmo)) {
    const st = await evalJs(`JSON.stringify({state: window.${G} && window.${G}.state, prog: window.${G} && window.${G}.prog})`);
    if (st !== last) { console.log(new Date().toTimeString().slice(0, 8), st); last = st; }
    let s = {}; try { s = JSON.parse(st || '{}'); } catch {}
    if (s.state && (s.state.startsWith('done') || s.state === 'error')) break;
    if (pageThrew && !s.state) break;
    await sleep(2000);
  }
  console.log('RESULTS', await evalJs(`JSON.stringify(window.${G} ? {results: window.${G}.results, error: window.${G}.error && String(window.${G}.error).slice(0, 2000), log: window.${G}.log} : {error: 'global ${G} never defined (page threw at load?)'}, null, 1)`));
} finally { try { ws?.close(); } catch {} chrome.kill('SIGTERM'); }
