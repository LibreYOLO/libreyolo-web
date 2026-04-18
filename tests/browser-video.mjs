/**
 * Browser regression test: predict(HTMLVideoElement) must return detections
 * matching predict(HTMLImageElement) for the same frame. Guards against the
 * "silent zero detections on video" bug where PreProcessor read image.width
 * (0 for video) instead of videoWidth.
 *
 * Spawns the vite test server itself, so this runs as a single command.
 */
import puppeteer from 'puppeteer';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const PORT = 5175;
const URL = `http://localhost:${PORT}/test-video.html`;
const TIMEOUT = 180_000;

function startVite() {
  const proc = spawn(
    'npx',
    ['vite', '-c', 'vite.config.test.ts', '--port', String(PORT), '--strictPort'],
    { cwd: REPO_ROOT, stdio: ['ignore', 'pipe', 'pipe'] }
  );
  proc.stdout.on('data', (b) => process.stdout.write(`[vite] ${b}`));
  proc.stderr.on('data', (b) => process.stderr.write(`[vite:err] ${b}`));

  return new Promise((resolve, reject) => {
    const onData = (buf) => {
      const s = buf.toString();
      if (s.includes(`localhost:${PORT}`) || s.includes('ready in')) {
        proc.stdout.off('data', onData);
        resolve(proc);
      }
    };
    proc.stdout.on('data', onData);
    proc.once('error', reject);
    proc.once('exit', (code) => reject(new Error(`vite exited early with code ${code}`)));
    setTimeout(() => reject(new Error('vite did not start in time')), 30_000);
  });
}

async function main() {
  console.log('Starting vite...');
  const vite = await startVite();
  // Give vite a moment after first-ready log to bind the port cleanly.
  await new Promise((r) => setTimeout(r, 500));

  console.log('Launching Chrome...');
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--autoplay-policy=no-user-gesture-required', '--enable-features=SharedArrayBuffer'],
  });

  let exitCode = 1;
  try {
    const page = await browser.newPage();
    page.on('console', (msg) => console.log(`[browser] ${msg.text()}`));
    page.on('pageerror', (err) => console.error(`[pageerror] ${err.message}`));

    console.log(`Navigating to ${URL}...`);
    await page.goto(URL, { waitUntil: 'networkidle0', timeout: 60_000 });

    await page.waitForFunction(
      () => document.getElementById('done')?.textContent === 'true',
      { timeout: TIMEOUT }
    );

    const results = await page.evaluate(() => window.__results);
    console.log('\n=== RESULTS ===');
    console.log(JSON.stringify(results, null, 2));

    const checks = [
      [results.pass === true, 'overall pass flag'],
      [results.video > 0, `video detections > 0 (got ${results.video})`],
      [
        Math.abs((results.video ?? 0) - (results.image ?? 0)) <= 1,
        `video (${results.video}) within ±1 of image (${results.image})`,
      ],
      [results.zeroDimThrew === true, 'predict on un-initialized video throws zero-dim error'],
    ];

    let allOk = true;
    for (const [ok, label] of checks) {
      console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}`);
      if (!ok) allOk = false;
    }

    exitCode = allOk ? 0 : 1;
  } finally {
    await browser.close().catch(() => {});
    vite.kill('SIGTERM');
    await once(vite, 'exit').catch(() => {});
  }

  process.exit(exitCode);
}

main().catch((err) => {
  console.error('Test runner failed:', err);
  process.exit(1);
});
