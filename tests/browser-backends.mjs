/**
 * Test all 3 backends (WebGPU, WebGL, WASM) with each model family.
 */
import puppeteer from 'puppeteer';

const URL = 'http://localhost:5174/test-backends.html';

async function main() {
  console.log('Launching Chrome...');
  const browser = await puppeteer.launch({
    headless: false,
    args: ['--enable-features=SharedArrayBuffer'],
  });

  const page = await browser.newPage();

  page.on('console', (msg) => console.log(`[BROWSER] ${msg.text()}`));
  page.on('pageerror', (err) => console.error(`[PAGE ERROR] ${err.message}`));

  await page.goto(URL, { waitUntil: 'networkidle0', timeout: 30000 });

  // Wait for all tests to complete
  console.log('Waiting for all backend tests to complete...');
  await page.waitForFunction(
    () => document.getElementById('done')?.textContent === 'true',
    { timeout: 300000 }
  );

  // Get results
  const results = await page.evaluate(() => window.__results);

  console.log('\n========== RESULTS ==========\n');

  let passed = 0, failed = 0;
  for (const r of results) {
    const status = r.error ? 'FAIL' : 'OK';
    if (r.error) failed++; else passed++;
    const info = r.error
      ? r.error
      : `${r.detections} detections in ${r.time}ms`;
    console.log(`[${status}] ${r.family} + ${r.backend}: ${info}`);
  }

  console.log(`\n${passed} passed, ${failed} failed out of ${results.length} tests`);

  await page.screenshot({
    path: '/Users/xuban.ceccon/Documents/GitHub/libreyolo-web/tests/backend-test-result.png',
    fullPage: true
  });

  await browser.close();
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => { console.error(err); process.exit(1); });
