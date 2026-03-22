import puppeteer from 'puppeteer';

const URL = 'http://localhost:5174/test-multi-image.html';

async function main() {
  console.log('Launching Chrome...');
  const browser = await puppeteer.launch({
    headless: false,
    args: ['--enable-features=SharedArrayBuffer'],
  });

  const page = await browser.newPage();
  page.on('console', (msg) => console.log(`[BROWSER] ${msg.text()}`));
  page.on('pageerror', (err) => console.error(`[PAGE ERROR] ${err.message}`));

  await page.goto(URL, { waitUntil: 'networkidle0', timeout: 60000 });

  console.log('Waiting for all tests...');
  await page.waitForFunction(
    () => document.getElementById('done')?.textContent === 'true',
    { timeout: 300000 }
  );

  const results = await page.evaluate(() => window.__results);

  // Print results as a table
  console.log('\n' + '='.repeat(90));
  console.log('RESULTS: 3 models x 6 images');
  console.log('='.repeat(90));

  const models = ['yolox', 'yolo9', 'rfdetr'];
  const images = [...new Set(results.map(r => r.image))];

  // Header
  console.log(`${'Image'.padEnd(20)} ${'YOLOX Nano'.padEnd(22)} ${'YOLO9 Tiny'.padEnd(22)} ${'RF-DETR Nano'.padEnd(22)}`);
  console.log('-'.repeat(90));

  for (const img of images) {
    const cols = models.map(m => {
      const r = results.find(x => x.model === m && x.image === img);
      return r ? `${r.detections} dets ${r.time}ms` : '-';
    });
    console.log(`${img.padEnd(20)} ${cols.map(c => c.padEnd(22)).join(' ')}`);
  }

  console.log('-'.repeat(90));

  // Detailed breakdown
  console.log('\nDetailed breakdown:');
  for (const r of results) {
    console.log(`  ${r.model.padEnd(8)} + ${r.image.padEnd(18)} → ${r.detections} dets in ${r.time}ms (${r.summary})`);
  }

  // Screenshot
  const path = '/Users/xuban.ceccon/Documents/GitHub/libreyolo-web/tests/multi-image-result.png';
  await page.screenshot({ path, fullPage: true });
  console.log(`\nScreenshot: ${path}`);

  await browser.close();
}

main().catch(e => { console.error(e); process.exit(1); });
