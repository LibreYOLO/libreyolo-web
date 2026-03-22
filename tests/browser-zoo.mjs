import puppeteer from 'puppeteer';

const URL = 'http://localhost:5174/test-zoo.html';

async function main() {
  console.log('Launching Chrome (no local ONNX files — testing HF auto-download)...');
  const browser = await puppeteer.launch({
    headless: false,
    args: ['--enable-features=SharedArrayBuffer'],
  });

  const page = await browser.newPage();
  page.on('console', msg => console.log(`[BROWSER] ${msg.text()}`));
  page.on('pageerror', err => console.error(`[PAGE ERROR] ${err.message}`));

  await page.goto(URL, { waitUntil: 'networkidle0', timeout: 60000 });

  // Wait for "Done!" to appear in the log
  console.log('Waiting for models to download from HF and run inference...');
  await page.waitForFunction(
    () => document.getElementById('log')?.textContent?.includes('Done!'),
    { timeout: 300000 }
  );

  const logText = await page.$eval('#log', el => el.textContent);
  console.log('\n=== FULL LOG ===');
  console.log(logText);

  const path = '/Users/xuban.ceccon/Documents/GitHub/libreyolo-web/tests/zoo-test-result.png';
  await page.screenshot({ path, fullPage: true });
  console.log(`Screenshot: ${path}`);

  const failed = logText.includes('FAILED');
  await browser.close();
  process.exit(failed ? 1 : 0);
}

main().catch(e => { console.error(e); process.exit(1); });
