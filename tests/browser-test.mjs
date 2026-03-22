/**
 * Automated browser test: opens Chrome, loads all 3 model families,
 * runs inference, and captures results from the console.
 */
import puppeteer from 'puppeteer';

const URL = 'http://localhost:5174/test-families.html';
const TIMEOUT = 180_000; // 3 min for RF-DETR (109MB)

async function main() {
  console.log('Launching Chrome...');
  const browser = await puppeteer.launch({
    headless: false,
    args: [
      '--enable-features=SharedArrayBuffer',
    ],
  });

  const page = await browser.newPage();

  // Capture all console logs
  const logs = [];
  page.on('console', (msg) => {
    const text = msg.text();
    logs.push(text);
    console.log(`[BROWSER] ${text}`);
  });

  page.on('pageerror', (err) => {
    console.error(`[PAGE ERROR] ${err.message}`);
  });

  console.log(`Navigating to ${URL}...`);
  await page.goto(URL, { waitUntil: 'networkidle0', timeout: 30_000 });

  // Wait for all models to be ready (button text changes from "Loading models...")
  console.log('Waiting for models to load...');
  await page.waitForFunction(
    () => {
      const btn = document.getElementById('run-all');
      return btn && !btn.disabled && btn.textContent.includes('Run All');
    },
    { timeout: TIMEOUT }
  );

  // Check status of each model
  for (const id of ['yolox', 'yolo9', 'rfdetr']) {
    const status = await page.$eval(`#status-${id}`, el => el.textContent);
    console.log(`\n[${id.toUpperCase()}] ${status}`);
  }

  // Click "Run All Detections"
  console.log('\n=== Running inference on all models ===\n');
  await page.click('#run-all');

  // Wait for all results (button re-enables with "Run Again")
  await page.waitForFunction(
    () => {
      const btn = document.getElementById('run-all');
      return btn && !btn.disabled && btn.textContent.includes('Run Again');
    },
    { timeout: TIMEOUT }
  );

  // Collect results
  console.log('\n=== RESULTS ===\n');
  for (const id of ['yolox', 'yolo9', 'rfdetr']) {
    const status = await page.$eval(`#status-${id}`, el => el.textContent);
    const dets = await page.$eval(`#dets-${id}`, el => el.textContent);
    const isError = await page.$eval(`#status-${id}`, el => el.classList.contains('error'));

    console.log(`--- ${id.toUpperCase()} ---`);
    console.log(`Status: ${status}`);
    if (dets) console.log(`Detections:\n${dets}`);
    if (isError) console.log('*** FAILED ***');
    console.log('');
  }

  // Get summary
  const summary = await page.$eval('#summary', el => el.textContent);
  console.log('=== SUMMARY ===');
  console.log(summary);

  // Take a screenshot
  const screenshotPath = '/Users/xuban.ceccon/Documents/GitHub/libreyolo-web/tests/browser-test-result.png';
  await page.screenshot({ path: screenshotPath, fullPage: true });
  console.log(`\nScreenshot saved to: ${screenshotPath}`);

  await browser.close();
  console.log('\nDone!');
}

main().catch((err) => {
  console.error('Test failed:', err);
  process.exit(1);
});
