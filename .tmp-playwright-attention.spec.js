const { test } = require('playwright/test');

test('inspect home', async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.goto('http://localhost:3000', { waitUntil: 'networkidle' });
  await page.screenshot({ path: '.tmp-live-home.png', fullPage: true });
  const buttons = await page.locator('button').evaluateAll((els) =>
    els.map((el) => (el.textContent || '').trim()).filter(Boolean)
  );
  const inputs = await page.locator('input, textarea, select').evaluateAll((els) =>
    els.map((el) => ({ tag: el.tagName, type: el.getAttribute('type'), placeholder: el.getAttribute('placeholder'), value: el.value }))
  );
  console.log(JSON.stringify({ url: page.url(), buttons, inputs }, null, 2));
});
