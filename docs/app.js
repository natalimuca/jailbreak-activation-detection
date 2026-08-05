// -- Theme toggle (same pattern as webapp/app.js) --

const themeToggle = document.getElementById("theme-toggle");
const iconSun = themeToggle.querySelector(".icon-sun");
const iconMoon = themeToggle.querySelector(".icon-moon");

function effectiveTheme() {
  return document.documentElement.dataset.theme || (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
}

function syncThemeIcon() {
  const isLight = effectiveTheme() === "light";
  iconSun.toggleAttribute("hidden", !isLight);
  iconMoon.toggleAttribute("hidden", isLight);
  themeToggle.setAttribute("aria-label", isLight ? "Switch to dark mode" : "Switch to light mode");
}

themeToggle.addEventListener("click", () => {
  const next = effectiveTheme() === "light" ? "dark" : "light";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("theme", next);
  syncThemeIcon();
});

syncThemeIcon();

// -- Split-flap hero heading --
//
// Each character gets its own flap-card element. A "flip" is two chained
// Web Animations: rotate 0deg -> -90deg (the card folds away from the
// viewer; backface-visibility:hidden makes it disappear at the midpoint),
// swap the character, then snap to 90deg and rotate 90deg -> 0deg (folds
// back in with the new glyph). Several flips per character, staggered by
// position, land on the real text left-to-right like a real split-flap sign.

const FLAP_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const FLIP_STEP_MS = 55;
// Explicit escape, not a literal space in source: a bare " " as the sole
// content of an inline-block can render at collapsed/zero width in some
// engines, and a literal NBSP byte in a source file is invisible and has
// already caused one silent string-match failure while editing this file.
const NBSP = "\u00A0";

function randomFlapChar(finalChar) {
  if (finalChar === " ") return NBSP;
  return FLAP_CHARSET[Math.floor(Math.random() * FLAP_CHARSET.length)];
}

// The 4s safety-net in runFlapboard force-cancels any animation still
// in flight, which rejects its `.finished` promise with AbortError -- an
// expected outcome of that cancellation, not a real failure. Swallowing it
// here (and only it) is what keeps a slow device or backgrounded tab from
// producing a page full of uncaught-rejection console errors.
function settled(animation) {
  return animation.finished.catch((err) => {
    if (err && err.name === "AbortError") return null;
    throw err;
  });
}

async function flipUnit(cardEl, sequence) {
  for (const ch of sequence) {
    const foldAway = cardEl.animate(
      [{ transform: "rotateX(0deg)" }, { transform: "rotateX(-90deg)" }],
      { duration: FLIP_STEP_MS, easing: "ease-in", fill: "forwards" }
    );
    if ((await settled(foldAway)) === null) return;
    cardEl.textContent = ch;
    cardEl.getAnimations().forEach((a) => a.cancel());
    const foldIn = cardEl.animate(
      [{ transform: "rotateX(90deg)" }, { transform: "rotateX(0deg)" }],
      { duration: FLIP_STEP_MS, easing: "ease-out", fill: "forwards" }
    );
    if ((await settled(foldIn)) === null) return;
  }
}

async function runFlapboard(el, text) {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const chars = [...text];

  const units = chars.map((ch) => {
    const unit = document.createElement("span");
    unit.className = "flap-unit";
    const card = document.createElement("span");
    card.className = "flap-card";
    card.textContent = reduceMotion ? ch : NBSP;
    card.setAttribute("aria-hidden", "true");
    unit.appendChild(card);
    return { unit, card };
  });

  el.textContent = "";
  el.setAttribute("aria-label", text);
  units.forEach(({ unit }) => el.appendChild(unit));

  if (reduceMotion) return;

  // Safety net, not a normal-path fallback: if Web Animations stalls for any
  // reason (a throttled background tab, a slow device, an engine that
  // doesn't advance .finished the way this assumes), the heading must never
  // sit blank -- force the real characters in after a generous bound.
  const settle = setTimeout(() => {
    units.forEach(({ card }, i) => {
      card.getAnimations().forEach((a) => a.cancel());
      card.textContent = chars[i];
    });
  }, 4000);

  const flips = units.map(({ card }, i) => {
    const finalChar = chars[i];
    const steps = 3 + Math.floor(Math.random() * 3);
    const sequence = Array.from({ length: steps }, () => randomFlapChar(finalChar));
    sequence.push(finalChar);
    return { card, sequence, delay: i * 34 };
  });

  await Promise.all(
    flips.map(
      ({ card, sequence, delay }) =>
        new Promise((resolve) => {
          setTimeout(() => flipUnit(card, sequence).then(resolve), delay);
        })
    )
  );
  clearTimeout(settle);
}

const heroFlap = document.getElementById("hero-flap");
if (heroFlap) {
  runFlapboard(heroFlap, heroFlap.getAttribute("aria-label") || "");
}
