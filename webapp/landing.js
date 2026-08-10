// -- Local dev: point "Run the detector" at the live app, not the repo --
//
// This page is served two ways: as the public GitHub Pages site (no backend,
// so the button should send visitors to the repo's reproduce instructions --
// the HTML's default href) and, when `src.api.main` is run locally, as the
// same origin's "/" route with the interactive detector mounted at "/app"
// (see src/api/main.py). Only the second case gets the direct link, since
// only then does "/app" actually resolve to something.

const runDetectorLink = document.getElementById("run-detector");
if (runDetectorLink && (location.hostname === "localhost" || location.hostname === "127.0.0.1")) {
  runDetectorLink.href = "/app/";
  runDetectorLink.removeAttribute("target");
  runDetectorLink.removeAttribute("rel");
}

// -- Scroll reveal: sections fade/slide in as they enter view --
//
// Cards within a grid (method-grid, findings-grid) get an incremental
// transition-delay so they cascade in left-to-right rather than all
// snapping into place at once. Toggled both ways (not a one-shot reveal),
// so scrolling back up past a section un-reveals it and scrolling back down
// plays it again.

const fadeSelectors = ".section > h2, .section-lead, .method-card, .finding-card, .table-wrap, .cta-section .hero-actions";
// .text-highlight isn't in fadeSelectors -- it keeps its own sweep-in look
// (see styles.css) instead of the generic fade/slide, but still shares the
// same observer so both trigger together as their section scrolls into view.
const revealTargets = document.querySelectorAll(`${fadeSelectors}, .text-highlight`);
if (revealTargets.length && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  document.querySelectorAll(fadeSelectors).forEach((el) => {
    el.classList.add("reveal");
    const siblings = el.parentElement.children;
    if (siblings.length > 1 && (el.classList.contains("method-card") || el.classList.contains("finding-card"))) {
      el.style.transitionDelay = `${[...siblings].indexOf(el) * 90}ms`;
    }
  });
  const revealObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        entry.target.classList.toggle("is-revealed", entry.isIntersecting);
      }
    },
    { threshold: 0.15, rootMargin: "0px 0px -60px 0px" }
  );
  revealTargets.forEach((el) => revealObserver.observe(el));
}

// -- Section dot nav: highlight whichever section is in view --

const sectionDotLinks = document.querySelectorAll(".section-dots a");
if (sectionDotLinks.length) {
  const sectionsById = new Map(
    [...sectionDotLinks].map((link) => [link.getAttribute("href").slice(1), link])
  );
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        const link = sectionsById.get(entry.target.id);
        if (link) link.classList.toggle("is-active", entry.isIntersecting);
      }
    },
    { rootMargin: "-40% 0px -40% 0px" }
  );
  for (const id of sectionsById.keys()) {
    const section = document.getElementById(id);
    if (section) observer.observe(section);
  }
}

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
// Words are grouped into their own non-wrapping span (see runFlapboard), so
// the space character itself never becomes a tile and never reaches here --
// this is only ever the pre-flip blank state of a real letter's card.
const NBSP = "\u00A0";

function randomFlapChar() {
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
  // Every char but the last is a scramble step, tinted amber; the final
  // fold-in lands on the real character and drops back to the normal color.
  cardEl.classList.add("is-scrambling");
  for (let i = 0; i < sequence.length; i++) {
    const ch = sequence[i];
    const foldAway = cardEl.animate(
      [{ transform: "rotateX(0deg)" }, { transform: "rotateX(-90deg)" }],
      { duration: FLIP_STEP_MS, easing: "ease-in", fill: "forwards" }
    );
    if ((await settled(foldAway)) === null) return;
    cardEl.textContent = ch;
    if (i === sequence.length - 1) cardEl.classList.remove("is-scrambling");
    cardEl.getAnimations().forEach((a) => a.cancel());
    const foldIn = cardEl.animate(
      [{ transform: "rotateX(90deg)" }, { transform: "rotateX(0deg)" }],
      { duration: FLIP_STEP_MS, easing: "ease-out", fill: "forwards" }
    );
    if ((await settled(foldIn)) === null) return;
  }
}

function buildFlapUnits(el, text) {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  el.textContent = "";
  el.setAttribute("aria-label", text);

  // Each word gets its own non-wrapping span (see .flap-word in styles.css),
  // so the flapboard only ever wraps between words, never mid-word -- the
  // space between words is that wrapper's own gap, not a tile of its own.
  const units = [];
  for (const word of text.split(" ")) {
    const wordEl = document.createElement("span");
    wordEl.className = "flap-word";
    for (const ch of word) {
      const unit = document.createElement("span");
      unit.className = "flap-unit";
      const card = document.createElement("span");
      card.className = "flap-card";
      card.textContent = reduceMotion ? ch : NBSP;
      card.setAttribute("aria-hidden", "true");
      unit.appendChild(card);
      wordEl.appendChild(unit);
      units.push({ card, ch });
    }
    el.appendChild(wordEl);
  }
  return { units, reduceMotion };
}

// Runs the staggered left-to-right flip wave over an already-built unit
// list. Shared by the initial page-load reveal and by playFlapWave below
// (the hover-retrigger), so both look identical.
async function playFlapWave(units) {
  // Safety net, not a normal-path fallback: if Web Animations stalls for any
  // reason (a throttled background tab, a slow device, an engine that
  // doesn't advance .finished the way this assumes), the heading must never
  // sit blank -- force the real characters in after a generous bound.
  const settle = setTimeout(() => {
    units.forEach(({ card, ch }) => {
      card.getAnimations().forEach((a) => a.cancel());
      card.textContent = ch;
      card.classList.remove("is-scrambling");
    });
  }, 4000);

  const flips = units.map(({ card, ch }, i) => {
    const steps = 3 + Math.floor(Math.random() * 3);
    const sequence = Array.from({ length: steps }, () => randomFlapChar());
    sequence.push(ch);
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
  const { units: heroFlapUnits, reduceMotion: heroFlapReduceMotion } = buildFlapUnits(
    heroFlap,
    heroFlap.getAttribute("aria-label") || ""
  );

  if (!heroFlapReduceMotion) {
    let heroFlapIsPlaying = false;
    playFlapWave(heroFlapUnits).finally(() => {
      heroFlapIsPlaying = false;
    });
    heroFlapIsPlaying = true;

    // Re-plays the same left-to-right wave on hover, but ignores re-entries
    // while a wave is already in flight instead of queuing them up.
    heroFlap.addEventListener("mouseenter", () => {
      if (heroFlapIsPlaying) return;
      heroFlapIsPlaying = true;
      playFlapWave(heroFlapUnits).finally(() => {
        heroFlapIsPlaying = false;
      });
    });
  }
}

// -- Hero fade-out on scroll --
//
// Scroll-linked, not a triggered transition: opacity/translateY are set
// directly from scroll position every frame, so the hero dissolves in step
// with the scroll instead of animating on a timer.

const heroSection = document.getElementById("hero");
if (heroSection && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  const HERO_FADE_DISTANCE = 420;
  let heroFadeQueued = false;
  const updateHeroFade = () => {
    const progress = Math.min(Math.max(window.scrollY / HERO_FADE_DISTANCE, 0), 1);
    heroSection.style.opacity = String(1 - progress);
    heroSection.style.transform = `translateY(${progress * -32}px)`;
    heroFadeQueued = false;
  };
  window.addEventListener(
    "scroll",
    () => {
      if (!heroFadeQueued) {
        requestAnimationFrame(updateHeroFade);
        heroFadeQueued = true;
      }
    },
    { passive: true }
  );
  updateHeroFade();
}

// -- Custom cursor dot over the card-grid sections --

const cursorDot = document.getElementById("cursor-dot");
if (cursorDot && window.matchMedia("(hover: hover) and (pointer: fine)").matches && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  document.addEventListener("mousemove", (e) => {
    cursorDot.style.transform = `translate(${e.clientX}px, ${e.clientY}px) translate(-50%, -50%)`;
  });
  document.querySelectorAll(".method-grid, .findings-grid").forEach((grid) => {
    grid.addEventListener("mouseenter", () => cursorDot.classList.add("is-active"));
    grid.addEventListener("mouseleave", () => cursorDot.classList.remove("is-active"));
  });
}
