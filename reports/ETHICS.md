# Ethics and safety handling

How this project handles harmful-instruction data, where that data comes from,
what leaves this machine, and what is deliberately not released.

## 1. What this project is

I am researching whether a jailbreak / harmful-intent prompt detector built
on an LLM's internal activations is more robust than surface-level (keyword
or perplexity-based) filtering. Surface filters can be evaded by disguising a
harmful request as fiction, roleplay, or a hypothetical -- an
activation-based detector, reading the model's internal state rather than the
prompt's surface wording, is intended to be harder to evade this way.

This is **defensive security research**. The goal is a better detector, not a
new attack technique. I am not developing, publishing, or disclosing any
novel jailbreak method as part of this work -- every adversarial technique I
use (PAIR, GCG) is an already-published attack I am reusing exactly as its
original authors intended, purely to test detector robustness.

## 2. Justification and expected contribution

The approach builds on a real, published finding (Arditi et al., NeurIPS
2024, "Refusal in Language Models Is Mediated by a Single Direction") showing
that refusal behaviour in open-weight chat models is controlled by a single
activation-space direction. My contribution is extending this into an
evaluated, calibrated detector -- with cross-model generalisation testing,
adversarial-robustness testing, and statistical significance testing -- which
the literature I surveyed had not yet done rigorously at the time I started
(see LITERATURE.md).

## 3. Data sources

All harmful-instruction prompts used in this project come from four
established, publicly published AI-safety benchmarks, used exactly as their
authors intend and cited throughout my write-up:

- AdvBench (Zou et al. 2023, github.com/llm-attacks/llm-attacks)
- HarmBench (Mazeika et al. 2024, github.com/centerforaisafety/HarmBench)
- JailbreakBench / JBB-Behaviors (Chao et al. 2024, NeurIPS Datasets and Benchmarks track)
- XSTest (Röttger et al. 2023), used for the over-refusal / false-positive side of evaluation

I have not authored, sourced, or scraped any harmful content from elsewhere.
Every harmful prompt in this project already exists in a public, citable
safety-research benchmark that is itself used by numerous other published
papers in this literature.

## 4. Methodology that generates harmful content

Testing whether a candidate activation direction *causally* controls refusal
(rather than merely correlating with it) requires running the benchmark
prompts through open-weight models with the refusal mechanism deliberately
disabled, and observing whether the model complies. This step does produce
real harmful completions, generated and stored temporarily on my own local
machine during experiments.

**How I handle this:**
- Raw model completions are never committed to the project's git repository
  or shared anywhere (`results/` is gitignored).
- Only aggregate statistics (refusal rates and confidence intervals, no
  actual harmful text) are recorded in the project's version-controlled
  results documentation.
- Generated harmful completions are not used for any purpose beyond
  confirming the causal effect exists for this research; they are not
  published, distributed, or repurposed.

## 5. Models used

All models are open-weight, instruction-tuned checkpoints already publicly
released and downloadable by anyone (models from the Qwen, Llama, Gemma,
SmolLM2, and DeepSeek families -- see DECISIONS.md for the full list and the
reasoning behind each addition). This research does not create any new
capability: these models' ability to produce harmful content once their
safety training is bypassed is already a known, publicly documented property
of each model, and is precisely the property the underlying published
research (Arditi et al. 2024) demonstrates and that I am building on.

## 6. Third-party API use

One component of this project sends data off my machine, and I want it stated
explicitly rather than left implicit in the "local" framing above.

To test whether activation-based detection outperforms *surface-level* text
analysis, I needed a strong text-only comparison rather than only the weak
baselines (a keyword lexicon and a perplexity filter). That comparison is a
frontier-scale model prompted as a classifier: **Llama-3.3-70B, accessed
through Groq's free API tier**, because a 70B model cannot run on my hardware.

**What is transmitted:** the prompt text being classified, and nothing else.
Every such prompt is an item already published in a public benchmark
(AdvBench, HarmBench, JailbreakBench/JBB-Behaviors, XSTest) or a published
adversarial artifact (PAIR and GCG attack strings from JailbreakBench). No
novel harmful text is authored or sent.

**What is never transmitted:** model activations, model-generated harmful
completions, any personal data, and any content originating from me rather
than from a published benchmark.

**What is stored locally:** only a cache mapping a SHA-256 hash of each
(model, prompt) pair to a numeric score (`results/llm_judge_cache.json`,
gitignored). The cache holds no prompt text.

**Provider terms, checked rather than assumed (verified 2026-08-04).** Groq's
Services Agreement defines Inputs (prompts) and Outputs as Customer Data, and
states that Groq "is not permitted to use Inputs or Outputs for training or
fine-tuning any AI Model Services or other models" without the customer's
explicit instruction. It commits to deleting Customer Data within 30 days of
termination, and offers a zero-data-retention setting to eligible customers.
The public Privacy Policy does not itself cover API request data, and draws no
free-tier/paid-tier distinction; the Services Agreement is the governing
document. Source: console.groq.com/docs/legal/services-agreement.

**Residual risk and my assessment.** The material question is whether sending
these prompts to a commercial provider creates harm. I assess it as low: the
prompts are already public, published precisely so that safety researchers can
evaluate against them; the request asks the provider's model to *rate* them
rather than to comply with them; and the provider is contractually barred from
training on them. Two caveats I state rather than gloss. First, I have not
confirmed whether a free-tier account qualifies as an "eligible customer" for
the zero-retention setting, so I assume prompts are retained during normal
operation. Second, the agreement contemplates provider access to Inputs for
operational reliability and acceptable-use enforcement, which means Groq's
abuse monitoring sees a stream of jailbreak prompts from my account; that is
expected for this kind of evaluation but could be flagged automatically even
though the purpose is defensive.

**Avoidability.** This component is a comparison baseline, not part of the
detector itself. The activation-based detectors, which are the actual
contribution, run entirely locally and send nothing anywhere. The judge
baseline can be omitted from a reproduction without affecting them.

## 7. Risk assessment

I assess this project as **low risk**:

- It uses only established public benchmarks and already-public open-weight
  models -- no new harmful capability or novel attack is created.
- No content generated during this research is intended for, or released
  for, any use outside verifying the detector's own effectiveness.
- The end goal is protective (a better jailbreak detector), not harmful.

Three points are worth stating explicitly rather than leaving implicit:

1. **Local storage of model-generated harmful text during experiments.** I
   mitigate this via `.gitignore` (never committed) and by not sharing raw
   completions with anyone; only aggregate statistics leave my machine.
2. **Eventual publication of a working detector's methodology.** This is
   standard practice in this research area -- every paper I surveyed in
   LITERATURE.md publishes comparable methodology -- but I want this
   explicitly acknowledged as part of my ethics review, not assumed.
3. **Sending published harmful prompts to a third-party API.** The LLM-judge
   baseline (section 6) transmits public benchmark prompts to Groq for
   scoring. I judge this low risk: the prompts are already public, the model
   is asked to rate rather than comply, and Groq's Services Agreement bars
   training on submitted Inputs. It remains the one part of this project where
   data leaves my machine, so I am flagging it for review rather than
   absorbing it into the "local processing" description.

## 8. Disclosure and release

This repository documents a defensive detection methodology and follows the
same disclosure norms as the published work it builds on: results and
methodology are published; no novel attack methodology is introduced; no raw
harmful completions are released.

All harmful-instruction data used here is drawn from established public
benchmarks rather than authored for this project, and no novel jailbreak
technique is disclosed by it.
