# Multilingual NLP: what a perplexity really measures

N-gram models, BLEU and decoding, written with no dependencies, to answer a question the coursework left open.

**Léo Mégret** — implementation, French technical notes.  
**Haeeul Hwang** — English documentation, Korean documentation, review.

> **Repository status** — step 2 of 3, landed by Léo Mégret.

---

## Why this repository

The lab that opened Léo's multilingual NLP course asked us to train one language
model per language and compare their perplexities. The central question in the
brief was whether perplexities compare across languages.

He answered no, without being able to prove it. This repository is the proof.

The brief asked for 41 languages, 40,000 training sentences each and a KenLM
install. We do the opposite: five languages, a hand-written parallel corpus of 40
sentences, everything in the code. The result runs on a laptop and will still run
in ten years.

---

## Steps

| | Step | Modules | Tests | Landed by | Detailed note (French) |
|---|---|---|---:|---|---|
| **1** | N-gram language models and perplexity | `src/corpus.py`, `src/ngrammes.py` | 18 | Léo | [`ngrammes-et-perplexite`](notes/ngrammes-et-perplexite.md) |
| **2** | BLEU: implementing the metric, then breaking it | `src/bleu.py` | 16 | Léo | [`bleu`](notes/bleu.md) |

**34** tests in total.

---

## What each step found

**1. N-gram language models and perplexity**  
Turkish gets the *best* perplexity (2.7) although it is the hardest language here — because it refuses to predict 80 % of its own words. The note documents the million-fold bug that led to this measurement.

**2. BLEU: implementing the metric, then breaking it**  
720 word permutations of one sentence average a BLEU of 75.22. The metric is largely blind to word order, which is exactly what it is used to judge.

---

## Running it

```bash
python -m src.bleu

python -m tests.test_ngrammes
python -m tests.test_bleu
```

---

## How we work

**Nothing to download.** The corpus is written into the code. All of
Léo's Master's notebooks began with a `wget` to a university server or a Google
Drive mount. Two years later, half of them no longer run.

**Nothing is claimed without a measurement.** Every figure in this file
corresponds to a command you can re-run.

**Mistakes in the original coursework are quoted, not erased.** Where a result
that was handed in was wrong or incomplete, we say so and give the correct one.

**Negative results stay.** When an experiment shows the opposite of what we
expected, we write down what we found.

**The code is commented in French**, because that is the language it was thought
out in. The documentation is in English, and in Korean where Haeeul has caught
up.

---

*Français : [README_FR.md](README_FR.md) · [CONTRIBUTORS.md](CONTRIBUTORS.md)*
