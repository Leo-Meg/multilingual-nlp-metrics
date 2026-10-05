# Multilingual NLP, what a perplexity really measures

N-gram models and perplexity, written with no dependencies, to answer a question my coursework left unanswered.

**Léo Mégret**, MSc Computational Linguistics, Université Paris Cité

> **Repository status, version 2.** I am doing this work in stages, each in its
> own folder. I publish as I go rather than once everything is finished.

---

## Why this repository

The lab that opened my multilingual NLP course asked us to train one language
model per language and compare their perplexities. The central question in the
brief was whether perplexities compare across languages.

I answered no, without being able to prove it. This repository is the proof.

The brief asked for 41 languages, 40,000 training sentences each and a KenLM
install. I do the opposite, five languages, a hand-written parallel corpus of 40
sentences, everything in the code. The result runs on a laptop and will still run
in ten years.

---

## Published versions

| | Folder | Contents | Tests |
|---|---|---|---:|
| **1** | `1.multilingue_python_projet` | Language models and perplexity | 18 |
| **2** | `2.multilingue_python_projet` | BLEU, implementing the metric then breaking it | 16 |

That is **34 tests** in total. Each folder contains everything the previous
one had, plus one step.

---

## Running the latest version

```bash
cd 2.multilingue_python_projet
python -m src.bleu
python -m tests.test_bleu
```

---

## What is still open

My measurements are about translations that have already been produced. How the
text is made from the model's probabilities is outside my scope.

I suspect that choice weighs as much as the model itself, and I have nothing to
check it with.

---

## How I work

Four rules I set myself at the start, and intend to keep across the whole
repository.

**Nothing to download.** The corpus is written into the code. All of my Master's
notebooks began with a `wget` to a university server or a Google Drive mount. Two
years later, half of them no longer run.

**Nothing is claimed without a measurement.** Every figure in this file
corresponds to a command you can re-run.

**Mistakes in my coursework are quoted, not erased.** Where a result I handed in
was wrong or incomplete, I say so and give the correct one.

**Negative results stay.** When an experiment shows the opposite of what I
expected, I write down what I found.

**The code is commented in French.**

---

---

*French version, which I wrote first, [README_FR.md](README_FR.md).*
