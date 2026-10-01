# Multilingual NLP — the metric that lies

N-gram models and perplexity, written with no dependencies, to answer a question my Master's submission left unanswered.

**Léo Mégret** — MSc Computational Linguistics, Université Paris Cité

> **Repository status: version 1.** This is the first step of a piece of work I
> am doing in stages, each in its own folder. Only version 1 exists so far. I am
> publishing as I go rather than once everything is finished, because the point
> of this work is precisely the way one question leads to the next.

---

## Why this repository

The lab that opened my multilingual NLP course asked us to train one language
model per language and compare their perplexities. The central question in the
brief was: can perplexities be compared across languages?

I answered no, without being able to prove it. This repository is the proof.

The brief asked for 41 languages, 40,000 training sentences each and a KenLM
install. I do the opposite: five languages, a hand-written parallel corpus of 40
sentences, and everything in the code. The result runs on a laptop and will still
run in ten years.

---

## What exists today

### Version 1 — Language models and perplexity

| File | What I do in it |
|---|---|
| `src/corpus.py` | A parallel corpus of 40 sentences translated into five languages: French, English, Spanish, German and Turkish. Word and character segmentation with Unicode normalisation, type-token ratio, descriptive statistics. |
| `src/ngrammes.py` | A complete n-gram model with three smoothing schemes: additive, backoff and interpolated Kneser-Ney. Out-of-vocabulary handling, perplexity, and the two methodological experiments. |
| `tests/test_ngrammes.py` | 18 tests, including a check that Kneser-Ney really does favour versatile words. |

Academic origin: lab 1 of multilingual NLP, *One model to rule them all*
(Guillaume Wisniewski, M2).

---

## Running the code

```bash
cd 1.multilingue_python_projet
python -m src.corpus
python -m src.ngrammes
python -m tests.test_ngrammes
```

No dependencies, not even NumPy. The corpus lives in the code.

---

## What I take from this step

**Turkish gets the best perplexity even though it is the hardest language in the
corpus.** The reason is that it refuses to predict 80% of its words: it sends
them to the out-of-vocabulary bucket, which does not count towards the metric. A
perplexity can therefore reward abstention.

That is precisely what makes cross-language comparison illegitimate, and it is
what I could not prove during the course.

**A limitation I document rather than hide.** I segment every language with the
same rule, deliberately, so that the only difference between my measurements
comes from the languages themselves. That has a cost: French clitics such as
`l'article` and German contractions such as `im` are not handled.

---

## What is still open

I can now show that a perplexity does not compare across languages, and why.

What I do not know is whether the other metrics in the field suffer from the same
flaw. I used several of them during my Master's without ever checking what they
actually measured.

---

## How I work

Four rules I set myself at the start, and intend to keep across the whole
repository.

**Nothing to download.** The corpus lives in the code. All of my Master's
notebooks began with a `wget` to a university server or a Google Drive mount.
Two years later, half of them no longer run.

**Nothing is claimed without a measurement.** Every figure in this file
corresponds to a command you can re-run.

**Mistakes in my submitted coursework are quoted, not erased.** Where a result I
handed in was wrong or incomplete, I say so and give the correct one.

**Negative results stay.** When an experiment shows the opposite of what I
expected, I change the conclusion, not the experiment.

**The code is commented in French.** This is a repository meant to be read as
much as run.

---

*French version, and the one I wrote first: [README_FR.md](README_FR.md).*
