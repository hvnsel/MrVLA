# Causal Generality in the Sparse-Autoencoder Features of a Vision-Language-Action Model

**A complete technical record of the experimental programme: motivation, methods, mathematics, results, and claims.**

Project: MrVLA. Model under study: OpenVLA-7B, fine-tuned on the LIBERO benchmark. Reference work under examination: Swann, McGranahan, Buurmeijer, Kennedy & Schwager, *Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models*.

## Who this document is for

This is written for a reader who has done roughly one or two years of an undergraduate degree in a quantitative subject. That means we assume you are comfortable with:

- basic algebra and summation notation,
- the idea of a vector as a list of numbers,
- the idea of an average, and roughly what a correlation is.

We assume **nothing** about machine learning, robotics, transformers, sparse autoencoders, or advanced statistics. Every one of those is built up from scratch. Where a formula appears, it is written out, every symbol is named, and the reason we compute that particular quantity rather than some other one is explained.

The aim is that you finish this document able to say, for every number in the results, three things: what was measured, how it was measured, and why that measurement answers the question it claims to answer.

> **A note on difficulty.** We have not simplified the science. The experiments involve some genuinely intricate reasoning — particularly about *controls*, which are the parts of an experiment designed to rule out boring explanations for exciting-looking results. Where something is hard, we slow down and break it into pieces rather than skipping it. If a section feels dense, the payoff is usually in the paragraph that starts "**Why we do this**".

## How this document is organised

| Chapter | What it covers | Read it if you want to know |
|---|---|---|
| 1 | The machines and the vocabulary | What a VLA is, what an SAE is, what a "feature" means |
| 2 | The scientific question | What "general" means and why the existing definition fails |
| 3 | The mathematical toolkit | Every statistic and algorithm used, explained from zero |
| 4 | Phase 1 — building the measures | How the two measurement paths were built and first tested |
| 5 | Phase 2 — auditing the measures | Whether those numbers survive their own controls |
| 6 | Complete results | Every number, in tables |
| 7 | Claims | What we assert, and what we deliberately do not |
| 8 | Limitations and known issues | Everything currently wrong or unresolved |
| 9 | Glossary and appendices | Definitions, file map, reproduction instructions |

Chapters 1–3 are background and can be skipped by a reader who already works in interpretability. Chapters 4–5 are the experimental record. Chapters 6–8 are the scientific content proper.

## A one-paragraph summary, for orientation

A robot policy is a neural network that turns camera images and a written instruction into motor commands. Inside it, information is stored in a way that does not line up neatly with human concepts, so researchers use a tool called a *sparse autoencoder* to break that information into interpretable pieces called *features*. A natural question is which features are **general** — reusable across many situations — and which are **memorised** — tied to one specific situation. The existing way of answering that question turns out to measure something else entirely. We replace it with a definition based on *causal influence*: a feature is general to the degree that removing it would change the robot's chosen action across many different tasks. We show this definition is measurable, replicates across four task suites, and reveals that causal influence is concentrated in roughly 100 of 2048 features. We then show that this within-model notion of generality does **not** predict whether a differently fine-tuned copy of the same base model will contain the same feature — and that this failure is not an artefact of how we matched features, which was the most plausible innocent explanation.
