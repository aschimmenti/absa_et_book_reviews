# Aspect-Based Sentiment Analysis for Literature Book Reviews

This repository contains the code and resources for the paper "Old Reviews, New Aspects: Aspect Based Sentiment Analysis and Entity Typing for Book Reviews with LLMs", accepted as conference paper at LDK 2025. ABSA-LIT addresses the limited availability of datasets for Aspect-Based Sentiment Analysis (ABSA) in the literary domain. We present:

- A methodology for generating semi-synthetic ABSA datasets by combining structured knowledge (Wikidata, OpenLibrary) with human-written book reviews
- A dataset of 10,000 book reviews with aspect-based sentiment annotations and DOLCE ontology entity typing
- A baseline model (fine-tuned Llama 3.1 8B) that performs both ABSA and entity typing

## Repository Structure
```shell
├── dataset_generation/              # Scripts for dataset generation
│   ├── generated prompts /              # examples of generated prompts (0-5)
│   ├── generated reviews /              # examples of generated reviews (0-5)
│   └── replaced_aspects_with_dolce /              # final reviews (0-5)
├── DOLCE-alignment/              # Scripts for data processing
│   ├── text2amr2fred.py          # Wrapper for Text2AMR2FRED tool
│   ├── dolce_alignment.py        # Aligns aspects with DOLCE ontology
│   └── data_validation.py        # Validates dataset quality
│
├── unsloth_llama3_1.ipynb/                        # Model training and evaluation
│
├── absa_ollama_inference.py/                   # Inference and evaluation (creates folder w/ json output)
│
├── metric_llama_answers.py/                     # computes metrics on output folder w jsons
│
├── LICENSE                       # Apache 2.0 license
└── README.md                     # This file```
