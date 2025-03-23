# Aspect-Based Sentiment Analysis for Literature Book Reviews

This repository contains the code and resources for the paper: "TBD". ABSA-LIT addresses the limited availability of datasets for Aspect-Based Sentiment Analysis (ABSA) in the literary domain. We present:

- A methodology for generating semi-synthetic ABSA datasets by combining structured knowledge (Wikidata, OpenLibrary) with human-written book reviews
- A dataset of 10,000 book reviews with aspect-based sentiment annotations and DOLCE ontology entity typing
- A baseline model (fine-tuned Llama 3.1 8B) that performs both ABSA and entity typing

## Repository Structure
    ```shell
├── data_generation/              # Scripts for dataset generation
│   ├── wikidata_extraction.py    # Extracts book data from Wikidata
│   └── generate_reviews.py       # Generates reviews using LLMs
│
├── data_processing/              # Scripts for data processing
│   ├── text2amr2fred.py          # Wrapper for Text2AMR2FRED tool
│   ├── dolce_alignment.py        # Aligns aspects with DOLCE ontology
│   └── data_validation.py        # Validates dataset quality
│
├── model/                        # Model training and evaluation
│   ├── train.py                  # Fine-tunes Llama 3.1 8B
│   └── evaluate.py              # Runs inference on new reviews
│
├── evaluation/                   # Evaluation scripts and results
│   ├── metrics.py                # Implements evaluation metrics
│   └── results/                  # Detailed evaluation results
│
├── examples/                     # Example usage and demonstrations
│
├── utils/                        # Utility functions
│   ├── format_converters.py      # Convert between different formats (e.g., gemma_to_phi_format.py)
│   └── dataset_upload.py         # Scripts for uploading to Hugging Face
├── LICENSE                       # Apache 2.0 license
└── README.md                     # This file
```
