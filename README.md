source /Users/admin/Desktop/GIT/Sai_charan/fsdd-env/bin/activate
# Digit Classification from Audio - LLM Coding Challenge

## Overview

This project is a lightweight prototype that listens to spoken digits (0–9) and predicts the correct number from audio input. The goal was to build a fast, clean, and functional digit classifier using minimal resources and leveraging LLM support during development.

## Approach

- **Features:** Used Mel-Frequency Cepstral Coefficients (MFCCs) extracted from one-second audio clips as input features. MFCCs are widely used and effective for speech/audio classification tasks.
- **Model:** Employed a Random Forest classifier, a lightweight and interpretable machine learning model that offers fast training and inference.
- **Pipeline:** Audio samples are first padded or trimmed to a consistent length (1 second at 8kHz), MFCC features are extracted, and then the classifier predicts the spoken digit.

## Dataset

- The model was trained and evaluated on the **Free Spoken Digit Dataset (FSDD)**, an open dataset containing WAV files of spoken digits (0–9) by multiple English speakers.
- Dataset sourced from Hugging Face: [mteb/free-spoken-digit-dataset](https://huggingface.co/datasets/mteb/free-spoken-digit-dataset)

## Results

- Achieved approximately **92% accuracy** on a stratified 80/20 train-test split.
- Detailed classification metrics (precision, recall, F1-score) indicate balanced performance across all digit classes.
- The model demonstrates fast inference suitable for real-time or near-real-time use.

## Noteworthy Decisions

- Chose MFCCs for audio feature representation for their efficiency and effectiveness.
- Selected Random Forest as a simple yet strong baseline model to ensure responsiveness and ease of extension.
- Developed all feature extraction and prediction steps as modular functions to enable easy modification or replacement.
- Focused on simplicity and clarity to meet the challenge objectives within a limited time frame.

## Running the Code

1. Ensure Python 3.10–3.11 environment with required packages is set up:


2. Run the main script:


3. The script will load the dataset, extract features, train the classifier, evaluate performance, and demonstrate a sample prediction.

## Development Process and LLM Collaboration

- Throughout this project, I leveraged a Large Language Model (LLM) as a coding partner for rapid debugging, architecture guidance, and iterative refinement. 
- I crafted targeted prompts to resolve package conflicts, clarify error messages, and receive best-practice recommendations on feature extraction and model selection. 
- Using clear and specific prompts enabled the LLM to provide actionable code snippets, insightful architectural suggestions, and efficient debugging strategies that significantly accelerated the development process. 
- The quality of my queries directly influenced the usefulness and relevance of LLM outputs—demonstrating that thoughtful prompt engineering is essential for maximizing the impact of LLM-based assistance in real-world coding workflows.
---

## Evaluation Table

| Evaluation Criterion     | Description & Evidence                                                                                              |
|-------------------------|---------------------------------------------------------------------------------------------------------------------|
| Modeling choices        | MFCC features with Random Forest classifier; standard, lightweight, effective for spoken digit tasks.                |
| Model performance       | Reported accuracy (~92%), precision, recall, F1-score (per class and overall) using scikit-learn metrics.           |
| Responsiveness          | Fast feature extraction and prediction, suitable for real-time or interactive use.                                  |
| Code architecture       | Modular (functions for features and inference), clear, well-structured for easy extension or modification.           |
| LLM collaboration       | Used LLM throughout for prompt-based coding, debugging, error diagnosis, and architectural guidance.                 |
| Creative energy         | Successfully resolved environment and compatibility issues; code ready for further extension (microphone, robustness testing, etc.). |


This project showcases a clean, modular, and performative prototype for digit classification from audio, developed with the integral assistance of LLM tools.
