---
title: Transformer QA System
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.11.0"
app_file: app.py
pinned: false
---

# Transformer Question Answering System

BERT for extraction and GPT-2 for generation.

## Features
- Extractive QA using DistilBERT
- Generative explanations using DistilGPT-2
- Confidence scores

## Usage
Enter question and context, click Submit to get answer with explanation.

## Technical Details
- QA: distilbert-base-uncased-distilled-squad
- Generation: distilgpt2
- Framework: PyTorch + Transformers + Gradio

Built with Hugging Face
