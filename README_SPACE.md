---
title: Transformer QA System
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.11.0
app_file: app.py
pinned: false
license: mit
---

# 🤖 Transformer-Based Question Answering System

A powerful question-answering system that combines **BERT** for answer extraction and **GPT-2** for explanation generation.

## 🎯 Features

- **Extractive QA**: Uses DistilBERT to find exact answers in text
- **Generative Explanations**: Uses DistilGPT-2 to explain the answer
- **Confidence Scores**: Shows prediction confidence
- **Pre-trained Models**: Fast and accurate without fine-tuning

## 🚀 How to Use

1. **Enter a question** in the first text box
2. **Provide context** (passage containing the answer) in the second box
3. **Click Submit** to get:
   - Extracted answer with confidence score
   - Natural language explanation

## 💡 Example

**Question:** "Which country contains the majority of the Amazon rainforest?"

**Context:** "The Amazon rainforest covers most of the Amazon basin of South America. The majority of the forest is contained within Brazil, with 60% of the rainforest..."

**Output:**
- **Answer:** Brazil (95% confidence)
- **Explanation:** The answer is Brazil because the context states that the majority of the forest is contained within Brazil, with 60% of the rainforest.

## 🔧 Technical Details

### Models Used
- **QA Model**: `distilbert-base-uncased-distilled-squad`
  - Fine-tuned on SQuAD dataset
  - Optimized for answer extraction
  
- **Generation Model**: `distilgpt2`
  - Generates natural language explanations
  - Creative and contextual responses

### Technology Stack
- **Transformers**: Hugging Face library
- **PyTorch**: Deep learning framework
- **Gradio**: Interactive UI

## 📊 Performance

- **Accuracy**: High accuracy on factual questions
- **Speed**: ~1-2 seconds per query
- **Languages**: Optimized for English

## 🎓 About

This project was created as part of an NLP course to demonstrate:
- Transformer architectures (BERT, GPT-2)
- Transfer learning
- Question answering systems
- Interactive ML deployment

## 📝 License

MIT License - Free to use and modify

## 🔗 Links

- [GitHub Repository](#)
- [Project Documentation](#)
- [Research Paper: BERT](https://arxiv.org/abs/1810.04805)
- [Research Paper: GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

**Built with ❤️ using Hugging Face Transformers and Gradio**
