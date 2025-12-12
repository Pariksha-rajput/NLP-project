import gradio as gr
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    pipeline
)

# Set device
device = 0 if torch.cuda.is_available() else -1

print("Loading models...")

# Load QA model (BERT for answer extraction)
try:
    qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
    qa_pipeline = pipeline(
        "question-answering",
        model=qa_model,
        tokenizer=qa_tokenizer,
        device=device
    )
    print("✓ QA model loaded")
except Exception as e:
    print(f"Error loading QA model: {e}")
    qa_pipeline = None

# Load generation model (GPT-2 for explanations)
try:
    gen_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    gen_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    gen_pipeline = pipeline(
        "text-generation",
        model=gen_model,
        tokenizer=gen_tokenizer,
        device=device
    )
    print("✓ Generation model loaded")
except Exception as e:
    print(f"Error loading generation model: {e}")
    gen_pipeline = None

def generate_explanation(question, context, answer, max_length=100):
    """Generate explanation using GPT-2"""
    if gen_pipeline is None:
        return f"The answer '{answer}' was found in the provided context."
    
    try:
        prompt = f"Question: {question}\nContext: {context[:200]}\nAnswer: {answer}\n\nExplanation:"
        
        result = gen_pipeline(
            prompt,
            max_length=len(prompt.split()) + max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=gen_tokenizer.eos_token_id
        )
        
        generated_text = result[0]['generated_text']
        explanation = generated_text.split("Explanation:")[-1].strip()
        
        # Clean up
        if len(explanation) > 300:
            explanation = explanation[:300] + "..."
        
        return explanation if explanation else f"The answer '{answer}' is found in the context."
    
    except Exception as e:
        return f"The answer '{answer}' was extracted from the provided context."

def qa_system(question, context):
    """Main QA function combining extraction and generation"""
    
    if not question or not context:
        return "⚠️ Please provide both question and context.", ""
    
    if qa_pipeline is None:
        return "❌ QA model not loaded. Please try again.", ""
    
    try:
        # Extract answer using BERT
        result = qa_pipeline(question=question, context=context)
        answer = result['answer']
        confidence = result['score']
        
        # Generate explanation using GPT-2
        explanation = generate_explanation(question, context, answer)
        
        # Format outputs
        extracted_answer = f"""### 🎯 Extracted Answer
**{answer}**

**Confidence Score:** {confidence:.2%}
"""
        
        generated_explanation = f"""### 💡 Explanation
{explanation}
"""
        
        return extracted_answer, generated_explanation
    
    except Exception as e:
        return f"❌ Error: {str(e)}", ""

# Example questions
examples = [
    [
        "Which country contains the majority of the Amazon rainforest?",
        "The Amazon rainforest, also called Amazon jungle or Amazonia, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. The majority of the forest, 60%, is in Brazil, followed by Peru with 13%, Colombia with 10%, and smaller portions in Venezuela, Ecuador, Bolivia, Guyana, Suriname, and French Guiana."
    ],
    [
        "Who developed the theory of relativity?",
        "The theory of relativity usually encompasses two interrelated physics theories by Albert Einstein: special relativity and general relativity, proposed and published in 1905 and 1915, respectively. Special relativity applies to all physical phenomena in the absence of gravity. General relativity explains the law of gravitation and its relation to the forces of nature."
    ],
    [
        "When did construction of the Great Wall begin?",
        "The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states and Imperial China as protection against various nomadic groups. Several walls were built from as early as the 7th century BC, with selective stretches later joined by Qin Shi Huang (220–206 BC), the first emperor of China."
    ],
    [
        "What is the speed of light?",
        "The speed of light in vacuum, commonly denoted c, is a universal physical constant that is exactly equal to 299,792,458 metres per second (approximately 300,000 kilometres per second or 186,000 miles per second). According to the special theory of relativity, c is the upper limit for the speed at which conventional matter or energy can travel through space."
    ]
]

# Create Gradio interface
demo = gr.Interface(
    fn=qa_system,
    inputs=[
        gr.Textbox(
            label="❓ Question",
            placeholder="Enter your question here...",
            lines=2
        ),
        gr.Textbox(
            label="📄 Context/Passage",
            placeholder="Paste the context or passage here...",
            lines=10
        )
    ],
    outputs=[
        gr.Markdown(label="Extracted Answer"),
        gr.Markdown(label="Generated Explanation")
    ],
    title="🤖 Transformer-Based Question Answering System",
    description="""
    **BERT for Extraction + GPT-2 for Generation** | Trained on SQuAD 2.0
    
    This system uses two transformer models:
    - 🎯 **DistilBERT**: Extracts the answer span from the context
    - 💡 **DistilGPT-2**: Generates a natural language explanation
    
    Simply enter a question and provide relevant context to get started!
    """,
    examples=examples,
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
