# What is LLaMA?
    =>
    1. LLaMA (Large Language Model Meta AI) is a family of open-source decoder-only transformer models developed by Meta (Facebook). It's similar in architecture to GPT, optimized for efficiency and performance.
    2. Think of LLaMA as a powerful, general-purpose text generator, trained to predict the next word/token in a sentence.


# 📦 LLaMA Model Variants
    =>
    1. LLaMA 1 (2023): Sizes from 7B to 65B.
    2. LLaMA 2 (July 2023): Improved training, alignment, and licensing (more permissive).
    3. LLaMA 3 (April 2024): State-of-the-art reasoning and performance.
    They are all decoder-only transformers, meaning they generate text left-to-right, unlike BERT-style models that read text bidirectionally.


# 🧠 How Does LLaMA Work?
    =>
    Architecture: Decoder-only Transformer
    Input tokens (text) are embedded.
    Passed through multiple transformer blocks with:
    Causal self-attention (only looks at previous tokens)
    Feed-forward networks
    Output: Predicts the next token in the sequence.
    Repeat token-by-token to generate longer text.

# 🔄 Training Objective: Autoregressive Language Modeling
    =>
    It learns to predict the next word, given all previous ones:
    Input: "The capital of France is"
    Target: "Paris"
    
    # It tries to maximize the probability of Paris being the correct next word.

# ✅ LLaMA Use Cases
    =>
    Text generation (chatbots, storytelling, autocomplete)
    Code generation
    Question answering
    Instruction following (when fine-tuned)
    Translation or summarization (with fine-tuning)
