# 🔍 What is BART?
    => 
    BART (Bidirectional and Auto-Regressive Transformers) is a sequence-to-sequence (seq2seq) model developed by Facebook AI.
    It's designed to combine the strengths of:
      1. BERT(good at understanding input using bidirectional attention).
      2. GPT(good at generating o/p with autoregressive decoding).


# 🏗️ BART Architecture:
    => BART has 2 main components:
      1. Encoder (like BERT):
        - Reads the entire input sentance at once.
        - Uses Bidirectional self-attention.
        - Outputs a sequence of hidden states (Context-rich representation of the input).

      2. Decoder (like GPT):
        - Generates the output one token at a time.
        - Uses autoregressive decoding (only attends to past tokens when generating).
        - Has cross-attention layers that attend to encoder outputs-this concept the input to the output.


# ⚙️ How It Works (Simplified Flow)
    =>
      1. Input text → Tokenized → Passed to encoder.
      2. Encoder processes it bidirectionally → Outputs context-aware embeddings.
      3. Decoder receives these embeddings and generates output token-by-token.
          
          

# 🧠 Why is BART Useful?
    =>
      # BART is powerful for tasks like:
      
      1. Text summarization
      2. Text generation
      3. Translation
      4. Paraphrasing
      5. Question answering
      Because it learns both how to understand and generate text.




📌 Summary of Key Features

| Feature     | Description                                                                 |
| ----------- | --------------------------------------------------------------------------- |
| Model type  | Sequence-to-sequence (encoder-decoder)                                      |
| Encoder     | Bidirectional (like BERT)                                                   |
| Decoder     | Autoregressive (like GPT)                                                   |
| Pretraining | Corrupting input (e.g., token masking, sentence permutation) then denoising |
| Use cases   | Summarization, translation, question answering, generation                  |

      
