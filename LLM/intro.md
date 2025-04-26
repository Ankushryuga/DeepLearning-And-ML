# intro of llm.
## Pipe Line of LLM: Data Extraction, Processing, and Model Training.

## Tokens:
Anything can be added as a token that AI can generate (predict based on preceding text):
1. "car" (words token)
2. "b"  (a letter token)
3. "123" (a number token)
4. "💯" (an emoji token)
5. "魑" ( a chinese character token)
6. "The sun sinks in the wast (a sentance token)
7. " 魑魅魍魉" (a chinese sentance tokens)
8. "💯192 魑魅魍魉" (an arbitrary token)

Question: Which way of tokenizing is best?
1. whole sentace?
2. by spliting sentance into word?
3. or by spliting into each letter or character?
etc..

## NOTE: Each token requires same amount of compute to process.
## Computational cost:
"The quick brown fox jumps over the lazy dog."
1. Letter tokenizer: 44 tokens=44* compute.
2. Word tokenizer: 9 tokens=9* compute.
3. Sentance tokenizer: 1 token=1* compute.

## issue with word and sentance tokenizer:
1. Massive vocabulary
2. if you forget to add any word, the AI will not learn what it means.
3. Can't handle misspelling
4. Most words are rare and will not appear enough times in the training data for model to learn their meaning well.
....etc.

### Building a Byte pair encoding (BPE) tokenizer from scaratch..

## Step 1: Prepare data.
The first step in building any tokenizer is to have a corpus of text train it on. The tokenizer learns merge rules based on the frequency character pairs in this data.
i: 1
s: 2
is: 3

Example:
corpus=[
"This is the first document.",
"This document is the second document.",
"And this is the third one.",
"Is this the first document?",
]


print("training corpus")
for doc in corpus
print(doc)


unique_chars=set()
for doc in corpus:
  for char in doc:
    unique_chars.add(char)


vocab=list(unique_chars)
vocab.sort()

end_of_word="</w>"
vocab.append(end_of_word)

print("Initial vocabulary")
print(vocab)
print(f"vocabulary size: {len(vocab)}")

