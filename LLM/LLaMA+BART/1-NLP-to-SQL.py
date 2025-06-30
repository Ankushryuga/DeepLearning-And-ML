# ✅ LLaMA (via a Hugging Face-compatible variant like LLaMA 2) for text generation
# ✅ BART (e.g. facebook/bart-base or facebook/bart-large) for natural language to SQL query generation



# 🧩 Use Case Example
# Your app should:
# Understand whether the user wants general text output or SQL.
# If it's a chat request → use LLaMA
# If it's a database question → use BART to generate SQL.


# Step 1: Install dep
# pip install transformers accelerate sentencepiece


# Step 2: Load Models:
from transformers import AutoTokenizer, AutoModelForCausalLM, BartTokenizer, BartForConditionalGeneration

# Load LLaMA (can also use TinyLLaMA or LLaMA 2 - must be from Hugging Face with access)
llama_model_name = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
llama_tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b")
llama_model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_3b")

# Load BART model for SQL generation
bart_model_name = "facebook/bart-base"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)


# Step 3: Define Utility functions:
def generate_with_llama(prompt: str):
    inputs = llama_tokenizer(prompt, return_tensors="pt")
    outputs = llama_model.generate(**inputs, max_new_tokens=100)
    return llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_sql_with_bart(question: str):
    inputs = bart_tokenizer([question], return_tensors="pt", max_length=128, truncation=True)
    outputs = bart_model.generate(**inputs, max_new_tokens=64, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(outputs[0], skip_special_tokens=True)



# Step 4: Define a Simple Task Router:
def is_sql_related(text: str) -> bool:
    keywords = ['list', 'show', 'select', 'table', 'from', 'database', 'query']
    return any(kw in text.lower() for kw in keywords)


# Step 5: Build the combined Interface:
  def combined_nl_processor(user_input: str):
    if is_sql_related(user_input):
        sql = generate_sql_with_bart(user_input)
        return f"🧾 Generated SQL:\n```sql\n{sql}\n```"
    else:
        response = generate_with_llama(user_input)
        return f"💬 Response:\n{response}"


# Step 6: # Sample inputs
print(combined_nl_processor("List all employees in the Sales department"))
print(combined_nl_processor("Tell me a joke about programming"))




# 🧾 Generated SQL:
# ```sql
# SELECT * FROM employees WHERE department = 'Sales';
