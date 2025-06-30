# Generate Text with LLaMA (using Hugging Face):
# Let’s use a public LLaMA-like model (e.g., LLaMA 2 or TinyLLaMA):

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-chat-hf"  # You can use a smaller one like TinyLLaMA too

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Input prompt
prompt = "What are the benefits of exercise?"

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt")

# Generate response
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.7)

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)



# What are the benefits of exercise?
# Exercise offers many benefits, including improved cardiovascular health, increased energy levels, better mood, enhanced sleep quality, and reduced risk of chronic diseases such as diabetes and obesity.

