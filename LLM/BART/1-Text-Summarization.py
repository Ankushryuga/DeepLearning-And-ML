from transformers import BartForConditionalGeneration, BartTokenizer

# Load pretrained BART model and tokenizer
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Input text to summarize
text = "The Apollo missions were a series of space missions conducted by NASA, aimed at landing humans on the Moon and bringing them back safely. Apollo 11 was the first mission to achieve this goal in 1969."

# Tokenize input
inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)

# Generate summary
summary_ids = model.generate(inputs['input_ids'], max_length=50, num_beams=4, early_stopping=True)

# Decode output tokens to string
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("🔹 Summary:", summary)



# o/p: => 🔹 Summary: Apollo missions were NASA's efforts to land humans on the Moon. Apollo 11 achieved that in 1969.

