# Step 1: Setup your environment:
# install libraries:    pip install transformers datasets sentencepiece accelerate


# Step 2: Prepare your datasets:
# Let’s say you have a CSV (nl_sql_pairs.csv) with two columns: question and query.
# | question                           | query                                                |
# | ---------------------------------- | ---------------------------------------------------- |
# | Show all employees in the HR dept. | SELECT \* FROM employees WHERE department = 'HR';    |
# | List customers in California       | SELECT \* FROM customers WHERE state = 'California'; |



# Convert this csv to a Hugging face Dataset:
from datasets import load_dataset

dataset = load_dataset('csv', data_files='nl_sql_pairs.csv')
dataset = dataset['train'].train_test_split(test_size=0.1)



# Step 3: Load BART and Tokenizer:
from transformers import BartTokenizer, BartForConditionalGeneration

model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)


# Step 4: Preprocess the dataset:
def preprocess(example):
    inputs = tokenizer(example["question"], truncation=True, padding="max_length", max_length=128)
    targets = tokenizer(example["query"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=["question", "query"])


# Step 5: Fine-Tune BART:

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./bart-nl2sql",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    save_total_limit=1,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
)

trainer.train()


# Step 6: Inference: Use Trained Model

def generate_sql(question):
    input_ids = tokenizer(question, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Try it
print(generate_sql("List all customers from California"))




# 📁 Example Prompt (with schema awareness)

# Input:
# Tables: customers(id, name, state)
# Question: List all customers from California.

# Output:
# SELECT * FROM customers WHERE state = 'California';



# 🎁 Bonus: Prebuilt Datasets

# You can use datasets like:

# spider → complex cross-domain SQL
# wikisql → simple single-table SQL
# Install via:

# from datasets import load_dataset
# dataset = load_dataset("spider")

