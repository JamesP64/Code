import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from Code.McCanceLoader import loadMcCance
from datasets import DatasetDict

# Load dataset
dataset = loadMcCance()
dataset = dataset.train_test_split(test_size=0.1)
dataset = DatasetDict({
    "train": dataset["train"],
    "validation": dataset["test"]
})

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    inputs['labels'] = inputs['input_ids'].copy()
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Load the model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Set the EOS token for padding
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define training arguments
training_args = TrainingArguments(
    output_dir="C:\\Users\\james\\Documents\\.UNI\\Year_3\\Individual Project\\results",
    eval_strategy="epoch",
    log_level="info",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=100,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

# Train the model
trainer.train()

# save the model and tokenizer explicitly
model_output_dir = '/mnt/disks/disk1/results/model'

model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)