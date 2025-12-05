import torch
import tiktoken
from transformers import AutoModelForCausalLM
from torch.utils.data import random_split

from datasetLoader import InstructionTrainingLoader
from gpt2InstructionTrainer import Gpt2Trainer

# Device and tokenizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = tiktoken.get_encoding("gpt2")

# Prep the instruction tuning data
print("Loading Data...")
dataset = InstructionTrainingLoader(tokenizer)
dataset.load()

# Split the data
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])
    
# Get Padding func   
collate_fn = lambda b: InstructionTrainingLoader.custom_collate(b, device=DEVICE)
    
# loaders to feed the model dataset
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=4, shuffle=True, collate_fn=collate_fn
)
val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=4, shuffle=False, collate_fn=collate_fn
)

# Get the model
print("Initializing Model...")
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")

# Full trainer
trainer = Gpt2Trainer(
    model=model, 
    train_loader=train_loader, 
    val_loader=val_loader,
    learning_rate=5e-4,
    device=DEVICE
)

# run
trainer.train(num_epochs=2)
trainer.save_model()