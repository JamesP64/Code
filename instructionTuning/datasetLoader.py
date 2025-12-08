import json
import os
import urllib.request
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

# Loads and tokenizers the instruction dataset
class InstructionTrainingLoader(Dataset):
    def __init__(self, tokenizer):
        self.encoded_texts = []
        self.file_path = "alpagasus_chatgpt_9k.json"
        self.url = (
            "https://raw.githubusercontent.com/gpt4life/alpagasus"
            "/main/data/filtered/chatgpt_9k.json"
        )
        self.tokenizer = tokenizer

    # Load in the dataset
    def download_and_load_file(self):
        if not os.path.exists(self.file_path):
            with urllib.request.urlopen(self.url) as response:
                text_data = response.read().decode("utf-8")
            with open(self.file_path, "w", encoding="utf-8") as file:
                file.write(text_data)

        with open(self.file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data

    # Get just the istruction an input so you can test the model
    # Format in Alpaca style
    @staticmethod
    def format_input(entry):
        instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        )

        input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

        return instruction_text + input_text

    # Get the correct output for testing
    @staticmethod
    def format_output(entry):
        return f"\n\n### Response:\n{entry['output']}"

    # Turn the data into strings of tokens
    def tokenize_dataset(self,data):
       for entry in data:
            encoded = self.tokenizer.encode(
                self.format_input(entry) + self.format_output(entry), 
                allowed_special={'<|endoftext|>'}
            )
            
            if len(encoded) <= 1024:
                self.encoded_texts.append(encoded)
            else:
                pass


    @staticmethod
    def custom_collate(batch, pad_token_id=50256, ignore_index=-100, device="cpu"):

        longest_input_len = max(len(item) for item in batch)

        padded_inputs = [] 
        target_outputs = []
        for item in batch:
            # See how much padding it needs
            padding_needed = longest_input_len - len(item)

            if (padding_needed > 0):
                padded_item = item.copy()
                # Add padding
                for _ in range(padding_needed):
                    padded_item.append(pad_token_id)

                # Convert to tensor
                inputs = torch.tensor(padded_item)
                padded_inputs.append(inputs)

                # Build the targets
                # Add 1 padding and remove the first item (shift once)
                target_item = item[1:].copy()
                target_item.append(pad_token_id)
                for _ in range(padding_needed):
                    target_item.append(ignore_index)
                
                outputs = torch.tensor(target_item)
                target_outputs.append(outputs)
            else:
                padded_item = item.copy()
                # Convert to tensor
                inputs = torch.tensor(padded_item)
                padded_inputs.append(inputs)

                # Add 1 padding and remove the first item (shift once)
                target_item = item[1:] + [pad_token_id]
                outputs = torch.tensor(target_item)
                target_outputs.append(outputs)
            

        # Convert list of inputs to tensor and transfer to target device
        inputs_tensor = torch.stack(padded_inputs).to(device)
        targets_tensor = torch.stack(target_outputs).to(device)

        return inputs_tensor, targets_tensor

    def load(self):
        data = self.download_and_load_file()
        self.tokenize_dataset(data)  

    def __len__(self):
        return len(self.encoded_texts)

    def __getitem__(self, idx):
        return self.encoded_texts[idx]
    
    def get_data_loader(self, batch_size=4, shuffle=True, device="cpu"):
        # Pass device and batch to padder
        custom_collate = lambda batch: self.pad_batch(
            batch, 
            pad_token_id=50256, 
            device=device
        )
        
        # Return the data loader
        return DataLoader(
            self, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=custom_collate
        )

if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    loader = InstructionTrainingLoader(tokenizer)
    data = loader.download_and_load_file()
    loader.tokenize_dataset(data)
    print(loader.custom_collate([loader.encoded_texts[52]]))

