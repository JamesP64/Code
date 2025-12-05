import torch
from transformers import AutoModelForCausalLM

class Gpt2Trainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=5e-4, device="cpu"):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizing
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def train_epoch(self, epoch_index, eval_freq=50):
        self.model.train()
        total_loss = 0
        
        for step, (inputs, targets) in enumerate(self.train_loader):
            # Move to GPU/MPS/CPU
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Reset Gradients
            self.optimizer.zero_grad()
            
            # Forward Pass 
            outputs = self.model(inputs, labels=targets)
            loss = outputs.loss
            
            # Backward Pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()

            # Print progress every few steps
            if step % eval_freq == 0:
                print(f"   Step {step}: Loss = {loss.item():.4f}")

        avg_train_loss = total_loss / len(self.train_loader)
        return avg_train_loss

    def validate(self):
        self.model.eval() 
        total_val_loss = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs, labels=targets)
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(self.val_loader)
        self.model.train()
        return avg_val_loss

    def train(self, num_epochs):
        print(f"Starting training on {self.device}...")
        
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            
            train_loss = self.train_epoch(epoch)
        
            val_loss = self.validate()
            
            print(f"--> Epoch {epoch+1} Summary:")
            print(f"    Avg Train Loss: {train_loss:.4f}")
            print(f"    Avg Val Loss:   {val_loss:.4f}")

    def save_model(self, output_dir="my_gpt2_model"):
        print(f"Saving model to {output_dir}...")
        self.model.save_pretrained(output_dir)