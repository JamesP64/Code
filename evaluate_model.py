import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def ask(model, tokenizer, question):
    prompt = f"Nutrition question: {question}\nNutrition answer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.7,
        do_sample=True
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("Nutrition answer:")[-1].strip()

def evaluate():
    scores = []
    with open("evaluation_questions.jsonl") as f:
        for line in f:
            row = json.loads(line)
            q = row["question"]
            correct = row["answer"]
            pred = ask(model, tokenizer, q)

            print("\nQUESTION:", q)
            print("GPT-2 ANSWER:", pred)
            print("EXPECTED:", correct)

if __name__ == "__main__":
    evaluate()
