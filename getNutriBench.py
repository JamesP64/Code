import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re   

def makeQuestion(i, dataset):
    description = dataset[i]['meal_description']
    prompt = (
        "Review the meal and calculate total carbs as a number.\n"
        "Description: I ate a banana and an apple.\n"
        "Carbs: 52.0\n\n"
        "Description: I had a bowl of plain rice.\n"
        "Carbs: 45.0\n\n"
        f"Description: {description}\n"
        "Carbs:"
    )
    correctAnswer = dataset[i]['carb']
    return prompt, correctAnswer

def extract_number(text):
    # Look for integer answer, finds the last one
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if matches:
        return float(matches[-1])
    return None

def ask(model, tokenizer, i, dataset):
    prompt, correctAnswer = makeQuestion(i, dataset)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        pad_token_id=tokenizer.eos_token_id,    
        temperature=0.7,
        do_sample=False
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Nutrition answer:" in text:
        generated_text = text.split("Nutrition answer:")[-1].strip()
    else:
        generated_text = text 

    return generated_text, correctAnswer

def evaluateModel():
    # NutriBench
    dataset = datasets.load_dataset('dongx1997/NutriBench', 'v2', split='train')

    # GPT 2
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    rightwrong = [0,0]
    for i in range(len(dataset)):
        print(f"\nCase {i}")
        givenAnswer, correctAnswer = ask(model, tokenizer, i, dataset)

        numericVal = extract_number(givenAnswer)
        # Handle cases where no number is found
        if numericVal is None:
            print(f"No number found")
            rightwrong[1] += 1
            continue

        error = correctAnswer - numericVal
        if error > 7.5 or error < -7.5:
            rightwrong[1] += 1
        else:
            rightwrong[0] += 1
        print(f"Given Answer: {givenAnswer}")
        print(f"Correct Answer: {correctAnswer}")
    print(f"Right/Wrong {rightwrong}")

if __name__ == "__main__":
    evaluateModel()
    

