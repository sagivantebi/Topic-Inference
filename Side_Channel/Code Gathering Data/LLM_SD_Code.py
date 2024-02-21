import logging
import sys

from datasets import load_dataset
import time
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

torch.cuda.empty_cache()
import os

logging.getLogger("transformers").setLevel(logging.ERROR)
print("Code QA")

# Load the new dataset
print("Loading dataset...")
dataset = load_dataset("lissadesu/code_qa_updated", split='train[:7000]')
model_name = str(sys.argv[1])
your_token = "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"  # Replace with your actual token
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=your_token)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=your_token)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model is running on: {next(model.parameters()).device}")


def generate_response(combined_question):
    inputs = tokenizer.encode(combined_question, return_tensors="pt")
    inputs = inputs.to(device)  # Move input tensors to the same device as the model
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(inputs, max_length=512)
        end_time = time.time()

    return tokenizer.decode(outputs[0], skip_special_tokens=True), end_time - start_time

# Function to process a single sample
def process_sample(sample, writer):
    # Combine the question and the code into one section for the model input
    q = sample['question']
    combined_question = f"Code: {sample['code']} \n \n {q}"
    answer, time_gen = generate_response(combined_question)
    # Attempt to remove the combined question and code from the beginning of the answer
    # This is a naive approach and may need adjustments
    words_in_combined_question = len(combined_question.split())
    index = answer.find(q)
    if index != -1:
        answer = answer[index+len(q):]

    words_in_answer = len(answer.split())
    tokens_in_combined_question = len(tokenizer.tokenize(combined_question))
    tokens_in_answer = len(tokenizer.tokenize(answer))

    # Writing data to CSV
    writer.writerow(
        [combined_question, answer, words_in_combined_question, tokens_in_combined_question, words_in_answer,
         tokens_in_answer, time_gen])



# Make sure to adjust the CSV headers accordingly since now we have combined question and code into a single entry
def process_dataset():
    filename = f'CodeQA_Combined_{model_name.replace("/", "-")}.csv'
    print(f"Processing dataset... Output file: {filename}")

    file_exists = os.path.isfile(filename)  # Check if file already exists

    with open(filename, mode='a', newline='', encoding='utf-8') as file:  # Open in append mode
        writer = csv.writer(file)

        # Writing headers only if file does not exist
        if not file_exists:
            writer.writerow(
                ["Question", "Answer", "Words in Question", "Tokens in Question",
                 "Words in Answer", "Tokens in Answer", "Time to Generate"])

        for i, sample in enumerate(dataset):
            process_sample(sample, writer)
            if i % 5 == 0:  # Print progress every few samples
                print(f"Processed {i} samples...")


if __name__ == '__main__':
    process_dataset()
    print("Dataset processing complete.")
