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
print("Started Running")

# print("Finished Loading model and tokenizer")

# Function to generate response from the model
def generate_response(question):
    inputs = tokenizer.encode(question, return_tensors="pt")
    inputs = inputs.to(device)  # Move input tensors to the same device as the model
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(inputs, max_length=512)
        end_time = time.time()

    return tokenizer.decode(outputs[0], skip_special_tokens=True), end_time - start_time

def generate_response_one_token(question):
    inputs = tokenizer.encode(question, return_tensors="pt")
    inputs = inputs.to(device)  # Move input tensors to the same device as the model
    token_times = []  # List to store time taken for each token

    with torch.no_grad():
        outputs = inputs
        while True:
            start_time = time.time()
            next_token_logits = model(outputs).logits[:, -1, :]  # Get logits for the last token
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  # Choose the most likely next token
            end_time = time.time()

            token_times.append(end_time - start_time)  # Append time taken for this token

            outputs = torch.cat((outputs, next_token), dim=-1)  # Append the new token to the output

            if next_token.item() == tokenizer.eos_token_id or outputs.shape[1] >= 512:
                break  # Stop if EOS token is generated or max_length is reached

    return tokenizer.decode(outputs[0], skip_special_tokens=True), token_times


def remove_question_from_answer(question, answer):
    # Check if the answer starts with the question
    if answer.startswith(question):
        # Remove the question part from the answer
        return answer[len(question):].strip()
    else:
        # If the question is not at the start, return the answer as is
        return answer


# Function to process a single sample
def process_sample(sample, writer, q_type):
    # pre_prompt = "Keep the answer to the following question short but best: "
    question = sample[q_type]

    answer,time_gen = generate_response(question)
    # Remove the question from the answer (The model repeats it in his answer)
    answer = remove_question_from_answer(question, answer)

    # Remove the question from the answer (The model repeats it in his answer)
    answer = remove_question_from_answer(question, answer)

    words_in_question = len(question.split())
    words_in_answer = len(answer.split())
    tokens_in_question = len(tokenizer.tokenize(question))
    tokens_in_answer = len(tokenizer.tokenize(answer))

    # Writing data to CSV
    writer.writerow(
        [question, answer, words_in_question, tokens_in_question, words_in_answer, tokens_in_answer, time_gen])


def process_sample_one_token(sample, writer, q_type):
    question = sample[q_type]
    answer, token_times = generate_response_one_token(question)
    answer = remove_question_from_answer(question, answer)
    words_in_question = len(question.split())
    words_in_answer = len(answer.split())
    tokens_in_question = len(tokenizer.tokenize(question))
    tokens_in_answer = len(tokenizer.tokenize(answer))

    # Writing data to CSV
    writer.writerow(
        [question, answer, words_in_question, tokens_in_question, words_in_answer, tokens_in_answer, sum(token_times), token_times])


# Function to loop over the dataset
def process_dataset(d):
    if d == "PedroCJardim/QASports":
        filename = f'Test_Time_QASports_ONE_TOKEN_{model_name.replace("/", "-")}.csv'
        print(f"Processing dataset... Output file: {filename}")
    else:
        filename = d + f'Test_Time__ONE_TOKEN_{model_name.replace("/", "-")}.csv'
        print(f"Processing dataset... Output file: {filename}")

    file_exists = os.path.isfile(filename)  # Check if file already exists

    with open(filename, mode='a', newline='', encoding='utf-8') as file:  # Open in append mode
        writer = csv.writer(file)

        # Writing headers only if file does not exist
        if not file_exists:
            writer.writerow(
                ["Question", "Answer", "Words in Question", "Tokens in Question", "Words in Answer", "Tokens in Answer",
                 "Time to Generate"])

        for i, sample in enumerate(dataset):
            if d == "medical_questions_pairs":
                if i % 2 == 0:
                    process_sample_one_token(sample, writer, "question_1")
                process_sample_one_token(sample, writer, "question_2")
                if i % 5 == 0:  # Print progress every 10 samples
                    print(f"Processed {i * 2} samples...")
            elif d == "math_qa":
                process_sample(sample, writer, "Problem")
                if i % 10 == 0:  # Print progress every 10 samples
                    print(f"Processed {i} samples...")
            else:
                process_sample(sample, writer, "question")
                if i % 10 == 0:  # Print progress every 10 samples
                    print(f"Processed {i} samples...")



if __name__ == '__main__':
    # Load the dataset
    print("Loading dataset...")
    model_name = str(sys.argv[1])
    your_token = "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"  # Replace with your actual token
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=your_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=your_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model is running on: {next(model.parameters()).device}")

    dataset_list = ["medical_questions_pairs", "PedroCJardim/QASports", "trivia_qa", "math_qa"]
    for d in dataset_list:
        torch.cuda.empty_cache()

        if d == "trivia_qa":
            dataset = load_dataset(d, "rc", split='train[:10000]')
        elif d == "medical_questions_pairs":
            dataset = load_dataset(d, split='train[1873:3040]')
        else:
            dataset = load_dataset(d, split='train[:10000]')

        process_dataset(d)
        print(d + " Dataset processing complete.")
