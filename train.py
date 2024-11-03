import torch
from datasets import load_dataset, Dataset
from transformers import RobertaForQuestionAnswering, RobertaTokenizer, RobertaTokenizerFast, AdamW, Trainer, TrainingArguments
import numpy as np
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

datasetData = load_dataset("issai/kazqad")
df = datasetData["train"].to_pandas()
df = df.head(1000)
dataset = Dataset.from_pandas(df)
df = datasetData["validation"].to_pandas()
df = df.head(1000)
dataset2 = Dataset.from_pandas(df)

model = RobertaForQuestionAnswering.from_pretrained("nur-dev/roberta-kaz-large").to(device)

tokenizer = RobertaTokenizer.from_pretrained("nur-dev/roberta-kaz-large")

def tokenize_function(examples):
    # Tokenize the question and context
    tokenized = tokenizer(
        examples['question'],
        examples['context'],
        truncation=True,
        padding="max_length"
    )

    # Initialize lists for start and end positions
    start_positions = []
    end_positions = []

    # Loop through the answers in the batch
    for i in range(len(examples['answers'])):
        answer_text = examples['answers'][i]['text'][0]  # Get the first answer text
        answer_start = examples['answers'][i]['answer_start'][0]  # Get the corresponding start position

        # Calculate the end position of the answer
        end_position = answer_start + len(answer_text) - 1

        start_positions.append(answer_start)
        end_positions.append(end_position)

    # Add start and end positions to the tokenized dataset
    tokenized['start_positions'] = start_positions
    tokenized['end_positions'] = end_positions
    return tokenized

tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_datasets2 = dataset2.map(tokenize_function, batched=True)

print (tokenized_datasets)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets2,
)

trainer.train()
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, "./saved_model")
model.save_pretrained(model_path)
tokenizer.save_pretrained("./saved_model")