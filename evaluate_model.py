import torch
from datasets import load_dataset, Dataset
from evaluate_model import load
from transformers import RobertaForQuestionAnswering, RobertaTokenizer
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = RobertaForQuestionAnswering.from_pretrained("./saved_model").to(device)
tokenizer = RobertaTokenizer.from_pretrained("./saved_model")

datasetData = load_dataset("issai/kazqad")
df_val = datasetData["validation"].to_pandas().head(1000)
val_dataset = Dataset.from_pandas(df_val)

def tokenize_function(examples):
    return tokenizer(
        examples['question'],
        examples['context'],
        truncation=True,
        padding="max_length"
    )

# Tokenize validation dataset
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# Generate predictions
model.eval()
start_logits = []
end_logits = []
with torch.no_grad():
    for item in tokenized_val:
        input_ids = torch.tensor([item["input_ids"]]).to(device)
        outputs = model(input_ids)
        start_logits.append(outputs.start_logits.cpu().numpy())
        end_logits.append(outputs.end_logits.cpu().numpy())

# Convert logits to indices
predicted_start_indices = np.argmax(start_logits, axis=-1).flatten()
predicted_end_indices = np.argmax(end_logits, axis=-1).flatten()

# Post-process predictions
predicted_answers = []
true_answers = []
for i in range(len(tokenized_val)):
    input_ids = tokenized_val[i]["input_ids"]
    start_idx = predicted_start_indices[i]
    end_idx = predicted_end_indices[i]
    predicted_answer = tokenizer.decode(input_ids[start_idx:end_idx+1], skip_special_tokens=True)
    predicted_answers.append(predicted_answer)
    true_answer = val_dataset[i]["answers"]["text"][0]
    true_answers.append(true_answer)

# Load F1 and EM metrics
metric_f1 = load("f1")
metric_em = load("exact_match")

# Calculate F1 and EM
f1_score = metric_f1.compute(predictions=predicted_answers, references=true_answers)
em_score = metric_em.compute(predictions=predicted_answers, references=true_answers)

print(f"F1 Score: {f1_score['f1']}")
print(f"Exact Match Score: {em_score['exact_match']}")