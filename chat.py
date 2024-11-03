from transformers import RobertaForQuestionAnswering, RobertaTokenizerFast
import torch
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = os.path.dirname(__file__)
print(script_dir)
model_path = os.path.join(script_dir, "./results/checkpoint-375")
print(model_path) 
model = RobertaForQuestionAnswering.from_pretrained(model_path, torch_dtype = torch.float16, use_safetensors=True).to(device)
tokenizer = RobertaTokenizerFast.from_pretrained("nur-dev/roberta-kaz-large")

def chat_with_model(question, context):
    # Tokenize the input question and context
    inputs = tokenizer(
        question,
        context,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    ).to(device)
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the most probable start and end positions
    start_logits = outputs.start_logits.cpu().numpy()
    end_logits = outputs.end_logits.cpu().numpy()
    start_index = np.argmax(start_logits)
    end_index = np.argmax(end_logits)

    # Decode the answer
    input_ids = inputs["input_ids"].cpu().numpy()[0]
    answer = tokenizer.decode(input_ids[start_index:end_index + 1], skip_special_tokens=True)
    
    return answer

print("Start chatting with the model (type 'exit' to stop).")

question = "Әйгілі 'Мона Лиза' картинасы қайда қойылған?"
context = "Мо́на Ли́за (Mona Lisa) — бұл шамамен 1503 жылдары итальяндық суретші Леонардо да Винчидің салған суреті. Бұл сурет әлемдегі ең танымалы көркем суреттердің бірі болып табылады. Қайта өркендеу заманына жатады. Луврда (Франция, Париж) қойылған."

answer = chat_with_model(question, context)
print("Model:", answer)