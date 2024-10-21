import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset (make sure to replace this with your dataset path)
data = [
    {"text": "I want to enter the milk", "intent": "milk_entry"},
    {"text": "Can you record milk entry?", "intent": "milk_entry"},
    {"text": "I want to add milk", "intent": "milk_entry"},
    {"text": "Improve the milk quality", "intent": "milk_quality_improvement"},
    {"text": "Enhance milk quality", "intent": "milk_quality_improvement"},
    # Add more examples for each intent
]

df = pd.DataFrame(data)
train_texts, val_texts, train_labels, val_labels = train_test_split(df["text"], df["intent"], test_size=0.2)

# Tokenizer and Model Initialization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Encoding data
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True, max_length=128)

# Converting labels to numeric format
label_mapping = {"milk_entry": 0, "milk_quality_improvement": 1}
train_labels = [label_mapping[label] for label in train_labels]
val_labels = [label_mapping[label] for label in val_labels]

# Create torch datasets
class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IntentDataset(train_encodings, train_labels)
val_dataset = IntentDataset(val_encodings, val_labels)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Training
trainer.train()

# Save the trained model
trainer.save_model("./intent_classification_model")
tokenizer.save_pretrained("./intent_classification_model")

# Load model for inference
model = BertForSequenceClassification.from_pretrained("./intent_classification_model")
tokenizer = BertTokenizer.from_pretrained("./intent_classification_model")

def predict_intent(text, threshold=0.6):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    
    # Get the predicted class and its probability
    predicted_class_id = torch.argmax(probabilities).item()
    confidence_score = probabilities[0][predicted_class_id].item()
    
    # Reverse mapping from ID to intent
    intent_mapping = {0: "milk_entry", 1: "milk_quality_improvement"}
    
    if confidence_score < threshold:
        return "fallback_to_gemini", confidence_score
    return intent_mapping[predicted_class_id], confidence_score

# Example usage
user_input = "I want to enter the milk"
recognized_intent, confidence = predict_intent(user_input)

if recognized_intent == "fallback_to_gemini":
    # Call Google Gemini to handle the response
    print("Using Google Gemini for response due to low confidence.")
else:
    print(f"Recognized Intent: {recognized_intent} with confidence {confidence}")
