import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loading the dataset
texts = [
    "There is a cow in the picture.",
    "I see a tiger.",
    "A dog is running in the park.",
    "An elephant is walking on the road."
]
labels = [
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # cow
    [0, 0, 0, 2, 0],  # tiger
    [0, 1, 0, 0, 0, 0, 0, 0],  # dog
    [0, 1, 0, 0, 0, 0, 0, 0, 0]  # elephant
]

# Convert to DataFrame
df = pd.DataFrame({'text': texts, 'labels': labels})
# Convert to Dataset
dataset = Dataset.from_pandas(df)

# Divide into train/test
split_dataset = dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Defining the model and tokenizer
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)

# Tokenization function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["text"], padding=True, truncation=True, is_split_into_words=False)
    labels_aligned = []
    
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = []
        
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)  # Skip special tokens
            else:
                aligned_labels.append(label[word_id] if word_id < len(label) else -100)
        
        labels_aligned.append(aligned_labels)
    
    tokenized_inputs["labels"] = labels_aligned
    return tokenized_inputs

# Dataset tokenization
train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
eval_dataset = eval_dataset.map(tokenize_and_align_labels, batched=True)

# Setting up training arguments
training_args = TrainingArguments(
    output_dir="./ner_model",
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1_000,
    use_cpu=True, 
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Model training
trainer.train()

# Saving the model and tokenizer
model.save_pretrained("./ner_model")
tokenizer.save_pretrained("./ner_model")