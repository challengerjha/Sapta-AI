# sapta/train.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch

def main():
    # Load dataset — using a small built-in dataset for demo
    dataset = load_dataset("imdb", split={"train": "train[:5%]", "test": "test[:2%]"})

    # Load tokenizer and tokenize the dataset
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_total_limit=1,
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()