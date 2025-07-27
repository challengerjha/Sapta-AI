# sapta/evaluate.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import load_dataset, load_metric
import numpy as np

def main():
    # Load model and tokenizer
    model_name_or_path = "./checkpoints"  # or use a model hub name like "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)

    # Load evaluation data
    dataset = load_dataset("imdb", split="test[:2%]")

    # Tokenize dataset
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # Load metric
    metric = load_metric("accuracy")

    # Define compute_metrics function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Setup Trainer for evaluation
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Evaluate
    results = trainer.evaluate(eval_dataset=tokenized_dataset)
    print("Evaluation Results:", results)

if __name__ == "__main__":
    main()
