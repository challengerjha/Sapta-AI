# examples/inference.py

from transformers import pipeline

# Load sentiment analysis pipeline (uses pretrained model from Hugging Face)
classifier = pipeline("sentiment-analysis")

# Test input
result = classifier("Sapta AI is going open-source! 🚀")

# Output result
print(result)
