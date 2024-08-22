import os
import numpy as np
from datasets import load_dataset
from transformers import (
  AutoModelForSequenceClassification,
  AutoTokenizer,
  Trainer,
  TrainingArguments,
  DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score

model_name = "microsoft/deberta-v3-xsmall"
task = "mnli"
num_labels = 3

dataset = load_dataset("nyu-mll/multi_nli")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

def preprocess_function(examples):
  return tokenizer(
    examples["hypothesis"],
    examples["premise"],
    truncation=True,
    max_length=256,
  )

tokinzed_dataset = dataset.map(preprocess_function, batched=True)

def compute_metrics(eval_pred):
  predictions, labels = eval_pred
  predictions = np.argmax(predictions, axis=1)
  return {
    "accuracy": accuracy_score(labels, predictions),
    "f1": f1_score(labels, predictions, average="weighted"),
  }

training_args = TrainingArguments(
  output_dir="./results",
  num_train_epochs=3,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=64,
  warmup_steps=500,
  logging_dir="./logs",
  logging_steps=100,
  evaluation_strategy="steps",
  eval_steps=500,
  save_steps=1000,
  load_best_model_at_end=True,
)

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokinzed_dataset["train"],
  eval_dataset=tokinzed_dataset["validation"],
  compute_metrics=compute_metrics,
)