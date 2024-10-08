import os
import numpy as np
import wandb
from datasets import load_dataset
from transformers import (
  AutoModelForSequenceClassification,
  AutoTokenizer,
  Trainer,
  TrainingArguments,
  DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score

wandb.init(project="deberta-v3-xsmall-mnli")

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
  accuracy = accuracy_score(labels, predictions)
  f1 = f1_score(labels, predictions, average="weighted")
  wandb.log({"accuracy": accuracy, "f1": f1})
  return {
    "accuracy": accuracy,
    "f1": f1,
  }

training_args = TrainingArguments(
  output_dir="./results",
  num_train_epochs=3,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=32,
  warmup_steps=500,
  logging_dir="./logs",
  logging_steps=100,
  evaluation_strategy="steps",
  eval_steps=500,
  save_steps=1000,
  load_best_model_at_end=True,
  report_to="wandb",
)

trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokinzed_dataset["train"],
  eval_dataset=tokinzed_dataset["validation_matched"],
  tokenizer=tokenizer,
  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
  compute_metrics=compute_metrics,
)

trainer.train()

eval_results = trainer.evaluate()
print(eval_results)

wandb.log(eval_results)

model.save_pretrained("./fine_tuned_deberta_xsmall_mnli")
tokenizer.save_pretrained("./fine_tuned_deberta_xsmall_mnli")

wandb.finish()