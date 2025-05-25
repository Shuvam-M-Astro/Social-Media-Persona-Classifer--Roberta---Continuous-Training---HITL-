import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from datasets import Dataset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    if torch.cuda.is_available():
        logger.info("GPU in use: %s", torch.cuda.get_device_name(0))
    else:
        logger.info("GPU not available. Running on CPU.")

    # Load datasets
    try:
        df_main = pd.read_csv("persona_dataset.csv")
        df_main["text"] = df_main["bio"] + " " + df_main["posts"]
        df_main = df_main[["text", "label"]]
        logger.info("Loaded main dataset with %d records.", len(df_main))
    except Exception as e:
        logger.error("Failed to load persona_dataset.csv: %s", e)
        return

    if os.path.isfile("result.csv"):
        try:
            df_feedback = pd.read_csv("result.csv")
            df_feedback["text"] = df_feedback["bio"] + " " + df_feedback["posts"]
            df_feedback = df_feedback[["text", "label"]]
            logger.info("Loaded %d feedback samples.", len(df_feedback))
        except Exception as e:
            logger.warning("Failed to load result.csv: %s", e)
            df_feedback = pd.DataFrame(columns=["text", "label"])
    else:
        logger.info("No feedback data found.")
        df_feedback = pd.DataFrame(columns=["text", "label"])

    # Combine and deduplicate
    df = pd.concat([df_main, df_feedback], ignore_index=True)
    df.drop_duplicates(subset=["text", "label"], inplace=True)
    logger.info("Total samples after merge and deduplication: %d", len(df))

    # Oversample underrepresented classes
    min_count = 10
    counts = df["label"].value_counts()
    for label, count in counts.items():
        if count < min_count:
            to_add = df[df["label"] == label]
            repeats = (min_count - count) // count + 1
            df = pd.concat([df, to_add] * repeats, ignore_index=True)
    logger.info("Total samples after oversampling: %d", len(df))

    # Save processed dataset
    df.to_csv("processed_personas.csv", index=False)
    df["label"].value_counts().to_csv("label_distribution.csv")

    # Label mappings
    label2id_path = "label2id.json"
    id2label_path = "id2label.json"
    if os.path.isfile(label2id_path) and os.path.isfile(id2label_path):
        with open(label2id_path, "r") as f:
            label2id = json.load(f)
        with open(id2label_path, "r") as f:
            id2label = {int(k): v for k, v in json.load(f).items()}
        logger.info("Loaded existing label mappings.")
    else:
        label2id = {}
        id2label = {}
        logger.info("No existing label mappings found. Starting fresh.")

    next_id = max(id2label.keys(), default=-1) + 1
    for label in sorted(df["label"].unique()):
        if label not in label2id:
            label2id[label] = next_id
            id2label[next_id] = label
            next_id += 1

    with open(label2id_path, "w") as f:
        json.dump(label2id, f)
    with open(id2label_path, "w") as f:
        json.dump(id2label, f)
    logger.info("Updated and saved label mappings.")

    # Encode labels
    def encode_labels(example):
        example["label"] = label2id[example["label"]]
        return example

    # Tokenize
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(encode_labels)
    tokenized_dataset = dataset.map(tokenize_function)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, shuffle=True)

    # Load model
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"models/roberta_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    os.makedirs("final_roberta_persona", exist_ok=True)
    model.save_pretrained("final_roberta_persona")
    tokenizer.save_pretrained("final_roberta_persona")
    logger.info("Final model saved to final_roberta_persona")

    with open(f"{output_dir}/training_summary.txt", "w") as f:
        f.write(f"Trained on {len(df)} samples across {len(label2id)} classes\n")
        for i, label in id2label.items():
            f.write(f"{i}: {label}\n")

if __name__ == "__main__":
    train_model()