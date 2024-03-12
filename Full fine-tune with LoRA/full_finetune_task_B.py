from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments

from random import randrange

import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

from datasets import load_dataset
import evaluate

nltk.download("punkt")

# Expt and Base model
expt_id = "full_finetune_task_B" #replace it for task_A
model_id= "google/flan-t5-large"
repository_id = f"{model_id.split('/')[-1]}-{expt_id}"

# Data
"""
This script is a full-finetune on task_B, where we consider that we have already full-finetuned on task_A as our base model.

For Task_A full-finetune would be:
train_file = "/content/data/task_A/task_A_train_ds.jsonl"
validation_file = "/content/data/task_A/task_A_test_ds.jsonl"

Results of this task aren't required as part of the current research, since we have a test set of A that we are evaluating over each epoch
to measure the performance of the finetuned model over the base model knowledge representations
"""

train_file = "/content/data/task_A/task_B_train_ds.jsonl" 
validation_file = "/content/data/task_A/task_A_test_ds.jsonl"

# Hyperparameters
LEARNING_RATE = 2e-5
NUM_EPOCHS = 4
MAX_SOURCE_LENGTH =  768
MAX_TARGET_LENGTH = 512

# Load dataset
dataset = load_dataset(
    "json", data_files={"train": train_file, "eval": validation_file}
)

# Data preprocessing
tokenizer = AutoTokenizer.from_pretrained(model_id)

def preprocess_function_details(sample, padding="max_length", prefix=''):
    # add prefix to the input for t5
    inputs = [prefix + item for item in sample["input"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=MAX_SOURCE_LENGTH, padding=padding, truncation=True, return_tensors="pt")

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["output"], max_length=MAX_TARGET_LENGTH, padding=padding, truncation=True, return_tensors="pt")

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset_task_B = dataset['train'].map(
    lambda sample: preprocess_function_details(sample, prefix="Question: "),
    batched=True,
    remove_columns=["input", "output"],
    desc="Running tokenizer on dataset"
)

tokenized_dataset_task_A = dataset['eval'].map(
    lambda sample: preprocess_function_details(sample, prefix="Summarize: "),
    batched=True,
    remove_columns=["input", "output"],
    desc="Running tokenizer on dataset"
)

train_dataset = tokenized_dataset_task_B
eval_dataset = tokenized_dataset_task_A

# Load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to('cuda')

# Metric
metric = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
bleuscore = evaluate.load("bleu")

# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(references=decoded_labels, predictions=decoded_preds, use_stemmer=True)
    bleu_score = bleuscore.compute(references=decoded_labels, predictions=decoded_preds)
    bert_score = bertscore.compute(references = decoded_labels, predictions=decoded_preds, model_type="distilbert-base-uncased")
    result = {k: 100 for k, v in result.items()}
    result['bleuscore'] = bleu_score
    result['bertscore'] = {"precision": sum(bert_score['precision'])/len(bert_score['precision']), 'f1': sum(bert_score['f1'])/len(bert_score['f1'])}
    return result

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=repository_id,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    predict_with_generate=True,
    fp16=False,
    evaluation_strategy="epochs",
    save_strategy="epochs",
    save_total_limit=1,
    push_to_hub=False
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

model.config.use_cache = False  

# train model
result = trainer.train()

# Save
# trainer.model.save_pretrained(repository_id)
# tokenizer.save_pretrained(repository_id)
# trainer.create_model_card()