from random import randrange
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments
import json
import evaluate
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize

nltk.download("punkt")

LEARNING_RATE = 2e-5
NUM_EPOCHS = 20
MAX_SOURCE_LENGTH =  768
MAX_TARGET_LENGTH = 512

"""
Current script assumes that the base-pretrained-model has be previously finetuned on task_A using LoRA
"""
expt_id = "lora_task_B" #replace it for task_A
model_id= "/content/flan-t5-large-task-A-lora-finetune" #for task A use the base model "google/flan-t5-large"
repository_id = f"{model_id.split('/')[-1]}-{expt_id}"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)

#load ds
ds_A = []
with open("/content/data/task_A/task_A_test_ds.jsonl", 'r') as file:
  for line in file:
    ds_A.append("Summarize: " + json.loads(line)['input'])

ds_B = load_dataset(
    "json", data_files={"train": "/content/data/task_B/task_B_train_ds.jsonl"}
)

def preprocess_function_details(sample, padding="max_length", prefix=''):
    # add prefix to the input for t5
    inputs = [prefix + item for item in sample["input"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=MAX_SOURCE_LENGTH, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["output"], max_length=MAX_TARGET_LENGTH, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset_ds_B = ds_B.map(
    lambda sample: preprocess_function_details(sample, prefix="Question: "),
    batched=True,
    remove_columns=["input", "output"],
    desc="Running tokenizer on dataset"
)

tokenized_dataset_ds_A = ds_A.map(
    lambda sample: preprocess_function_details(sample, prefix=""), #prefix is empty since the "Summarize: " is already included
    batched=True,
    remove_columns=["input", "output"],
    desc="Running tokenizer on dataset"
)

train_dataset = tokenized_dataset_ds_B
eval_dataset = tokenized_dataset_ds_A

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


label_pad_token_id = -100

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

training_args = Seq2SeqTrainingArguments(
    output_dir="flant5-large-task_B-peft-lora",
    auto_find_batch_size=True,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    evaluation_strategy="epochs",
    logging_steps=500,
    save_strategy="no",
)

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

result = trainer.train()

# Save
# trainer.model.save_pretrained(repository_id)
# tokenizer.save_pretrained(repository_id)
# trainer.create_model_card()