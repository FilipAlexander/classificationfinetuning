import wandb
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from custom_trainer import CustomTrainer
import json
import sys
import math
import torch

torch.manual_seed(42)

MODEL_NAME = 'xlm-roberta-large'
DEVICE = 'cuda'

dataset_test = load_from_disk('./data/emotion_dataset_test_ready.jsonl')
dataset_train = load_from_disk('./data/emotion_dataset_train_ready.jsonl')

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, use_fast=True, add_prefix_space=True, truncation=True, padding=False)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer, max_length=512
)

with open('./data/class_name_to_labels.json', 'r') as fh:
    class_name_to_labels = json.load(fh)

num_classes = len(class_name_to_labels)
id2label = {v: k for k, v in class_name_to_labels.items()}
label2id = class_name_to_labels

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    wandb.log({
        'accuracy': acc,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_micro': f1_micro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro
    })
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_micro': f1_micro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro
    }


if __name__ == '__main__':
    wandb.init(project='labcafe-finetuning-emotion')

    if len(sys.argv) == 1:
        wandb.config.epochs = 5
        wandb.config.batch_size = 16
        wandb.config.learning_rate = 1e-5
        wandb.config.gradient_accumulation = 1
        wandb.config.weight_decay = 0.3

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes,
                                                                id2label=id2label,
                                                                label2id=label2id).to(DEVICE)
    
    run_name = f'{MODEL_NAME}-lr_{wandb.config.learning_rate}-bs_{wandb.config.batch_size}-epochs_{wandb.config.epochs}-grad_acc_{wandb.config.gradient_accumulation}-weight_decay_{wandb.config.weight_decay}'
    
    training_args = TrainingArguments(
        output_dir=f'./results/{run_name}',
        num_train_epochs=wandb.config.epochs,
        per_device_train_batch_size=wandb.config.batch_size,
        per_device_eval_batch_size=wandb.config.batch_size,
        warmup_steps=math.ceil(len(dataset_train) *
                            wandb.config.epochs / wandb.config.batch_size * 0.1),
        logging_dir='./logs',
        load_best_model_at_end=True,
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=1000,
        learning_rate=wandb.config.learning_rate,
        weight_decay=wandb.config.weight_decay,
        save_steps=1000,
        save_total_limit=2,
        metric_for_best_model='f1_macro',
        gradient_accumulation_steps=wandb.config.gradient_accumulation,
        # fp16=True
    )

    # Do not use custom class weights
    
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset_train,
    #     eval_dataset=dataset_test,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    # )

    # Use custom class weights
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
    
    trainer.save_model(f'./models/{run_name}_sota')







    
    