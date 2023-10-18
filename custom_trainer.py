from torch import nn
from transformers import Trainer
import torch
import json


with open('./data/class_weights.json', 'r') as fh:
    class_weights = json.load(fh)

cv_list = sorted(class_weights.items(), key=lambda x: int(x[0]), reverse=False)
weight_list = [item[1] for item in cv_list]


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(weight_list).to('cuda'))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss