from torch import nn
import torch
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from collections import defaultdict
from typing import Optional
from datetime import datetime

class ToxicCommentsTransformer(pl.LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        id2label = {0: "NON-TOXIC", 1: "TOXIC"}
        label2id = {"NON-TOXIC": 0, "TOXIC": 1}

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels, id2label=id2label, label2id=label2id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.outputs = defaultdict(list)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)

        labels = batch["labels"]

        self.outputs[dataloader_idx].append({"loss": val_loss, "preds": preds, "labels": labels})

    def on_validation_epoch_end(self):
        for dataloader_idx, output_list in self.outputs.items():
            losses = torch.stack([x["loss"] for x in output_list])
            preds = torch.cat([x["preds"] for x in output_list])
            labels = torch.cat([x["labels"] for x in output_list])
            
            avg_loss = losses.mean()
            accuracy = (preds == labels).float().mean()
            
            self.log(f"val_loss_{dataloader_idx}", avg_loss, prog_bar=True)
            self.log(f"val_accuracy_{dataloader_idx}", accuracy, prog_bar=True)
        self.outputs.clear()

    # def configure_optimizers(self):
    #     """Prepare optimizer and schedule (linear warmup and decay)."""
    #     model = self.model
    #     no_decay = ["bias", "LayerNorm.weight"]
    #     optimizer_grouped_parameters = [
    #         {
    #             "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #             "weight_decay": self.hparams.weight_decay,
    #         },
    #         {
    #             "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
    #             "weight_decay": 0.0,
    #         },
    #     ]
    #     optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=self.hparams.warmup_steps,
    #         num_training_steps=self.trainer.estimated_stepping_batches,
    #     )
    #     scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
    #     return [optimizer], [scheduler]
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        return optimizer


if __name__ == "__main__":
    model = ToxicCommentsTransformer(
        model_name_or_path="vinai/bertweet-base",
        num_labels=2,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "vinai/bertweet-base",
    )

    test_sentence = "This is a test sentence."
    inputs = tokenizer(test_sentence, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits

    predicted_class_id = predictions.argmax().item()
    print('Label:', model.model.config.id2label[predicted_class_id])
    print(f"Output shape of model: {model(**inputs).logits.shape}")
