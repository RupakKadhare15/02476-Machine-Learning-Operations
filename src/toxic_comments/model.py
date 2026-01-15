from collections import defaultdict

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


class ToxicCommentsTransformer(pl.LightningModule):

    """PyTorch Lightning Module for toxic comments classification using a transformer model."""

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        **kwargs,
    ):
        """
        Initialize the ToxicCommentsTransformer.

        Args:
        ----
            model_name_or_path: Pretrained model identifier
            num_labels: Number of labels for classification
            learning_rate: Learning rate for the optimizer
            adam_epsilon: Epsilon value for the AdamW optimizer
            **kwargs: Additional keyword arguments
        """
        super().__init__()

        self.save_hyperparameters()

        id2label = {0: 'NON-TOXIC', 1: 'TOXIC'}
        label2id = {'NON-TOXIC': 0, 'TOXIC': 1}

        self.config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=num_labels, id2label=id2label, label2id=label2id
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.model.train()  # Ensure model is in training mode
        self.outputs = defaultdict(list)

    def forward(self, **inputs):
        """Forward pass through the model."""
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        """Perform a training step."""
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Perform a validation step."""
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)

        labels = batch['labels']

        self.outputs[dataloader_idx].append({'loss': val_loss, 'preds': preds, 'labels': labels})

    def test_step(self, batch, batch_idx):
        """Perform a test step."""
        outputs = self(**batch)
        test_loss, logits = outputs[:2]
        preds = torch.argmax(logits, axis=1)

        labels = batch['labels']

        self.log("test_loss", test_loss, prog_bar=True)
        accuracy = (preds == labels).float().mean()
        self.log("test_accuracy", accuracy, prog_bar=True)

    def on_validation_epoch_end(self):
        """Compute and log validation metrics at the end of the epoch."""
        for dataloader_idx, output_list in self.outputs.items():
            losses = torch.stack([x['loss'] for x in output_list])
            preds = torch.cat([x['preds'] for x in output_list])
            labels = torch.cat([x['labels'] for x in output_list])

            avg_loss = losses.mean()
            accuracy = (preds == labels).float().mean()

            self.log("val_loss", avg_loss, prog_bar=True)
            self.log("val_accuracy", accuracy, prog_bar=True)
        self.outputs.clear()

    def configure_optimizers(self):
        """Configure the optimizer."""
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        return optimizer


if __name__ == '__main__':
    model = ToxicCommentsTransformer(
        model_name_or_path='vinai/bertweet-base',
        num_labels=2,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        'vinai/bertweet-base',
    )

    test_sentence = 'This is a test sentence.'
    inputs = tokenizer(test_sentence, return_tensors='pt')
    outputs = model(**inputs)
    predictions = outputs.logits

    predicted_class_id = predictions.argmax().item()
    print('Label:', model.model.config.id2label[predicted_class_id])
    print(f'Output shape of model: {model(**inputs).logits.shape}')
