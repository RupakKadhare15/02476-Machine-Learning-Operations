import hydra
import pytorch_lightning as pl

from toxic_comments.datamodule import ToxicCommentsDataModule
from toxic_comments.model import ToxicCommentsTransformer


@hydra.main(version_base=None, config_path='../../configs', config_name='training.yaml')
def main(cfg):
    """Train the model."""
    # Initialize datamodule
    datamodule = ToxicCommentsDataModule(
        model_name_or_path=cfg.model_name_or_path,
        data_dir='data',
        train_batch_size=cfg.batch_size,
        eval_batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        num_workers=cfg.num_workers,
    )

    # Initialize model
    model = ToxicCommentsTransformer(
        model_name_or_path=cfg.model_name_or_path,
        num_labels=2,
        learning_rate=cfg.learning_rate,
        adam_epsilon=cfg.adam_epsilon,
    )

    # Train with PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=cfg.epochs, limit_train_batches=10, limit_val_batches=10, log_every_n_steps=10)
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    main()
