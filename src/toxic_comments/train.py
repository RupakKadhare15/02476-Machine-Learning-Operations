from toxic_comments.datamodule import ToxicCommentsDataModule
from toxic_comments.model import ToxicCommentsTransformer
import pytorch_lightning as pl

def main():
    # Initialize datamodule
    datamodule = ToxicCommentsDataModule(
        data_dir="data",
        train_batch_size=32,
        eval_batch_size=32,
        max_length=128,
        num_workers=0
    )

    # Initialize model
    model = ToxicCommentsTransformer(
        model_name_or_path="vinai/bertweet-base",
        num_labels=2
    )

    # Train with PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=1,
                        limit_train_batches=10,
                        limit_val_batches=10,
                        log_every_n_steps=10)
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()
