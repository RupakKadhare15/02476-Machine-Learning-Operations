import wandb
import hydra
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from toxic_comments.model import ToxicCommentsTransformer
from toxic_comments.datamodule import ToxicCommentsDataModule


@hydra.main(version_base=None, config_path='../../configs', config_name='evaluation.yaml')
def main(cfg):
    """Evaluate the model using the best checkpoint from WandB Artifacts."""
    with wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group='evaluation'
    ) as run:
        artifact = run.use_artifact(cfg.model_artifact_path, type='model')
        artifact_dir = artifact.download()
        model = ToxicCommentsTransformer.load_from_checkpoint(f"{artifact_dir}/best-checkpoint.ckpt")
        model.eval()

        # initialize the Trainer
        trainer = Trainer(logger=WandbLogger(project=cfg.wandb.project))
        dm = ToxicCommentsDataModule()

        # test the model
        trainer.test(model,dm)
        logger.info("Model evaluation completed.")


if __name__ == "__main__":
    main()