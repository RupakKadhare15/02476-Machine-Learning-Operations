"""Script to find the latest model and upload it to wandb model registry."""

import os
from pathlib import Path

import hydra
from dotenv import load_dotenv
from loguru import logger

import wandb


def find_latest_model(model_dir: str = 'models', extensions: list[str] = None) -> Path | None:
    """
    Find the latest model file in the specified directory.

    Args:
        model_dir: Directory to search for models (default: "models")
        extensions: List of file extensions to consider (default: [".ckpt", ".pth", ".pt"])

    Returns:
        Path to the latest model file, or None if no models found

    """
    if extensions is None:
        extensions = ['.ckpt', '.pth', '.pt']

    model_path = Path(model_dir)
    if not model_path.exists():
        logger.error(f'Model directory {model_dir} does not exist')
        return None

    # Find all model files with specified extensions
    model_files = []
    for dirpath, _, filenames in os.walk(model_path):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                model_files.append(Path(dirpath) / filename)

    if not model_files:
        logger.warning(f'No model files found in {model_dir}')
        return None

    # Sort by modification time and get the latest
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    logger.info(f'Found latest model: {latest_model}')
    logger.info(f'Last modified: {latest_model.stat().st_mtime}')

    return latest_model


def upload_model_to_registry(
    model_path: Path,
    artifact_name: str = 'bert-toxic-comments-classifier',
    artifact_type: str = 'model',
    description: str = None,
    aliases: list[str] = None,
    metadata: dict = None,
) -> None:
    """
    Upload a model to wandb model registry.

    Args:
        model_path: Path to the model file
        artifact_name: Name for the artifact in wandb
        artifact_type: Type of artifact (default: "model")
        description: Optional description for the artifact
        aliases: Optional list of aliases (e.g., ["latest", "production"])
        metadata: Optional metadata dictionary

    """
    if not model_path.exists():
        logger.error(f'Model file {model_path} does not exist')
        return

    # Initialize wandb run
    with wandb.init(
        entity=os.environ.get('WANDB_ENTITY'), project=os.environ.get('WANDB_PROJECT'), job_type='model-upload'
    ) as run:
        logger.info(f'Uploading model {model_path.name} to wandb model registry...')

        # Create artifact
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            description=description or f'Model checkpoint: {model_path.name}',
            metadata=metadata or {'model_file': model_path.name},
        )

        # Add model file to artifact
        artifact.add_file(str(model_path), name=model_path.name)

        # Log artifact with aliases
        if aliases is None:
            aliases = ['latest']

        run.log_artifact(artifact, aliases=aliases)

        # Specify the collection and registry to link the artifact to
        REGISTRY_NAME = 'ToxicTweets'
        COLLECTION_NAME = 'models'
        target_path = f'wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}'

        # Link the artifact to the collection
        run.link_artifact(artifact=artifact, target_path=target_path)

        logger.success(f'Successfully uploaded {model_path.name} to wandb model registry')
        logger.info(f'Artifact name: {artifact_name}')
        logger.info(f'Aliases: {aliases}')


@hydra.main(version_base=None, config_path='../../configs', config_name='training.yaml')
def main(cfg):
    """Main function to find latest model and upload to wandb registry."""
    # Find the latest model
    latest_model = find_latest_model()
    load_dotenv()  # Load environment variables from .env file
    os.environ['WANDB_ENTITY'] = cfg.wandb.entity
    os.environ['WANDB_PROJECT'] = cfg.wandb.project
    wandb.login()

    if latest_model is None:
        logger.error('No model found to upload')
        return

    # Upload to wandb model registry
    upload_model_to_registry(
        model_path=latest_model,
        artifact_name='bert-toxic-comments-classifier',
        artifact_type='model',
        description=f'Latest BERT toxic comments classifier model: {latest_model.name}',
        aliases=['latest', 'production'],
        metadata={
            'model_file': latest_model.name,
            'file_size_bytes': latest_model.stat().st_size,
        },
    )


if __name__ == '__main__':
    main()
