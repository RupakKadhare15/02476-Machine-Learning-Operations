import os

from dotenv import load_dotenv
from invoke import Context, task

WINDOWS = os.name == 'nt'
PROJECT_NAME = 'toxic_comments'
PYTHON_VERSION = '3.12'


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f'uv run src/{PROJECT_NAME}/data.py data/raw data/processed', echo=True, pty=not WINDOWS)


@task
def train(ctx: Context, data_dir: str | None = None) -> None:
    """Train model."""
    command = f'uv run src/{PROJECT_NAME}/train.py'
    if data_dir:
        # Pass data_dir using Hydra override syntax (no leading `--`, uses `=`).
        command += f' data_dir={data_dir}'
    ctx.run(command, echo=True, pty=not WINDOWS)
    ctx.run(f'uv run src/{PROJECT_NAME}/log_artifact.py', echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run('uv run coverage run -m pytest tests/', echo=True, pty=not WINDOWS)
    ctx.run('uv run coverage report -m -i', echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = 'plain') -> None:
    """Build docker images."""
    ctx.run(
        f'docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress} --platform linux/amd64',
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f'docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress} --platform linux/amd64',
        echo=True,
        pty=not WINDOWS,
    )


@task
def docker_push(ctx: Context) -> None:
    """Push docker images to artifact registry."""
    load_dotenv(override=True)
    artifactory = os.getenv('ARTIFACTORY')
    if not artifactory:
        raise ValueError('ARTIFACTORY environment variable is not set.')

    ctx.run(f'docker tag train:latest {artifactory}/train:latest', echo=True, pty=not WINDOWS)
    ctx.run(f'docker push {artifactory}/train:latest', echo=True, pty=not WINDOWS)

    ctx.run(f'docker tag api:latest {artifactory}/api:latest', echo=True, pty=not WINDOWS)
    ctx.run(f'docker push {artifactory}/api:latest', echo=True, pty=not WINDOWS)


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run('uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build', echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run('uv run mkdocs serve --config-file docs/mkdocs.yaml', echo=True, pty=not WINDOWS)
