# tests/test_tasks.py
"""Tests for tasks.py invoke commands."""

import os
from unittest.mock import Mock, patch

import pytest
from invoke import Context


class TestDockerPush:
    """Test suite for docker_push invoke task."""

    @pytest.fixture
    def mock_context(self):
        """Create a mocked Context for testing."""
        context = Mock(spec=Context)
        context.run = Mock()
        return context

    def test_docker_push_missing_artifactory_env(self, mock_context):
        """Test docker_push raises ValueError when ARTIFACTORY env var is not set."""
        import tasks

        # Mock load_dotenv and ensure ARTIFACTORY is not set
        with patch('tasks.load_dotenv'), patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match='ARTIFACTORY environment variable is not set.'):
                tasks.docker_push(mock_context)

    def test_docker_push_calls_docker_tag(self, mock_context):
        """Test docker_push calls docker tag with correct arguments."""
        import tasks

        artifactory = 'us-central1-docker.pkg.dev/my-project/my-repo'

        with patch('tasks.load_dotenv'), patch.dict(os.environ, {'ARTIFACTORY': artifactory}):
            tasks.docker_push(mock_context)

            # Check that docker tag was called correctly
            expected_tag_command = f'docker tag train:latest {artifactory}/train:latest'
            mock_context.run.assert_any_call(expected_tag_command, echo=True, pty=not tasks.WINDOWS)

    def test_docker_push_calls_docker_push(self, mock_context):
        """Test docker_push calls docker push with correct arguments."""
        import tasks

        artifactory = 'us-central1-docker.pkg.dev/my-project/my-repo'

        with patch('tasks.load_dotenv'), patch.dict(os.environ, {'ARTIFACTORY': artifactory}):
            tasks.docker_push(mock_context)

            # Check that docker push was called correctly
            expected_push_command = f'docker push {artifactory}/train:latest'
            mock_context.run.assert_any_call(expected_push_command, echo=True, pty=not tasks.WINDOWS)

    def test_docker_push_calls_commands_in_order(self, mock_context):
        """Test docker_push calls tag before push."""
        import tasks

        artifactory = 'us-central1-docker.pkg.dev/my-project/my-repo'

        with patch('tasks.load_dotenv'), patch.dict(os.environ, {'ARTIFACTORY': artifactory}):
            tasks.docker_push(mock_context)

            # Verify the order of calls
            assert mock_context.run.call_count == 2
            calls = mock_context.run.call_args_list

            # First call should be docker tag
            assert 'docker tag train:latest' in calls[0][0][0]
            # Second call should be docker push
            assert 'docker push' in calls[1][0][0]

    def test_docker_push_with_different_artifactory_values(self, mock_context):
        """Test docker_push works with different ARTIFACTORY values."""
        import tasks

        test_artifactories = [
            'gcr.io/my-project',
            'us-west1-docker.pkg.dev/project/repo',
            'europe-docker.pkg.dev/another-project/images',
        ]

        for artifactory in test_artifactories:
            mock_context.reset_mock()

            with patch('tasks.load_dotenv'), patch.dict(os.environ, {'ARTIFACTORY': artifactory}):
                tasks.docker_push(mock_context)

                # Verify correct artifactory is used in commands
                expected_tag_command = f'docker tag train:latest {artifactory}/train:latest'
                expected_push_command = f'docker push {artifactory}/train:latest'

                mock_context.run.assert_any_call(expected_tag_command, echo=True, pty=not tasks.WINDOWS)
                mock_context.run.assert_any_call(expected_push_command, echo=True, pty=not tasks.WINDOWS)

    def test_docker_push_loads_dotenv(self, mock_context):
        """Test docker_push calls load_dotenv with override=True."""
        import tasks

        artifactory = 'us-central1-docker.pkg.dev/my-project/my-repo'

        with patch('tasks.load_dotenv') as mock_load_dotenv, patch.dict(
            os.environ, {'ARTIFACTORY': artifactory}
        ):
            tasks.docker_push(mock_context)

            # Verify load_dotenv was called with override=True
            mock_load_dotenv.assert_called_once_with(override=True)
