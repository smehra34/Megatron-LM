# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

from megatron.training import wandb_utils


class _Artifact:
    def __init__(self, name, type, metadata):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.references = []

    def add_reference(self, path, checksum):
        self.references.append((path, checksum))

    def add_file(self, path):
        raise AssertionError("mutable checkpoint tracker must not be added to the artifact")


class _Run:
    entity = "entity"
    project = "project"

    def __init__(self):
        self.logged_artifacts = []
        self.used_artifacts = []

    def log_artifact(self, artifact, aliases):
        self.logged_artifacts.append((artifact, aliases))

    def use_artifact(self, artifact_path):
        self.used_artifacts.append(artifact_path)


def _writer():
    return SimpleNamespace(Artifact=_Artifact, run=_Run())


def test_checkpoint_artifacts_are_disabled_without_touching_wandb(monkeypatch):
    def fail_if_called():
        raise AssertionError("W&B writer must not be accessed when artifacts are disabled")

    monkeypatch.setattr(wandb_utils, "get_wandb_writer", fail_if_called)

    wandb_utils.on_save_checkpoint_success("checkpoint", "save", 10)
    wandb_utils.on_load_checkpoint_success("checkpoint", "save")


def test_opt_in_checkpoint_artifact_uses_reference_without_hashing_tracker(
    monkeypatch, tmp_path
):
    writer = _writer()
    monkeypatch.setattr(wandb_utils, "get_wandb_writer", lambda: writer)
    checkpoint = tmp_path / "save" / "iter_0000010"
    checkpoint.mkdir(parents=True)

    wandb_utils.on_save_checkpoint_success(
        str(checkpoint), str(checkpoint.parent), 10, enabled=True
    )

    artifact, aliases = writer.run.logged_artifacts[0]
    assert artifact.references == [(f"file://{checkpoint.resolve()}", False)]
    assert aliases == ["iter_0000010"]
    assert (checkpoint.parent / "latest_wandb_artifact_path.txt").read_text() == (
        "entity/project"
    )
