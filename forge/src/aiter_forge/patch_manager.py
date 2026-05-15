"""Apply and rollback kernel patches on remote GPU nodes."""
from __future__ import annotations

import difflib
import shlex
import sys
import tempfile
from pathlib import Path

from .remote import RemoteRunner


class PatchManager:
    """Manage kernel file modifications on a remote host with rollback support."""

    def __init__(self, runner: RemoteRunner, remote_kernel_path: str):
        self.runner = runner
        self.remote_kernel_path = remote_kernel_path
        self._backup: str | None = None
        self._modified: str | None = None

    @property
    def has_backup(self) -> bool:
        return self._backup is not None

    def _read_remote(self) -> str:
        """Read the current kernel file from the remote host."""
        quoted_path = shlex.quote(self.remote_kernel_path)
        result = self.runner.run(f"cat {quoted_path}", label="read_kernel")
        if not result.ok:
            raise RuntimeError(f"Failed to read remote kernel: {result.stderr}")
        return result.stdout

    def apply(self, modified_source: str) -> None:
        """Apply modified kernel to remote host. Saves backup of original."""
        # Read and backup original
        original = self._read_remote()

        # Write modified to temp file, upload
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(modified_source)
            tmp_path = f.name

        try:
            self.runner.upload(tmp_path, self.remote_kernel_path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

        Path(tmp_path).unlink(missing_ok=True)
        self._backup = original
        self._modified = modified_source
        print(f"[patch] Applied to {self.remote_kernel_path}", file=sys.stderr)

    def accept(self) -> None:
        """Accept the current modification as the new baseline for future rollbacks."""
        if self._modified is None:
            raise ValueError("No modification to accept. Call apply() first.")
        self._backup = self._modified
        self._modified = None

    def rollback(self) -> None:
        """Restore the original kernel from backup."""
        if self._backup is None:
            raise ValueError("No backup available. Call apply() first.")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(self._backup)
            tmp_path = f.name

        try:
            self.runner.upload(tmp_path, self.remote_kernel_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        print(f"[patch] Rolled back {self.remote_kernel_path}", file=sys.stderr)
        self._modified = None

    def save_patch(self, patch_path: str) -> None:
        """Save the diff between original and modified as a unified patch file."""
        if self._backup is None:
            raise ValueError("No backup available. Call apply() first.")

        diff = difflib.unified_diff(
            self._backup.splitlines(keepends=True),
            self._modified.splitlines(keepends=True),
            fromfile="original",
            tofile="modified",
        )
        Path(patch_path).write_text("".join(diff))
