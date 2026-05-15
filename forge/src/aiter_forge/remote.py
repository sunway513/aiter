# src/aiter_forge/remote.py
"""SSH remote command execution for MI355X GPU nodes."""
from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class RemoteResult:
    """Result of a remote command execution."""
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def output(self) -> str:
        return self.stdout if self.stdout else self.stderr


class RemoteRunner:
    """Execute commands on a remote GPU node via SSH, or locally as fallback."""

    def __init__(self, host: str, user: str, timeout: int = 600,
                 key_filename: str | None = None, jump_host: str | None = None,
                 gpu_id: int | None = None,
                 auto_add_host_keys: bool = False):
        self.host = host
        self.user = user
        self.timeout = timeout
        self.key_filename = key_filename
        self.jump_host = jump_host
        self.gpu_id = gpu_id
        self.auto_add_host_keys = auto_add_host_keys

    def _is_local(self) -> bool:
        return self.host in ("localhost", "127.0.0.1", "")

    def _get_client(self):
        """Create and return a connected paramiko SSHClient."""
        import paramiko
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        if self.auto_add_host_keys:
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        else:
            client.set_missing_host_key_policy(paramiko.RejectPolicy())

        connect_kwargs = {
            "hostname": self.host,
            "username": self.user,
            "timeout": self.timeout,
        }
        if self.key_filename:
            connect_kwargs["key_filename"] = self.key_filename

        if self.jump_host:
            jump = paramiko.SSHClient()
            jump.load_system_host_keys()
            if self.auto_add_host_keys:
                jump.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            else:
                jump.set_missing_host_key_policy(paramiko.RejectPolicy())
            jump.connect(self.jump_host, username=self.user)
            transport = jump.get_transport()
            channel = transport.open_channel(
                "direct-tcpip", (self.host, 22), ("127.0.0.1", 0)
            )
            connect_kwargs["sock"] = channel

        client.connect(**connect_kwargs)
        return client

    def _wrap_cmd(self, cmd: str) -> str:
        """Prepend HIP_VISIBLE_DEVICES if gpu_id is set."""
        if self.gpu_id is not None:
            return f"HIP_VISIBLE_DEVICES={self.gpu_id} {cmd}"
        return cmd

    def run(self, cmd: str, label: str = "") -> RemoteResult:
        """Run a command. Uses SSH for remote hosts, subprocess for localhost."""
        cmd = self._wrap_cmd(cmd)
        print(f"[{label}] Running on {self.host}: {cmd}", file=sys.stderr)

        if self._is_local():
            # Use shell=True for local mode so env var prefixes like
            # HIP_VISIBLE_DEVICES=N work correctly as shell assignments.
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=self.timeout,
            )
            return RemoteResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

        client = self._get_client()
        try:
            stdin, stdout, stderr = client.exec_command(cmd, timeout=self.timeout)
            rc = stdout.channel.recv_exit_status()
            out = stdout.read().decode("utf-8", errors="replace")
            err = stderr.read().decode("utf-8", errors="replace")
            print(f"[{label}] Exit code: {rc}", file=sys.stderr)
            return RemoteResult(returncode=rc, stdout=out, stderr=err)
        finally:
            client.close()

    def upload(self, local_path: str, remote_path: str) -> None:
        """Upload a local file to the remote host via SFTP (or local copy for localhost)."""
        if self._is_local():
            import shutil
            shutil.copy2(local_path, remote_path)
            return
        client = self._get_client()
        try:
            sftp = client.open_sftp()
            sftp.put(local_path, remote_path)
            sftp.close()
        finally:
            client.close()

    def download(self, remote_path: str, local_path: str) -> None:
        """Download a file from the remote host via SFTP (or local copy for localhost)."""
        if self._is_local():
            import shutil
            shutil.copy2(remote_path, local_path)
            return
        client = self._get_client()
        try:
            sftp = client.open_sftp()
            sftp.get(remote_path, local_path)
            sftp.close()
        finally:
            client.close()
