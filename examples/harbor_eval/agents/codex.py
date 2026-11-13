from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
import tempfile
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Annotated, Any, ClassVar

from harbor.agents.installed.codex import Codex, CodexOptions
from harbor.agents.options import Cli
from harbor.environments.base import BaseEnvironment

from .prebuilt import Prebuilt

VERSION = "0.167.1"
SANDBOX = "/opt/tokin-codex"
VENDOR = "package/vendor/x86_64-unknown-linux-musl/"
# What `codex exec` runs on: the binary, the `rg` it puts on its commands' path, and the manifest it finds both by.
# The voice and code-mode hosts are the TUI's and an off-by-default feature's, and bwrap is the sandbox's, which
# Harbor's run command bypasses.
KEEP = ("codex-package.json", "bin/codex", "codex-path/rg")
# Codex's metadata for a model it has none for (`model_info_from_slug`), stated so it no longer warns that it fell
# back. The empty slug is a prefix of every model name, as tokin answers whatever the name. Text is the only input,
# so Codex replaces an image with a note rather than send one tokin refuses. `apply_patch` stays a command run through
# the shell, the form its instructions show, not a freeform tool.
MODEL: dict[str, Any] = {
    "slug": "",
    "display_name": "",
    "supported_reasoning_levels": [],
    "shell_type": "unified_exec",
    "visibility": "none",
    "supported_in_api": True,
    "priority": 99,
    "support_verbosity": False,
    "truncation_policy": {"mode": "bytes", "limit": 10_000},
    "experimental_supported_tools": [],
    "include_apps_usage_instructions": False,
    "input_modalities": ["text"],
}


def build(root: Path) -> None:
    """Codex's static musl build from npm, the platform package `npm install -g @openai/codex` takes on linux x64.

    Its catalog, `models.json`, is `MODEL` with the instructions Codex sends a model it has no metadata for, read from
    its first request to an endpoint that refuses it. A catalog entry has to state its instructions, and the prompt in
    Codex's source is not them: Codex drops the sections on its plan tool, which is off, only from prompts of its own.
    """
    url = f"https://registry.npmjs.org/@openai/codex/-/codex-{VERSION}-linux-x64.tgz"
    with tarfile.open(fileobj=io.BytesIO(urllib.request.urlopen(url).read()), mode="r:gz") as npm:
        for name in KEEP:
            member = npm.getmember(VENDOR + name)
            member.name = name
            npm.extract(member, root, filter="tar")
    codex = str(root / "bin" / "codex")
    version = subprocess.run([codex, "--version"], capture_output=True, text=True, check=True)
    if version.stdout.split() != ["codex-cli", VERSION]:
        raise RuntimeError(f"the archive's Codex is not {VERSION}: {version.stdout}{version.stderr}")
    sent: list[dict[str, Any]] = []

    class Refuse(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            sent.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_error(400)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Refuse)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    provider = f'{{name = "refuse", base_url = "http://127.0.0.1:{server.server_port}/v1", wire_api = "responses"}}'
    with tempfile.TemporaryDirectory() as home:
        tried = subprocess.run(
            [codex, "exec", "--skip-git-repo-check", "--model", "unknown", "-c", 'model_provider="refuse"']
            + ["-c", f"model_providers.refuse={provider}", "hi"],
            env={"PATH": os.environ["PATH"], "CODEX_HOME": home, "OPENAI_API_KEY": "none"},
            cwd=home,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=180,
        )
    server.shutdown()
    if not sent or "Model metadata for `unknown` not found" not in tried.stdout + tried.stderr:
        raise RuntimeError(f"Codex {VERSION} sent no request as a model it has no metadata for: {tried.stderr}")
    instructions = sent[0]["input"][0]["content"][0]["text"]
    model = MODEL | {"model_messages": {"instructions_template": instructions}}
    (root / "models.json").write_text(json.dumps({"models": [model]}))


PREBUILT = Prebuilt("codex", VERSION, SANDBOX, build)


class PrebuiltCodexOptions(CodexOptions):
    archive: str
    # Codex compacts at 90% of it and cannot ask tokin, so it is the context the engine serves.
    context_window: Annotated[int, Cli("-c", format="-c model_context_window={value}")]


class PrebuiltCodex(Codex):
    """Harbor's Codex, installed from a prebuilt archive, calling tokin's Responses API at the trial's session URL.

    The session URL goes to a provider of its own rather than to Harbor's `openai_base_url`: the built-in `openai`
    provider opens a websocket, which tokin does not serve.
    """

    options_model: ClassVar[type[CodexOptions]] = PrebuiltCodexOptions
    options: PrebuiltCodexOptions

    async def install(self, environment: BaseEnvironment) -> None:
        await PREBUILT.carry(environment, self.options.archive)
        # Where Harbor's run command finds it; Codex resolves the link to find its manifest and `rg`.
        await self.exec_as_root(
            environment, command=f"mkdir -p /usr/local/bin && ln -sf {SANDBOX}/bin/codex /usr/local/bin/"
        )

    def _build_effective_config(self, openai_base_url: str | None = None) -> dict[str, Any]:
        if openai_base_url is None:
            raise ValueError("Codex has no tokin session to call: OPENAI_BASE_URL and OPENAI_API_BASE are unset")
        provider = {
            "name": "tokin",
            "base_url": openai_base_url,
            "wire_api": "responses",
            "supports_websockets": False,
            "stream_idle_timeout_ms": 600_000,
        }
        return super()._build_effective_config() | {
            "model_provider": "tokin",
            "model_providers": {"tokin": provider},
            "model_catalog_json": f"{SANDBOX}/models.json",
            # The model's commands would otherwise see the session URL and key, and could call tokin themselves.
            "shell_environment_policy": {"exclude": ["OPENAI_*"]},
        }
