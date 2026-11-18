"""The local provider: talks to an Ollama server, faked here."""
import io
import json
import urllib.error

import pytest

from vtd.ai import ollama_provider
from vtd.ai.base import SectionInput
from vtd.ai.ollama_provider import OllamaProvider
from vtd.config import VTDConfig
from vtd.utils import VTDError


class FakeOllama:
    """Stands in for urllib.request.urlopen, answering like an Ollama server."""

    def __init__(self, models=("qwen3.6:35b",), capabilities=("completion", "vision", "thinking"),
                 answer="Una respuesta.", done_reason="stop"):
        self.models, self.capabilities = models, list(capabilities)
        self.answer, self.done_reason = answer, done_reason
        self.chats = []

    def __call__(self, request, timeout=None):
        path = request.full_url.split("11434", 1)[1]
        payload = json.loads(request.data) if request.data else None
        if path == "/api/tags":
            body = {"models": [{"name": m} for m in self.models]}
        elif path == "/api/show":
            if payload["model"].lower() not in [m.lower() for m in self.models]:
                error = json.dumps({"error": f"model '{payload['model']}' not found"}).encode()
                raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {},
                                             io.BytesIO(error))
            body = {"capabilities": self.capabilities}
        else:
            self.chats.append(payload)
            body = {"message": {"content": self.answer}, "done_reason": self.done_reason}
        return _Response(body)


class _Response:
    def __init__(self, body):
        self._body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self):
        return self._body


class _Opener:
    def __init__(self, handler):
        self.open = handler


def _serve(monkeypatch, handler):
    """Answer the provider's requests with `handler`, proxied or not."""
    monkeypatch.setattr(ollama_provider.urllib.request, "urlopen", handler)
    monkeypatch.setattr(ollama_provider.urllib.request, "build_opener",
                        lambda *handlers: _Opener(handler))


@pytest.fixture
def ollama(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    def install(**kwargs):
        fake = FakeOllama(**kwargs)
        _serve(monkeypatch, fake)
        return fake
    return install


def _config(model="qwen3.6:35b"):
    config = VTDConfig()
    config.ai.provider, config.ai.model = "ollama", model
    return config


def _section(tmp_path):
    frame = tmp_path / "frame_0001.jpg"
    frame.write_bytes(b"jpeg-bytes")
    return SectionInput(index=0, title="T", start_s=0, end_s=10,
                        transcript_chunk="hola a todos", frame_paths=[frame])


def test_section_body_is_asked_from_the_local_server_with_its_frames(tmp_path, ollama):
    server = ollama()
    body = OllamaProvider(_config(), tmp_path).write_section_body(_section(tmp_path))

    assert body == "Una respuesta."
    chat = server.chats[0]
    assert chat["model"] == "qwen3.6:35b" and chat["stream"] is False
    assert chat["messages"][0]["role"] == "system"
    assert "hola a todos" in chat["messages"][1]["content"]
    assert len(chat["messages"][1]["images"]) == 1  # the frame, base64
    assert chat["options"]["num_ctx"] == ollama_provider.NUM_CTX


def test_thinking_is_switched_off_and_given_room_in_case_it_cannot_be(tmp_path, ollama):
    server = ollama()
    OllamaProvider(_config(), tmp_path).write_section_body(_section(tmp_path))
    assert server.chats[0]["think"] is False
    assert server.chats[0]["options"]["num_predict"] == 2000 + ollama_provider.THINKING_ROOM


def test_a_model_without_thinking_is_not_sent_the_thinking_switch(tmp_path, ollama):
    server = ollama(capabilities=("completion", "vision"))
    OllamaProvider(_config(), tmp_path).write_section_body(_section(tmp_path))
    assert "think" not in server.chats[0]
    assert server.chats[0]["options"]["num_predict"] == 2000


def test_frames_are_not_sent_to_a_model_that_cannot_read_images(tmp_path, ollama, capsys):
    server = ollama(capabilities=("completion",))
    OllamaProvider(_config(), tmp_path).write_section_body(_section(tmp_path))
    assert "images" not in server.chats[0]["messages"][1]
    assert "cannot read images" in capsys.readouterr().err + capsys.readouterr().out


def test_an_answer_cut_at_the_token_limit_is_reported_and_not_cached(tmp_path, ollama):
    server = ollama(done_reason="length")
    provider = OllamaProvider(_config(), tmp_path)
    provider.write_section_body(_section(tmp_path))
    provider.write_section_body(_section(tmp_path))
    assert len(server.chats) == 2  # asked again: the first answer was not kept
    assert provider.degraded == ["body §1"]


def test_ollama_not_running_is_explained(tmp_path, monkeypatch):
    def refused(request, timeout=None):
        raise ConnectionRefusedError("Connection refused")
    _serve(monkeypatch, refused)
    with pytest.raises(VTDError, match="ollama serve"):
        OllamaProvider(_config(), tmp_path)


def test_a_model_that_is_not_installed_is_explained(tmp_path, ollama):
    ollama(models=("llama3.2:3b",))
    with pytest.raises(VTDError, match="not found.*ollama pull qwen3.6:35b.*Installed: llama3.2:3b"):
        OllamaProvider(_config(), tmp_path)


def test_ollama_decides_whether_it_knows_a_model_name(tmp_path, ollama):
    ollama(models=("llama3.2:3b",))
    OllamaProvider(_config("LLAMA3.2:3b"), tmp_path)  # Ollama accepts it, so vtd does


def test_a_local_server_is_reached_without_the_configured_proxy(tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    handlers = []
    fake = FakeOllama()
    monkeypatch.setattr(ollama_provider.urllib.request, "build_opener",
                        lambda *given: handlers.extend(given) or _Opener(fake))
    monkeypatch.setattr(ollama_provider.urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("went through the proxy-aware opener"))
    OllamaProvider(_config(), tmp_path).write_section_body(_section(tmp_path))
    assert len(handlers) == 1 and handlers[0].proxies == {}
    assert len(fake.chats) == 1


@pytest.mark.parametrize("value, expected", [
    (None, "http://localhost:11434"),
    ("127.0.0.1:11434", "http://127.0.0.1:11434"),
    ("http://box.local:11434/", "http://box.local:11434"),
    ("0.0.0.0", "http://127.0.0.1:11434"),        # the value exported for `ollama serve`
    ("localhost", "http://localhost:11434"),
    ("http://box.local", "http://box.local:11434"),
    ("https://ollama.example.com", "https://ollama.example.com"),
    ("box.local:8080", "http://box.local:8080"),
])
def test_server_address_follows_ollama_host(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
    else:
        monkeypatch.setenv("OLLAMA_HOST", value)
    assert ollama_provider.ollama_host() == expected
