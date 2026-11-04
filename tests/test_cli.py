"""The whole command line against a local fake OpenAI-compatible server and embedded Qdrant on disk."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from conftest import fake_vector

from legal_rag_eval.cli import main


class FakeOpenAI(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out: dict[str, object]
        if self.path.endswith("/embeddings"):
            out = {"data": [{"index": i, "embedding": fake_vector(t)} for i, t in enumerate(body["input"])]}
        else:
            prompt = body["messages"][-1]["content"]
            answer = "Sim. O prazo é de 15 (quinze) dias, art. 335 do CPC." if "contestação" in prompt else "Não sei."
            out = {"choices": [{"message": {"content": answer}}], "usage": {"prompt_tokens": 100}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def server() -> Iterator[str]:
    httpd = HTTPServer(("127.0.0.1", 0), FakeOpenAI)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    httpd.shutdown()


def test_validate(key_path: Path, corpus_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", "--key", str(key_path), "--corpus", str(corpus_dir)]) == 0
    assert "0 problem(s)" in capsys.readouterr().out


def test_ingest_retrieval_e2e_rescore_compare(
    server: str,
    tmp_path: Path,
    key_path: Path,
    corpus_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for var in ("LANGFUSE_HOST", "LANGFUSE_BASE_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    conn = [
        "--embed-url",
        server,
        "--embed-model",
        "fake",
        "--qdrant-path",
        str(tmp_path / "qdrant"),
        "--collection",
        "t",
        "--out",
        str(tmp_path / "results"),
    ]
    data = ["--key", str(key_path), "--corpus", str(corpus_dir)]

    assert main(["ingest", *data, *conn]) == 0
    assert "articles in 't'" in capsys.readouterr().out

    assert main(["retrieval", *data, *conn, "--k", "1,5"]) == 0
    assert "| k | hit@k | recall@k | coverage@k |" in capsys.readouterr().out
    assert len(list((tmp_path / "results").glob("retrieval-fake-*.json"))) == 1

    e2e = ["e2e", *data, *conn, "--llm-url", server, "--llm-model", "fake-llm", "--only", "cpc-01,cpc-02,cdc-01"]
    assert main(e2e) == 0
    assert "passed. Report:" in capsys.readouterr().out
    [run] = sorted((tmp_path / "results").glob("e2e-fake-llm-rag-*.jsonl"))
    rows = [json.loads(line) for line in run.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["type"] == "meta" and rows[0]["model"] == "fake-llm" and rows[0]["top_k"] == 5
    assert "http" not in json.dumps(rows[0])  # endpoints are not written to results that may be shared
    assert [r["id"] for r in rows[1:]] == ["cdc-01", "cpc-01", "cpc-02"]
    assert run.with_suffix(".md").exists()

    assert main(["rescore", "--key", str(key_path), str(run)]) == 0
    assert "stored:" in capsys.readouterr().out
    assert main(["compare", str(run), str(run)]) == 0
    assert "pass -> fail: 0" in capsys.readouterr().out


def test_wrong_embedding_model_is_refused(server: str, tmp_path: Path, key_path: Path, corpus_dir: Path) -> None:
    conn = ["--embed-url", server, "--qdrant-path", str(tmp_path / "q"), "--collection", "t", "--out", str(tmp_path)]
    data = ["--key", str(key_path), "--corpus", str(corpus_dir)]
    assert main(["ingest", *data, *conn, "--embed-model", "fake"]) == 0
    with pytest.raises(SystemExit, match="built with 'fake'"):
        main(["retrieval", *data, *conn, "--embed-model", "other"])
