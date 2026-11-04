"""Command line: validate, ingest, retrieval, e2e, rescore, compare. Run with --help for the options."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import httpx

from . import __version__
from .answer_key import file_sha256, load_answer_key
from .clients import ChatClient, EmbeddingClient
from .config import Settings
from .corpus import CHUNKER_VERSION, chunk_corpus, load_corpus
from .e2e_eval import Run, load_run, print_progress, run_local, save_run
from .rag import BlackBoxPipeline, DenseRetriever, Pipeline, RagPipeline, load_template
from .report import render_run
from .rescore import compare, rescore
from .retrieval_eval import evaluate_retrieval, render_markdown
from .store import VectorStore, connect
from .tracing import run_with_langfuse
from .validate import validate

DEFAULT_KEY = "data/answer_key.json"
DEFAULT_CORPUS = "data/corpus"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "model"


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings.from_env()
    for name in (
        "llm_base_url",
        "llm_model",
        "embed_base_url",
        "embed_model",
        "qdrant_url",
        "qdrant_path",
        "collection",
    ):
        value = getattr(args, name, None)
        if value:
            setattr(s, name, value)
    if getattr(args, "temperature", None) is not None:
        s.temperature = args.temperature
    return s


def _store(s: Settings) -> VectorStore:
    return VectorStore(connect(s.qdrant_url, s.qdrant_path), s.collection)


def _retriever(s: Settings, http: httpx.Client) -> DenseRetriever:
    store = _store(s)
    problem = None
    if not store.exists():
        problem = f"collection {s.collection!r} not found: run `legal-rag-eval ingest` first"
    elif (built_with := store.embed_model()) and built_with != s.embed_model:
        problem = f"collection {s.collection!r} was built with {built_with!r}, not {s.embed_model!r}: re-run ingest"
    if problem:
        store.close()
        sys.exit(problem)
    return DenseRetriever(EmbeddingClient(s.embed_base_url, s.embed_model, s.embed_api_key, http), store)


def cmd_validate(args: argparse.Namespace) -> int:
    items, articles = load_answer_key(args.key), load_corpus(args.corpus)
    problems = validate(items, articles)
    for p in problems:
        print(p)
    used = {ref for it in items for ref in it.legal_basis}
    print(
        f"{len(items)} questions, {len(articles)} articles ({len(used)} used as legal basis), "
        f"{len(problems)} problem(s)"
    )
    return 1 if problems else 0


def cmd_ingest(args: argparse.Namespace) -> int:
    s = _settings(args)
    chunks = chunk_corpus(load_corpus(args.corpus), args.max_tokens)
    with httpx.Client(timeout=300) as http:
        vectors = EmbeddingClient(s.embed_base_url, s.embed_model, s.embed_api_key, http).embed(
            [c.embed_text for c in chunks]
        )
    store = _store(s)
    try:
        store.recreate(len(vectors[0]))  # the corpus is small: rebuilding is simpler than diffing
        store.upsert(chunks, vectors, s.embed_model)
        print(
            f"{store.count()} chunks from {len({c.article_id for c in chunks})} articles in {s.collection!r} "
            f"({s.embed_model}, {len(vectors[0])} dimensions, {CHUNKER_VERSION})"
        )
    finally:
        store.close()
    return 0


def cmd_retrieval(args: argparse.Namespace) -> int:
    s = _settings(args)
    items = load_answer_key(args.key)
    ks = sorted({int(k) for k in args.k.split(",")})
    meta = {
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "embed_model": s.embed_model,
        "collection": s.collection,
        "answer_key": f"{Path(args.key).name} ({file_sha256(args.key)})",
        "version": __version__,
    }
    with httpx.Client(timeout=120) as http:
        retriever = _retriever(s, http)
        try:
            report = evaluate_retrieval(items, retriever, ks, meta)
        finally:
            retriever.store.close()
    md = render_markdown(report)
    print(md)
    stem = Path(args.out) / f"retrieval-{_slug(s.embed_model)}-{time.strftime('%Y%m%d-%H%M%S')}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    stem.with_suffix(".json").write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {stem}.md and .json", file=sys.stderr)
    return 0


def cmd_e2e(args: argparse.Namespace) -> int:
    s = _settings(args)
    if not s.llm_model:
        sys.exit("set LLM_MODEL or --llm-model")
    items = load_answer_key(args.key)
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - {it.id for it in items}
        if unknown:
            sys.exit(f"not in the answer key: {', '.join(sorted(unknown))}")
        items = [it for it in items if it.id in wanted]
    k = args.k or s.top_k
    with httpx.Client(timeout=600) as http:
        chat = ChatClient(s.llm_base_url, s.llm_model, s.llm_api_key, http, temperature=s.temperature)
        pipeline: Pipeline
        if args.blackbox:
            pipeline, template = BlackBoxPipeline(chat), "-"
        else:
            pipeline, template = RagPipeline(_retriever(s, http), chat, load_template(args.template), k), args.template
        when = time.strftime("%Y%m%d-%H%M%S")
        label = "" if args.blackbox else Path(template).stem
        run_name = " ".join(x for x in (s.llm_model, pipeline.name, label, when) if x)
        meta = {
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "pipeline": pipeline.name,
            "model": s.llm_model,
            "temperature": "server default" if s.temperature is None else s.temperature,
            "answer_key": f"{Path(args.key).name} ({file_sha256(args.key)})",
            "version": __version__,
        }
        if not args.blackbox:
            meta.update({"template": template, "top_k": k, "embed_model": s.embed_model})
        results = None
        try:
            if not args.no_langfuse:
                results = run_with_langfuse(
                    items,
                    pipeline,
                    Path(args.key).stem,
                    run_name,
                    {k_: str(v) for k_, v in meta.items()},
                    print_progress,
                )
            if results is None:
                results = run_local(items, pipeline)
        finally:
            if isinstance(pipeline, RagPipeline) and isinstance(pipeline.retriever, DenseRetriever):
                pipeline.retriever.store.close()
    run = Run(meta, results)
    stem = Path(args.out) / f"e2e-{_slug(s.llm_model)}-{pipeline.name}-{when}"
    save_run(run, stem.with_suffix(".jsonl"))
    md = render_run(run)
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    print(f"\n== {run.passed}/{len(results)} passed. Report: {stem}.md, answers: {stem}.jsonl")
    return 0


def cmd_rescore(args: argparse.Namespace) -> int:
    items = load_answer_key(args.key)
    for path in args.runs:
        print(f"{path}:\n" + rescore(load_run(path), items).render("stored", "current key"))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    cmp = compare(load_run(args.before), load_run(args.after))
    print(cmp.render(Path(args.before).stem, Path(args.after).stem))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="legal-rag-eval", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    data = argparse.ArgumentParser(add_help=False)
    data.add_argument("--key", default=DEFAULT_KEY, help=f"answer key (default {DEFAULT_KEY})")
    data.add_argument("--corpus", default=DEFAULT_CORPUS, help=f"corpus directory (default {DEFAULT_CORPUS})")

    conn = argparse.ArgumentParser(add_help=False)
    conn.add_argument("--embed-url", dest="embed_base_url", help="OpenAI-compatible base URL for /embeddings")
    conn.add_argument("--embed-model")
    conn.add_argument("--qdrant-url")
    conn.add_argument("--qdrant-path", help="use embedded Qdrant at this path instead of a server")
    conn.add_argument("--collection")
    conn.add_argument("--out", default="results", help="output directory (default results/)")

    sp = sub.add_parser("validate", parents=[data], help="check the answer key against the corpus (offline)")
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("ingest", parents=[data, conn], help="chunk, embed and index the corpus")
    sp.add_argument("--max-tokens", type=int, default=400)
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("retrieval", parents=[data, conn], help="retrieval-only evaluation (no LLM)")
    sp.add_argument("--k", default="1,3,5", help="comma-separated cut-offs (default 1,3,5)")
    sp.set_defaults(func=cmd_retrieval)

    sp = sub.add_parser("e2e", parents=[data, conn], help="end-to-end evaluation with the rule-based checks")
    sp.add_argument("--llm-url", dest="llm_base_url", help="OpenAI-compatible base URL for /chat/completions")
    sp.add_argument("--llm-model")
    sp.add_argument("--temperature", type=float)
    sp.add_argument("--k", type=int, help="passages per question (default TOP_K or 5)")
    sp.add_argument("--template", default="basis", help="basis | no-basis | path to a template file")
    sp.add_argument(
        "--blackbox", action="store_true", help="the endpoint does its own retrieval: send only the question"
    )
    sp.add_argument("--only", help="comma-separated question ids")
    sp.add_argument("--no-langfuse", action="store_true", help="do not send the run to Langfuse even if configured")
    sp.set_defaults(func=cmd_e2e)

    sp = sub.add_parser("rescore", parents=[data], help="grade saved runs again with the current key")
    sp.add_argument("runs", nargs="+", help="run files (.jsonl)")
    sp.set_defaults(func=cmd_rescore)

    sp = sub.add_parser("compare", help="question-by-question changes between two runs")
    sp.add_argument("before")
    sp.add_argument("after")
    sp.set_defaults(func=cmd_compare)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
