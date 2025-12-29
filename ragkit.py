#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local RAG Kit (MVP v0)
- index: 로컬 문서 로드 → 청크 → 임베딩(TF-IDF or SBERT) → 인덱스 저장
- query: 질문 → top_k 청크 검색 → 근거 출력
- eval: (query, gold_doc) jsonl로 Recall@K 평가

주의:
- 기본은 TF-IDF (requirements-min.txt만으로 동작)
- SBERT는 requirements-sbert.txt 추가 설치 필요
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import yaml

# TF-IDF dependencies (min)
from joblib import dump, load
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import inspect
from sentence_transformers import SentenceTransformer


# -----------------------------
# Config
# -----------------------------
@dataclass
class LoaderConfig:
    allowed_exts: List[str]

@dataclass
class ChunkConfig:
    size: int
    overlap: int

@dataclass
class RetrievalConfig:
    top_k: int

@dataclass
class SbertConfig:
    model_name: str

@dataclass
class EmbedderConfig:
    type: str # "tfidf" or "sbert"
    sbert: SbertConfig

@dataclass
class AppConfig:
    data_dir: str
    index_dir: str
    loader: LoaderConfig
    chunk: ChunkConfig
    retrieval: RetrievalConfig
    embedder: EmbedderConfig

def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    
    def _must(d, k):
        if k not in d:
            raise ValueError(f"config missing key: {k}")
        return d[k]
    
    loader = LoaderConfig(allowed_exts=list(_must(raw["loader"], "allowed_exts")))
    chunk = ChunkConfig(size=int(_must(raw["chunk"], "size")), overlap=int(_must(raw["chunk"], "overlap")))
    retrieval = RetrievalConfig(top_k=int(_must(raw["retrieval"], "top_k")))
    
    sbert_raw = raw.get("embedder", {}).get("sbert", {}) or {}
    sbert = SbertConfig(model_name=str(sbert_raw.get("model_name", "intfloat/multilingual-e5-small")))
    embedder = EmbedderConfig(type=str(_must(raw["embedder"], "type")).lower(), sbert=sbert)
    
    return AppConfig(
        data_dir=str(_must(raw, "data_dir")),
        index_dir=str(_must(raw, "index_dir")),
        loader=loader,
        chunk=chunk,
        retrieval=retrieval,
        embedder=embedder,
    )


# -----------------------------
# Utils
# -----------------------------
def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def iter_files(root: Path, allowed_exts: List[str]) -> Iterable[Path]:
    allowed = {e.lower() for e in allowed_exts}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in allowed:
            yield p

def read_text_file(path: Path) -> str:
    # BOM/인코딩 이슈 최소 대응
    data = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    # 최후: 대충 살리기
    return data.encode("utf-8", errors="replace")


# -----------------------------
# Chunking
# -----------------------------
@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    relpath: str
    chunk_index: int
    start_char: int
    end_char: int
    text: str

def chunk_text(doc_text: str, doc_id: str, relpath: str, size: int, overlap: int) -> List[Chunk]:
    txt = normalize_whitespace(doc_text)
    if not txt:
        return []
    
    if overlap >= size:
        raise ValueError(f"chunk.overlap must be < chunk.size (overlap={overlap}, size={size})")
    
    chunks: List[Chunk] = []
    step = size - overlap
    i = 0
    idx = 0
    n = len(txt)
    
    while i < n:
        j = min(i + size, n)
        piece = txt[i:j].strip()
        if piece:
            cid = sha1_text(f"{doc_id}:{idx}:{i}:{j}:{piece[:32]}")
            chunks.append(Chunk(
                chunk_id=cid,
                doc_id=doc_id,
                relpath=relpath,
                chunk_index=idx,
                start_char=i,
                end_char=j,
                text=piece
            ))
            idx += 1
        i += step
    
    return chunks


# -----------------------------
# Embedders
# -----------------------------
class TfidfIndex:
    """
    Stores:
        - vectorizer.joblib
        - tfidf_matrix.npz  (chunk vectors)
    """
    def __init__(self, vectorizer: TfidfVectorizer, matrix):
        self.vectorizer = vectorizer
        self.matrix = matrix
    
    @staticmethod
    def build(texts: List[str]) -> "TfidfIndex":
        vec = TfidfVectorizer(
            lowercase=False,        # 한국어 고려
            # token_pattern=r"(?u)\b\w+\b",   # 한글 토큰도 최대한 수용
            ngram_range=(1, 2),
            max_features=200_000
        )
        mat = vec.fit_transform(texts)
        return TfidfIndex(vec, mat)
    
    def query(self, q: str):
        qv = self.vectorizer.transform([q])
        sims = cosine_similarity(qv, self.matrix).ravel()   # shape (N,)
        return sims
    
    def save(self, index_dir: Path):
        index_dir.mkdir(parents=True, exist_ok=True)
        dump(self.vectorizer, index_dir / "tfidf_vectorizer.joblib")
        save_npz(index_dir / "tfidf_matrix.npz", self.matrix)
    
    @staticmethod
    def load(index_dir: Path) -> "TfidfIndex":
        vec = load(index_dir / "tfidf_vectorizer.joblib")
        mat = load_npz(index_dir / "tfidf_matrix.npz")
        return TfidfIndex(vec, mat)


def st_encode_safe(model, sentences, **kwargs):
    """
    sentence-transformers 버전/모델별로 encode()가 받는 kwargs가 달라서
    지원되는 인자만 골라 호출한다.
    """
    params = set(inspect.signature(model.encode).parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in params}
    
    try:
        return model.encode(sentences, **filtered)
    except Exception as e:
        # 특정 버전에서 'normalize_whitespace' 같은 인자 때문에 터지는 경우가 있어 재시도
        msg = str(e)
        if "normalize_whitespace" in msg:
            filtered.pop("normalize_whitespace", None)
            return model.encode(sentences, **filtered)
        raise

_SBERT_MODEL_CACHE = {}

def get_sbert_model(model_name: str):
    m = _SBERT_MODEL_CACHE.get(model_name)
    if m is None:
        # device를 강제하고 싶으면 여기서 지정 가능: SentenceTransformer(model_name, device="cpu")
        m = SentenceTransformer(model_name)
        _SBERT_MODEL_CACHE[model_name] = m
    return m

class SbertIndex:
    """
    Stores:
        - embeddings.npy    (float32, shape (N, dim))
        - sbert_meta.json   (model_name, dim)
    """
    def __init__(self, model_name: str, emb: np.ndarray):
        self.model_name = model_name
        self.emb = emb
        self._norm = None
    
    @staticmethod
    def _load_model(model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "SBERT not installed. Install requirements-sbert.txt first."
            ) from e
        return SentenceTransformer(model_name)
    
    @staticmethod
    def build(texts: List[str], model_name: str) -> "SbertIndex":
        model = get_sbert_model(model_name)
        # E5 계열은 "query: ...", "passage: ..." prefix가 성능에 중요할 수 있음.
        # 여기서는 간단히 passage prefix만 적용.
        passages = [f"passage: {t}" for t in texts]
        emb = st_encode_safe(
            model,
            passages,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        emb = np.asarray(emb, dtype=np.float32)
        return SbertIndex(model_name, emb)
    
    def query(self, q: str) -> np.ndarray:
        model = get_sbert_model(self.model_name)
        qq = f"query: {q}"
        qv = st_encode_safe(model, [qq], normalize_embeddings=True)
        qv = np.asarray(qv, dtype=np.float32)[0]
        sims = self.emb @ qv # cosine (이미 normalize_embeddings=True)
        return sims
    
    def save(self, index_dir: Path):
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "embeddings.npy", self.emb)
        meta = {"model_name": self.model_name, "dim": int(self.emb.shape[1])}
        (index_dir / "sbert_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    
    @staticmethod
    def load(index_dir: Path) -> "SbertIndex":
        meta = json.loads((index_dir / "sbert_meta.json").read_text(encoding="utf-8"))
        emb = np.load(index_dir / "embeddings.npy")
        return SbertIndex(meta["model_name"], emb)


# -----------------------------
# Index format (shared)
# -----------------------------
def save_chunks(chunks: List[Chunk], index_dir: Path):
    index_dir.mkdir(parents=True, exist_ok=True)
    out = index_dir / "chunks.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(dataclasses.asdict(c), ensure_ascii=False) + "\n")

def load_chunks(index_dir: Path) -> List[Chunk]:
    out = index_dir / "chunks.jsonl"
    chunks: List[Chunk] = []
    with out.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            chunks.append(Chunk(**d))
    return chunks

def save_meta(index_dir: Path, meta: Dict):
    (index_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def load_meta(index_dir: Path) -> Dict:
    return json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))


def compute_data_fingerprint(files: List[Path], data_root: Path) -> str:
    """
    문서 변경 감지용 fingerprint.
    - 상대경로 + mtime + size 조합을 정렬해 SHA1으로 요약.
    - 내용 전체 해시보다 훨씬 빠르고, 대부분의 변경을 잘 잡는다.
    """
    items = []
    for fp in files:
        st = fp.stat()
        rel = fp.relative_to(data_root).as_posix()
        items.append(f"{rel}|{int(st.st_mtime)}|{st.st_size}")
    items.sort()
    return sha1_text("\n".join(items))


# -----------------------------
# Commands
# -----------------------------
def cmd_index(cfg: AppConfig):
    data_root = Path(cfg.data_dir)
    index_dir = Path(cfg.index_dir)

    if not data_root.exists():
        raise FileNotFoundError(f"data_dir not found: {data_root}")

    files = sorted(list(iter_files(data_root, cfg.loader.allowed_exts)))
    if not files:
        raise RuntimeError(f"No documents found in {data_root} with {cfg.loader.allowed_exts}")

    # --- 문서 변경 감지 fingerprint 생성 ---
    data_fingerprint = compute_data_fingerprint(files, data_root)

    # --- 기존 인덱스 fingerprint와 비교(상태만 출력) ---
    prev_fp = None
    prev_meta_path = index_dir / "meta.json"
    if prev_meta_path.exists():
        try:
            prev_meta = load_meta(index_dir)
            prev_fp = prev_meta.get("data_fingerprint")
        except Exception:
            prev_fp = None

    if prev_fp is None:
        change_msg = "no_previous_index"
    elif prev_fp == data_fingerprint:
        change_msg = "unchanged"
    else:
        change_msg = "changed"

    # --- 문서 로드 & 청킹 ---
    all_chunks: List[Chunk] = []
    for fp in files:
        rel = fp.relative_to(data_root).as_posix()
        doc_id = rel  # doc_id는 상대경로로 고정 (평가/재현에 유리)
        text = read_text_file(fp)
        chunks = chunk_text(
            text,
            doc_id=doc_id,
            relpath=rel,
            size=cfg.chunk.size,
            overlap=cfg.chunk.overlap
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("All documents were empty after normalization/chunking.")

    texts = [c.text for c in all_chunks]

    meta = {
        "created_at": now_iso(),
        "data_dir": cfg.data_dir,
        "doc_count": len(files),
        "chunk_count": len(all_chunks),
        "chunk": {"size": cfg.chunk.size, "overlap": cfg.chunk.overlap},
        "embedder": {"type": cfg.embedder.type},
        "data_fingerprint": data_fingerprint,
        "data_change": change_msg,  # 상태(unchanged/changed/no_previous_index)
    }

    print(f"[index] docs={len(files)} chunks={len(all_chunks)} embedder={cfg.embedder.type} data={change_msg}")
    save_chunks(all_chunks, index_dir)

    # --- 임베딩/인덱스 빌드 ---
    if cfg.embedder.type == "tfidf":
        tf = TfidfIndex.build(texts)
        tf.save(index_dir)
        meta["embedder"]["details"] = {"vectorizer": "TfidfVectorizer(1-2gram)"}
    elif cfg.embedder.type == "sbert":
        sb = SbertIndex.build(texts, model_name=cfg.embedder.sbert.model_name)
        sb.save(index_dir)
        meta["embedder"]["details"] = {"model_name": cfg.embedder.sbert.model_name}
    else:
        raise ValueError(f"Unknown embedder.type: {cfg.embedder.type}")

    save_meta(index_dir, meta)
    print(f"[index] saved -> {index_dir.resolve()}")

def _load_index(cfg: AppConfig):
    index_dir = Path(cfg.index_dir)
    if not (index_dir / "meta.json").exists():
        raise RuntimeError(f"Index not found. Run index first. ({index_dir}/meta.json missing)")
    
    meta = load_meta(index_dir)
    chunks = load_chunks(index_dir)
    
    etype = meta.get("embedder", {}).get("type", "").lower()
    if etype == "tfidf":
        emb_index = TfidfIndex.load(index_dir)
    elif etype == "sbert":
        emb_index = SbertIndex.load(index_dir)
    else:
        raise RuntimeError(f"Unknown embedder in meta.json: {etype}")
    
    return meta, chunks, emb_index


def cmd_query(cfg: AppConfig, question: str):
    meta, chunks, emb_index = _load_index(cfg)
    
    if not chunks:
        raise RuntimeError("No chunks in index. Rebuild index first.")
    
    # top_k safety
    top_k = min(cfg.retrieval.top_k, len(chunks))
    
    sims = emb_index.query(question)
    if len(sims) != len(chunks):
        raise RuntimeError(f"Index corrupted: sims({len(sims)}) != chunks({len(chunks)})")
    
    # rank chunks
    order = np.argsort(-sims)[:top_k]
    
    # ---------- header ---------
    print(f"\nQ: {question}\n")
    print(
        f"Index embedder: {meta.get('embedder', {}).get('type', 'unknown')}"
        f" | chunks={len(chunks)} | top_k={top_k}\n"
    )
    
    # ---------- doc-level summary (max score per doc) ----------
    best_doc_score: Dict[str, float] = {}
    best_doc_chunk: Dict[str, int] = {}
    
    for i, c in enumerate(chunks):
        s = float(sims[i])
        if (c.doc_id not in best_doc_score) or (s > best_doc_score[c.doc_id]):
            best_doc_score[c.doc_id] = s
            best_doc_chunk[c.doc_id] = i
    
    # show top docs (up to 5)
    top_docs = sorted(best_doc_score.items(), key=lambda x: x[1], reverse=True)
    top_docs = top_docs[: min(5, len(top_docs))]
    
    print("Top Docs (max score per doc):")
    for rank, (doc_id, score) in enumerate(top_docs, start=1):
        cidx = best_doc_chunk[doc_id]
        c = chunks[cidx]
        print(f"  [{rank}] score={score:.4f} doc={c.relpath} (best_chunk={c.chunk_index})")
    print("")
    
    # ---------- naive answer (rule-based from top chunk) ----------
    def _naive_answer_from_chunk_text(q: str, chunk_text: str) -> str:
        q = q.strip()
        lines = [ln.strip() for ln in chunk_text.split("\n") if ln.strip()]
        
        # Prefer bullet/sentence lines that share keywords with the question.
        # (Korean tokenization is hard; keep it simple and robust.)
        keywords = []
        for w in re.split(r"\s+", q):
            w = w.strip(".,?!:;\"'()[]{}")
            if len(w) >= 2:
                keywords.append(w)
        keywords = keywords[:6] # cap
        
        # 1) bullets first
        for ln in lines:
            if ln.startswith(("-", "*", "•")):
                hit = sum(1 for k in keywords if k in ln)
                if hit >= 1:
                    return ln.lstrip("-*• ").strip()
        
        # 2) any line with keyword overlap
        best = ("", -1)
        for ln in lines:
            if ln.startswith("#"):
                continue
            hit = sum(1 for k in keywords if k in ln)
            if hit > best[1]:
                best = (ln, hit)
        if best[0] and best[1] >= 1:
            return best[0]
        
        # 3) fallback: first non-heading line
        for ln in lines:
            if not ln.startswith("#"):
                return ln.lstrip("-*• ").strip()
        
        return ""
    
    top1_text = chunks[int(order[0])].text if len(order) else ""
    naive_answer = _naive_answer_from_chunk_text(question, top1_text)
    
    if naive_answer:
        print(f"Answer (naive): {naive_answer}\n")
    
    # ---------- chunk-level evidence ----------
    print("Top Chunks (evidence):")
    for rank, idx in enumerate(order, start=1):
        c = chunks[int(idx)]
        score = float(sims[int(idx)])
        
        snippet = c.text.strip()

        # 너무 길면 보기 좋게 "줄 단위"로 컷
        max_chars = 700
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "\n..."

        # 들여쓰기해서 보기 좋게
        indented = "\n".join("    " + line for line in snippet.splitlines())
        print(f"[{rank}] score={score:.4f} doc={c.relpath} chunk={c.chunk_index} ({c.start_char}:{c.end_char})")
        print(indented + "\n")


def cmd_eval(cfg: AppConfig, dataset_path: str, ks: List[int], fail_k: int | None = None):
    meta, chunks, emb_index = _load_index(cfg)
    index_dir = Path(cfg.index_dir)
    
    ds_path = Path(dataset_path)
    if not ds_path.exists():
        raise FileNotFoundError(f"dataset not found: {ds_path}")
    
    # chunk -> doc mapping
    chunk_docs = [c.doc_id for c in chunks]
    
    def top_docs_for_query(q: str, k: int) -> List[str]:
        sims = emb_index.query(q)
        order = np.argsort(-sims)
        # doc-level score: max sim over chunks
        best: Dict[str, float] = {}
        for idx in order:
            d = chunk_docs[int(idx)]
            s = float(sims[int(idx)])
            if d not in best or s > best[d]:
                best[d] = s
        # sort docs by best score
        docs_sorted = sorted(best.items(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in docs_sorted[:k]]
    
    total = 0
    hit = {k: 0 for k in ks}
    
    failures = []
    with ds_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            q = row["query"]
            gold = row["gold_doc"]
            total += 1
            for k in ks:
                topd = top_docs_for_query(q, k)
                if gold in topd:
                    hit[k] += 1
            # store failure sample for max k
            fk = fail_k if fail_k is not None else max(ks)
            if gold not in top_docs_for_query(q, fk):
                failures.append({
                    "query": q,
                    "gold_doc": gold,
                    "top_docs": top_docs_for_query(q, fk)
                })
    
    print(f"\n[eval] dataset={ds_path} total={total} embedder={meta['embedder']['type']}")
    for k in ks:
        r = (hit[k] / total) if total else 0.0
        print(f"  Recall@{k}: {r:.4f} ({hit[k]}/{total})")
    
    # save failures
    out = index_dir / f"eval_failures_at_{fk}.json"
    out.write_text(json.dumps(failures[:50], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] failures(saved, up to 50): {out.resolve()}\n")

def _chunks_signature(chunks: List[Chunk]) -> List[tuple]:
    # alignment check용 서명(순서 포함)
    return [(c.doc_id, c.chunk_index, c.start_char, c.end_char) for c in chunks]

def load_two_indices_for_hybrid(sbert_dir: str, tfidf_dir: str):
    sdir = Path(sbert_dir)
    tdir = Path(tfidf_dir)
    
    if not (sdir / "meta.json").exists():
        raise RuntimeError(f"SBERT index not found: {sdir}/meta.json")
    if not (tdir / "meta.json").exists():
        raise RuntimeError(f"TFIDF index not found: {tdir}/meta.json")
    
    s_meta = load_meta(sdir)
    t_meta = load_meta(tdir)
    
    # chunks 로드 및 정합성 체크
    s_chunks = load_chunks(sdir)
    t_chunks = load_chunks(tdir)
    
    if len(s_chunks) != len(t_chunks):
        raise RuntimeError(f"Chunk mismatch: sbert={len(s_chunks)} tfidf={len(t_chunks)}. Rebuild both indices with same data/chunk config.")
    
    if _chunks_signature(s_chunks) != _chunks_signature(t_chunks):
        raise RuntimeError("Chunk order/content mismatch between indices. Rebuild both indices with identical data_dir and chunk params.")
    
    # index 로드
    s_index = SbertIndex.load(sdir)
    t_index = TfidfIndex.load(tdir)
    
    return s_meta, t_meta, s_chunks, s_index, t_index

def cmd_eval_hybrid(
    dataset_path: str,
    ks: List[int],
    sbert_index_dir: str,
    tfidf_index_dir: str,
    rerank_docs_k: int = 10,
    fail_k: int | None = None,
):
    s_meta, t_meta, chunks, s_index, t_index = load_two_indices_for_hybrid(sbert_index_dir, tfidf_index_dir)
    ds_path = Path(dataset_path)
    if not ds_path.exists():
        raise FileNotFoundError(f"dataset not found: {ds_path}")
    
    chunk_docs = [c.doc_id for c in chunks]
    
    def hybrid_top_docs(q: str, k: int) -> List[str]:
        # 1) SBERT: doc 후보 뽑기 (doc별 max sim)
        s_sims = s_index.query(q)
        best_s: Dict[str, float] = {}
        for i, d in enumerate(chunk_docs):
            s = float(s_sims[i])
            if (d not in best_s) or (s > best_s[d]):
                best_s[d] = s
        cand_docs = [d for d, _ in sorted(best_s.items(), key=lambda x: x[1], reverse=True)[: max(1, rerank_docs_k)]]
        cand_set = set(cand_docs)
        
        # 2) TF-IDF: 후보 문서만 대상으로 doc rerank (doc별 max sim)
        t_sims = t_index.query(q)
        best_t: Dict[str, float] = {}
        for i, d in enumerate(chunk_docs):
            if d not in cand_set:
                continue
            s = float(t_sims[i])
            if (d not in best_t) or (s > best_t[d]):
                best_t[d] = s
        
        # 3) 최종 정렬: TF-IDF 우선, 동점이면 SBERT로 타이브레이크
        def key(doc_id: str):
            return (best_t.get(doc_id, -1e9), best_s.get(doc_id, -1e9))
        
        ranked = sorted(cand_docs, key=key, reverse=True)
        return ranked[:k]
    
    total = 0
    hit = {k: 0 for k in ks}
    failures = []
    fk = fail_k if fail_k is not None else max(ks)
    
    with ds_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            q = row["query"]
            gold = row["gold_doc"]
            total += 1
            
            for k in ks:
                topd = hybrid_top_docs(q, k)
                if gold in topd:
                    hit[k] += 1
            
            top_fk = hybrid_top_docs(q, fk)
            if gold not in top_fk:
                failures.append({"query": q, "gold_doc": gold, "top_docs": top_fk})
    
    print(f"\n[eval-hybrid] dataset={ds_path} total={total} rerank_docs_k={rerank_docs_k}")
    print(f"  sbert_index_dir={sbert_index_dir}")
    print(f"  tfidf_index_dir={tfidf_index_dir}")
    for k in ks:
        r = (hit[k] / total) if total else 0.0
        print(f"  Recall@{k}: {r:.4f} ({hit[k]}/{total})")

    out = Path(sbert_index_dir) / f"eval_failures_hybrid_at_{fk}.json"
    out.write_text(json.dumps(failures[:50], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval-hybrid] failures(saved, up to 50): {out.resolve()}\n")


def cmd_query_json(cfg: AppConfig, question: str):
    meta, chunks, emb_index = _load_index(cfg)
    
    if not chunks:
        raise RuntimeError("No chunks in index. Rebuild index first.")
    
    top_k = min(cfg.retrieval.top_k, len(chunks))
    
    sims = emb_index.query(question)
    if len(sims) != len(chunks):
        raise RuntimeError(f"Index corrupted: sims({len(sims)}) != chunks({len(chunks)})")
    
    order = np.argsort(-sims)[:top_k]
    
    # doc-level summary (max score per doc)
    best_doc_score: Dict[str, float] = {}
    best_doc_chunk: Dict[str, int] = {}
    for i, c in enumerate(chunks):
        s = float(sims[i])
        if (c.doc_id not in best_doc_score) or (s > best_doc_score[c.doc_id]):
            best_doc_score[c.doc_id] = s
            best_doc_chunk[c.doc_id] = i
    
    top_docs = sorted(best_doc_score.items(), key=lambda x: x[1], reverse=True)
    top_docs = top_docs[: min(10, len(top_docs))]
    
    # naive answer from top chunk
    def _naive_answer_from_chunk_text(q: str, chunk_text: str) -> str:
        q = q.strip()
        lines = [ln.strip() for ln in chunk_text.split("\n") if ln.strip()]
        
        keywords = []
        for w in re.split(r"\s+", q):
            w = w.strip(".,?!:;\"'()[]{}")
            if len(w) >= 2:
                keywords.append(w)
        keywords = keywords[:6]
        
        for ln in lines:
            if ln.lstrip().startswith(("-", "*", "•")):
                hit = sum(1 for k in keywords if k in ln)
                if hit >= 1:
                    return ln.lstrip().lstrip("-*• ").strip()
        
        best = ("", -1)
        for ln in lines:
            if ln.startswith("#"):
                continue
            hit = sum(1 for k in keywords if k in ln)
            if hit > best[1]:
                best = (ln, hit)
        if best[0] and best[1] >= 1:
            return best[0]
        
        for ln in lines:
            if not ln.startswith("#"):
                return ln.lstrip("-*• ").strip()
        
        return ""
    
    top1_text = chunks[int(order[0])].text if len(order) else ""
    answer = _naive_answer_from_chunk_text(question, top1_text)
    
    # chunks evidence list
    evidence = []
    for rank, idx in enumerate(order, start=1):
        c = chunks[int(idx)]
        score = float(sims[int(idx)])
        evidence.append({
            "rank": rank,
            "score": round(score, 6),
            "doc": c.relpath,
            "chunk_index": c.chunk_index,
            "start_char": c.start_char,
            "end_char": c.end_char,
            "text": c.text,
        })
    
    docs = []
    for rank, (doc_id, score) in enumerate(top_docs, start=1):
        cidx = best_doc_chunk[doc_id]
        c = chunks[cidx]
        docs.append({
            "rank": rank,
            "score": round(float(score), 6),
            "doc": c.relpath,
            "best_chunk_index": c.chunk_index,
        })
    
    out = {
        "query": question,
        "answer_naive": answer,
        "embedder": meta.get("embedder", {}),
        "top_k": top_k,
        "docs": docs,
        "chunks": evidence,
    }
    
    print(json.dumps(out, ensure_ascii=False, indent=2))


# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser(prog="ragkit", description="Local RAG Kit MVP")
    p.add_argument("--config", default="config.yaml", help="cinfg yaml path")
    
    sub = p.add_subparsers(dest="cmd", required=True)
    
    p_index = sub.add_parser("index", help="build index")
    p_query = sub.add_parser("query", help="query index")
    p_query_json = sub.add_parser("query-json", help="query index (json output)")
    
    p_query.add_argument("question", help="your question text")
    p_query_json.add_argument("question", help="your question text")
    
    p_eval = sub.add_parser("eval", help="evaluate recall@k with jsonl dataset")
    p_eval.add_argument("--dataset", required=True, help="path to jsonl dataset")
    p_eval.add_argument("--ks", default="1,3,5,10", help="comma-separated K values, e.g. 1,3,5,10")
    p_eval.add_argument("--fail_k",  type=int, default=None, help="store failures at this K (e.g., 1). default: max(ks)")
    
    p_eval_h = sub.add_parser("eval-hybrid", help="evaluate hybrid: SBERT candidates -> TF-IDF rerank")
    p_eval_h.add_argument("--dataset", required=True, help="path to jsonl dataset")
    p_eval_h.add_argument("--ks", default="1,3,5,10", help="comma-separated K values")
    p_eval_h.add_argument("--sbert_index_dir", default="index_sbert", help="SBERT index dir")
    p_eval_h.add_argument("--tfidf_index_dir", default="index_tfidf", help="TF-IDF index dir")
    p_eval_h.add_argument("--rerank_docs_k", default=10, type=int, help="how many docs to keep from SBERT before rerank")
    p_eval_h.add_argument("--fail_k", default=None, type=int, help="store failures at this K (default=max(ks))")
    
    args = p.parse_args()
    cfg = load_config(args.config)
    
    if args.cmd == "index":
        cmd_index(cfg)
    elif args.cmd == "query":
        cmd_query(cfg, args.question)
    elif args.cmd == "query-json":
        cmd_query_json(cfg, args.question)
    elif args.cmd == "eval":
        ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
        cmd_eval(cfg, args.dataset, ks, args.fail_k)
    elif args.cmd == "eval-hybrid":
        ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
        cmd_eval_hybrid(
            dataset_path=args.dataset,
            ks=ks,
            sbert_index_dir=args.sbert_index_dir,
            tfidf_index_dir=args.tfidf_index_dir,
            rerank_docs_k=args.rerank_docs_k,
            fail_k=args.fail_k,
        )
    else:
        raise RuntimeError("unknown command")

if __name__ == "__main__":
    main()