from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

import numpy as np
import re
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ragkit import (
    load_config,
    cmd_index,
    load_meta,
    load_chunks,
    compute_data_fingerprint,
    TfidfIndex,
    SbertIndex,
)

app = FastAPI(title="Local RAG Kit API (Hybrid)", version="0.2.0")

STATE: Dict[str, Any] = {
    "cfg_path": None,
    "cfg": None,
    "sbert": {"meta": None, "chunks": None, "index": None, "index_dir": None},
    "tfidf": {"meta": None, "chunks": None, "index": None, "index_dir": None},
}

class IndexResponse(BaseModel):
    status: str
    index_dir: str

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    
    # retrieval
    top_k: Optional[int] = Field(None, ge=1)
    top_docs_k: int = Field(5, ge=1, le=20)
    
    # response size
    include_text: bool = Field(False)
    max_chars: int = Field(700, ge=1, le=5000)
    
    # engine mode
    mode: Literal["tfidf", "sbert", "hybrid"] = Field(default="hybrid")
    
    # hybrid params
    rerank_docs_k: int = Field(5, ge=1, le=50)
    
    # index dirs
    sbert_index_dir: str = Field("index_sbert")
    tfidf_index_dir: str = Field("index_tfidf")


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


def _chunks_signature(chunks) -> List[tuple]:
    return [(c.doc_id, c.chunk_index, c.start_char, c.end_char) for c in chunks]


def _ensure_loaded_dual(cfg_path: str, sbert_index_dir: str, tfidf_index_dir: str):
    cfg = load_config(cfg_path)
    STATE["cfg_path"] = cfg_path
    STATE["cfg"] = cfg
    
    # SBERT
    if STATE["sbert"]["index_dir"] != sbert_index_dir:
        sdir = Path(sbert_index_dir)
        s_meta = load_meta(sdir)
        s_chunks = load_chunks(sdir)
        s_index = SbertIndex.load(sdir)
        STATE["sbert"] = {"meta": s_meta, "chunks": s_chunks, "index": s_index, "index_dir": sbert_index_dir}
    
    # TFIDF
    if STATE["tfidf"]["index_dir"] != tfidf_index_dir:
        tdir = Path(tfidf_index_dir)
        t_meta = load_meta(tdir)
        t_chunks = load_chunks(tdir)
        t_index = TfidfIndex.load(tdir)
        STATE["tfidf"] = {"meta": t_meta, "chunks": t_chunks, "index": t_index, "index_dir": tfidf_index_dir}
    
    # alignment check
    s_sig = _chunks_signature(STATE["sbert"]["chunks"])
    t_sig = _chunks_signature(STATE["tfidf"]["chunks"])
    if s_sig != t_sig:
        raise RuntimeError("SBERT/TF-IDF chunks mismatch. Rebuild both indices with same data_dir and chunk params.")


def _run_query(req: QueryRequest, config_path: str) -> Dict[str, Any]:
    _ensure_loaded_dual(config_path, req.sbert_index_dir, req.tfidf_index_dir)
    
    cfg = STATE["cfg"]
    assert cfg is not None
    
    s_meta = STATE["sbert"]["meta"]
    t_meta = STATE["tfidf"]["meta"]
    chunks = STATE["sbert"]["chunks"]  # aligned
    s_index = STATE["sbert"]["index"]
    t_index = STATE["tfidf"]["index"]
    
    assert s_meta and t_meta and chunks and s_index and t_index
    
    top_k = req.top_k if req.top_k is not None else cfg.retrieval.top_k
    top_k = min(int(top_k), len(chunks))
    if top_k <= 0:
        top_k = 1
    
    chunk_docs = [c.doc_id for c in chunks]
    
    def doc_best_from_sims(sims) -> Dict[str, float]:
        best = {}
        for i, d in enumerate(chunk_docs):
            sc = float(sims[i])
            if (d not in best) or (sc > best[d]):
                best[d] = sc
        return best
    
    mode = (req.mode or "hybrid").lower().strip()
    if mode not in ("tfidf", "sbert", "hybrid"):
        mode = "hybrid"
    
    if mode == "tfidf":
        sims = t_index.query(req.query)
        meta = t_meta
        best_doc = doc_best_from_sims(sims)
        ranked_docs = [d for d, _ in sorted(best_doc.items(), key=lambda x: x[1], reverse=True)]
    
    elif mode == "sbert":
        sims = s_index.query(req.query)
        meta = s_meta
        best_doc = doc_best_from_sims(sims)
        ranked_docs = [d for d, _ in sorted(best_doc.items(), key=lambda x: x[1], reverse=True)]
    
    else:
        # hybrid: SBERT candidates -> TFIDF rerank (doc-level)
        s_sims = s_index.query(req.query)
        s_best = doc_best_from_sims(s_sims)
        cand_docs = [d for d, _ in sorted(s_best.items(), key=lambda x: x[1], reverse=True)[: max(1, req.rerank_docs_k)]]
        cand_set = set(cand_docs)
        
        t_sims = t_index.query(req.query)
        t_best = {}
        for i, d in enumerate(chunk_docs):
            if d not in cand_set:
                continue
            sc = float(t_sims[i])
            if (d not in t_best) or (sc > t_best[d]):
                t_best[d] = sc
        
        def doc_key(d: str):
            return (t_best.get(d, -1e9), s_best.get(d, -1e9))
        
        ranked_docs = sorted(cand_docs, key=doc_key, reverse=True)
        
        # chunk scoring: use TFIDF sims for stability
        sims = t_sims
        meta = s_meta # 대표로 sbert meta
    
    ranked_docs = ranked_docs[: min(req.top_docs_k, len(ranked_docs))]
    doc_filter = set(ranked_docs) if mode == "hybrid" else None
    
    # doc block (best chunk per doc)
    docs_out = []
    for rank, d in enumerate(ranked_docs, start=1):
        best_idx = None
        best_score = None
        for i, c in enumerate(chunks):
            if c.doc_id != d:
                continue
            sc = float(sims[i])
            if best_score is None or sc > best_score:
                best_score = sc
                best_idx = i
        
        c = chunks[int(best_idx)]
        docs_out.append({
            "rank": rank,
            "score": round(float(best_score or 0.0), 6),
            "doc": c.relpath,
            "best_chunk_index": c.chunk_index,
        })
    
    # chunks evidence
    order = np.argsort(-sims)
    evidence = []
    top1_text = ""
    for idx in order:
        c = chunks[int(idx)]
        if doc_filter is not None and c.doc_id not in doc_filter:
            continue
        
        score = float(sims[int(idx)])
        item = {
            "rank": len(evidence) + 1,
            "score": round(score, 6),
            "doc": c.relpath,
            "chunk_index": c.chunk_index,
            "start_char": c.start_char,
            "end_char": c.end_char,
        }
        if req.include_text:
            t = c.text
            if len(t) > req.max_chars:
                t = t[: req.max_chars].rstrip() + "\n..."
            item["text"] = t
        
        evidence.append(item)
        if not top1_text:
            top1_text = c.text
        
        if len(evidence) >= top_k:
            break
    
    answer = _naive_answer_from_chunk_text(req.query, top1_text) if top1_text else ""
    
    return {
        "query": req.query,
        "answer_naive": answer,
        "embedder": meta.get("embedder", {}),
        "top_k": top_k,
        "docs": docs_out,
        "chunks": evidence,
        "mode": mode,
        "score_type": (
            "sbert_cosine" if mode == "sbert"
            else "tfidf_cosine" if mode == "tfidf"
            else "tfidf_cosine_rerank"
        ),
        "rerank_docs_k": req.rerank_docs_k if mode == "hybrid" else None,
    }

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status(config_path: str = "config.yaml"):
    try:
        cfg = load_config(config_path)
        
        s_meta = None
        t_meta = None
        sdir = Path("index_sbert")
        tdir = Path("index_tfidf")
        if (sdir / "meta.json").exists():
            s_meta = load_meta(sdir)
        if (tdir / "meta.json").exists():
            t_meta = load_meta(tdir)
        
        def slim(meta):
            if not meta:
                return None
            return {
                "created_at": meta.get("created_at"),
                "doc_count": meta.get("doc_count"),
                "chunk_count": meta.get("chunk_count"),
                "data_fingerprint": meta.get("data_fingerprint"),
                "data_change": meta.get("data_change"),
                "embedder": meta.get("embedder"),
            }
        
        return {
            "status": "ok",
            "config_path": config_path,
            "cfg_index_dir": cfg.index_dir,
            "loaded": {
                "sbert_index_dir": STATE["sbert"]["index_dir"],
                "tfidf_index_dir": STATE["tfidf"]["index_dir"],
            },
            "meta": {
                "sbert": slim(s_meta),
                "tfidf": slim(t_meta),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/check-data")
def check_data(config_path: str = "config.yaml", deep: bool = False):
    """
    Check whether current data_dir differs from indexed fingerprints.
    This does NOT rebuild indexes. It only reports status.
    """
    try:
        cfg = load_config(config_path)
        
        data_dir = getattr(cfg, "data_dir", None)
        if not data_dir:
            raise RuntimeError("config has no data_dir")
        
        allowed_exts = []
        if hasattr(cfg, "loader") and getattr(cfg.loader, "allowed_exts", None):
            allowed_exts = list(cfg.loader.allowed_exts)
        
        current_fp = compute_data_fingerprint(data_dir, allowed_exts, deep=deep)
        
        def read_index_fp(index_dir: str):
            idir = Path(index_dir)
            if not (idir / "meta.json").exists():
                return {"indexed_fingerprint": None, "changed": None, "status": "missing_index"}
            meta = load_meta(idir)
            indexed_fp = meta.get("data_fingerprint")
            return {
                "indexed_fingerprint": indexed_fp,
                "changed": (indexed_fp != current_fp),
                "status": "ok",
            }
        
        s = read_index_fp("index_sbert")
        t = read_index_fp("index_tfidf")
        
        # overall 판단: 둘 중 하나라도 changed=True면 changed
        any_missing = (s["status"] != "ok") or (t["status"] != "ok")
        any_changed = (s.get("changed") is True) or (t.get("changed") is True)
        
        overall = "missing_index" if any_missing else ("changed" if any_changed else "unchanged")
        
        return {
            "data_dir": data_dir,
            "deep": deep,
            "current_fingerprint": current_fp,
            "indexes": {
                "index_sbert": s,
                "index_tfidf": t,
            },
            "overall_change": overall,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index", response_model=IndexResponse)
def build_index(config_path: str = "config.yaml"):
    try:
        cfg = load_config(config_path)
        cmd_index(cfg)
        # 캐시 리셋
        STATE["sbert"]["index_dir"] = None
        STATE["tfidf"]["index_dir"] = None
        return {"status": "indexed", "index_dir": cfg.index_dir}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query(req: QueryRequest, config_path: str = "config.yaml"):
    try:
        out = _run_query(req, config_path)
        return JSONResponse(content=out, media_type="application/json; charset=utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default=8000, type=int)
    args = ap.parse_args()

    import uvicorn
    uvicorn.run("api_hybrid:app", host=args.host, port=args.port, reload=False)