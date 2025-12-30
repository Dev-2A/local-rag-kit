from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Tuple

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

@app.on_event("startup")
def _startup_load_indexes():
    try:
        _ensure_loaded_dual("config.yaml", "index_sbert", "index_tfidf")
        print("[startup] indexes loaded:", STATE["sbert"]["index_dir"], STATE["tfidf"]["index_dir"])
    except Exception as e:
        print("[startup] failed to load indexes:", repr(e))

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


_BAD_LINE_PATTERNS = [
    r"^#+\s",                 # markdown header
    r"^\s*```",               # code fence
    r"^(curl|pip|python)\b",  # command-like
    r"Authorization:\s*Bearer\b",
    r"<token>",
]

def _looks_bad_line(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return True
    for pat in _BAD_LINE_PATTERNS:
        if re.search(pat, s, flags=re.IGNORECASE):
            return True
    return False

def _score_answer_candidate(line: str, q_keywords: List[str]) -> float:
    s = (line or "").strip()
    if _looks_bad_line(s):
        return -1e9

    # strip bullet / numbering prefix
    s2 = re.sub(r"^\s*([-*•]+)\s*", "", s)
    s2 = re.sub(r"^\s*\d+[\).\]]\s*", "", s2).strip()
    if not s2 or _looks_bad_line(s2):
        return -1e9

    n = len(s2)
    score = 0.0

    # 1) query overlap: 가장 중요 (도메인 룰 아님)
    # - 토큰이 포함되면 가산
    # - 여러 토큰이 들어가면 크게 가산
    hits = 0
    for k in q_keywords:
        if k and k in s2:
            hits += 1

    # hits가 1 이상이면 급격히 유리하게 (정답 라인을 잡아내기 위해)
    if hits > 0:
        score += 3.0 + (hits - 1) * 1.2  # hits=1:+3.0, hits=2:+4.2, hits=3:+5.4 ...
    else:
        score -= 0.5  # 전혀 안 맞으면 살짝 페널티

    # 2) length preference
    if n < 12:
        score -= 1.5
    elif 16 <= n <= 140:
        score += 1.0
    elif n > 220:
        score -= 1.0

    # 3) sentence-ish signals (generic)
    if ":" in s2:
        score += 0.4
    if any(ch in s2 for ch in [".", "!", "?", "…"]):
        score += 0.2
    if re.search(r"(다\.?$|요\.?$|합니다\.?$|한다\.?$)", s2):
        score += 0.2

    # 4) Penalize bare section labels
    if n < 25 and re.fullmatch(r"[\w\s\-/()]+", s2):
        score -= 0.6

    return score

def _naive_answer_from_evidence_texts(question: str, ranked_texts: List[Tuple[int, str]]) -> str:
    """
    ranked_texts: [(rank, text)] where rank starts at 1 (higher is worse).
    We bias toward higher-ranked evidence (rank=1).
    """
    q = (question or "").strip()
    if not ranked_texts:
        return ""

    q_keywords: List[str] = []
    for w in re.split(r"\s+", q):
        w = w.strip(".,?!:;\"'()[]{}")
        if len(w) >= 2:
            q_keywords.append(w)
    q_keywords = q_keywords[:6]

    best_line = ""
    best_score = -1e18

    for rank, txt in ranked_texts:
        if not txt:
            continue

        # rank bonus: rank=1 gets +1.2, rank=5 gets +0.0 (linear)
        # clamp rank into [1,5] for stable behavior
        r = max(1, min(int(rank), 5))
        rank_bonus = (5 - r) * 0.3  # rank1:+1.2, rank2:+0.9, ..., rank5:+0.0

        for raw in txt.splitlines():
            ln = raw.strip()
            if not ln:
                continue
            if ln.startswith("#"):
                continue

            sc = _score_answer_candidate(ln, q_keywords) + rank_bonus
            if sc > best_score:
                best_score = sc
                best_line = ln

    best_line = re.sub(r"^\s*([-*•]+)\s*", "", best_line).strip()
    best_line = re.sub(r"^\s*\d+[\).\]]\s*", "", best_line).strip()
    return best_line


def _chunks_signature(chunks) -> List[tuple]:
    return [(c.doc_id, c.chunk_index, c.start_char, c.end_char) for c in chunks]


def _ensure_loaded_dual(cfg_path: str, sbert_index_dir: str, tfidf_index_dir: str):
    if not sbert_index_dir or not tfidf_index_dir:
        raise RuntimeError(f"index_dir not set: sbert_index_dir={sbert_index_dir}, tfidf_index_dir={tfidf_index_dir}")

    if not Path(sbert_index_dir).exists():
        raise RuntimeError(f"sbert index_dir not found: {sbert_index_dir}")
    if not Path(tfidf_index_dir).exists():
        raise RuntimeError(f"tfidf index_dir not found: {tfidf_index_dir}")
    
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
    # Always load both indices (hybrid-capable server)
    _ensure_loaded_dual(config_path, "index_sbert", "index_tfidf")

    cfg = STATE["cfg"]
    assert cfg is not None

    s_meta = STATE["sbert"]["meta"]
    t_meta = STATE["tfidf"]["meta"]
    chunks = STATE["sbert"]["chunks"]  # aligned
    s_index = STATE["sbert"]["index"]
    t_index = STATE["tfidf"]["index"]
    assert s_meta and t_meta and chunks and s_index and t_index

    # -------- params / defaults --------
    default_top_k = getattr(getattr(cfg, "retrieval", None), "top_k", 5)
    top_k = int(req.top_k if req.top_k is not None else default_top_k)
    top_k = max(1, min(top_k, len(chunks)))

    default_max_chars = getattr(getattr(cfg, "retrieval", None), "max_chars", 800)
    max_chars = int(req.max_chars if getattr(req, "max_chars", None) is not None else default_max_chars)

    # "docs" block size (doc ranking output)
    tdk = req.top_docs_k if getattr(req, "top_docs_k", None) is not None else top_k
    tdk = max(1, int(tdk))

    # hybrid candidate docs count
    rk = req.rerank_docs_k if getattr(req, "rerank_docs_k", None) is not None else 5
    rk = max(1, int(rk))

    include_text = bool(getattr(req, "include_text", True))

    # -------- helpers --------
    chunk_docs = [c.doc_id for c in chunks]

    def doc_best_from_sims(sims) -> Dict[str, float]:
        best: Dict[str, float] = {}
        for i, d in enumerate(chunk_docs):
            sc = float(sims[i])
            if (d not in best) or (sc > best[d]):
                best[d] = sc
        return best

    def ranked_docs_from_best(best: Dict[str, float]) -> List[str]:
        return [d for d, _ in sorted(best.items(), key=lambda x: x[1], reverse=True)]

    def best_chunk_for_doc(d: str, sims) -> Tuple[int, float]:
        best_idx = -1
        best_score = -1e18
        for i, c in enumerate(chunks):
            if c.doc_id != d:
                continue
            sc = float(sims[i])
            if sc > best_score:
                best_score = sc
                best_idx = i
        return best_idx, best_score

    mode = (getattr(req, "mode", None) or "hybrid").lower().strip()
    if mode not in ("tfidf", "sbert", "hybrid"):
        mode = "hybrid"

    # -------- retrieval / ranking --------
    meta = s_meta
    score_type = "sbert_cosine"
    doc_filter = None

    if mode == "tfidf":
        sims = t_index.query(req.query)
        meta = t_meta
        score_type = "tfidf_cosine"
        ranked_docs = ranked_docs_from_best(doc_best_from_sims(sims))

    elif mode == "sbert":
        sims = s_index.query(req.query)
        meta = s_meta
        score_type = "sbert_cosine"
        ranked_docs = ranked_docs_from_best(doc_best_from_sims(sims))

    else:
        # hybrid: SBERT candidates -> TFIDF rerank (doc-level)
        s_sims = s_index.query(req.query)
        s_best = doc_best_from_sims(s_sims)

        cand_docs = ranked_docs_from_best(s_best)[:rk]
        cand_set = set(cand_docs)

        t_sims = t_index.query(req.query)

        # TF-IDF has no signal => fallback to SBERT for both doc order and evidence order
        if float(np.max(t_sims)) <= 0.0:
            sims = s_sims
            meta = s_meta
            ranked_docs = cand_docs
            score_type = "sbert_cosine_fallback"
        else:
            # rerank candidates by TF-IDF best chunk per doc (tie-break by SBERT)
            t_best: Dict[str, float] = {}
            for i, d in enumerate(chunk_docs):
                if d not in cand_set:
                    continue
                sc = float(t_sims[i])
                if (d not in t_best) or (sc > t_best[d]):
                    t_best[d] = sc

            def doc_key(d: str):
                return (t_best.get(d, -1e9), s_best.get(d, -1e9))

            ranked_docs = sorted(cand_docs, key=doc_key, reverse=True)
            sims = t_sims
            meta = s_meta
            score_type = "tfidf_cosine_rerank"

        # hybrid: restrict outputs to ranked_docs
        doc_filter = set(ranked_docs)

    # limit docs output
    ranked_docs = ranked_docs[: min(tdk, len(ranked_docs))]
    if mode == "hybrid":
        doc_filter = set(ranked_docs)

    # -------- docs block (best chunk per doc) --------
    docs_out = []
    for rank, d in enumerate(ranked_docs, start=1):
        best_idx, best_sc = best_chunk_for_doc(d, sims)
        if best_idx < 0:
            continue
        c = chunks[int(best_idx)]
        docs_out.append({
            "rank": rank,
            "score": round(float(best_sc), 6) if best_sc > -1e17 else 0.0,
            "doc": c.relpath,
            "best_chunk_index": c.chunk_index,
        })

    # -------- chunks evidence (top_k) --------
    order = np.argsort(-sims)

    evidence = []
    ranked_texts_for_answer = []  # [(rank, full_text)]
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

        # collect text for answer selection even if include_text=false
        ranked_texts_for_answer.append((len(evidence) + 1, c.text))

        if include_text:
            t = c.text
            if len(t) > max_chars:
                t = t[:max_chars].rstrip() + "\n..."
            item["text"] = t

        evidence.append(item)
        if not top1_text:
            top1_text = c.text

        if len(evidence) >= top_k:
            break

    # -------- answer_naive: choose best line from evidence, biasing higher-ranked evidence --------
    answer = _naive_answer_from_evidence_texts(req.query, ranked_texts_for_answer) if ranked_texts_for_answer else ""
    if not answer and top1_text:
        # final fallback (should rarely happen)
        answer = _naive_answer_from_chunk_text(req.query, top1_text)

    return {
        "query": req.query,
        "answer_naive": answer,
        "embedder": meta.get("embedder", {}),
        "top_k": top_k,
        "docs": docs_out,
        "chunks": evidence,
        "mode": mode,
        "score_type": score_type,
        "rerank_docs_k": rk if mode == "hybrid" else None,
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