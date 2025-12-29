@echo off
chcp 65001 >nul

echo ===== STATUS =====
curl -s http://127.0.0.1:8000/status
echo.

echo ===== QUERY (SBERT) =====
curl -s -H "Content-Type: application/json; charset=utf-8" -d "{\"query\":\"부분 성공/중복 처리 점검 내용 어 디있어?\",\"include_text\":false,\"mode\":\"sbert\"}" http://127.0.0.1:8000/query
echo.

echo ===== QUERY (HYBRID, rerank_docs_k=5) =====
curl -s -H "Content-Type: application/json; charset=utf-8" -d "{\"query\":\"부분 성공/중복 처리 점검 내용 어 디있어?\",\"include_text\":false,\"mode\":\"hybrid\",\"rerank_docs_k\":5}" http://127.0.0.1:8000/query
echo.

pause
