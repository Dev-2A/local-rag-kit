@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo ===== STATUS =====
curl -s http://127.0.0.1:8000/status
echo.
echo.

echo ===== CHECK-DATA =====
curl -s http://127.0.0.1:8000/check-data
echo.
echo.

set MAXCH=220

call :RUN_ONE "부분 성공/중복 처리 점검 내용 어 디있어?"
call :RUN_ONE "롤백은 어떻게 해?"
call :RUN_ONE "DB 연결 오류 원인 뭐야?"

endlocal
exit /b 0

:RUN_ONE
set Q=%~1
echo ===== QUERY (SBERT) : %Q% =====
curl -s -H "Content-Type: application/json; charset=utf-8" -d "{\"query\":\"%Q%\",\"include_text\":true,\"mode\":\"sbert\",\"top_k\":5,\"max_chars\":%MAXCH%}" http://127.0.0.1:8000/query
echo.
echo.
echo ===== QUERY (HYBRID, rerank_docs_k=5) : %Q% =====
curl -s -H "Content-Type: application/json; charset=utf-8" -d "{\"query\":\"%Q%\",\"include_text\":true,\"mode\":\"hybrid\",\"rerank_docs_k\":5,\"top_k\":5,\"max_chars\":%MAXCH%}" http://127.0.0.1:8000/query
echo.
echo.
exit /b 0
