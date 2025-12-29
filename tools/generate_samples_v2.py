from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "data" / "docs_v2"
DS_DIR = ROOT / "data" / "datasets"
DS_PATH = DS_DIR / "eval_40_v2.jsonl"

def write(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.strip() + "\n", encoding="utf-8")

def main():
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    DS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Docs (6) ----
    write(DOC_DIR / "incident_runbook.md", """
# 장애 대응 런북 (Incident Runbook)

## 우선순위
1) 영향 범위(사용자/서비스/매출) 파악
2) 최근 변경(배포/설정/배치) 확인
3) 로그/메트릭/트레이스 확인

## 즉시 조치
- 장애 공지: 커뮤니케이션 채널 고정(슬랙/전화), 타임라인 기록
- 임시 완화: 트래픽 차단, 기능 플래그 off, 캐시 우회/적용
- 롤백: 배포 파이프라인에서 이전 태그로 되돌린다

## 체크리스트
- 에러율(5xx), 지연시간, 트래픽 급변 확인
- 의존성 장애(DB/Redis/외부 API) 여부 확인
- 재현 가능 여부 및 재현 조건 정리
""")

    write(DOC_DIR / "release_notes.md", """
# 릴리즈 노트 템플릿

## 변경점(What changed)
- 기능 추가/개선
- 버그 수정
- 성능/비용 최적화

## 영향도(Impact)
- 영향 범위(사용자/서비스/데이터)
- 다운타임 여부/슬로우다운 가능성

## 롤백(Rollback)
- 이전 태그로 롤백
- DB 스키마 변경이 있으면 롤백 절차와 주의사항 명시

## 점검 체크리스트(Post-check)
- API 정상 응답 확인
- 주요 배치 동작 확인
- 에러율/지연시간 모니터링 확인
""")

    write(DOC_DIR / "db_troubleshooting.md", """
# DB 트러블슈팅 가이드

## 흔한 증상
- DB 연결 오류 증가
- 커넥션 타임아웃
- 쿼리 지연/락 경합

## 점검 포인트
- 커넥션 풀 고갈(풀 사이즈, 대기열, 누수) 여부 확인
- 장기 트랜잭션/락(LOCK) 확인
- 느린 쿼리(인덱스 누락) 확인

## 완화 방법
- 풀 사이즈 조정(일시적)
- 문제 쿼리 수정/인덱스 추가
- 장기 트랜잭션 종료 및 배치 시간 조정
""")

    write(DOC_DIR / "api_auth.md", """
# API 인증/인가 가이드

## 토큰 인증(JWT 등)
- Authorization: Bearer <token>
- 토큰 만료 시 401 발생, 리프레시 토큰 또는 재로그인 필요

## 자주 나는 이슈
- 401 Unauthorized: 토큰 만료/누락/서명 불일치
- 403 Forbidden: 권한 부족(roles/scopes)
- Clock skew로 인한 만료 오판(서버 시간 동기화 필요)

## 운영 팁
- 만료 시간(exp)과 서버 시간(NTP) 점검
- 권한 정책 변경 시 영향도 공지 및 점진적 롤아웃
""")

    write(DOC_DIR / "batch_ops.md", """
# 배치 운영 가이드

## 기본 원칙
- 배치는 idempotent(재실행 가능)하게 만든다
- 재시도 정책(지수 백오프)과 데드레터 큐(DLQ) 고려

## 장애 시 대응
- 실패 지점 확인(입력 데이터/외부 의존성/DB 락)
- 부분 성공/중복 처리 여부 점검
- 필요 시 특정 구간만 재처리(리런)

## 체크리스트
- 스케줄러(cron) 정상 동작
- 배치 실행 시간 급증 여부
- 실패율/재시도 횟수 모니터링
""")

    write(DOC_DIR / "observability.md", """
# 관측(Observability) 가이드

## 3대 신호
- 로그(Logs): 에러 스택/요청 ID/사용자 영향
- 메트릭(Metrics): 에러율, 지연시간, QPS, 리소스(CPU/메모리)
- 트레이스(Traces): 분산 추적, 병목 구간

## 알람 설계
- SLO 기반(에러율/지연시간) 알람 우선
- 알람 폭주 방지(집계/쿨다운/우선순위)

## 실무 팁
- 요청 ID를 로그/트레이스에 통일
- 대시보드: 서비스 개요 → 의존성 → 상세로 단계화
""")

    # ---- Eval dataset (40) ----
    # gold_doc must match relative path under DOC_DIR, so use filenames only.
    eval_rows = [
        # incident_runbook.md (8)
        ("장애 나면 제일 먼저 뭐부터 해?", "incident_runbook.md"),
        ("장애 대응 우선순위가 뭐야?", "incident_runbook.md"),
        ("장애 때 최근 변경은 어떻게 확인해?", "incident_runbook.md"),
        ("롤백은 어디서 어떻게 해?", "incident_runbook.md"),
        ("기능 플래그로 임시 완화하는 내용은 어디 있어?", "incident_runbook.md"),
        ("에러율이랑 지연시간 확인하라는 런북 문서가 뭐야?", "incident_runbook.md"),
        ("의존성 장애(DB/Redis) 점검 얘기 나온 문서?", "incident_runbook.md"),
        ("장애 타임라인 기록과 커뮤니케이션 채널 고정은 어디에 적혀있어?", "incident_runbook.md"),

        # release_notes.md (7)
        ("릴리즈 노트 템플릿 있어?", "release_notes.md"),
        ("릴리즈 노트에 변경점은 어디에 써?", "release_notes.md"),
        ("릴리즈 영향도(impact) 적는 섹션이 뭐야?", "release_notes.md"),
        ("롤백 절차는 릴리즈 노트 어디에 넣어?", "release_notes.md"),
        ("배포 후 점검 체크리스트 항목 알려줘", "release_notes.md"),
        ("API 정상 응답 확인은 어떤 문서에 있어?", "release_notes.md"),
        ("주요 배치 동작 확인은 릴리즈 문서에 있어?", "release_notes.md"),

        # db_troubleshooting.md (8)
        ("DB 연결 오류가 늘었는데 뭘 점검해야 해?", "db_troubleshooting.md"),
        ("커넥션 풀 고갈 여부 확인은 어디 문서?", "db_troubleshooting.md"),
        ("쿼리 지연이 있으면 어떤 원인을 봐야 해?", "db_troubleshooting.md"),
        ("락 경합/장기 트랜잭션 점검 내용 어디 있어?", "db_troubleshooting.md"),
        ("느린 쿼리 인덱스 누락 관련 가이드 문서?", "db_troubleshooting.md"),
        ("커넥션 타임아웃 대응은 어디에 있어?", "db_troubleshooting.md"),
        ("풀 사이즈 조정 같은 완화 방법 나오는 문서?", "db_troubleshooting.md"),
        ("배치 시간 조정이 DB 문제 완화에 도움이 된다는 내용 어디야?", "db_troubleshooting.md"),

        # api_auth.md (6)
        ("401 Unauthorized가 나오는 흔한 이유는?", "api_auth.md"),
        ("403 Forbidden은 뭐가 문제야?", "api_auth.md"),
        ("JWT 토큰 만료로 401 나는 내용 어디 있어?", "api_auth.md"),
        ("Authorization Bearer 토큰 헤더 형식 문서?", "api_auth.md"),
        ("서버 시간 차이(clock skew)로 인증 문제 난다는 내용 어디야?", "api_auth.md"),
        ("권한 정책 변경 시 운영 팁 있는 문서?", "api_auth.md"),

        # batch_ops.md (6)
        ("배치는 왜 idempotent하게 만들라고 해?", "batch_ops.md"),
        ("배치 재시도 정책(지수 백오프) 언급한 문서?", "batch_ops.md"),
        ("배치 실패하면 어디부터 확인해?", "batch_ops.md"),
        ("부분 성공/중복 처리 점검 내용 어디있어?", "batch_ops.md"),
        ("특정 구간만 재처리(리런)하라는 가이드 문서?", "batch_ops.md"),
        ("cron 스케줄러 체크리스트 있는 문서?", "batch_ops.md"),

        # observability.md (5)
        ("관측의 3대 신호가 뭐야?", "observability.md"),
        ("알람 폭주 방지(쿨다운/집계) 내용 어디있어?", "observability.md"),
        ("SLO 기반 알람 우선이라는 문서?", "observability.md"),
        ("요청 ID를 로그/트레이스에 통일하라는 팁 어디야?", "observability.md"),
        ("대시보드 설계(개요→의존성→상세) 가이드 문서?", "observability.md"),
    ]

    with DS_PATH.open("w", encoding="utf-8") as f:
        for q, gold in eval_rows:
            f.write(json.dumps({"query": q, "gold_doc": gold}, ensure_ascii=False) + "\n")

    print(f"[ok] wrote docs -> {DOC_DIR}")
    print(f"[ok] wrote dataset -> {DS_PATH} (rows={len(eval_rows)})")

if __name__ == "__main__":
    main()
