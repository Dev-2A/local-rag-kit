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
