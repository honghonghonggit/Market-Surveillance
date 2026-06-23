# 프로젝트: 이상거래 탐지(시장감시) 엔진

실시간 주문 스트림에서 스푸핑·레이어링·워시트레이딩 같은 시세조종 패턴을 탐지하는 시스템.
핵심은 합성 데이터에 조작 패턴을 직접 주입해 ground truth를 확보하고, 탐지 모델을 정밀도·재현율로 엄밀하게 평가하는 것.
전체 기획/스코프/스택은 @docs/PROJECT_BRIEF.md 참고. MINI-Exchange, RA-Testbed와 동일한 운영 원칙을 따른다.

## 기술 스택
- 언어: Python 3.12
- 데이터 처리: pandas, numpy
- 탐지 모델: scikit-learn (통계 기반 룰 + ML 분류기)
- 시각화/대시보드: Streamlit + Plotly
- 테스트: pytest
- 배포: Streamlit Community Cloud
- (Kafka는 기획 초안에 있었으나 의도적으로 제외 — 인메모리 스트림으로 탐지 로직 완전 검증 가능, 스코프 판단. 근거: README §4(9))

## 빌드 / 테스트
- `pip install -r requirements.txt`
- `pytest`
- `python -m streamlit run src/surveillance/app.py`
- `python scripts/run_phase1.py` / `run_phase2.py` / `run_phase3.py`

## 개발 순서 (실제 진행)
1. 합성 주문 데이터 생성기 + 패턴 주입(ground truth 라벨 포함) — TDD
2. 피처 엔지니어링(주문취소율, 가격충격, 레이어링 지표 등)
3. 탐지 모델(룰 기반 → ML) + 정밀도·재현율·ROC 평가
4. Streamlit 대시보드
5. Phase 3: 패턴×강도 분해, 탐지 지연, 오탐 분석

## 핵심 원칙
- 범위는 Phase1(MVP) → Phase2(차별화) → Phase3(스트레치) 순서로 단계적으로 확장한다. Phase1이 끝나기 전 다음 단계에 손대지 않는다.
- 이 프로젝트의 가장 중요한 차별점은 "평가 방법론의 엄밀함"이다. 합성 데이터에 패턴을 주입해 ground truth를 알고 있는 상태에서 모델을 평가하므로, 정밀도/재현율/혼동행렬/ROC가 진짜 의미를 갖는다. 이 점을 README 설계 결정에서 가장 비중 있게 다룬다.
- 실제 특정 종목/거래소/투자자를 지목하거나, 실거래 데이터를 사용했다는 식의 주장은 절대 하지 않는다. 100% 합성 데이터 기반임을 README와 화면에 명시한다.
- 코스콤이 운영하는 시장감시시스템(CAMS)을 "그대로 재현했다"는 단정적 표현은 쓰지 않는다. "그 개념을 이해하고 시세조종 탐지의 핵심 원리를 직접 구현해본 것"이라고 서술한다.
- 커밋 메시지에 "Co-Authored-By: Claude"나 "Generated with Claude Code" 같은 attribution을 절대 넣지 않는다. (전역 설정 includeCoAuthoredBy:false와 함께 이중 안전장치)
- 큰 설계 변경 전에는 plan mode로 먼저 합의받는다.

## 폴더 구조
```
market-surveillance/
├── src/surveillance/
│   ├── generator/              # 합성 데이터 생성 + 패턴 주입
│   ├── features/               # 피처 엔지니어링
│   ├── detection/              # 탐지 모델 + 평가 + Phase 3 분석
│   ├── viz.py                  # matplotlib 한글 폰트 공통 설정
│   └── app.py                  # Streamlit 대시보드
├── scripts/                    # run_phase1/2/3.py
├── tests/
├── docs/                       # PROJECT_BRIEF.md, architecture.png
├── requirements.txt
└── README.md
```
