# 삼성전자 주가 예측 파이프라인

삼성전자(005930.KS) 주가를 예측하기 위한 멀티에이전트 리서치 파이프라인입니다.
데이터 수집 → 피처 엔지니어링 → walk-forward 백테스트 → 평가까지 end-to-end로 동작합니다.

> **현재 상태:** 파이프라인은 정상 작동하지만, 예측 성능은 아직 낮습니다(방향성 정확도 ~48–52%).
> 이는 출발점으로서 예상된 결과이며, 피처 개선과 외부 데이터 추가를 통해 개선할 예정입니다.

---

## 핵심 기능

- **멀티에이전트 파이프라인:** `PlannerAgent → DataAgent → ModelingAgent → EvaluationAgent → ReportAgent`
- **Walk-forward 백테스트:** 미래 데이터 누수 없이 실제 트레이딩 재학습 방식으로 검증
- **외부 데이터 연동:** KOSPI, USD/KRW, VIX, 한국은행 기준금리 등 (선택적 활성화)
- **LLM 통합:** OpenAI / Anthropic으로 실험 제안 및 결과 해석 (선택 사항)
- **실험 리포트:** 매 실행마다 `reports/<experiment_id>.json`으로 결과 자동 저장

---

## 실행 방법

### 1. 설치

```bash
git clone https://github.com/coo001/stock-forecast.git
cd stock-forecast
pip install -r requirements.txt
```

### 2. 실행

```bash
# 기본 실행 (실시간 데이터 다운로드)
python run_pipeline.py

# 오프라인 테스트 (네트워크 불필요)
python run_pipeline.py --synthetic

# 캐시된 데이터만 사용
python run_pipeline.py --no-download

# 데이터 강제 재다운로드
python run_pipeline.py --force-refresh
```

### 3. LLM 연결 (선택 사항)

API 키는 **반드시 환경변수로만** 설정합니다. `config/default.yaml`의 `api_key_env` 필드에는 실제 키가 아니라 **환경변수 이름**을 넣습니다.

```bash
# Windows Command Prompt
set OPENAI_API_KEY=sk-...

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

`config/default.yaml`:
```yaml
llm:
  provider: "openai"          # none | openai | anthropic
  model: "gpt-4o-mini"
  api_key_env: "OPENAI_API_KEY"   # ← 키 자체가 아닌 환경변수 이름
```

LLM 없이 완전 결정론적으로 실행하려면 `provider: "none"`으로 설정하면 됩니다.

---

## 결과물

실행할 때마다 `reports/<experiment_id>.json`에 리포트가 저장되고, 터미널에 요약이 출력됩니다.

```
======================================================================
EXPERIMENT REPORT  [exp_20260319T071656]
Generated: 2026-03-19T07:17:08+00:00
======================================================================
Ticker     : 005930.KS
Target     : next_day_log_return  (horizon=1d)
Data       : 2015-01-02 → 2026-03-19  (2,747 rows)
Features   : 10
Model      : LGBMForecaster
OOS obs.   : 2,193  over 35 folds
----------------------------------------------------------------------
METRICS (mean across folds)
  directional_accuracy         0.4797
  sharpe                      -0.5348
  mae                          0.0141
  ic                           0.0046
----------------------------------------------------------------------
VERDICT    : POOR
```

리포트에는 폴드별 상세 지표, 피처 중요도, LLM 해석(활성화된 경우)이 포함됩니다.

---

## 현재 한계

- 방향성 정확도가 랜덤 수준(~50%)에 가깝고, Sharpe ratio가 음수
- 현재 피처셋(기술 지표 10개)만으로는 예측 신호가 충분하지 않음
- 1일 수익률 예측은 노이즈가 많아 본질적으로 어려운 태스크

---

## 다음 목표

- **외부 데이터 활성화:** KOSPI, USD/KRW, VIX 등 매크로 피처 추가 (`external_data.enabled: true`)
- **피처 개선:** 긴 리턴 윈도우(60d, 120d), 변동성 레짐, 캘린더 효과
- **타겟 변경:** 5일·20일 수익률, 방향성 분류(AUC 기준) 실험
- **모델 다양화:** Ridge 회귀, XGBoost, LSTM 등 비교 실험

---

## 테스트

```bash
pytest                     # 전체 테스트
pytest --cov=src           # 커버리지 포함
```

테스트는 합성 데이터와 MockLLMClient를 사용하므로 API 키나 인터넷 연결이 필요 없습니다.

---

## 스택

Python 3.11 · LightGBM · pandas · yfinance · Pydantic v2 · pytest · OpenAI/Anthropic (선택)

---

## 라이선스

MIT
