"""Phase 2 파이프라인 — 3종 패턴, ML 분류기 vs 룰, ROC/PR 시각화.

생성(스푸핑·워시·레이어링 + ground truth) → 피처 → 시간분할(lookahead 방지)
→ 룰/ML 탐지 → 동일 test 구간에서 정밀도·재현율 비교 → ROC/PR/임계값 플롯 저장.

실행:  python scripts/run_phase2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows 콘솔(cp949)에서도 한글/특수문자 출력이 깨지거나 죽지 않도록 UTF-8로.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from surveillance.detection.curves import binary_metrics, curve_summary, save_curves  # noqa: E402
from surveillance.detection.evaluate import build_truth  # noqa: E402
from surveillance.detection.ml import (  # noqa: E402
    MLConfig,
    build_labeled_frame,
    feature_importances,
    time_split,
    train_and_score,
)
from surveillance.detection.rules import RuleConfig, predict  # noqa: E402
from surveillance.features.account_features import build_features  # noqa: E402
from surveillance.generator.dataset import DatasetConfig, generate_dataset  # noqa: E402
from surveillance.generator.injectors import InjectionConfig  # noqa: E402


def main() -> None:
    # 난이도(위장) 모드: 강도 랜덤화 + 조작 계좌의 정상 거래 위장 → 탐지가 불완전해져
    # 정밀도/재현율이 실제로 맞교환되고 ROC/PR 커브가 의미를 갖는다.
    # 약한 끝을 정상 영역까지 넓혀(예: 스푸핑 소량, 레이어링 2레벨) 일부 조작이
    # 정상과 피처 공간에서 겹치게 한다 → ML도 완벽 분리하지 못해 ROC/PR가 휜다.
    inj = InjectionConfig(
        randomize_intensity=True,
        camouflage=True,
        spoof_qty_range=(10, 90),
        layering_levels_range=(2, 8),
        wash_trades_range=(1, 7),
    )
    cfg = DatasetConfig(
        seed=42, duration=40_000,
        num_spoofing_episodes=20, num_wash_episodes=20, num_layering_episodes=20,
        injection=inj,
    )
    result = generate_dataset(cfg)
    features = build_features(result.events)
    truth = build_truth(result.events, result.labels)

    labeled = build_labeled_frame(features, truth)
    train, test = time_split(labeled, train_fraction=0.7)

    print("=" * 60)
    print("Phase 2 — 룰 기반 vs ML (시간 분할: 과거로 학습, 미래를 탐지)")
    print("=" * 60)
    print(
        f"학습 표본 {len(train)} (조작 {int(train['is_manip'].sum())})  |  "
        f"평가 표본 {len(test)} (조작 {int(test['is_manip'].sum())})"
    )
    print("→ 무작위가 아닌 시간 분할로 lookahead 누설을 차단한다.\n")

    # ── 룰 기반 (동일 test 구간) ──
    rule_pred = predict(test, RuleConfig())
    rm = binary_metrics(test["is_manip"], rule_pred["is_alert"])

    # ── ML (시간 분할 학습) ──
    ml = train_and_score(train, test, MLConfig())
    mm = binary_metrics(ml.test["is_manip"], ml.test["ml_pred"])
    summary = curve_summary(ml.test["is_manip"], ml.test["score"])

    print(f"{'방법':<10}{'정밀도':>10}{'재현율':>10}{'F1':>10}   (TP/FP/FN)")
    print("-" * 60)
    print(f"{'룰 기반':<10}{rm.precision:>10.3f}{rm.recall:>10.3f}{rm.f1:>10.3f}"
          f"   ({rm.tp}/{rm.fp}/{rm.fn})")
    print(f"{'ML':<10}{mm.precision:>10.3f}{mm.recall:>10.3f}{mm.f1:>10.3f}"
          f"   ({mm.tp}/{mm.fp}/{mm.fn})")
    print()
    print(f"ML ROC AUC = {summary['roc_auc']:.3f}   |   평균정밀도(PR AUC) = "
          f"{summary['average_precision']:.3f}")
    print("→ 핵심: 위장(camouflage) 아래에서 고정 임계값 룰은 재현율이 급락하지만,")
    print("  학습된 ML은 다중 피처 조합으로 약화된 신호까지 잡아낸다(룰의 취약성 = ML의 근거).")

    paths = save_curves(ml.test["is_manip"], ml.test["score"], outdir="artifacts")
    print("\n저장된 플롯:")
    for name, p in paths.items():
        print(f"  - {name}: {p}")

    print("\n피처 중요도(상위 5):")
    for feat, imp in feature_importances(ml.model).head(5).items():
        print(f"  {feat:<22} {imp:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
