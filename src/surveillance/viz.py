"""시각화 공통 설정.

matplotlib 한글 폰트를 한 곳에서 설정한다. 로컬(Windows)은 맑은 고딕, 배포
(Linux/Streamlit Cloud)는 나눔고딕(packages.txt의 fonts-nanum)을 쓴다. matplotlib는
설치된 첫 폰트를 선택하므로 양쪽을 모두 나열하고, 둘 다 없으면 DejaVu Sans로 떨어진다.

대시보드(app.py)와 분석 스크립트가 동일한 설정을 공유하도록 이 함수를 호출한다.
"""

from __future__ import annotations

import matplotlib
from matplotlib import font_manager

# 선호 순서: Windows(맑은 고딕) → Linux 배포(나눔고딕) → macOS(애플고딕) → 최종 폴백.
_PREFERRED = ["Malgun Gothic", "NanumGothic", "AppleGothic", "DejaVu Sans"]


def setup_korean_font() -> None:
    """차트 한글 깨짐 방지용 폰트를 전역 설정한다.

    실제 *설치된* 폰트만 골라 지정한다. 없는 폰트를 나열하면 matplotlib이 매 렌더마다
    'findfont ... not found' 경고를 뿜으므로, 플랫폼별로 존재하는 것만 남긴다.
    """
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [name for name in _PREFERRED if name in available] or ["DejaVu Sans"]
    matplotlib.rcParams["font.family"] = chosen
    matplotlib.rcParams["axes.unicode_minus"] = False
