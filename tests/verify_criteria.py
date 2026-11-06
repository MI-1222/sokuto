"""Exit Criteria 総合自動検証テストスクリプト。

Dual-Sequence 長文 `State` 分離 `Cross-Attention` および中間特徴量キャッシュの、
定量的 Exit Criteria を機械的に一括合否判定するエントリポイントスクリプト。
"""

import sys
from pathlib import Path

# ワークスペースルートおよび train ディレクトリをモジュールパスへ登録する。
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "train"))

from eval.verify_phase8_criteria import main

if __name__ == "__main__":
    main()
