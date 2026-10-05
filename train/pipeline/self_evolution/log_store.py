"""エスカレーション難例ログストアモジュール。

System 1 (sokuto) から System 2 (上位 LLM) へのエスカレーション難例ログを収集・蓄積し、
Pareto Gating 不確実性シグネチャに基づくフィルタリング、Parquet / JSONL への永続化、
および匿名化パイプラインへの受け渡しを担う。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pipeline.self_evolution.anonymizer import EntityConsistentAnonymizer


@dataclass
class HardSampleRecord:
    """エスカレーション難例ログレコード。

    Attributes:
        sample_id (str): サンプル固有識別子。
        question_type (str): 決定プリミティブ種別 ('choice', 'score', 'noul')。
        state (str): 入力非構造化文脈。
        instructions (str): 指示文。
        criteria (dict[str, str]): 選択肢辞書。
        escalation_reason (str): エスカレーション発生理由。
        system1_confidence (float): sokuto 側の実効確信度。
        top_margin (float | None): 上位2候補の確率差。
        normalized_entropy (float | None): 正規化シャノンエントロピー。
        free_energy (float | None): ヘルムホルツ自由エネルギー。
        is_ood (bool): OOD (分布外) 検知フラグ。
        system2_decision (str | None): 上位 LLM が下した決定ラベル。
        system2_thinking (str | None): 上位 LLM の思考ログ (CoT)。
        metadata (dict[str, Any]): その他付随情報。
    """

    sample_id: str
    question_type: str
    state: str
    instructions: str
    criteria: dict[str, str]
    escalation_reason: str
    system1_confidence: float
    top_margin: float | None = None
    normalized_entropy: float | None = None
    free_energy: float | None = None
    is_ood: bool = False
    system2_decision: str | None = None
    system2_thinking: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """辞書オブジェクトへ変換する。

        Returns:
            dict[str, Any]: シリアライズ可能な辞書。
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HardSampleRecord:
        """辞書からレコードを復元する。

        Args:
            data (dict[str, Any]): レコード辞書。

        Returns:
            HardSampleRecord: 復元されたレコードインスタンス。
        """
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


class HardSampleStore:
    """難例ログストア管理クラス。

    ログのファイル永続化、インメモリバッファリング、およびフィルタリングを提供する。
    """

    def __init__(
        self,
        store_path: str | Path,
        anonymizer: EntityConsistentAnonymizer | None = None,
    ) -> None:
        """ログストアを初期化する。

        Args:
            store_path (str | Path): ログファイルまたはディレクトリのパス。
            anonymizer (EntityConsistentAnonymizer | None): 自動仮名化エンジン。
        """
        self.store_path = Path(store_path)
        self.anonymizer = anonymizer or EntityConsistentAnonymizer()
        self._records: list[HardSampleRecord] = []

    def add_record(self, record: HardSampleRecord, auto_anonymize: bool = True) -> None:
        """新規難例レコードを追加する。

        Args:
            record (HardSampleRecord): 追加する難例レコード。
            auto_anonymize (bool): 追加時に自動的に仮名化を適用するか。
        """
        if auto_anonymize:
            rec_dict = record.to_dict()
            anon_dict = self.anonymizer.anonymize_dict(
                rec_dict, custom_salt=record.sample_id
            )
            processed_record = HardSampleRecord.from_dict(anon_dict)
        else:
            processed_record = record

        self._records.append(processed_record)

    def filter_by_uncertainty(
        self,
        max_margin: float = 0.15,
        min_entropy: float = 0.65,
        min_energy: float = -1.0,
        include_ood: bool = True,
    ) -> list[HardSampleRecord]:
        """Pareto ゲーティング不確実性条件に合致する難例を抽出する。

        Args:
            max_margin (float): Top-Margin 上限値。
            min_entropy (float): 正規化エントロピー下限値。
            min_energy (float): 自由エネルギー下限値。
            include_ood (bool): OOD 検知サンプルを無条件に含めるか。

        Returns:
            list[HardSampleRecord]: 抽出された難例レコードのリスト。
        """
        extracted: list[HardSampleRecord] = []
        for rec in self._records:
            is_candidate = False

            if (
                include_ood
                and rec.is_ood
                or rec.top_margin is not None
                and rec.top_margin <= max_margin
                or (
                    rec.normalized_entropy is not None
                    and rec.normalized_entropy >= min_entropy
                )
                or rec.free_energy is not None
                and rec.free_energy >= min_energy
            ):
                is_candidate = True

            if is_candidate:
                extracted.append(rec)

        return extracted

    def save_jsonl(self, file_path: str | Path | None = None) -> Path:
        """保持している全難例レコードを JSONL ファイルへ保存する。

        Args:
            file_path (str | Path | None): 出力先パス。None の場合はストアパスを使用。

        Returns:
            Path: 保存されたファイルの絶対パス。
        """
        target = Path(file_path) if file_path else self.store_path
        target.parent.mkdir(parents=True, exist_ok=True)

        with target.open("w", encoding="utf-8") as f:
            for rec in self._records:
                line = json.dumps(rec.to_dict(), ensure_ascii=False)
                f.write(line + "\n")

        return target.resolve()

    def load_jsonl(self, file_path: str | Path | None = None) -> list[HardSampleRecord]:
        """JSONL ファイルから難例レコードを読み込む。

        Args:
            file_path (str | Path | None): 読み込み元パス。None の場合はストアパスを使用。

        Returns:
            list[HardSampleRecord]: 読み込まれたレコードのリスト。
        """
        source = Path(file_path) if file_path else self.store_path
        if not source.exists():
            return []

        loaded: list[HardSampleRecord] = []
        with source.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    data = json.loads(stripped)
                    loaded.append(HardSampleRecord.from_dict(data))

        self._records = loaded
        return loaded

    @property
    def records(self) -> list[HardSampleRecord]:
        """蓄積された難例レコードのリストを取得する。"""
        return list(self._records)

    def __len__(self) -> int:
        """蓄積された難例レコード数を返す。"""
        return len(self._records)
