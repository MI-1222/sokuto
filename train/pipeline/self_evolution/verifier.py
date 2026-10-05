"""Dual-LLM ブラインドクロスチェック検証モジュール。

Claude 3.7 Sonnet と GPT-4o 等の 2 系統の独立したフロンティア LLM を用い、
System 1 の事前判定を完全隠蔽したブラインド並列推論を実行する。
両モデルの離散決定ラベルが完全一致したサンプルのみを Silver Data (高品質擬似正解) へ昇格させ、
不一致サンプルは人間レビューキューへ隔離することでモデル崩壊とバイアス転写を防止する。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol

from pipeline.self_evolution.config import VerifierConfig
from pipeline.self_evolution.log_store import HardSampleRecord


@dataclass
class LLMVerificationResponse:
    """単一フロンティア LLM からの推論レスポンス。

    Attributes:
        model_name (str): モデル識別子。
        decision_label (str): 判定された離散決定ラベル。
        thinking_process (str): 生成された思考連鎖 (CoT)。
        probabilities (dict[str, float] | None): 選択肢ごとの確率分布 (利用可能な場合)。
        raw_response (dict[str, Any] | None): 生の API レスポンス辞書。
    """

    model_name: str
    decision_label: str
    thinking_process: str
    probabilities: dict[str, float] | None = None
    raw_response: dict[str, Any] | None = None


@dataclass
class SilverSample:
    """Dual-LLM による検証を通過した Silver Data (準正解データ)。

    Attributes:
        sample_id (str): サンプル識別子。
        question_type (str): 決定プリミティブ種別 ('choice', 'score', 'noul')。
        state (str): 匿名化済み文脈テキスト。
        instructions (str): 指示文。
        criteria (dict[str, str]): 選択肢辞書。
        consensus_target (str): 両モデルが完全一致した合意ラベル。
        soft_labels (dict[str, float]): アンサンブルソフト確率分布。
        primary_thinking (str): 第 1 モデルの思考ログ (CoT)。
        secondary_thinking (str): 第 2 モデルの思考ログ (CoT)。
        metadata (dict[str, Any]): 検証時のメタデータ。
    """

    sample_id: str
    question_type: str
    state: str
    instructions: str
    criteria: dict[str, str]
    consensus_target: str
    soft_labels: dict[str, float]
    primary_thinking: str
    secondary_thinking: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IsolatedSample:
    """判定不一致により隔離された人間レビュー対象サンプル。

    Attributes:
        sample_id (str): サンプル識別子。
        record: 元の難例レコード。
        primary_response: 第 1 モデルのレスポンス。
        secondary_response: 第 2 モデルのレスポンス。
        isolation_reason (str): 隔離理由。
    """

    sample_id: str
    record: HardSampleRecord
    primary_response: LLMVerificationResponse
    secondary_response: LLMVerificationResponse
    isolation_reason: str


class LLMClientProtocol(Protocol):
    """フロンティア LLM 推論クライアントのインターフェースプロトコル。"""

    async def complete(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1500,
    ) -> LLMVerificationResponse:
        """非同期で LLM 推論を実行する。

        Args:
            model_name (str): モデル名。
            system_prompt (str): システムプロンプト。
            user_prompt (str): ユーザープロンプト。
            temperature (float): 温度パラメータ。
            max_tokens (int): 最大生成トークン数。

        Returns:
            LLMVerificationResponse: レスポンスオブジェクト。
        """
        ...


class MockLLMClient:
    """テストおよびオフライン実験用の決定論的モック LLM クライアント。"""

    def __init__(
        self,
        default_label: str = "0",
        mismatch_sample_ids: set[str] | None = None,
    ) -> None:
        """モッククライアントを初期化する。

        Args:
            default_label (str): デフォルトで返却する決定ラベル。
            mismatch_sample_ids (set[str] | None): モデル間でラベルを不一致にするサンプルIDの集合。
        """
        self.default_label = default_label
        self.mismatch_sample_ids = mismatch_sample_ids or set()

    async def complete(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1500,
    ) -> LLMVerificationResponse:
        """モック推論結果を返却する。"""
        # プロンプト内の sample_id の簡易検出
        is_mismatch = False
        for s_id in self.mismatch_sample_ids:
            if s_id in user_prompt:
                is_mismatch = True
                break

        if is_mismatch and "gpt" in model_name.lower():
            # GPT 側で意図的に異なるラベルを返す
            label = "MISMATCH_LABEL"
        else:
            label = self.default_label

        return LLMVerificationResponse(
            model_name=model_name,
            decision_label=label,
            thinking_process=(
                f"[{model_name}] 思考ログ:\n"
                "<premises>\n"
                f"- [TRUE] 文脈と指示に基づく前提条件の充足\n"
                "</premises>\n"
                f"論理的帰結として {label} を選択。"
            ),
            probabilities={label: 0.92},
        )


class DualLLMVerifier:
    """Claude 3.7 Sonnet と GPT-4o のブラインド並列検証エンジン。"""

    def __init__(
        self,
        config: VerifierConfig | None = None,
        client: LLMClientProtocol | None = None,
    ) -> None:
        """検証エンジンを初期化する。

        Args:
            config (VerifierConfig | None): 検証設定。
            client (LLMClientProtocol | None): LLM 呼び出しクライアント。未指定時はモックを使用。
        """
        self.config = config or VerifierConfig()
        self.client: LLMClientProtocol = client or MockLLMClient()

    def build_blind_prompt(self, record: HardSampleRecord) -> tuple[str, str]:
        """アンカリング思考汚染を防止する完全ブラインドプロンプトを構築する。

        System 1 の推論結果、不確実性スコア、エスカレーション理由は一切含めず、
        純粋な入力文脈と指示のみを提示する。

        Args:
            record (HardSampleRecord): 難例レコード。

        Returns:
            tuple[str, str]: (システムプロンプト, ユーザープロンプト)。
        """
        system_prompt = (
            "あなたは厳密かつ批判的な意思決定を行う最高位の論理推論エンジンです。\n"
            "与えられた文脈(State)と業務ルール(Instructions)のみに立脚して推論してください。\n"
            "推論過程(Chain-of-Thought)を論理的かつ簡潔に展開し、最後に最終決定ラベルを出力してください。"
        )

        criteria_lines = [f"- 候補 '{k}': {v}" for k, v in record.criteria.items()]
        user_prompt = (
            f"【サンプルID】\n{record.sample_id}\n\n"
            f"【入力文脈 (State)】\n{record.state}\n\n"
            f"【判定指示 (Instructions)】\n{record.instructions}\n\n"
            f"【判定選択肢 (Criteria)】\n" + "\n".join(criteria_lines) + "\n\n"
            "【出力指示】\n"
            "決定に至る論理的思考ログを展開してください。その際、判断の基礎となる前提命題・条件を以下のタグ形式で列挙してください：\n"
            "<premises>\n"
            "- [TRUE] 満たされている前提条件や事実\n"
            "- [FALSE] 満たされていない前提条件や例外\n"
            "</premises>\n\n"
            "そして、必ず最終行に\n"
            "DECISION: <選択肢キー>\n"
            "の形式で該当する唯一のキーを出力してください。"
        )
        return system_prompt, user_prompt

    def _extract_label(self, raw_label_or_text: str, valid_keys: list[str]) -> str:
        """レスポンスから決定ラベルを正規化抽出する。"""
        # "DECISION: foo" 形式の探索
        for line in reversed(raw_label_or_text.splitlines()):
            line_str = line.strip()
            if line_str.startswith("DECISION:"):
                cand = line_str.replace("DECISION:", "").strip()
                if cand in valid_keys:
                    return cand

        # そのまま一致するか確認
        stripped = raw_label_or_text.strip()
        if stripped in valid_keys:
            return stripped

        # 部分一致のフォールバック
        for k in valid_keys:
            if k in stripped:
                return k

        return stripped

    async def verify_sample(
        self, record: HardSampleRecord
    ) -> SilverSample | IsolatedSample:
        """単一の難例レコードに対して Dual-LLM ブラインドクロスチェックを実行する。

        Args:
            record (HardSampleRecord): 難例レコード。

        Returns:
            SilverSample | IsolatedSample: 一致時は SilverSample、不一致時は IsolatedSample。
        """
        sys_p, user_p = self.build_blind_prompt(record)
        valid_keys = list(record.criteria.keys())

        # Claude と GPT への並列推論リクエスト
        res_primary, res_secondary = await asyncio.gather(
            self.client.complete(
                model_name=self.config.primary_model,
                system_prompt=sys_p,
                user_prompt=user_p,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ),
            self.client.complete(
                model_name=self.config.secondary_model,
                system_prompt=sys_p,
                user_prompt=user_p,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ),
        )

        label_p = self._extract_label(res_primary.decision_label, valid_keys)
        label_s = self._extract_label(res_secondary.decision_label, valid_keys)

        # 完全一致 (Strict Match) の判定
        if label_p == label_s and label_p in valid_keys:
            # アンサンブルソフト確率分布の合成
            num_cands = max(len(valid_keys), 1)
            soft_dist: dict[str, float] = {}

            # 正解ラベルに高い確率を割り当て、残りを平滑化
            main_prob = 0.90
            rem_prob = (1.0 - main_prob) / max(num_cands - 1, 1)

            for k in valid_keys:
                soft_dist[k] = main_prob if k == label_p else rem_prob

            return SilverSample(
                sample_id=record.sample_id,
                question_type=record.question_type,
                state=record.state,
                instructions=record.instructions,
                criteria=record.criteria,
                consensus_target=label_p,
                soft_labels=soft_dist,
                primary_thinking=res_primary.thinking_process,
                secondary_thinking=res_secondary.thinking_process,
                metadata={
                    "primary_model": self.config.primary_model,
                    "secondary_model": self.config.secondary_model,
                    "strict_match": True,
                },
            )

        # 不一致サンプルの隔離
        reason = (
            f"判定ラベル不一致: {self.config.primary_model}='{label_p}' "
            f"vs {self.config.secondary_model}='{label_s}'"
        )
        return IsolatedSample(
            sample_id=record.sample_id,
            record=record,
            primary_response=res_primary,
            secondary_response=res_secondary,
            isolation_reason=reason,
        )

    async def verify_batch(
        self, records: list[HardSampleRecord]
    ) -> tuple[list[SilverSample], list[IsolatedSample]]:
        """複数レコードの一括ブラインド検証を実行する。

        Args:
            records (list[HardSampleRecord]): 難例レコード列。

        Returns:
            tuple[list[SilverSample], list[IsolatedSample]]: (認定 Silver サンプル列, 隔離サンプル列)。
        """
        tasks = [self.verify_sample(rec) for rec in records]
        results = await asyncio.gather(*tasks)

        silver_samples: list[SilverSample] = []
        isolated_samples: list[IsolatedSample] = []

        for res in results:
            if isinstance(res, SilverSample):
                silver_samples.append(res)
            else:
                isolated_samples.append(res)

        return silver_samples, isolated_samples
