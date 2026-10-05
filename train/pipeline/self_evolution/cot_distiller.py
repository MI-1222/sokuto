"""CoT 前提条件の中間 DAG 射影蒸留モジュール。

上位 LLM (System 2) の長大な思考連鎖 (CoT: Chain-of-Thought) から、
決定に至る前提条件 (Premise)、例外規定の該否、および中間判定命題 (Sub-claims) を抽出し、
インプロセス DAG の各ノードで評価可能な原子的マイクロ決定タスク (Noul / Choice / Score)
の UnifiedSample へ射影・分解する。
非自己回帰型モデル (completion_tokens = 0) との構造的ギャップを解消し、無理のない模倣学習を実現する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from data.schema import QuestionType, UnifiedSample
from pipeline.self_evolution.verifier import SilverSample


@dataclass
class MicroDecision:
    """CoT から抽出された単一の原子的マイクロ決定ステップ。

    Attributes:
        node_id (str): DAG ノード識別子 (例: 'premise_check_1', 'state_eval_2')。
        question_type (QuestionType): 決定プリミティブ種別 (NOUL, CHOICE, SCORE)。
        instruction (str): マイクロ決定タスク指示文。
        criteria (dict[str, str]): 選択肢または段階の辞書。
        target (str): 正解ラベル ('true'/'false', 選択肢キー, または段階数値)。
        rationale (str): CoT 内での当該ステップの推論根拠。
    """

    node_id: str
    question_type: QuestionType
    instruction: str
    criteria: dict[str, str]
    target: str
    rationale: str = ""


@dataclass
class DistilledDAGProjection:
    """単一の難例から分解・射影されたマイクロ決定 DAG 構造。

    Attributes:
        parent_sample_id (str): 元の難例サンプル識別子。
        micro_decisions (list[MicroDecision]): 分解されたマイクロ決定ステップ列。
        final_decision: 最終決定のマイクロ決定。
    """

    parent_sample_id: str
    micro_decisions: list[MicroDecision] = field(default_factory=list)
    final_decision: MicroDecision | None = None


class CoTDistiller:
    """思考連鎖 (CoT) から前提条件を抽出し DAG マイクロ決定へ射影する蒸留器。"""

    def __init__(self) -> None:
        """CoT 蒸留器を初期化する。"""
        # 構造化タグ抽出用パターン (<premises> ... </premises>)
        self._premises_tag_pattern = re.compile(
            r"<premises>(.*?)</premises>", re.DOTALL | re.IGNORECASE
        )
        self._tagged_item_pattern = re.compile(
            r"-\s*\[(TRUE|FALSE)\]\s*([^\n]+)", re.IGNORECASE
        )

        # 自由記述思考ログ抽出用のフォールバックパターン
        self._premise_pattern = re.compile(
            r"(?:前提|条件|要件|ステップ|規定)(?:[0-9１-９一二三四五]|[:：])\s*[:：]?\s*([^。\n]+[。]?)"
        )
        self._bool_pattern = re.compile(
            r"(?:適合|該当|満たす|クリア|あり|真|適用される|True|true)"
        )
        self._neg_pattern = re.compile(
            r"(?:不適合|非該当|満たさない|除外|なし|偽|適用されない|False|false)"
        )

    def extract_premises_from_text(self, thinking_text: str) -> list[tuple[str, str]]:
        """思考ログから前提・条件文の候補と真偽ラベルを抽出する。

        <premises> タグの構造化出力を最優先で解析し、
        存在しない場合は自由記述テキストから正規表現でフォールバック抽出する。

        Args:
            thinking_text (str): フロンティア LLM の推論思考テキスト。

        Returns:
            list[tuple[str, str]]: 抽出された (前提命題文, 真偽ラベル 'true'/'false') のリスト。
        """
        # 1. 構造化タグ <premises> の優先探索
        tag_match = self._premises_tag_pattern.search(thinking_text)
        if tag_match:
            tag_content = tag_match.group(1)
            tagged_items = self._tagged_item_pattern.findall(tag_content)
            if tagged_items:
                extracted: list[tuple[str, str]] = []
                for flag, text in tagged_items:
                    is_true = flag.upper() == "TRUE"
                    target = "true" if is_true else "false"
                    extracted.append((text.strip(), target))
                return extracted

        # 2. 自由記述思考ログからのフォールバック抽出
        premises_fallback: list[tuple[str, str]] = []
        for line in thinking_text.splitlines():
            line_str = line.strip()
            # 箇条書きや番号付き行の検出
            if re.match(r"^(?:[-*・]|\d+[.)])\s*", line_str):
                cleaned = re.sub(r"^(?:[-*・]|\d+[.)])\s*", "", line_str)
                if len(cleaned) >= 5:
                    is_true = bool(self._bool_pattern.search(cleaned)) and not bool(
                        self._neg_pattern.search(cleaned)
                    )
                    premises_fallback.append((cleaned, "true" if is_true else "false"))
            else:
                matches = self._premise_pattern.findall(line_str)
                for m in matches:
                    is_true = bool(self._bool_pattern.search(m)) and not bool(
                        self._neg_pattern.search(m)
                    )
                    premises_fallback.append((m, "true" if is_true else "false"))

        return premises_fallback

    def project_to_dag(self, sample: SilverSample) -> DistilledDAGProjection:
        """Silver Data の CoT ログを解析し、DAG マイクロ決定構造へ射影する。

        Args:
            sample (SilverSample): 検証通過済みの難例サンプル。

        Returns:
            DistilledDAGProjection: 射影されたマイクロ決定 DAG。
        """
        projection = DistilledDAGProjection(parent_sample_id=sample.sample_id)

        # 思考ログから前提命題を抽出 (第 1 モデルと第 2 モデルの両方を走査)
        combined_thinking = f"{sample.primary_thinking}\n{sample.secondary_thinking}"
        extracted_pairs = self.extract_premises_from_text(combined_thinking)

        # 前提条件命題を Noul プリミティブ(真偽決定ノード)へマッピング
        for i, (clause, target) in enumerate(
            extracted_pairs[:3]
        ):  # 最大 3 個のマイクロ前提ノード
            node = MicroDecision(
                node_id=f"{sample.sample_id}_premise_{i + 1}",
                question_type=QuestionType.NOUL,
                instruction=f"以下の条件が満たされているか真偽で判定してください: {clause}",
                criteria={},  # Noul 型契約に基づき空辞書を設定 (formatter 側で暗黙展開)
                target=target,
                rationale=clause,
            )
            projection.micro_decisions.append(node)

        # 最終決定ノードの構築
        final_qtype = (
            QuestionType(sample.question_type)
            if sample.question_type in [t.value for t in QuestionType]
            else QuestionType.CHOICE
        )
        final_node = MicroDecision(
            node_id=f"{sample.sample_id}_final",
            question_type=final_qtype,
            instruction=sample.instructions,
            criteria=sample.criteria,
            target=sample.consensus_target,
            rationale="上位 LLM 思考連鎖の合意最終決定。",
        )
        projection.final_decision = final_node
        return projection

    def to_unified_samples(
        self, projection: DistilledDAGProjection, state: str
    ) -> list[UnifiedSample]:
        """DAG 射影結果を学習用 UnifiedSample のリストへ変換する。

        マイクロ決定ごとに 1 つの UnifiedSample を生成し、
        非自己回帰型モデルのマルチタスク訓練データとして出力する。

        Args:
            projection (DistilledDAGProjection): マイクロ決定 DAG。
            state (str): 入力文脈テキスト (State)。

        Returns:
            list[UnifiedSample]: 生成された UnifiedSample のリスト。
        """
        samples: list[UnifiedSample] = []

        # 中間マイクロ決定ノード
        for step in projection.micro_decisions:
            sample = UnifiedSample(
                dataset_name="cot_dag_subclaim",
                sample_id=step.node_id,
                question_type=step.question_type,
                state=state,
                instructions=step.instruction,
                criteria=step.criteria,
                target=step.target,
                metadata={
                    "parent_sample_id": projection.parent_sample_id,
                    "is_subclaim": True,
                    "rationale": step.rationale,
                },
            )
            samples.append(sample)

        # 最終決定ノード
        if projection.final_decision:
            f_node = projection.final_decision
            final_sample = UnifiedSample(
                dataset_name="cot_dag_final",
                sample_id=f_node.node_id,
                question_type=f_node.question_type,
                state=state,
                instructions=f_node.instruction,
                criteria=f_node.criteria,
                target=f_node.target,
                metadata={
                    "parent_sample_id": projection.parent_sample_id,
                    "is_subclaim": False,
                },
            )
            samples.append(final_sample)

        return samples
