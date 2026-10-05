"""Multi-hop 意思決定シナリオ合成モジュール。

条項依存関係 DAG の複数ホップ経路 (2〜3 条項以上の連鎖) を探索し、
単一の条項検索では回答できず、複数条項のクロスリファレンスを必要とする
複合実務シナリオおよび型付き設問 (Choice / Noul) を合成する。
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from data.schema import QuestionType
from data.synthetic.long_context.dag_extractor import (
    ClauseDAG,
    ClauseRelationType,
)

logger = logging.getLogger(__name__)


@dataclass
class MultiHopScenario:
    """Multi-hop 意思決定シナリオ。

    Attributes:
        scenario_id (str): シナリオ識別子。
        doc_id (str): 参照元ドキュメント ID。
        path_clause_ids (list[str]): 関与する条項 ID 列 (例: ['第3条', '第6条']).
        fact_situation (str): 事実関係・問い合わせ状況説明文。
        question_text (str): 判定指示文。
        question_type (QuestionType): プリミティブ種別 (Choice / Noul).
        criteria (dict[str, str]): 選択肢辞書 (Choice 時).
        target (str): 正解キーまたは 'true' / 'false'.
        evidence_spans (list[tuple[int, int]]): 根拠となった各条項の文字スパン。
        metadata (dict[str, Any]): 追加メタデータ。
    """

    scenario_id: str
    doc_id: str
    path_clause_ids: list[str]
    fact_situation: str
    question_text: str
    question_type: QuestionType
    criteria: dict[str, str]
    target: str
    evidence_spans: list[tuple[int, int]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiHopSynthesizer:
    """条項依存 DAG に基づく Multi-hop 意思決定シナリオ合成器。

    条項間の参照・例外・手続連鎖を辿り、複数条項の突き合わせを必須とするシナリオを生成する。
    """

    def __init__(self, seed: int = 42) -> None:
        """合成器を初期化する。

        Args:
            seed (int): 乱数シード。
        """
        self.seed = seed

    def synthesize_scenarios(
        self,
        dag: ClauseDAG,
        max_scenarios: int = 5,
    ) -> list[MultiHopScenario]:
        """DAG から Multi-hop 意思決定シナリオを合成する。

        Args:
            dag (ClauseDAG): 対象ドキュメントの条項 DAG。
            max_scenarios (int): 最大生成シナリオ数。

        Returns:
            list[MultiHopScenario]: 合成されたシナリオリスト。
        """
        scenarios: list[MultiHopScenario] = []

        # 1. 2条項以上のエッジ・パスを抽出
        if dag.edges:
            for edge_idx, edge in enumerate(dag.edges):
                if len(scenarios) >= max_scenarios:
                    break

                src_node = dag.nodes.get(edge.source_id)
                tgt_node = dag.nodes.get(edge.target_id)
                if src_node is None or tgt_node is None:
                    continue

                spans = [src_node.char_span, tgt_node.char_span]
                scenario_id = (
                    f"synth_multihop_{dag.doc_id}_{src_node.clause_id}_"
                    f"{tgt_node.clause_id}_{edge_idx}"
                )

                if edge.relation_type == ClauseRelationType.EXCEPTION:
                    # 例外規定が絡むシナリオ (原則と例外の抵触)
                    fact = (
                        f"【状況】当事者間において、{tgt_node.clause_id}（{tgt_node.title}）に基づく基本義務の履行が問題となった。"
                        f"しかし、事案において{src_node.clause_id}（{src_node.title}）に定める特別事由が発生している。"
                    )
                    question = (
                        f"{tgt_node.clause_id}および{src_node.clause_id}の相互関係に基づき、"
                        f"本件において相手方の義務免責または例外要件が適用されるか判断せよ。"
                    )
                    criteria = {
                        "opt_exception_applies": f"{src_node.clause_id}の例外規定が適用され、相手方は責任を免れる",
                        "opt_principle_applies": f"{tgt_node.clause_id}の原則が優先され、免責は認められない",
                        "opt_both_breach": "双方が義務違反となり過失相殺される",
                        "none": "本規程に該当する条項なし",
                    }
                    target = "opt_exception_applies"

                    scenarios.append(
                        MultiHopScenario(
                            scenario_id=scenario_id,
                            doc_id=dag.doc_id,
                            path_clause_ids=[
                                tgt_node.clause_id,
                                src_node.clause_id,
                            ],
                            fact_situation=fact,
                            question_text=f"{fact}\n【指示】{question}",
                            question_type=QuestionType.CHOICE,
                            criteria=criteria,
                            target=target,
                            evidence_spans=spans,
                            metadata={
                                "relation_type": edge.relation_type.value,
                                "hops": 2,
                            },
                        )
                    )

                elif edge.relation_type == ClauseRelationType.PROCEDURE:
                    # 手続要件が絡むシナリオ (Noul 型)
                    fact = (
                        f"【状況】乙は{tgt_node.clause_id}の権利を行使しようとしているが、"
                        f"{src_node.clause_id}で義務付けられている事前の書面通知を行わず口頭で申告した。"
                    )
                    question = (
                        f"{tgt_node.clause_id}および{src_node.clause_id}の手続要件に照らし、"
                        f"乙による本権利行使は規程上適法かつ有効に成立するか。"
                    )
                    target = "false"

                    scenarios.append(
                        MultiHopScenario(
                            scenario_id=scenario_id,
                            doc_id=dag.doc_id,
                            path_clause_ids=[
                                tgt_node.clause_id,
                                src_node.clause_id,
                            ],
                            fact_situation=fact,
                            question_text=f"{fact}\n【指示】{question}",
                            question_type=QuestionType.NOUL,
                            criteria={},
                            target=target,
                            evidence_spans=spans,
                            metadata={
                                "relation_type": edge.relation_type.value,
                                "hops": 2,
                            },
                        )
                    )

        # 2. エッジが少ない場合のフォールバック (隣接2条項の突き合わせ)
        if not scenarios and len(dag.nodes) >= 2:
            node_ids = list(dag.nodes.keys())
            n1 = dag.nodes[node_ids[0]]
            n2 = dag.nodes[node_ids[1]]
            scenarios.append(
                MultiHopScenario(
                    scenario_id=f"synth_multihop_{dag.doc_id}_fallback",
                    doc_id=dag.doc_id,
                    path_clause_ids=[n1.clause_id, n2.clause_id],
                    fact_situation=f"【状況】{n1.clause_id}の対象業務に関して、{n2.clause_id}の順守状況を検証する事案が生じた。",
                    question_text=f"【指示】{n1.clause_id}および{n2.clause_id}を総合考慮し、手続が完了しているとみなせるか判定せよ。",
                    question_type=QuestionType.NOUL,
                    criteria={},
                    target="true",
                    evidence_spans=[n1.char_span, n2.char_span],
                    metadata={"relation_type": "fallback", "hops": 2},
                )
            )

        logger.info(
            "ドキュメント '%s' から Multi-hop シナリオ %d 件を合成しました。",
            dag.doc_id,
            len(scenarios),
        )
        return scenarios
