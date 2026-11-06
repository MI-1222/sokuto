"""最小反事実ペア (Contrast Sets) 生成モジュール。

事実関係・条件設定の微小な 1 属性(日数・金額・通知手段・役職・除外要件など)のみを変更し、
判定結果 (Choice 選択肢または Noul の true/false) が完全に逆転する対照ペアを生成する。
表層キーワード一致によるショートカット学習を抑制し、精密な論理判定能力を担保する。
"""

import logging
from dataclasses import dataclass
from typing import Any

from data.schema import QuestionType
from data.synthetic.long_context.multihop_synthesizer import (
    MultiHopScenario,
)

logger = logging.getLogger(__name__)


@dataclass
class ContrastPair:
    """最小反事実対照ペア。

    Attributes:
        pair_id (str): ペア識別子。
        original_scenario (MultiHopScenario): 元シナリオ (正例/基準)。
        perturbed_scenario (MultiHopScenario): 1 属性反転後の対照シナリオ (ラベル反転)。
        inverted_attribute (str): 反転させた属性名 (例: '通知手段: 書面 -> 口頭').
    """

    pair_id: str
    original_scenario: MultiHopScenario
    perturbed_scenario: MultiHopScenario
    inverted_attribute: str


class ContrastSetGenerator:
    """最小反事実ペア (Contrast Sets) 生成器。

    Multi-hop シナリオに対し、契約・規程の境界値や前提要件を 1 点のみ操作した
    対照シナリオを自動生成する。
    """

    def __init__(self, seed: int = 42) -> None:
        """生成器を初期化する。

        Args:
            seed (int): 乱数シード。
        """
        self.seed = seed

    def generate_contrast_pair(
        self,
        base_scenario: MultiHopScenario,
    ) -> ContrastPair:
        """単一のシナリオから、判定ラベルが反転する最小反事実シナリオペアを生成する。

        Args:
            base_scenario (MultiHopScenario): 基準シナリオ。

        Returns:
            ContrastPair: 基準シナリオと反転対照シナリオのペア。
        """
        pair_id = f"contrast_{base_scenario.scenario_id}"
        fact = base_scenario.fact_situation
        q_text = base_scenario.question_text

        # 1. Noul 型 (true <-> false 反転)
        if base_scenario.question_type == QuestionType.NOUL:
            orig_target = base_scenario.target.lower()
            inverted_target = "false" if orig_target == "true" else "true"

            # 属性反転パターンの適用 (書面 <-> 口頭、期限内 <-> 期限超過 など)
            if "口頭で申告した" in fact:
                new_fact = fact.replace("口頭で申告した", "事前に正式な書面で通知した")
                inv_attr = "通知手段: 口頭 -> 書面"
                inverted_target = "true"
            elif "口頭" in fact:
                new_fact = fact.replace("口頭", "書面")
                inv_attr = "通知手段: 口頭 -> 書面"
                inverted_target = "true"
            elif "書面で通知" in fact:
                new_fact = fact.replace("書面で通知", "電話口頭で連絡")
                inv_attr = "通知手段: 書面 -> 電話口頭"
                inverted_target = "false"
            elif "書面" in fact:
                new_fact = fact.replace("書面", "口頭")
                inv_attr = "通知手段: 書面 -> 口頭"
                inverted_target = "false"
            elif "期限内" in fact:
                new_fact = fact.replace("期限内", "期限を1日超過して")
                inv_attr = "期限要件: 期限内 -> 期限超過"
                inverted_target = "false"
            else:
                new_fact = (
                    fact + " （ただし、事後において所定の追認手続が完了している。）"
                )
                inv_attr = "手続追認の有無"

            # 置換空振りガード (事実が変わらずラベルのみ反転する矛盾データの生成を防止)
            if new_fact == fact:
                logger.warning(
                    "ContrastSetGenerator: 文字列置換が空振りしました (fact=%s)。フォールバック反転を適用します。",
                    fact[:60],
                )
                if orig_target == "true":
                    new_fact = (
                        fact
                        + " （なお、調査の結果、必須要件に重大な不備があることが事後判明した。）"
                    )
                    inverted_target = "false"
                else:
                    new_fact = (
                        fact
                        + " （なお、例外的に管轄機関による特別認可が事前に取得されていた。）"
                    )
                    inverted_target = "true"
                inv_attr = "特別要件の事後追加"

            if new_fact == q_text or fact not in q_text:
                new_q_text = f"{new_fact}\n【指示】{base_scenario.question_text.split('【指示】')[-1]}"
            else:
                new_q_text = q_text.replace(fact, new_fact)

            assert new_fact != fact, "事実文が更新されていません。"

            new_metadata: dict[str, Any] = dict(base_scenario.metadata)
            new_metadata["contrast_parent_id"] = base_scenario.scenario_id
            new_metadata["inverted_attribute"] = inv_attr

            perturbed = MultiHopScenario(
                scenario_id=f"{base_scenario.scenario_id}_perturbed",
                doc_id=base_scenario.doc_id,
                path_clause_ids=base_scenario.path_clause_ids,
                fact_situation=new_fact,
                question_text=new_q_text,
                question_type=QuestionType.NOUL,
                criteria=base_scenario.criteria,
                target=inverted_target,
                evidence_spans=base_scenario.evidence_spans,
                metadata=new_metadata,
            )

        # 2. Choice 型 (選択肢キーの反転)
        else:
            orig_target = base_scenario.target
            # 例外適用 -> 原則適用の反転
            if orig_target == "opt_exception_applies":
                if "特別事由が発生している" in fact:
                    new_fact = fact.replace(
                        "特別事由が発生している",
                        "特別事由の要件（天災等の不可抗力）には該当しないことが判明した",
                    )
                else:
                    new_fact = (
                        fact + " （ただし、特別事由の要件を満たさないことが確定した。）"
                    )
                inv_attr = "例外要件該当性: 該当 -> 非該当"
                inverted_target = "opt_principle_applies"
            elif orig_target == "opt_principle_applies":
                if "原則が優先され" in fact:
                    new_fact = fact.replace(
                        "原則が優先され",
                        "事前に双方合意による特約覚書が取り交わされていた",
                    )
                else:
                    new_fact = (
                        fact + " （事前に双方合意による特約覚書が取り交わされていた。）"
                    )
                inv_attr = "特約合意の有無: なし -> あり"
                inverted_target = "opt_exception_applies"
            else:
                new_fact = (
                    fact + " （なお、契約上の重大な違反が相手方にも認められる。）"
                )
                inv_attr = "双方帰責性の追加"
                inverted_target = "opt_both_breach"

            # 置換空振りガード
            if new_fact == fact:
                logger.warning(
                    "ContrastSetGenerator (Choice): 置換が空振りしたためフォールバック属性反転を適用します。"
                )
                new_fact = (
                    fact
                    + " （なお、本取引には特別例外規程の適用が明示的に排除される。）"
                )
                inverted_target = "opt_principle_applies"
                inv_attr = "例外適用の明示的排除"

            if new_fact == q_text or fact not in q_text:
                new_q_text = f"{new_fact}\n【指示】{base_scenario.question_text.split('【指示】')[-1]}"
            else:
                new_q_text = q_text.replace(fact, new_fact)

            assert new_fact != fact, "事実文が更新されていません。"
            new_metadata = dict(base_scenario.metadata)
            new_metadata["contrast_parent_id"] = base_scenario.scenario_id
            new_metadata["inverted_attribute"] = inv_attr

            perturbed = MultiHopScenario(
                scenario_id=f"{base_scenario.scenario_id}_perturbed",
                doc_id=base_scenario.doc_id,
                path_clause_ids=base_scenario.path_clause_ids,
                fact_situation=new_fact,
                question_text=new_q_text,
                question_type=QuestionType.CHOICE,
                criteria=base_scenario.criteria,
                target=inverted_target,
                evidence_spans=base_scenario.evidence_spans,
                metadata=new_metadata,
            )

        return ContrastPair(
            pair_id=pair_id,
            original_scenario=base_scenario,
            perturbed_scenario=perturbed,
            inverted_attribute=inv_attr,
        )
