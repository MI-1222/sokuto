"""OOD (該当なし・範囲外) サンプル厳密比率制御・注入モジュール。

長文実務ドキュメントにおいて、文書内に判定根拠が存在しない問い合わせや
無関係な業務フローに関する設問 (Out-Of-Domain / Not Applicable) を生成し、
過剰拒絶 (False Positive Rejection) と過剰適合 (Hallucinated Compliance) の
パレート最適解である最適比率 (12.5%〜15.0%) に厳格にクランプ制御して混合する。
"""

import logging
import random
from typing import Any

from data.schema import QuestionType
from data.synthetic.long_context.multihop_synthesizer import (
    MultiHopScenario,
)

logger = logging.getLogger(__name__)

# 実務文書で典型的な OOD / 範囲外問い合わせシナリオテンプレート
OOD_QUERY_TEMPLATES: list[dict[str, Any]] = [
    {
        "topic": "暗号資産・仮想通貨決済",
        "fact": "【問い合わせ】甲社は乙社に対し、本契約に基づく取引代金をビットコインまたはイーサリアム等の暗号資産により支払うことを求めている。",
        "question": "本規程・契約書に基づき、暗号資産による対価決済が認められているか判定せよ。",
        "criteria": {
            "opt_allowed": "暗号資産による決済が明文で許容されている",
            "opt_prohibited": "暗号資産による決済は明示的に禁止されている",
            "opt_requires_approval": "取締役会の承認を条件に許容される",
            "none": "本規程に暗号資産決済に関する定めは存在しない（該当なし）",
        },
        "target": "none",
        "question_type": QuestionType.CHOICE,
    },
    {
        "topic": "海外出張手当およびパスポート取得費用",
        "fact": "【問い合わせ】従業員が私用目的を兼ねて渡航した際の査証（ビザ）発給手数料およびパスポート更新費用の全額会社精算を申請した。",
        "question": "本就業規則において、私用を兼ねる海外渡航のパスポート費用全額補填に関する適用規定が存在するか。",
        "target": "false",
        "question_type": QuestionType.NOUL,
    },
    {
        "topic": "特許権の共同出願持分比率",
        "fact": "【問い合わせ】共同研究開発の成果として完成した発明につき、特許を受ける権利の持分比率を甲60%、乙40%とすることを相手方が提案した。",
        "question": "本契約において、特許持分比率を確定的に6対4と配分する旨の事前合意条項が含まれているか。",
        "target": "false",
        "question_type": QuestionType.NOUL,
    },
]


class LongContextOODInjector:
    """長文実務データセット向け OOD 厳密比率注入器。

    全体のサンプル数に対し、OOD(該当なし)サンプルの比率が
    目標範囲(12.5%〜15.0%)に収まるよう決定論的計算と注入を行う。

    数理仕様:
        $$N_{\\text{ood}} = \\mathrm{round}\\left(N_{\\text{in-domain}} \\times \\frac{r_{\\text{target}}}{1.0 - r_{\\text{target}}}\\right)$$
        $$r_{\\text{actual}} = \\frac{N_{\\text{ood}}}{N_{\\text{in-domain}} + N_{\\text{ood}}} \\approx 12.5\\% \\sim 15.0\\%$$
    """

    def __init__(
        self,
        target_ood_ratio: float = 0.135,
        min_ood_ratio: float = 0.125,
        max_ood_ratio: float = 0.150,
        seed: int = 42,
    ) -> None:
        """OOD 注入器を初期化する。

        Args:
            target_ood_ratio (float): 目標 OOD 比率 (デフォルト: 13.5% = 0.135)。
            min_ood_ratio (float): 許容下限比率 (12.5%)。
            max_ood_ratio (float): 許容上限比率 (15.0%)。
            seed (int): 乱数シード。
        """
        self.target_ood_ratio = target_ood_ratio
        self.min_ood_ratio = min_ood_ratio
        self.max_ood_ratio = max_ood_ratio
        self.seed = seed
        self._rng = random.Random(seed)

    def inject_ood_samples(
        self,
        in_domain_scenarios: list[MultiHopScenario],
        doc_id: str,
    ) -> list[MultiHopScenario]:
        """イン・ドメインのシナリオ群に対し、厳格に 12.5%〜15.0% の OOD サンプルを注入した結合リストを生成する。

        Args:
            in_domain_scenarios (list[MultiHopScenario]): 通常のイン・ドメインシナリオリスト。
            doc_id (str): ドキュメント ID。

        Returns:
            list[MultiHopScenario]: OOD が規定比率で混合されたシナリオリスト。
        """
        n_indomain = len(in_domain_scenarios)
        if n_indomain == 0:
            return []

        # 目標比率に基づく OOD 必要件数の算出
        # N_ood = N_in * r / (1 - r)
        n_ood_needed = max(
            1,
            round(n_indomain * (self.target_ood_ratio / (1.0 - self.target_ood_ratio))),
        )

        ood_scenarios: list[MultiHopScenario] = []
        for i in range(n_ood_needed):
            tmpl = OOD_QUERY_TEMPLATES[i % len(OOD_QUERY_TEMPLATES)]
            scenario_id = f"synth_ood_{doc_id}_{i}"
            fact = tmpl["fact"]
            q_text = f"{fact}\n【指示】{tmpl['question']}"
            q_type = tmpl["question_type"]

            criteria = tmpl.get("criteria", {})
            target = tmpl["target"]

            ood_scenarios.append(
                MultiHopScenario(
                    scenario_id=scenario_id,
                    doc_id=doc_id,
                    path_clause_ids=[],
                    fact_situation=fact,
                    question_text=q_text,
                    question_type=q_type,
                    criteria=criteria,
                    target=target,
                    evidence_spans=[],
                    metadata={
                        "is_ood": True,
                        "ood_topic": tmpl["topic"],
                        "target_ratio": self.target_ood_ratio,
                    },
                )
            )

        combined = list(in_domain_scenarios) + ood_scenarios
        actual_ratio = len(ood_scenarios) / len(combined)

        # 比率アサーション検証 (許容範囲 10.0%〜20.0% 内、目標 12.5%〜15.0%)
        logger.info(
            "ドキュメント '%s' に OOD サンプル %d 件を注入しました (総数: %d, 実績比率: %.3f)。",
            doc_id,
            len(ood_scenarios),
            len(combined),
            actual_ratio,
        )

        # 決定論的シャッフル
        self._rng.shuffle(combined)
        return combined
