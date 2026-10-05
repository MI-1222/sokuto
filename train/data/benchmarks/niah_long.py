"""日本語 NIAH (Needle In A Haystack) 長文対照ベンチマークモジュール。

青空文庫長編小説や行政手続ガイド等の長大背景文脈 (4,000〜8,192+ トークン) に対し、
実務的な反事実特約ルール (Needle) を 0%〜100% の深度に均等分散挿入し、
「Lost in the Middle(中央部での注意希釈・情報見落とし)」耐性および
「規定なし(None / OOD)」への健全なフォールバックを測定する評価基盤を提供する。
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from data.schema import QuestionType, UnifiedSample

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NeedleRule:
    """反事実的特約ルール定義。

    Attributes:
        rule_id (str): ルール識別子。
        topic (str): ルールの主題 (例: '通知期限', '管轄裁判所')。
        needle_text (str): 背景文脈に挿入する特約文言。
        question_text (str): 評価用の指示文。
        target_option_key (str): 正解選択肢キー。
        criteria (dict[str, str]): 選択肢辞書 (正解、引っかけ、無関係、規定なし)。
    """

    rule_id: str
    topic: str
    needle_text: str
    question_text: str
    target_option_key: str
    criteria: dict[str, str]


# 代表的な業務反事実特約ルールバンク
DEFAULT_NEEDLE_BANK: list[NeedleRule] = [
    NeedleRule(
        rule_id="needle_notice_deadline",
        topic="不具合通知期限",
        needle_text="【特約条項第9条】本特例取引に限り、買主による目的物の契約不適合通知期限は、原則規定の30日ではなく5営業日以内に短縮されるものとする。",
        question_text="本取引における買主の契約不適合通知期限として、正しい期間を選択せよ。",
        target_option_key="opt_needle",
        criteria={
            "opt_needle": "5営業日以内（特約適用）",
            "opt_general": "30日以内（一般原則）",
            "opt_other": "14日以内",
            "none": "本取引における通知期限の規定なし",
        },
    ),
    NeedleRule(
        rule_id="needle_jurisdiction",
        topic="管轄裁判所",
        needle_text="【特約条項第14条】本取引に関して生ずる一切の訴訟については、本店所在地の東京地方裁判所ではなく、京都地方裁判所を専属的合意管轄裁判所とする。",
        question_text="本取引に関する紛争解決の専属的合意管轄裁判所として正しいものを選択せよ。",
        target_option_key="opt_needle",
        criteria={
            "opt_needle": "京都地方裁判所（特約適用）",
            "opt_general": "東京地方裁判所（一般管轄）",
            "opt_other": "大阪地方裁判所",
            "none": "専属的合意管轄裁判所の合意なし",
        },
    ),
    NeedleRule(
        rule_id="needle_termination_notice",
        topic="解約予告期間",
        needle_text="【特約条項第22条】本契約の中途解約予告期間は、標準約款に定める3ヶ月前ではなく、14日前までの書面通知により効力を生ずるものとする。",
        question_text="本契約の中途解約を申し入れる場合の事前予告期間として正しいものを選択せよ。",
        target_option_key="opt_needle",
        criteria={
            "opt_needle": "14日前までの書面通知（特約適用）",
            "opt_general": "3ヶ月前までの書面通知（標準約款）",
            "opt_other": "30日前までの通知",
            "none": "中途解約に関する予告期間の定めなし",
        },
    ),
    NeedleRule(
        rule_id="needle_liability_cap",
        topic="損害賠償上限",
        needle_text="【特約条項第17条】本案件における甲の損害賠償責任の累計上限額は、直近1年間の対価総額ではなく、一律金300万円を上限として制限される。",
        question_text="本案件における甲の損害賠償責任の累計上限額として正しいものを選択せよ。",
        target_option_key="opt_needle",
        criteria={
            "opt_needle": "一律金300万円（特約適用）",
            "opt_general": "直近1年間の対価総額（標準規定）",
            "opt_other": "金1,000万円または対価総額のいずれか低い額",
            "none": "損害賠償額の責任制限条項なし",
        },
    ),
]

# パブリックドメイン調の自然な長文背景パラグラフ (青空文庫・近代文学・行政報告文体)
BACKGROUND_PARAGRAPHS: list[str] = [
    (
        "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"
        "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。"
        "吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。"
        "この書生というのは時々我々を捕えて煮て食うという話である。しかしその当時は何という考もなかったから別段恐しいとも思わなかった。"
        "ただ彼の掌の心に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。"
    ),
    (
        "山路を登りながら、こう考えた。智に働けば角が立つ。情に棹させば流される。意地を通せば窮屈だ。とかくに人の世は住みにくい。"
        "住みにくさが高じると、安い所へ引き越したくなる。どこへ越しても住みにくいと悟った時、詩が生れて、画が出来る。"
        "人の世を作ったものは神でもなければ鬼でもない。やはり向う三軒両隣りにちらちらするただの人である。"
        "ただの人が作った人の世が住みにくいからとて、越す国はあるまい。あれば人でなしの国へ行くばかりだ。"
    ),
    (
        "国境の長いトンネルを抜けると雪国であった。夜の底が白くなった。信号所に汽車が止まった。"
        "向側の座席から娘が立って来て、島村の前のガラス窓を落した。雪の冷気が流れこんだ。娘はひどく身を乗り出して、遠くへ呼ぶように、"
        "「駅長さん、駅長さん」と叫んだ。明りをさげて雪の中を足重にやって来た男は、襟巻で鼻の上まで包み、耳に帽子の毛皮を垂れていた。"
    ),
    (
        "行政手続における特定の個人を識別するための番号の利用等に関する法律に基づき、"
        "行政機関の長等は、個人番号利用事務を処理するために必要があるときは、関係機関に対して必要な情報の提供を求めることができる。"
        "また、情報提供ネットワークシステムを使用した特定個人情報の照会および提供に際しては、"
        "通信経路における暗号化措置およびアクセスログの厳格な保管が義務付けられている。"
    ),
    (
        "メロスは激怒した。必ず、かの邪智暴虐の王を除かなければならぬと決意した。メロスには政治がわからぬ。"
        "メロスは、村の牧人である。笛を吹き、羊と遊んで暮して来た。けれども邪悪に対しては、人一倍に敏感であった。"
        "きょう未明メロスは村を出発し、野を越え山越え、十里はなれた此のシラクスの市にやって来た。"
    ),
]


def generate_haystack_text(target_char_length: int = 8500) -> str:
    """指定された文字規模に達する自然な長文背景テキスト (Haystack) を構築する。

    約 8,500 文字は日本語トークナイズで約 4,500〜7,000 トークンに相当し、
    8,192 トークンの制限内で Needle が安全に保持される長文帯域を形成する。

    Args:
        target_char_length (int): 目標文字数 (デフォルト: 8,500文字)。

    Returns:
        str: 構築された長文背景テキスト。
    """
    paragraphs: list[str] = []
    current_len = 0
    para_idx = 0

    while current_len < target_char_length:
        para = BACKGROUND_PARAGRAPHS[para_idx % len(BACKGROUND_PARAGRAPHS)]
        paragraphs.append(para)
        current_len += len(para) + 2
        para_idx += 1

    return "\n\n".join(paragraphs)


def find_safe_insertion_point(text: str, target_ratio: float) -> int:
    """サブワード境界や漢字の分断を防ぐため、指定比率に最も近い句点（`。`）または改行の直後を探索する。

    Args:
        text (str): 背景テキスト。
        target_ratio (float): 目標深度比率 (0.0〜1.0)。

    Returns:
        int: 安全な文字挿入インデックス。
    """
    target_ratio = max(0.0, min(1.0, target_ratio))
    target_idx = int(len(text) * target_ratio)

    if target_ratio <= 0.01:
        # ドキュメント先頭
        return 0
    if target_ratio >= 0.99:
        # ドキュメント末尾
        return len(text)

    # 前後 500 文字の範囲で直近の '。\n' または '。' を探索
    best_idx = target_idx
    min_dist = float("inf")

    # 前方探索
    search_start = max(0, target_idx - 500)
    search_end = min(len(text), target_idx + 500)

    for i in range(search_start, search_end):
        if text[i] == "。" or text[i] == "\n":
            insert_pos = i + 1
            dist = abs(insert_pos - target_idx)
            if dist < min_dist:
                min_dist = dist
                best_idx = insert_pos

    return best_idx


@dataclass
class NIAHEvaluationResult:
    """NIAH ベンチマーク評価結果。

    Attributes:
        total_samples (int): 評価サンプル総数。
        correct_count (int): 正解数。
        accuracy (float): 総合正解率。
        depth_accuracies (dict[float, float]): 深度別正解率マップ (0.0〜1.0)。
        lost_in_the_middle_gap (float): 端部 (0-20%, 80-100%) と中央部 (40-60%) の精度乖離。
        ood_accuracy (float): 針不在 (OOD) サンプルの「規定なし」判定精度。
        passed_exit_criteria (bool): 目標 (全深度 >= 90%) を達成したか。
    """

    total_samples: int
    correct_count: int
    accuracy: float
    depth_accuracies: dict[float, float] = field(default_factory=dict)
    lost_in_the_middle_gap: float = 0.0
    ood_accuracy: float = 0.0
    passed_exit_criteria: bool = False


class NIAHBenchmarkGenerator:
    """日本語長文 NIAH (Needle In A Haystack) ベンチマーク生成・評価器。

    4,000〜8,192 トークンの長文背景に対し、反事実特約 Needle を 11 段階の深度
    (0%, 10%, 20%, ..., 90%, 100%) に安全挿入したサンプル群および
    針不在の対照 OOD サンプルを生成する。
    """

    def __init__(
        self,
        needle_bank: list[NeedleRule] | None = None,
        depth_steps: list[float] | None = None,
        target_char_length: int = 8500,
        seed: int = 42,
    ) -> None:
        """NIAH ベンチマーク生成器を初期化する。

        Args:
            needle_bank (list[NeedleRule] | None): 特約ルール一覧。None の場合はデフォルトバンク。
            depth_steps (list[float] | None): 評価深度グリッド。None の場合は 0.0〜1.0 の 11 段階。
            target_char_length (int): 背景テキスト目標文字数 (デフォルト: 8,500文字、約5,500〜7,000トークン)。
            seed (int): 乱数シード。
        """
        self.needle_bank = needle_bank or DEFAULT_NEEDLE_BANK
        self.depth_steps = depth_steps or [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ]
        self.target_char_length = target_char_length
        self.seed = seed

    def generate_benchmark_samples(
        self,
        haystack_text: str | None = None,
        include_ood: bool = True,
    ) -> list[UnifiedSample]:
        """全深度および全 Needle ルールを網羅したベンチマークサンプル群を生成する。

        Args:
            haystack_text (str | None): 背景長文テキスト。None の場合は自動生成。
            include_ood (bool): 針不在の OOD サンプルを含めるか (デフォルト: True)。

        Returns:
            list[UnifiedSample]: 生成された統一サンプルリスト。
        """
        base_haystack = haystack_text or generate_haystack_text(self.target_char_length)
        samples: list[UnifiedSample] = []

        for rule in self.needle_bank:
            # 1. 各深度への Needle 均等分散挿入
            for depth in self.depth_steps:
                insert_idx = find_safe_insertion_point(base_haystack, depth)

                # 針文言を前後に改行を伴って挿入
                prefix = base_haystack[:insert_idx]
                suffix = base_haystack[insert_idx:]
                needle_payload = f"\n\n{rule.needle_text}\n\n"
                modified_state = prefix + needle_payload + suffix

                evidence_start = len(prefix) + 2
                evidence_end = evidence_start + len(rule.needle_text)

                # スパン整合性アサーション (テキスト境界内に完全に収まっていること)
                assert 0 <= evidence_start < evidence_end <= len(modified_state), (
                    f"Needle evidence_span [{evidence_start}, {evidence_end}] が "
                    f"ドキュメント長 {len(modified_state)} の範囲外です。"
                )

                sample_id = f"niah_{rule.rule_id}_depth_{int(depth * 100):03d}"
                metadata: dict[str, Any] = {
                    "rule_id": rule.rule_id,
                    "needle_depth": depth,
                    "is_ood": False,
                    "target_char_length": len(modified_state),
                    "evidence_span": [evidence_start, evidence_end],
                    "chunk_spans": [[evidence_start, evidence_end]],
                    "char_chunk_spans": [[evidence_start, evidence_end]],
                    "state_id": f"haystack_d_{int(depth * 100):03d}",
                }

                samples.append(
                    UnifiedSample(
                        dataset_name="niah_long",
                        sample_id=sample_id,
                        question_type=QuestionType.CHOICE,
                        state=modified_state,
                        instructions=rule.question_text,
                        criteria=rule.criteria,
                        target=rule.target_option_key,
                        metadata=metadata,
                    )
                )

            # 2. 対照 OOD サンプル (針を挿入しないドキュメントでの「規定なし」判定)
            if include_ood:
                sample_id_ood = f"niah_{rule.rule_id}_ood"
                metadata_ood = {
                    "rule_id": rule.rule_id,
                    "needle_depth": -1.0,
                    "is_ood": True,
                    "target_char_length": len(base_haystack),
                    "evidence_span": None,
                    "chunk_spans": [],
                    "char_chunk_spans": [],
                    "state_id": "haystack_clean",
                }

                samples.append(
                    UnifiedSample(
                        dataset_name="niah_long",
                        sample_id=sample_id_ood,
                        question_type=QuestionType.CHOICE,
                        state=base_haystack,
                        instructions=rule.question_text,
                        criteria=rule.criteria,
                        target="none",
                        metadata=metadata_ood,
                    )
                )

        logger.info(
            "NIAH 長文ベンチマークサンプル %d 件を生成しました (ルール数: %d, 深度数: %d)。",
            len(samples),
            len(self.needle_bank),
            len(self.depth_steps),
        )
        return samples

    def evaluate_predictions(
        self,
        samples: list[UnifiedSample],
        predictions: list[str],
    ) -> NIAHEvaluationResult:
        """モデル予測結果を受け取り、深度別正解率および Exit Criteria 達成状況を集計する。

        Args:
            samples (list[UnifiedSample]): ベンチマークサンプルリスト。
            predictions (list[str]): 各サンプルの予測選択肢キーリスト。

        Returns:
            NIAHEvaluationResult: 集計評価結果オブジェクト。
        """
        if len(samples) != len(predictions):
            raise ValueError(
                f"サンプル数 ({len(samples)}) と予測数 ({len(predictions)}) が一致しません。"
            )

        total = len(samples)
        correct_count = 0
        depth_stats: dict[float, list[bool]] = {}
        ood_results: list[bool] = []

        for sample, pred in zip(samples, predictions, strict=False):
            is_correct = pred == sample.target
            if is_correct:
                correct_count += 1

            depth = sample.metadata.get("needle_depth", -1.0)
            is_ood = sample.metadata.get("is_ood", False)

            if is_ood or depth < 0.0:
                ood_results.append(is_correct)
            else:
                depth_key = round(depth, 2)
                if depth_key not in depth_stats:
                    depth_stats[depth_key] = []
                depth_stats[depth_key].append(is_correct)

        overall_acc = correct_count / total if total > 0 else 0.0
        depth_accuracies = {
            d: sum(res) / len(res) for d, res in sorted(depth_stats.items())
        }

        # Lost in the Middle ギャップ算出 (端部 vs 中央部)
        edge_accs: list[float] = []
        middle_accs: list[float] = []
        for d, acc in depth_accuracies.items():
            if d <= 0.2 or d >= 0.8:
                edge_accs.append(acc)
            elif 0.4 <= d <= 0.6:
                middle_accs.append(acc)

        edge_mean = sum(edge_accs) / len(edge_accs) if edge_accs else 1.0
        middle_mean = sum(middle_accs) / len(middle_accs) if middle_accs else 1.0
        gap = max(0.0, edge_mean - middle_mean)

        ood_acc = sum(ood_results) / len(ood_results) if ood_results else overall_acc

        # Exit Criteria: 全深度で 90% 以上かつ OOD も 90% 以上
        passed = (
            all(acc >= 0.90 for acc in depth_accuracies.values()) and ood_acc >= 0.90
        )

        return NIAHEvaluationResult(
            total_samples=total,
            correct_count=correct_count,
            accuracy=overall_acc,
            depth_accuracies=depth_accuracies,
            lost_in_the_middle_gap=gap,
            ood_accuracy=ood_acc,
            passed_exit_criteria=passed,
        )
