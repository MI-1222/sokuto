"""JBE-QA (司法試験短答式試験問題) データセットコンバータモジュール。

司法試験短答式試験の長文事例 (事実関係および関連法令文言、1,000〜3,500 トークン) を State とし、
各設問肢の真偽判定を Noul プリミティブ、または正解肢組み合わせ選択を Choice プリミティブへ射影する。
同一事例に対する複数質問の追跡メタデータ (state_id) および条文・段落スパン (chunk_spans) を保持する。
"""

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from data.converters.base import BaseDatasetConverter
from data.schema import QuestionType, UnifiedSample

logger = logging.getLogger(__name__)

# オフライン動作およびテスト用の司法試験代表事例データ
BUILTIN_JBE_QA_RECORDS: list[dict[str, Any]] = [
    {
        "case_id": "jbe_qa_civil_01",
        "category": "民法（契約不適合責任・詐欺取消と第三者保護）",
        "facts": (
            "【事実関係】\n"
            "第1項: 令和4年4月1日、甲は乙との間で、甲所有の甲土地（所在: 東京都千代田区、地目: 宅地、実測面積: 300平方メートル）を"
            "代金5,000万円で乙に売却する旨の売買契約（以下「本件売買契約」という）を締結した。"
            "本件売買契約の締結に際し、乙は甲に対し、甲土地の周辺一帯が近く市街化調整区域から商業地域へ用途変更される旨の虚偽の都市計画図を提示し、"
            "甲はその説明を真実であると誤信して売却を決意した。\n"
            "第2項: 令和4年4月15日、乙は本件売買契約に基づく売買代金全額を甲に支払い、甲は乙に対して甲土地の所有権移転登記手続きを完了した。"
            "その後、乙は令和4年5月10日、甲土地を用途変更の計画について何ら知らず過失もない善意無過失の丙に対し、代金6,000万円で転売し、"
            "丙への所有権移転登記手続きを完了した。\n"
            "第3項: 令和4年6月1日、甲は行政機関への問い合わせによって用途変更の事実が全く存在しないことを知るに至り、"
            "直ちに乙に対し民法第96条第1項に基づき詐欺による意思表示の取消しを通知した。"
            "さらに甲は、丙に対して甲土地の所有権は依然として甲に帰属する旨を主張し、丙名義の所有権移転登記の抹消および甲土地の明渡しを求めた。\n"
            "第4項: また、甲土地には売買契約締結前から地中3メートルの深さに産業廃棄物である大量のコンクリートガラが埋設されており、"
            "通常の建築基礎工事を行うには多額の撤去費用（約800万円）を要する状態であった。"
            "乙および丙はこの埋設物の存在を認識していなかったが、丙が基礎工事を着工した令和4年7月10日に至って初めて発見された。"
        ),
        "chunk_spans": [
            (0, 168),
            (169, 362),
            (363, 560),
            (561, 740),
        ],
        "sub_questions": [
            {
                "sub_id": "sub_a",
                "proposition": "甲は、乙の詐欺を理由として本件売買契約を取り消したとしても、善意無過失の第三者である丙に対して甲土地の所有権復帰を主張することはできない。",
                "is_correct": True,
                "evidence_span": (363, 560),
                "target_chunk_index": 2,
            },
            {
                "sub_id": "sub_b",
                "proposition": "甲が乙に対する詐欺取消しの意思表示をした後、丙が乙から甲土地を買い受けて登記を備えた場合であっても、丙が善意無過失であれば民法第96条第3項の第三者として保護される。",
                "is_correct": False,
                "evidence_span": (363, 560),
                "target_chunk_index": 2,
            },
            {
                "sub_id": "sub_c",
                "proposition": "甲土地の地中にコンクリートガラが埋設されていたことは契約の内容に適合しない欠陥であり、買主である丙は、売主である乙に対して民法第562条に基づく追完請求権を行使することができる。",
                "is_correct": True,
                "evidence_span": (561, 740),
                "target_chunk_index": 3,
            },
            {
                "sub_id": "sub_d",
                "proposition": "乙は甲に対して甲土地の代金を完済しているため、甲が詐欺取消しを主張する場合であっても、乙が得た転売差益1,000万円の返還を請求することは一切認められない。",
                "is_correct": True,
                "evidence_span": (169, 362),
                "target_chunk_index": 1,
            },
        ],
        "choice_question": {
            "instructions": "上記【事実関係】に基づき、各命題（ア〜エ）の正誤の組み合わせとして正しいものを後記の選択肢の中から1つ選べ。",
            "criteria": {
                "opt_1": "ア: 正、イ: 誤、ウ: 正、エ: 正",
                "opt_2": "ア: 正、イ: 正、ウ: 誤、エ: 正",
                "opt_3": "ア: 誤、イ: 誤、ウ: 正、エ: 誤",
                "opt_4": "ア: 誤、イ: 正、ウ: 誤、エ: 誤",
                "none": "該当する組み合わせなし",
            },
            "target": "opt_1",
        },
    },
    {
        "case_id": "jbe_qa_penal_02",
        "category": "刑法（共犯関係の離脱・正当防衛・誤想防衛）",
        "facts": (
            "【事実関係】\n"
            "第1項: 甲および乙は、令和4年10月1日夜間、Vが経営する貴金属店（以下「V店」という）に侵入し、"
            "陳列ケース内の高額腕時計を窃取することを計画した。甲が見張り役を担当し、乙が店舗内に侵入して現金を物色する役割分担を取り決めた。\n"
            "第2項: 同日午後11時30分、乙がV店の通用口の鍵を工具で破壊して店内に侵入した直後、店内の警備ブザーが作動した。"
            "店舗外で見張りをしていた甲は、警備員の駆けつけを恐れ、乙に対して「警察が来るからもうやめよう。逃げるぞ」と大声で叫び、"
            "自己の携帯電話から乙の携帯電話へ犯行の中止を促すメッセージを送信した上で、単独で現場から走り去った。\n"
            "第3項: 乙は甲の呼びかけおよびメッセージを認識したものの、犯行の継続を決意し、奥の事務室から出てきた店主Vに対し、"
            "所持していた鉄パイプを振り上げて脅迫し、金庫から現金200万円および腕時計3点を奪取して逃走した。\n"
            "第4項: 逃走の際、乙は近隣住民Wに発見されて取り押さえられそうになったため、Wの脚部を鉄パイプで殴打して全治3週間の骨折を負わせた。"
            "甲は現場離脱後、自宅において乙が強盗傷人行為に及んだ事実を知った。"
        ),
        "chunk_spans": [
            (0, 140),
            (141, 314),
            (315, 458),
            (459, 584),
        ],
        "sub_questions": [
            {
                "sub_id": "sub_a",
                "proposition": "甲は乙に対して犯行の中止を告げて現場から単独離脱しているため、当初の共謀に基づく心理的・物理的因果性を完全に遮断したものとして、その後の乙の強盗行為につき共犯責任を負わない。",
                "is_correct": False,
                "evidence_span": (141, 314),
                "target_chunk_index": 1,
            },
            {
                "sub_id": "sub_b",
                "proposition": "甲と乙の間で合意されていた当初の共謀は窃盗にとどまるため、乙が強盗および傷害に及んだ行為について、甲に強盗傷人罪の共同正犯としての罪責を問うことはできず、窃盗未遂罪等の限度で処断される。",
                "is_correct": True,
                "evidence_span": (315, 458),
                "target_chunk_index": 2,
            },
            {
                "sub_id": "sub_c",
                "proposition": "乙が逃走時に住民Wを殴打して負傷させた行為は、窃盗の機会において逮捕を免れるために暴行を加えたものであり、刑法第238条の事後強盗罪および刑法第240条の強盗致傷罪が成立する。",
                "is_correct": True,
                "evidence_span": (459, 584),
                "target_chunk_index": 3,
            },
        ],
        "choice_question": {
            "instructions": "上記【事実関係】における甲および乙の罪責に関する各命題の正誤組み合わせとして正しいものを1つ選べ。",
            "criteria": {
                "opt_1": "ア: 誤、イ: 正、ウ: 正",
                "opt_2": "ア: 正、イ: 正、ウ: 正",
                "opt_3": "ア: 誤、イ: 誤、ウ: 正",
                "opt_4": "ア: 正、イ: 誤、ウ: 誤",
                "none": "該当なし",
            },
            "target": "opt_1",
        },
    },
]


class JBEQAConverter(BaseDatasetConverter):
    """司法試験短答式 (JBE-QA) データセットコンバータ。

    長文事実関係 (1,000〜3,500 トークン) を State とし、
    設問肢の真偽判定 (Noul) および組み合わせ選択 (Choice) への射影を行う。
    同一事例 (state_id) に対する複数質問バッチ評価を支援する。

    Attributes:
        file_path (Path | None): 外部 JSON/JSONL ファイルパス。
        mode (Literal['both', 'noul', 'choice']): 変換対象プリミティブ指定。
        seed (int): 乱数シード。
    """

    def __init__(
        self,
        file_path: str | Path | None = None,
        mode: Literal["both", "noul", "choice"] = "both",
        seed: int = 42,
    ) -> None:
        """JBE-QA コンバータを初期化する。

        Args:
            file_path (str | Path | None): 読み込み対象の JBE-QA データファイルパス。
            mode (Literal['both', 'noul', 'choice']): 抽出プリミティブ。
            seed (int): 乱数シード。
        """
        super().__init__(seed=seed)
        self.file_path = Path(file_path) if file_path is not None else None
        self.mode = mode

    def _load_raw_records(self) -> list[dict[str, Any]]:
        """ローカルファイルまたは組み込みサンプルからレコード一覧を読み出す。

        Returns:
            list[dict[str, Any]]: 生データ辞書列。
        """
        if self.file_path is not None and self.file_path.is_file():
            records: list[dict[str, Any]] = []
            try:
                with open(self.file_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
                logger.info(
                    "JBE-QA ファイルから %d 件をロードしました: %s",
                    len(records),
                    self.file_path,
                )
                return records
            except (json.JSONDecodeError, OSError, ValueError) as e:
                logger.warning(
                    "JBE-QA ファイルの読み込みに失敗しました (%s)。ビルトイン事例を使用します。",
                    e,
                )

        return BUILTIN_JBE_QA_RECORDS

    def convert_split(self, split: str = "train") -> Iterator[UnifiedSample]:
        """指定スプリットを走査し、UnifiedSample を順次生成する。

        同一の事実関係 (State) に対し、各枝問を独立した Noul サンプルとして生成し、
        さらに全体の多肢選択設問を Choice サンプルとして生成する。
        各サンプルには同一の state_id が付与される。

        Args:
            split (str): スプリット名 ('train', 'validation', 'test')。

        Yields:
            Iterator[UnifiedSample]: 統一サンプル列。
        """
        records = self._load_raw_records()

        for rec_idx, rec in enumerate(records):
            case_id = rec.get("case_id", f"jbe_qa_{rec_idx}")
            state_text = rec["facts"]
            chunk_spans = rec.get("chunk_spans", [])
            category = rec.get("category", "司法試験短答式")

            # 決定論的スプリット分割 (zlib.crc32)
            # サンプル数が少ない (<= 5) 場合は train に全件含め、validation/test にもフォールバック提供
            if len(records) > 5:
                import zlib

                case_hash = zlib.crc32(case_id.encode()) % 100
                if split in ("validation", "test"):
                    if case_hash >= 20:
                        continue
                else:
                    if case_hash < 20:
                        continue

            # 1. 枝問ごとの Noul 判定サンプル (真偽確率)
            if self.mode in ("both", "noul"):
                for sub in rec.get("sub_questions", []):
                    sub_id = sub["sub_id"]
                    proposition = sub["proposition"]
                    target = "true" if sub["is_correct"] else "false"
                    evidence_span = sub.get("evidence_span")
                    target_chunk_index = sub.get("target_chunk_index")

                    instructions = (
                        f"以下の事実関係および関係法令に照らし、命題が法律上「真（正しい）」か"
                        f"「偽（誤り）」かを判定せよ。\n【命題】{proposition}"
                    )

                    sample_id = f"{case_id}_{sub_id}"
                    metadata: dict[str, Any] = {
                        "split": split,
                        "state_id": case_id,
                        "sub_id": sub_id,
                        "category": category,
                        "is_ood": False,
                        "chunk_spans": chunk_spans,
                    }
                    if evidence_span is not None:
                        metadata["evidence_span"] = list(evidence_span)
                    if target_chunk_index is not None:
                        metadata["target_chunk_index"] = target_chunk_index

                    yield UnifiedSample(
                        dataset_name="jbe_qa",
                        sample_id=sample_id,
                        question_type=QuestionType.NOUL,
                        state=state_text,
                        instructions=instructions,
                        criteria={},
                        target=target,
                        metadata=metadata,
                    )

            # 2. 全体設問の Choice 判定サンプル (正誤組み合わせ選択)
            if self.mode in ("both", "choice") and "choice_question" in rec:
                cq = rec["choice_question"]
                instructions = cq["instructions"]
                criteria = cq["criteria"]
                target = cq["target"]

                sample_id = f"{case_id}_choice"
                metadata = {
                    "split": split,
                    "state_id": case_id,
                    "category": category,
                    "is_ood": False,
                    "chunk_spans": chunk_spans,
                }

                yield UnifiedSample(
                    dataset_name="jbe_qa",
                    sample_id=sample_id,
                    question_type=QuestionType.CHOICE,
                    state=state_text,
                    instructions=instructions,
                    criteria=criteria,
                    target=target,
                    metadata=metadata,
                )
