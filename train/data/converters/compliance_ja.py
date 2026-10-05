"""コンプライアンス・就業規則 (金融庁ガイドライン・モデル就業規則) データセットコンバータモジュール。

金融庁監督指針・事務ガイドラインおよび厚生労働省モデル就業規則等の長文実務規程 (2,000〜8,192 トークン) を State とし、
インシデント重大度評価 (Score 型: 0〜4 段階)、
懲戒処分および適用条項の選択 (Choice 型)、
ならびに適法性・手続要件充足判定 (Noul 型) への標準化変換を行う。
"""

import json
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from data.converters.base import BaseDatasetConverter
from data.schema import QuestionType, UnifiedSample

logger = logging.getLogger(__name__)

# ビルトインモデル就業規則サンプル (厚生労働省標準準拠)
BUILTIN_WORK_RULES_TEXT = (
    "モデル就業規則（抜粋）\n\n"
    "第1章 総則\n"
    "第1条（目的）\n"
    "この就業規則（以下「規則」という。）は、労働基準法に基づき、株式会社ソクト（以下「会社」という。）の"
    "労働者の労働条件、服務規律その他の就業に関する事項を定めるものである。\n\n"
    "第2章 服務規律\n"
    "第2条（服務の基本原則）\n"
    "労働者は、会社が保有する情報資産および顧客情報を厳重に保護し、業務上の指揮命令に従い誠実に勤務しなければならない。\n"
    "第3条（情報セキュリティ義務）\n"
    "労働者は、会社の許可なく私用PC、USBメモリその他の記憶媒体を社内ネットワークに接続し、"
    "または機密データを外部へ送信・持ち出してはならない。\n\n"
    "第3章 懲戒\n"
    "第4条（懲戒の種別）\n"
    "会社は、労働者が本規則に違反した場合、その情状に応じ次の区分により懲戒処分を行う。\n"
    " (1) けん責（レベル0相当の厳重注意・始末書提出）: 始末書を提出させて将来を戒める。\n"
    " (2) 減給: 始末書を提出させ、1回の額が平均賃金の1日分の半額、総額が一賃金支払期の総額の10分の1を超えない範囲で減給する。\n"
    " (3) 出勤停止: 始末書を提出させ、14日以内の期間を定めて出勤を停止し、その期間の賃金は支給しない。\n"
    " (4) 降格: 役職または資格階級を1段階以上引き下げる。\n"
    " (5) 諭旨解雇: 退職を勧告し、勧告に従わない場合は懲戒解雇とする。\n"
    " (6) 懲戒解雇: 即時に予告手当を支給することなく解雇する。\n\n"
    "第5条（懲戒の事由および重大度基準）\n"
    "1. 正当な理由のない無断欠勤が連続3日以内の場合、または軽微な職責怠慢があった場合は、けん責に処する。\n"
    "2. 故意または過失により社外秘情報を誤送信したが、第三者への拡散前に回収・暗号化保護された場合は、減給に処する。\n"
    "3. 私用端末を用いて機密情報・個人情報を意図的に外部へ持ち出した場合、またはハラスメント行為が認定された場合は、出勤停止または降格に処する。\n"
    "4. 顧客情報の大量不正売買、横領、贈収賄その他の重大な犯罪行為に及んだ場合は、懲戒解雇に処する。\n\n"
    "第6条（懲戒の手続および弁明の機会）\n"
    "懲戒処分を行うにあたっては、懲戒委員会を開催し、対象となる労働者に対して事前に事実関係を通知し、十分な弁明の機会を与えなければならない。\n"
)


def parse_rules_clause_spans(text: str) -> list[tuple[int, int]]:
    """規程テキストから条項区切り文字スパン (start_char, end_char) を抽出する。

    行頭に配置された条項見出しのみを検出し、本文中の引用を除外する。
    また「第5条の2」のような枝番条項にも対応する。

    Args:
        text (str): 規程テキスト。

    Returns:
        list[tuple[int, int]]: 各条項の文字スパン。
    """
    pattern = re.compile(
        r"(?:^|\n)[ \t]*(第[0-9一二三四五六七八九十百]+条(?:の[0-9一二三四五六七八九十百]+)?(?:\s*（[^）\n]+）|[\s　]+|$))"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return [(0, len(text))]

    spans: list[tuple[int, int]] = []
    for i, m in enumerate(matches):
        clause_full = m.group(1).strip()
        start = text.find(clause_full, m.start())
        if start == -1:
            start = m.start()

        if i + 1 < len(matches):
            next_full = matches[i + 1].group(1).strip()
            end = text.find(next_full, matches[i + 1].start())
            if end == -1:
                end = matches[i + 1].start()
        else:
            end = len(text)

        spans.append((start, end))

    return spans


class ComplianceJAConverter(BaseDatasetConverter):
    """金融・就業規則コンプライアンスデータセットコンバータ。

    実務規程ドキュメント (2,000〜8,192 トークン) を State とし、
    インシデント重大度評価 (Score 型: 0〜4 段階)、
    懲戒処分選択 (Choice 型)、
    および手続要件充足の真偽判定 (Noul 型) への射影を行う。

    Attributes:
        file_path (Path | None): 外部規程データファイルパス。
        mode (Literal['all', 'score', 'choice', 'noul']): 出力プリミティブ種別。
        seed (int): 乱数シード。
    """

    def __init__(
        self,
        file_path: str | Path | None = None,
        mode: Literal["all", "score", "choice", "noul"] = "all",
        seed: int = 42,
    ) -> None:
        """コンプライアンス規程コンバータを初期化する。

        Args:
            file_path (str | Path | None): 規程データファイルパス。
            mode (Literal['all', 'score', 'choice', 'noul']): 出力モード。
            seed (int): 乱数シード。
        """
        super().__init__(seed=seed)
        self.file_path = Path(file_path) if file_path is not None else None
        self.mode = mode

    def _load_documents(self) -> list[dict[str, Any]]:
        """外部ファイルまたは組み込み規程コーパスからデータをロードする。

        Returns:
            list[dict[str, Any]]: 規程データ辞書列。
        """
        if self.file_path is not None and self.file_path.is_file():
            documents: list[dict[str, Any]] = []
            try:
                with open(self.file_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            documents.append(json.loads(line))
                logger.info(
                    "ComplianceJA ファイルから %d 件をロードしました: %s",
                    len(documents),
                    self.file_path,
                )
                return documents
            except (json.JSONDecodeError, OSError, ValueError) as e:
                logger.warning(
                    "ComplianceJA ファイルのパースに失敗しました (%s)。ビルトイン規程を使用します。",
                    e,
                )

        spans = parse_rules_clause_spans(BUILTIN_WORK_RULES_TEXT)
        return [
            {
                "doc_id": "compliance_work_rules_sokuto",
                "title": "株式会社ソクト 就業規則",
                "text": BUILTIN_WORK_RULES_TEXT,
                "clause_spans": spans,
                "questions": [
                    {
                        "q_id": "incident_severity_leak",
                        "type": "score",
                        "instructions": (
                            "以下のインシデント事案について、就業規則第5条の懲戒基準に基づき、"
                            "違反の重大度・処分レベルを 0 から 4 までの順序尺度で評価せよ。\n"
                            "【事案】従業員Aは、私用USBメモリを用いて顧客名簿5,000件の機密データを無断で社外へ持ち出した。"
                        ),
                        "criteria": {
                            "0": "軽微（無断欠勤・職責怠慢、けん責相当）",
                            "1": "中程度（過失による誤送信、減給相当）",
                            "2": "深刻（意図的機密持ち出し・ハラスメント、出勤停止・降格相当）",
                            "3": "極めて深刻（組織的信用毀損、諭旨解雇相当）",
                            "4": "重大犯罪・破滅的（顧客情報不正売買・横領、懲戒解雇相当）",
                        },
                        "target": "2",
                        "evidence_clause": "第5条第3項",
                    },
                    {
                        "q_id": "disciplinary_action_choice",
                        "type": "choice",
                        "instructions": (
                            "正当な理由なく連続2日間の無断欠勤を行った従業員に対し、"
                            "本規則第5条第1項に基づき適用されるべき懲戒処分の種別を選択せよ。"
                        ),
                        "criteria": {
                            "opt_reprimand": "けん責（始末書提出による将来の戒め）",
                            "opt_salary_cut": "減給処分",
                            "opt_suspension": "出勤停止処分",
                            "opt_dismissal": "懲戒解雇",
                            "none": "規則上の懲戒事由に該当しない",
                        },
                        "target": "opt_reprimand",
                        "evidence_clause": "第5条第1項",
                    },
                    {
                        "q_id": "due_process_defense_noul",
                        "type": "noul",
                        "instructions": (
                            "本規則において、懲戒処分を決定するにあたり、懲戒委員会を開催して"
                            "対象労働者に弁明の機会を与えることは必須の手続要件として義務付けられているか。"
                        ),
                        "target": "true",
                        "evidence_clause": "第6条",
                    },
                ],
            }
        ]

    def convert_split(self, split: str = "train") -> Iterator[UnifiedSample]:
        """指定スプリットを走査し、UnifiedSample を順次生成する。

        Args:
            split (str): スプリット名 ('train', 'validation', 'test')。

        Yields:
            Iterator[UnifiedSample]: 統一サンプル列。
        """
        docs = self._load_documents()

        for doc_idx, doc in enumerate(docs):
            doc_id = doc.get("doc_id", f"compliance_doc_{doc_idx}")
            state_text = doc["text"]
            clause_spans = doc.get("clause_spans") or parse_rules_clause_spans(
                state_text
            )

            # 決定論的スプリット振り分け (zlib.crc32)
            if len(docs) > 5:
                import zlib

                doc_hash = zlib.crc32(doc_id.encode()) % 100
                if split in ("validation", "test"):
                    if doc_hash >= 20:
                        continue
                else:
                    if doc_hash < 20:
                        continue

            for q in doc.get("questions", []):
                q_id = q["q_id"]
                q_type_str = q.get("type", "choice").lower()

                if q_type_str == "score":
                    if self.mode not in ("all", "score"):
                        continue
                    q_type = QuestionType.SCORE
                elif q_type_str == "choice":
                    if self.mode not in ("all", "choice"):
                        continue
                    q_type = QuestionType.CHOICE
                elif q_type_str == "noul":
                    if self.mode not in ("all", "noul"):
                        continue
                    q_type = QuestionType.NOUL
                else:
                    continue

                metadata: dict[str, Any] = {
                    "split": split,
                    "state_id": doc_id,
                    "doc_title": doc.get("title", ""),
                    "chunk_spans": clause_spans,
                    "char_chunk_spans": clause_spans,
                    "is_ood": False,
                }
                if "evidence_clause" in q:
                    metadata["evidence_clause"] = q["evidence_clause"]

                yield UnifiedSample(
                    dataset_name="compliance_ja",
                    sample_id=f"{doc_id}_{q_id}",
                    question_type=q_type,
                    state=state_text,
                    instructions=q["instructions"],
                    criteria=q.get("criteria", {}),
                    target=str(q["target"]),
                    metadata=metadata,
                )
