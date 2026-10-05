"""LegalRikai (企業間契約書・利用規約) データセットコンバータモジュール。

契約書・利用規約等の長文実務ドキュメント (2,000〜8,192 トークン) を State とし、
条項境界 (第○条) の構文解析による chunk_spans 抽出、
法的リスク・管轄裁判所等の特定 (Choice 型)、および免責・義務条項の有無判定 (Noul 型) への射影を行う。
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

# ビルトイン契約書サンプル (企業間秘密保持契約書 NDA)
BUILTIN_LEGAL_CONTRACT_NDA = (
    "秘密保持契約書\n\n"
    "株式会社アルファ（以下「甲」という）と株式会社ベータ（以下「乙」という）とは、"
    "両社間における業務提携および新規共同事業の検討（以下「本取引」という）に関し、"
    "甲または乙が相手方に開示する秘密情報の取扱いについて、以下のとおり秘密保持契約（以下「本契約」という）を締結する。\n\n"
    "第1条（目的）\n"
    "本契約は、甲および乙が本取引を検討・実施するにあたり、相互に開示される秘密情報の保護および取扱いに関する条件を定めることを目的とする。\n\n"
    "第2条（秘密情報の定義）\n"
    "1. 本契約において「秘密情報」とは、甲または乙が開示にあたり秘密である旨を書面、電子メール等の電磁的方法により明示した技術上、営業上、財務上その他の業務に関する一切の情報をいう。\n"
    "2. 口頭または視覚的手段により開示された情報については、開示時に秘密である旨が告知され、かつ開示後14日以内に書面または電磁的方法により内容を要約して確認された場合に限り、秘密情報とみなす。\n"
    "3. 前二項の規定にかかわらず、次の各号の一に該当する情報については、秘密情報に含まれないものとする。\n"
    " (1) 開示を受けた時点で、既に公知であった情報\n"
    " (2) 開示を受けた時点で、既に自己が適法に保有していた情報\n"
    " (3) 開示を受けた後、自己の責によらず公知となった情報\n"
    " (4) 正当な権限を有する第三者から秘密保持義務を負うことなく適法に取得した情報\n"
    " (5) 相手方の秘密情報を利用することなく独自に開発した情報\n\n"
    "第3条（秘密保持義務および例外）\n"
    "1. 甲および乙は、相手方の事前の書面による承諾を得ることなく、相手方の秘密情報を第三者に開示または漏洩してはならない。\n"
    "2. 甲および乙は、相手方の秘密情報を、本取引の検討および遂行に必要な自己の役員および従業員（以下「対象者」という）にのみ開示できるものとする。"
    "この場合、甲および乙は、対象者に対して本契約と同等以上の厳格な秘密保持義務を課すものとする。\n"
    "3. 法令の定めに基づき、または裁判所、行政官庁その他の公的機関からの適法な命令・要請により秘密情報の開示を求められた場合、"
    "相手方は、必要最小限の範囲で当該秘密情報を開示することができる。ただし、当該要請を受けた当事者は、事前に（緊急やむを得ない場合は事後速やかに）その旨を相手方に書面で通知しなければならない。\n\n"
    "第4条（目的外使用の禁止）\n"
    "甲および乙は、相手方の事前の書面による承諾を得ることなく、相手方の秘密情報を本取引の検討および遂行以外のいかなる目的にも使用・複製してはならない。\n\n"
    "第5条（秘密情報の返還または破棄）\n"
    "本契約が終了した場合、または相手方から書面による要請があった場合、甲および乙は、相手方の指示に従い、秘密情報（その複製物を含む）を直ちに返還または自己の責任において完全に消去・破棄し、その旨の証明書を提出しなければならない。\n\n"
    "第6条（損害賠償および責任制限）\n"
    "1. 甲または乙が本契約の規定に違反し相手方に損害を与えた場合、故意または重大な過失がある場合を除き、損害賠償の累計上限額は直近1年間に両社間で支払われた対価の総額、または金1,000万円のいずれか低い額とする。\n"
    "2. 前項の規定にかかわらず、秘密情報の不正競争防止法違反を伴う意図的な漏洩・盗用については、前項の責任上限は適用されない。\n\n"
    "第7条（有効期間）\n"
    "本契約の有効期間は、契約締結日から2年間とする。ただし、第3条（秘密保持義務）および第6条（損害賠償）の効力は、本契約終了後もさらに3年間存続するものとする。\n\n"
    "第8条（反社会的勢力の排除）\n"
    "甲および乙は、自らまたはその役員が反社会的勢力に該当しないこと、および反社会的勢力と社会的に非難されるべき関係を有しないことを確約する。本条に違反した場合、相手方は何らの催告を要せず直ちに本契約を解除できる。\n\n"
    "第9条（準拠法および管轄裁判所）\n"
    "1. 本契約の解釈および適用は、日本法に準拠するものとする。\n"
    "2. 本契約に関して甲乙間に生じた一切の紛争については、東京地方裁判所を第一審の専属的合意管轄裁判所とする。\n"
)

# ビルトイン利用規約サンプル (SaaSクラウドサービス利用規約)
BUILTIN_LEGAL_TERMS_SAAS = (
    "SaaSプラットフォーム利用規約\n\n"
    "本利用規約（以下「本規約」といいます。）は、株式会社クラウド（以下「当社」といいます。）が提供する"
    "AI業務自動化プラットフォームサービス（以下「本サービス」といいます。）の利用条件を定めるものです。"
    "登録ユーザーの皆様（以下「ユーザー」といいます。）には、本規約に従って本サービスをご利用いただきます。\n\n"
    "第1条（適用および変更）\n"
    "1. 本規約は、本サービスの利用に関する当社とユーザーとの間の一切の関係に適用されます。\n"
    "2. 当社は、民法第548条の4の規定に基づき、事前に当社ウェブサイト上への掲載または電子メールによる通知をもって、"
    "ユーザーの個別合意を得ることなく本規約を変更できるものとします。変更後の規約は効力発生日から適用されます。\n\n"
    "第2条（利用登録および拒絶事由）\n"
    "1. 登録希望者が当社の定める方法によって利用登録を申請し、当社がこれを承認することによって、利用登録が完了します。\n"
    "2. 当社は、登録希望者が過去に規約違反等により処分を受けたことがある場合、または未成年者等で法定代理人の同意を得ていない場合、利用登録を拒否できるものとします。\n\n"
    "第3条（アカウント管理および不正利用）\n"
    "ユーザーは、自己の責任において本サービスのアカウント情報（IDおよびパスワード）を厳重に管理しなければなりません。"
    "当社は、登録されたアカウント情報を用いて行われた一切の操作について、当該アカウントを保有するユーザー自身の行為とみなします。\n\n"
    "第4条（禁止事項）\n"
    "ユーザーは、本サービスの利用にあたり、以下の行為を行ってはなりません。\n"
    " (1) 法令または公序良俗に違反する行為\n"
    " (2) 当社または第三者の知的財産権、名誉、プライバシーを侵害する行為\n"
    " (3) 本サービスのサーバーまたはネットワークの機能を破壊・妨害する行為（スクレイピング、過度なAPI連続リクエスト等）\n"
    " (4) リバースエンジニアリング、逆アセンブル等による本サービス基盤ソフトウェアの解析行為\n"
    " (5) 当社の事前の書面承諾を得ないアカウントの第三者への貸与、譲渡または再販売\n\n"
    "第5条（サービスの停止・中断）\n"
    "当社は、システムの定期保守、火災・停電等の不可抗力、または地震等の天災地変が発生した場合、"
    "ユーザーに事前通知することなく本サービスの全部または一部の提供を停止または中断することができます。"
    "これにより生じた損害について、当社は故意・重過失を除き一切の責任を負いません。\n\n"
    "第6条（保証の否認および免責事項）\n"
    "1. 当社は、本サービスに事実上または法律上の瑕疵（安全性、信頼性、正確性、完全性、有効性、特定目的適合性等を含む）がないことを明示的にも黙示的にも保証しません。\n"
    "2. 当社が損害賠償責任を負う場合であっても、その賠償範囲はユーザーに直接かつ現実に生じた通常損害に限定され、"
    "逸失利益、間接損害、特別損害については予見可能性の有無を問わず免責されるものとします。\n"
    "3. 前項の損害賠償額の上限は、当該損害の発生原因となった事由が生じた月において、ユーザーが当社に支払った月額利用料の直近3ヶ月分相当額を上限とします。\n\n"
    "第7条（契約解除および利用制限）\n"
    "ユーザーが第4条の禁止事項に該当した場合、当社は何らの催告を要せず、直ちにアカウントの凍結または本利用契約を解除できるものとします。\n\n"
    "第8条（準拠法および裁判管轄）\n"
    "本規約の準拠法は日本法とし、本サービスに関する一切の紛争については、当社の本店所在地を管轄する地方裁判所（東京地方裁判所）を第一審の専属的合意管轄裁判所とします。\n"
)


def parse_contract_clause_spans(text: str) -> list[tuple[int, int]]:
    """契約書テキストから「第○条」「第○条の○」の条項開始・終了文字スパンを抽出する。

    行頭に配置された条項見出しのみを検出し、本文中の引用を除外する。
    また「第5条の2」のような枝番条項にも対応する。

    Args:
        text (str): 契約書テキスト。

    Returns:
        list[tuple[int, int]]: 各条項の文字スパン (start_char, end_char) のリスト。
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
        # 条文ヘッダー文字自体の開始位置 (改行や空白を除外)
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


class LegalRikaiConverter(BaseDatasetConverter):
    """LegalRikai (企業間契約書・利用規約) データセットコンバータ。

    長文契約書 (2,000〜8,192 トークン) を State とし、
    条項構文解析による chunk_spans の追跡、
    法的リスク・管轄裁判所等の特定 (Choice 型)、
    および免責・義務等の有無判定 (Noul 型) への標準化変換を行う。

    Attributes:
        file_path (Path | None): 外部契約書データファイルパス。
        mode (Literal['both', 'choice', 'noul']): 出力プリミティブ種別。
        seed (int): 乱数シード。
    """

    def __init__(
        self,
        file_path: str | Path | None = None,
        mode: Literal["both", "choice", "noul"] = "both",
        seed: int = 42,
    ) -> None:
        """LegalRikai コンバータを初期化する。

        Args:
            file_path (str | Path | None): 契約書 JSONL またはテキストファイルパス。
            mode (Literal['both', 'choice', 'noul']): 出力モード。
            seed (int): 乱数シード。
        """
        super().__init__(seed=seed)
        self.file_path = Path(file_path) if file_path is not None else None
        self.mode = mode

    def _load_documents(self) -> list[dict[str, Any]]:
        """外部ファイルまたは組み込み契約書コーパスからドキュメントをロードする。

        Returns:
            list[dict[str, Any]]: 契約書・規約データ辞書列。
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
                    "LegalRikai ファイルから %d 件をロードしました: %s",
                    len(documents),
                    self.file_path,
                )
                return documents
            except (json.JSONDecodeError, OSError, ValueError) as e:
                logger.warning(
                    "LegalRikai ファイルのパースに失敗しました (%s)。ビルトイン契約書を使用します。",
                    e,
                )

        # ビルトイン契約書ドキュメントと典型的な設問定義
        nda_spans = parse_contract_clause_spans(BUILTIN_LEGAL_CONTRACT_NDA)
        saas_spans = parse_contract_clause_spans(BUILTIN_LEGAL_TERMS_SAAS)

        return [
            {
                "doc_id": "legal_nda_alpha_beta",
                "title": "秘密保持契約書 (NDA)",
                "contract_type": "NDA",
                "text": BUILTIN_LEGAL_CONTRACT_NDA,
                "clause_spans": nda_spans,
                "questions": [
                    {
                        "q_id": "nda_choice_jurisdiction",
                        "type": "choice",
                        "instructions": "本秘密保持契約において、紛争発生時の専属的合意管轄裁判所として規定されている裁判所を選択せよ。",
                        "criteria": {
                            "opt_tokyo": "東京地方裁判所",
                            "opt_osaka": "大阪地方裁判所",
                            "opt_kyoto": "京都地方裁判所",
                            "opt_arbitration": "日本商事仲裁協会（JCAA）の仲裁判断による",
                            "none": "専属的合意管轄の明記なし",
                        },
                        "target": "opt_tokyo",
                        "evidence_clause": "第9条",
                    },
                    {
                        "q_id": "nda_choice_liability_cap",
                        "type": "choice",
                        "instructions": "本契約第6条における損害賠償責任の累計上限額として正しいものを選択せよ（故意・重過失の場合を除く）。",
                        "criteria": {
                            "opt_unlimited": "上限なし（全額賠償義務）",
                            "opt_1000m_or_paid": "直近1年間の対価総額、または金1,000万円のいずれか低い額",
                            "opt_paid_only": "直近1年間に支払われた対価の総額",
                            "opt_500m": "一律金500万円を上限とする",
                            "none": "責任限定条項は定められていない",
                        },
                        "target": "opt_1000m_or_paid",
                        "evidence_clause": "第6条",
                    },
                    {
                        "q_id": "nda_noul_oral_disclosure",
                        "type": "noul",
                        "instructions": "本契約において、口頭で開示された情報であっても、開示時に秘密である旨が告知され、開示後14日以内に書面等で要約確認された場合は秘密情報として保護されるか。",
                        "target": "true",
                        "evidence_clause": "第2条",
                    },
                    {
                        "q_id": "nda_noul_court_order",
                        "type": "noul",
                        "instructions": "公的機関または裁判所からの適法な命令により秘密情報の開示を求められた場合であっても、相手方の事前書面承諾を得られない限り開示は一切免責されないか。",
                        "target": "false",
                        "evidence_clause": "第3条",
                    },
                ],
            },
            {
                "doc_id": "legal_saas_platform_terms",
                "title": "SaaSプラットフォーム利用規約",
                "contract_type": "TermsOfService",
                "text": BUILTIN_LEGAL_TERMS_SAAS,
                "clause_spans": saas_spans,
                "questions": [
                    {
                        "q_id": "saas_choice_liability_scope",
                        "type": "choice",
                        "instructions": "本利用規約において、当社が損害賠償責任を負う場合の損害範囲に関する規定として正しいものを選択せよ。",
                        "criteria": {
                            "opt_direct_normal": "直接かつ現実に生じた通常損害に限定され、逸失利益・間接損害は免責される",
                            "opt_all_including_lost_profits": "逸失利益および特別損害を含め、発生した全損害を賠償する",
                            "opt_completely_exempt": "いかなる損害についても一切の責任を負わない完全免責",
                            "none": "賠償範囲に関する限定規定は存在しない",
                        },
                        "target": "opt_direct_normal",
                        "evidence_clause": "第6条",
                    },
                    {
                        "q_id": "saas_noul_terms_change",
                        "type": "noul",
                        "instructions": "当社は、民法第548条の4に基づき、事前通知を行うことでユーザーの個別合意を得ることなく本規約を変更することができるか。",
                        "target": "true",
                        "evidence_clause": "第1条",
                    },
                    {
                        "q_id": "saas_noul_reverse_eng",
                        "type": "noul",
                        "instructions": "ユーザーがソフトウェアの解析を目的としてリバースエンジニアリングを行う行為は、本規約上禁止事項に該当するか。",
                        "target": "true",
                        "evidence_clause": "第4条",
                    },
                ],
            },
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
            doc_id = doc.get("doc_id", f"legal_doc_{doc_idx}")
            state_text = doc["text"]
            clause_spans = doc.get("clause_spans") or parse_contract_clause_spans(
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

                if q_type_str == "choice":
                    if self.mode not in ("both", "choice"):
                        continue
                    q_type = QuestionType.CHOICE
                elif q_type_str == "noul":
                    if self.mode not in ("both", "noul"):
                        continue
                    q_type = QuestionType.NOUL
                else:
                    continue

                metadata: dict[str, Any] = {
                    "split": split,
                    "state_id": doc_id,
                    "doc_title": doc.get("title", ""),
                    "contract_type": doc.get("contract_type", "General"),
                    "chunk_spans": clause_spans,
                    "char_chunk_spans": clause_spans,
                    "is_ood": False,
                }
                if "evidence_clause" in q:
                    metadata["evidence_clause"] = q["evidence_clause"]

                yield UnifiedSample(
                    dataset_name="legal_rikai",
                    sample_id=f"{doc_id}_{q_id}",
                    question_type=q_type,
                    state=state_text,
                    instructions=q["instructions"],
                    criteria=q.get("criteria", {}),
                    target=str(q["target"]),
                    metadata=metadata,
                )
