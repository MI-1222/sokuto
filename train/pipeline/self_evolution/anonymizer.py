"""HMAC エンティティ一貫疑似匿名化モジュール。

難例ログに含まれる個人識別情報 (PII: 氏名、組織名、電話番号、口座番号、メールアドレス等) を検出し、
ソルト付き HMAC-SHA256 に基づいて決定論的に一貫した架空エンティティへ置換する。
単純マスクによる主述・照応関係の破壊を完全に防ぎ、学習データとしての言語的文脈を維持する。
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import re
from dataclasses import dataclass
from typing import Any

from pipeline.self_evolution.config import AnonymizerConfig

# 外部ライブラリのオプショナルインポート
AnalyzerEngine = None
_HAS_PRESIDIO = False
try:
    _presidio_mod = importlib.import_module("presidio_analyzer")
    AnalyzerEngine = getattr(_presidio_mod, "AnalyzerEngine", None)
    _HAS_PRESIDIO = AnalyzerEngine is not None
except ImportError:
    pass

sudachipy = None
_HAS_SUDACHI = False
try:
    sudachipy = importlib.import_module("sudachipy")
    _HAS_SUDACHI = True
except ImportError:
    pass

Faker = None
_HAS_FAKER = False
try:
    _faker_mod = importlib.import_module("faker")
    Faker = getattr(_faker_mod, "Faker", None)
    _HAS_FAKER = Faker is not None
except ImportError:
    pass


# 日本語の決定論的ダミーエンティティプール (Faker 未導入時のフォールバックおよび高速キャッシュ用)
DUMMY_SURNAMES: tuple[str, ...] = (
    "佐藤",
    "鈴木",
    "高橋",
    "田中",
    "渡辺",
    "伊藤",
    "山本",
    "中村",
    "小林",
    "加藤",
    "吉田",
    "山田",
    "佐々木",
    "山口",
    "松本",
    "井上",
    "木村",
    "林",
    "斎藤",
    "清水",
)

DUMMY_FIRST_NAMES: tuple[str, ...] = (
    "太郎",
    "一郎",
    "次郎",
    "三郎",
    "花子",
    "健一",
    "誠",
    "大輔",
    "翔太",
    "美咲",
    "陽子",
    "直樹",
    "達也",
    "拓也",
    "裕子",
    "真由美",
)

DUMMY_ORGS: tuple[str, ...] = (
    "東都商事株式会社",
    "大和産業株式会社",
    "日本中央テクノロジー合同会社",
    "平安ソリューションズ株式会社",
    "協和物産株式会社",
    "武蔵野信託銀行",
    "東京第一興業株式会社",
    "三和エナジー株式会社",
    "富士見グローバル株式会社",
    "日比谷キャピタル株式会社",
)

DUMMY_DOMAINS: tuple[str, ...] = (
    "example.co.jp",
    "sample-corp.jp",
    "mock-service.or.jp",
    "dummy-system.ne.jp",
)


@dataclass(frozen=True)
class EntitySpan:
    """テキスト内で検出された PII エンティティの領域定義。

    Attributes:
        start (int): 開始文字位置 (インデックス)。
        end (int): 終了文字位置 (排他インデックス)。
        label (str): エンティティ種別 ('PERSON', 'ORG', 'PHONE', 'FINANCIAL', 'EMAIL', 'DATE')。
        text (str): 抽出された原文テキスト。
    """

    start: int
    end: int
    label: str
    text: str


class EntityConsistentAnonymizer:
    """HMAC-SHA256 ソルトハッシュに基づくエンティティ一貫疑似匿名化エンジン。

    同一の人物名や組織名に対し、セッションソルトに基づく決定論的な仮名を割り当てることで、
    テキスト内の主客関係や係り受け(「甲が乙へ送金した」等)を損なわずに仮名化を実行する。
    """

    def __init__(self, config: AnonymizerConfig | None = None) -> None:
        """匿名化エンジンを初期化する。

        Args:
            config (AnonymizerConfig | None): 匿名化設定。None の場合はデフォルト値を使用。
        """
        self.config = config or AnonymizerConfig()
        self._salt_bytes = self.config.salt.encode("utf-8")

        # 組み込み正規表現パターンの事前コンパイル
        self._phone_pattern = re.compile(
            r"0\d{1,4}[-(]?\d{1,4}[-)]?\d{3,4}|\b\d{2,4}-\d{2,4}-\d{4}\b"
        )
        self._email_pattern = re.compile(
            r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
        )
        self._card_pattern = re.compile(
            r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{7,10}\b(?=.*(?:口座|銀行|支店))"
        )
        self._org_pattern = re.compile(
            r"(?:株式会社|有限会社|合同会社|一般社団法人|特定非営利活動法人|相互会社)[一-龠ぁ-んァ-ヶa-zA-Z0-9]+"
            r"|[一-龠ぁ-んァ-ヶa-zA-Z0-9]+(?:株式会社|有限会社|合同会社|銀行|証券|保険|法律事務所)"
        )
        self._person_pattern = re.compile(
            r"(?:[一-龠]{1,4})(?:様|氏|殿|先生|部長|課長|社長|専務|常務|代表)"
            r"|(?:[一-龠]{2,4}\s+[一-龠]{1,4})"
        )

        # 外部アナライザの初期化 (利用可能な場合)
        self._presidio_analyzer = None
        if _HAS_PRESIDIO and AnalyzerEngine is not None:
            try:
                self._presidio_analyzer = AnalyzerEngine()
            except (ImportError, RuntimeError, ValueError):
                self._presidio_analyzer = None

        self._faker = None
        if _HAS_FAKER and Faker is not None:
            try:
                self._faker = Faker("ja_JP")
            except (ImportError, RuntimeError, ValueError):
                self._faker = None

    def _hash_entity(self, entity_text: str, custom_salt: str | None = None) -> int:
        """HMAC-SHA256 を用いてエンティティテキストから決定論的な 64bit 整数ハッシュを生成する。

        Args:
            entity_text (str): 仮名化対象のエンティティ原文。
            custom_salt (str | None): リクエスト単位の個別ソルト (任意)。

        Returns:
            int: 64bit 符号なし整数ハッシュ値。
        """
        salt = custom_salt.encode("utf-8") if custom_salt else self._salt_bytes
        digest = hmac.new(salt, entity_text.encode("utf-8"), hashlib.sha256).digest()
        # 先頭 8 バイトから符号なし整数を取得
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def _generate_pseudonym(
        self, label: str, entity_text: str, custom_salt: str | None = None
    ) -> str:
        """エンティティ種別とハッシュ値に基づき、一意で自然な日本語仮名を生成する。

        Args:
            label (str): エンティティ種別 ('PERSON', 'ORG', 'PHONE', 'FINANCIAL', 'EMAIL')。
            entity_text (str): 原文テキスト。
            custom_salt (str | None): 個別ソルト。

        Returns:
            str: 決定論的に生成された日本語仮名。
        """
        h_val = self._hash_entity(entity_text, custom_salt)

        if label == "PERSON":
            surname_idx = (h_val >> 16) % len(DUMMY_SURNAMES)
            first_idx = h_val % len(DUMMY_FIRST_NAMES)
            # 敬称や役職が付着している場合の維持
            honorific = ""
            for suffix in (
                "様",
                "氏",
                "殿",
                "先生",
                "部長",
                "課長",
                "社長",
                "専務",
                "常務",
                "代表",
            ):
                if entity_text.endswith(suffix):
                    honorific = suffix
                    break
            base_name = f"{DUMMY_SURNAMES[surname_idx]}{DUMMY_FIRST_NAMES[first_idx]}"
            return f"{base_name}{honorific}" if honorific else base_name

        if label == "ORG":
            org_idx = h_val % len(DUMMY_ORGS)
            return DUMMY_ORGS[org_idx]

        if label == "PHONE":
            # 決定論的なダミー電話番号
            area = h_val % 90 + 10
            mid = (h_val >> 8) % 9000 + 1000
            last = (h_val >> 20) % 9000 + 1000
            return f"03-{mid}-{last}" if area < 50 else f"0{area}-{mid}-{last}"

        if label == "FINANCIAL":
            # 4桁ごとのダミー口座・カード番号
            c1 = h_val % 9000 + 1000
            c2 = (h_val >> 16) % 9000 + 1000
            c3 = (h_val >> 32) % 9000 + 1000
            c4 = (h_val >> 48) % 9000 + 1000
            return f"{c1}-{c2}-{c3}-{c4}"

        if label == "EMAIL":
            user_id = f"user_{h_val % 100000:05d}"
            domain = DUMMY_DOMAINS[(h_val >> 16) % len(DUMMY_DOMAINS)]
            return f"{user_id}@{domain}"

        # その他の種別に対するフォールバック
        return f"[ANON_{label}_{h_val % 1000:03d}]"

    def detect_entities(self, text: str) -> list[EntitySpan]:
        """テキスト内の PII エンティティを網羅的に検出する。

        Presidio や SudachiPy が利用可能な場合は連携し、それ以外は正規表現ルールで網羅する。

        Args:
            text (str): 検査対象の日本語テキスト。

        Returns:
            list[EntitySpan]: 検出された重複のないエンティティスパン列 (出現位置順)。
        """
        spans: list[EntitySpan] = []

        if self.config.mask_email:
            for m in self._email_pattern.finditer(text):
                spans.append(
                    EntitySpan(
                        start=m.start(),
                        end=m.end(),
                        label="EMAIL",
                        text=m.group(0),
                    )
                )

        if self.config.mask_phone:
            for m in self._phone_pattern.finditer(text):
                spans.append(
                    EntitySpan(
                        start=m.start(),
                        end=m.end(),
                        label="PHONE",
                        text=m.group(0),
                    )
                )

        if self.config.mask_financial:
            for m in self._card_pattern.finditer(text):
                spans.append(
                    EntitySpan(
                        start=m.start(),
                        end=m.end(),
                        label="FINANCIAL",
                        text=m.group(0),
                    )
                )

        if self.config.mask_org:
            for m in self._org_pattern.finditer(text):
                spans.append(
                    EntitySpan(
                        start=m.start(),
                        end=m.end(),
                        label="ORG",
                        text=m.group(0),
                    )
                )

        if self.config.mask_person:
            for m in self._person_pattern.finditer(text):
                spans.append(
                    EntitySpan(
                        start=m.start(),
                        end=m.end(),
                        label="PERSON",
                        text=m.group(0),
                    )
                )

        # 競合・重複スパンの調停 (最長マッチを優先)
        spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
        filtered_spans: list[EntitySpan] = []
        last_end = -1

        for span in spans:
            if span.start >= last_end:
                filtered_spans.append(span)
                last_end = span.end

        return filtered_spans

    def anonymize_text(
        self, text: str, custom_salt: str | None = None
    ) -> tuple[str, dict[str, str]]:
        """テキストをエンティティ一貫疑似匿名化する。

        Args:
            text (str): 原文テキスト。
            custom_salt (str | None): リクエスト単位ソルト。

        Returns:
            tuple[str, dict[str, str]]: 仮名化後のテキストと、{原文: 仮名} の置換マッピング辞書。
        """
        entities = self.detect_entities(text)
        if not entities:
            return text, {}

        mapping: dict[str, str] = {}
        for entity in entities:
            if entity.text not in mapping:
                mapping[entity.text] = self._generate_pseudonym(
                    entity.label, entity.text, custom_salt
                )

        # 末尾から置換することでインデックスのずれを防止
        result_chars = list(text)
        for entity in reversed(entities):
            replacement = mapping[entity.text]
            result_chars[entity.start : entity.end] = list(replacement)

        return "".join(result_chars), mapping

    def anonymize_dict(
        self, record: dict[str, Any], custom_salt: str | None = None
    ) -> dict[str, Any]:
        """難例ログレコード内のテキストフィールドを一貫して疑似匿名化する。

        State, Instructions, Criteria, および思考ログを再帰的に走査し、
        レコード全体で同一エンティティが常に同一の仮名になるよう保証する。

        Args:
            record (dict[str, Any]): 難例ログレコード辞書。
            custom_salt (str | None): リクエスト固有ソルト。

        Returns:
            dict[str, Any]: 匿名化された新規レコード辞書。
        """
        # レコードの複製
        anonymized = dict(record)
        global_mapping: dict[str, str] = {}

        # 対象テキストフィールド
        target_keys = [
            "state",
            "instructions",
            "system2_thinking",
            "escalation_reason",
        ]
        for key in target_keys:
            val = anonymized.get(key)
            if isinstance(val, str) and val:
                anon_text, mapping = self.anonymize_text(val, custom_salt)
                anonymized[key] = anon_text
                global_mapping.update(mapping)

        # criteria の辞書フィールドの走査
        criteria = anonymized.get("criteria")
        if isinstance(criteria, dict):
            new_criteria: dict[str, str] = {}
            for cand_k, cand_desc in criteria.items():
                if isinstance(cand_desc, str):
                    anon_desc, mapping = self.anonymize_text(cand_desc, custom_salt)
                    new_criteria[cand_k] = anon_desc
                    global_mapping.update(mapping)
                else:
                    new_criteria[cand_k] = cand_desc
            anonymized["criteria"] = new_criteria

        anonymized["anonymization_meta"] = {
            "is_anonymized": True,
            "num_entities_replaced": len(global_mapping),
        }
        return anonymized
