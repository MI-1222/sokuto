"""HMAC エンティティ一貫疑似匿名化モジュールのテスト。"""

from pipeline.self_evolution.anonymizer import (
    AnonymizerConfig,
    EntityConsistentAnonymizer,
)


def test_entity_detection_and_consistency() -> None:
    """同一人物名・組織名が決定論的に同一の仮名へ一貫して置換されることを検証する。"""
    anonymizer = EntityConsistentAnonymizer(
        AnonymizerConfig(salt="test-secret-salt-1234")
    )

    text = (
        "東都商事株式会社の代表取締役である佐藤太郎氏は、"
        "取引先である佐藤太郎氏の個人口座へ送金を行った。"
        "連絡先は 03-1234-5678、メールは satoh@example.com である。"
    )

    anon_text, mapping = anonymizer.anonymize_text(text)

    # 原文の機密情報が含まれていないことを確認
    assert "03-1234-5678" not in anon_text
    assert "satoh@example.com" not in anon_text

    # 同一人物名「佐藤太郎氏」が同一の仮名へ一貫して置換されていることを確認
    # mapping に登録されており、置換後のテキスト内でも置換先が複数回出現すること
    assert len(mapping) >= 3
    for orig, pseudonym in mapping.items():
        assert orig not in anon_text
        assert pseudonym in anon_text


def test_anonymize_dict_record() -> None:
    """難例ログレコード辞書の全テキストフィールドが一貫して仮名化されることを検証する。"""
    anonymizer = EntityConsistentAnonymizer()
    record = {
        "sample_id": "test_rec_001",
        "state": "山田花子様は株式会社テストへ返金を要求した。",
        "instructions": "山田花子様の要求が利用規約に合致するか判定せよ。",
        "criteria": {
            "accept": "山田花子様の要求を承認する。",
            "reject": "山田花子様の要求を却下する。",
        },
        "escalation_reason": "山田花子様の取引履歴に不審点あり。",
    }

    anon_record = anonymizer.anonymize_dict(record, custom_salt="test_rec_001")

    # 全フィールドで「山田花子」が同一の仮名に置換されていること
    assert "山田花子" not in anon_record["state"]
    assert "山田花子" not in anon_record["instructions"]
    assert "山田花子" not in anon_record["criteria"]["accept"]
    assert "山田花子" not in anon_record["criteria"]["reject"]
    assert "山田花子" not in anon_record["escalation_reason"]

    # 匿名化メタデータが付与されていること
    assert anon_record["anonymization_meta"]["is_anonymized"] is True
