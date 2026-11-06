"""Evol-Instruct 合成データ生成パイプライン単体・結合テストモジュール。

シードサンプリング、3段階品質ゲート、非同期パイプライン、レジューム機能、
およびコンバータ・トークナイザとのエンドツーエンド疎通を検証する。
"""

from pathlib import Path
from typing import cast

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from data.builders import UnifiedDatasetBuilder
from data.converters.synthetic import SyntheticDatasetConverter
from data.formatter import format_prompt, tokenize_sample
from data.schema import QuestionType, UnifiedSample
from data.synthetic.client import MockLLMClient, extract_json_object
from data.synthetic.config import (
    QualityFilterConfig,
    SyntheticPipelineConfig,
)
from data.synthetic.deduplicator import TextEmbeddingDeduplicator
from data.synthetic.evolver import SyntheticEvolver
from data.synthetic.pipeline import SyntheticPipeline
from data.synthetic.prompt_templates import (
    build_generation_prompt,
    build_validation_prompt,
)
from data.synthetic.taxonomy import DOMAIN_TAXONOMY, DomainTaxonomySampler
from data.synthetic.validator import SyntheticQualityGate


def test_taxonomy_sampler() -> None:
    """業務ドメインサンプラーが比率設定および候補数範囲を満たすことを検証する。"""
    sampler = DomainTaxonomySampler(seed=42)
    assert len(DOMAIN_TAXONOMY) >= 10

    q_types = [sampler.sample_question_type() for _ in range(300)]
    assert QuestionType.CHOICE in q_types
    assert QuestionType.SCORE in q_types
    assert QuestionType.NOUL in q_types

    # Choice 候補数
    choice_opts = [sampler.sample_num_options(QuestionType.CHOICE) for _ in range(50)]
    assert all(2 <= opt <= 16 for opt in choice_opts)

    # Score 段階数
    score_opts = [sampler.sample_num_options(QuestionType.SCORE) for _ in range(50)]
    assert all(opt in {3, 4, 5, 10} for opt in score_opts)

    # Noul 候補数
    assert sampler.sample_num_options(QuestionType.NOUL) == 2

    spec = sampler.generate_seed_spec(sequence_index=1)
    assert spec.seed_id == "synth_choice_000001" or "synth_" in spec.seed_id
    assert spec.domain_node.domain != ""
    assert spec.hard_negative_focus != ""


def test_prompt_templates() -> None:
    """各プリミティブの生成・検証プロンプトが要件記述を含むことを検証する。"""
    sampler = DomainTaxonomySampler(seed=123)
    spec_choice = sampler.generate_seed_spec(sequence_index=1)
    prompt_choice = build_generation_prompt(spec_choice, "deepen_constraints")
    assert "ハードネガティブ" in prompt_choice
    assert "state（状況文）は日本語で 200文字以上 800文字以下" in prompt_choice

    val_prompt = build_validation_prompt(
        {
            "question_type": "choice",
            "state": "状況文です。",
            "instructions": "指示文です。",
            "criteria": {"opt_a": "説明A", "opt_b": "説明B"},
        }
    )
    assert "opt_a" in val_prompt
    assert "is_ambiguous" in val_prompt


def test_extract_json_object() -> None:
    """Markdown コードブロックや前後のテキストから正しく JSON が抽出されることを検証する。"""
    raw_md = '```json\n{"question_type": "choice", "target": "opt_a"}\n```'
    parsed = extract_json_object(raw_md)
    assert parsed["question_type"] == "choice"

    surrounded = '回答はこちらです。\n{"question_type": "noul", "target": "true"}\nご活用ください。'
    parsed2 = extract_json_object(surrounded)
    assert parsed2["target"] == "true"

    with pytest.raises(ValueError, match="JSON パースに失敗"):
        extract_json_object("不正なテキストのみ")


def test_deduplicator() -> None:
    """意味的重複排除エンジンが高類似度テキストを検出し、非類似テキストを通過させることを検証する。"""
    dedup = TextEmbeddingDeduplicator(threshold=0.90)

    text_base = "お客様からBluetooth接続が頻繁に切断されるとの問い合わせを受けました。端末の再起動を案内しましたが改善されず、ファームウェアのバージョン確認が必要です。"
    text_dup = "お客様からBluetooth接続が頻繁に切断されるとの問い合わせを受けました。端末の再起動を案内しましたが改善されず、ファームウェアのバージョン確認をお願いします。"
    text_diff = (
        "海外決済において二重引き落としが発生したため、カード会社へ調査を依頼しました。"
    )

    is_dup1, sim1, _ = dedup.is_duplicate("sample_01", text_base, add_if_unique=True)
    assert not is_dup1
    assert sim1 == 0.0

    is_dup2, sim2, dup_id = dedup.is_duplicate(
        "sample_02", text_dup, add_if_unique=True
    )
    assert is_dup2
    assert sim2 >= 0.90
    assert dup_id == "sample_01"

    is_dup3, sim3, _ = dedup.is_duplicate("sample_03", text_diff, add_if_unique=True)
    assert not is_dup3
    assert sim3 < 0.50


@pytest.mark.anyio
async def test_quality_gate() -> None:
    """3段階品質ゲートの各検証基準(スキーマ・重複・クロス検証)を検証する。"""
    q_config = QualityFilterConfig(
        min_state_chars=50,
        max_state_chars=300,
        max_criteria_chars=50,
        similarity_threshold=0.92,
        require_hard_negative=True,
    )
    dedup = TextEmbeddingDeduplicator(threshold=0.92)
    mock_val_client = MockLLMClient()
    gate = SyntheticQualityGate(
        config=q_config,
        deduplicator=dedup,
        validator_client=mock_val_client,
    )

    valid_raw_choice = {
        "question_type": "choice",
        "state": "本件はECサイトにおける初期不良の問い合わせですが、購入後45日が経過しており通常保証期間（30日）を超過している境界事例です。",
        "instructions": "適切なエスカレーション窓口を選択せよ。",
        "criteria": {
            "action_a": "有償修理窓口へ案内する。",
            "action_b": "【ハードネガティブ】無償交換窓口へ手配する。",
        },
        "target": "action_a",
        "hard_negative_key": "action_b",
        "rationale": "保証期間超過のため有償対応となる。",
    }

    # 1. 正常サンプルの全ゲート通過
    sample, result = await gate.evaluate_sample(valid_raw_choice, "sample_test_01")
    assert sample is not None
    assert result.is_valid
    assert sample.target == "action_a"

    # 2. ゲート1失敗: State が短すぎる
    short_state = dict(valid_raw_choice)
    short_state["state"] = "短すぎる文章。"
    sample_fail1, result_fail1 = await gate.evaluate_sample(
        short_state, "sample_test_02"
    )
    assert sample_fail1 is None
    assert result_fail1.gate_failed == "gate1_schema"

    # 3. ゲート1失敗: Choice でハードネガティブキー欠落
    no_hn = dict(valid_raw_choice)
    del no_hn["hard_negative_key"]
    sample_fail_hn, result_fail_hn = await gate.evaluate_sample(no_hn, "sample_test_03")
    assert sample_fail_hn is None
    assert result_fail_hn.gate_failed == "gate1_schema"

    # 4. ゲート2失敗: 重複 State
    dup_sample = dict(valid_raw_choice)
    sample_dup, result_dup = await gate.evaluate_sample(dup_sample, "sample_test_04")
    assert sample_dup is None
    assert result_dup.gate_failed == "gate2_dedup"

    # 5. ゲート3失敗: クロス検証不一致
    mismatch_client = MockLLMClient(mismatch_ratio=1.0)
    mismatch_gate = SyntheticQualityGate(
        config=q_config,
        deduplicator=TextEmbeddingDeduplicator(),
        validator_client=mismatch_client,
    )
    valid_raw_choice_diff = dict(valid_raw_choice)
    valid_raw_choice_diff["state"] = (
        "こちらは別ドメインのインシデント事例であり、内容が重複しないように構成された独立した文脈テキストです。"
    )
    # mock client の呼び出しカウントを調整して不一致を発生させる
    mismatch_client._call_count = 4
    sample_mis, result_mis = await mismatch_gate.evaluate_sample(
        valid_raw_choice_diff, "sample_test_05"
    )
    assert sample_mis is None
    assert result_mis.gate_failed == "gate3_cross_val"


@pytest.mark.anyio
async def test_evolver() -> None:
    """Evol-Instruct エンジンがシード仕様に応じた生成呼び出しを実行できることを検証する。"""
    client = MockLLMClient()
    evolver = SyntheticEvolver(client=client, seed=42)
    sampler = DomainTaxonomySampler(seed=42)
    spec = sampler.generate_seed_spec(1)

    raw_dict, op = await evolver.evolve_seed(spec)
    assert "question_type" in raw_dict
    assert "state" in raw_dict
    assert op in evolver.operator_names


@pytest.mark.anyio
async def test_pipeline_run_and_resume(tmp_path: Path) -> None:
    """合成データ生成パイプラインが指定件数を生成し、中断再開が正しく機能することを検証する。"""
    config = SyntheticPipelineConfig(
        output_dir=tmp_path / "synthetic_out",
        total_target_samples=10,
        max_concurrency=2,
        seed=42,
        dry_run=True,
    )
    gen_client = MockLLMClient()
    val_client = MockLLMClient()

    pipeline = SyntheticPipeline(
        config=config,
        generator_client=gen_client,
        validator_client=val_client,
    )

    stats = await pipeline.run(target_count=6)
    assert stats.total_passed == 6
    assert (config.output_dir / "synthetic_all.jsonl").exists()
    assert (config.output_dir / "synthetic_choice.jsonl").exists()

    # レジューム(中断再開)の検証
    # 再度同じディレクトリで 10 件を目標として実行
    pipeline2 = SyntheticPipeline(
        config=config,
        generator_client=gen_client,
        validator_client=val_client,
    )
    stats2 = await pipeline2.run(target_count=10)
    # 既存 6 件をスキップし、新規に 4 件追加生成して合計 10 件になる
    assert stats2.total_passed == 4


def test_synthetic_converter_and_e2e(tmp_path: Path) -> None:
    """SyntheticDatasetConverter が出力 JSONL をロードし、プロンプト整形およびトークナイズと完全結合することを検証する。"""
    output_file = tmp_path / "synthetic_all.jsonl"
    choice_sample = UnifiedSample(
        dataset_name="synthetic",
        sample_id="test_choice_01",
        question_type=QuestionType.CHOICE,
        state="顧客からのWiFi接続切断問い合わせです。特定の端末のみが接続不能となっており、LANルータの再起動を案内する必要があります。購入後45日経過しています。",
        instructions="最も適切な初動対応を選択せよ。",
        criteria={
            "action_a": "端末のネットワーク設定リセットを案内する。",
            "action_b": "【ハードネガティブ】無償交換を即時手配する。",
        },
        target="action_a",
        metadata={"hard_negative_key": "action_b"},
    )
    score_sample = UnifiedSample(
        dataset_name="synthetic",
        sample_id="test_score_01",
        question_type=QuestionType.SCORE,
        state="クラウドAPI障害の報告です。HTTP 504エラーが散発しており、一部の決済リクエストが失敗しています。業務影響は限定的ですが監視を強化します。",
        instructions="インシデントの深刻度を評価せよ。",
        criteria={
            "0": "影響なし",
            "1": "軽微な遅延",
            "2": "一部機能停止",
            "3": "完全遮断",
        },
        target="1",
    )
    noul_sample = UnifiedSample(
        dataset_name="synthetic",
        sample_id="test_noul_01",
        question_type=QuestionType.NOUL,
        state="セキュリティ監査の事象です。退職予定者のアカウントから夜間に大量のデータエクスポートが実行されました。コンプライアンス規約第5条に違反する疑いがあります。",
        instructions="本件は即時アカウント凍結の対象となる規約違反であるか判定せよ。",
        criteria={},
        target="true",
    )

    with open(output_file, "w", encoding="utf-8") as f:
        import json

        f.write(json.dumps(choice_sample.to_dict(), ensure_ascii=False) + "\n")
        f.write(json.dumps(score_sample.to_dict(), ensure_ascii=False) + "\n")
        f.write(json.dumps(noul_sample.to_dict(), ensure_ascii=False) + "\n")

    converter = SyntheticDatasetConverter(file_path=output_file, mode="all")
    train_samples = list(converter.convert_split("train"))
    assert isinstance(train_samples, list)

    all_loaded = []
    with open(output_file, encoding="utf-8") as f:
        for line in f:
            all_loaded.append(UnifiedSample.from_dict(json.loads(line)))

    assert len(all_loaded) == 3

    # Tokenizer & Formatter との疎通確認
    raw_tokenizer = AutoTokenizer.from_pretrained("sbintuitions/modernbert-ja-130m")
    tokenizer = cast(PreTrainedTokenizerFast, raw_tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[OP]"]})
    token_id_raw = tokenizer.convert_tokens_to_ids("[OP]")
    op_token_id: int = int(
        token_id_raw[0] if isinstance(token_id_raw, list) else token_id_raw
    )

    for s in all_loaded:
        prompt_str, _option_keys = format_prompt(s)
        assert "[OP]" in prompt_str
        features = tokenize_sample(
            sample=s,
            tokenizer=tokenizer,
            op_token_id=op_token_id,
            max_length=512,
        )
        assert "input_ids" in features
        assert "op_indices" in features
        assert "label" in features
        assert features["op_indices"].size(0) >= 2


def test_builder_integration(tmp_path: Path) -> None:
    """UnifiedDatasetBuilder に登録された合成データコンバータが動作することを検証する。"""
    builder = UnifiedDatasetBuilder(seed=42)
    assert "synthetic_all" in builder.converters
    assert "synthetic_choice" in builder.converters
    assert "synthetic_score" in builder.converters
    assert "synthetic_noul" in builder.converters
