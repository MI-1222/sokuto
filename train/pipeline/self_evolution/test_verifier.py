"""Dual-LLM ブラインドクロスチェック検証モジュールのテスト。"""

import asyncio

from pipeline.self_evolution.log_store import HardSampleRecord
from pipeline.self_evolution.verifier import (
    DualLLMVerifier,
    IsolatedSample,
    MockLLMClient,
    SilverSample,
)


def test_blind_prompt_anchoring_prevention() -> None:
    """ブラインドプロンプトに System 1 の事前判定やロジットが一切含まれないことを検証する。"""
    verifier = DualLLMVerifier()
    record = HardSampleRecord(
        sample_id="sample_01",
        question_type="choice",
        state="契約文脈テキスト",
        instructions="違反があるか判定せよ",
        criteria={"0": "違反なし", "1": "軽微な違反", "2": "重大な違反"},
        escalation_reason="Pareto Top-Margin 不足: margin=0.08",
        system1_confidence=0.45,
        top_margin=0.08,
        normalized_entropy=0.82,
        free_energy=-0.5,
    )

    sys_p, user_p = verifier.build_blind_prompt(record)

    # System 1 の事前判定や不確実性スコアがプロンプトから完全に排除されていること
    assert "0.45" not in sys_p and "0.45" not in user_p
    assert "0.08" not in sys_p and "0.08" not in user_p
    assert "0.82" not in sys_p and "0.82" not in user_p
    assert "Pareto" not in user_p


def test_strict_match_promotes_to_silver() -> None:
    """両モデルの離散ラベルが完全一致したとき Silver Data へ昇格することを検証する。"""

    async def _run() -> None:
        client = MockLLMClient(default_label="1")
        verifier = DualLLMVerifier(client=client)

        record = HardSampleRecord(
            sample_id="sample_match_01",
            question_type="choice",
            state="テスト文脈",
            instructions="指示文",
            criteria={"0": "A", "1": "B"},
            escalation_reason="Entropy 超過",
            system1_confidence=0.5,
        )

        result = await verifier.verify_sample(record)

        assert isinstance(result, SilverSample)
        assert result.consensus_target == "1"
        assert result.soft_labels["1"] > 0.8
        assert "思考ログ" in result.primary_thinking

    asyncio.run(_run())


def test_mismatch_isolates_sample() -> None:
    """判定ラベル不一致時にサンプルが隔離されることを検証する。"""

    async def _run() -> None:
        client = MockLLMClient(
            default_label="0", mismatch_sample_ids={"sample_mismatch_02"}
        )
        verifier = DualLLMVerifier(client=client)

        record = HardSampleRecord(
            sample_id="sample_mismatch_02",
            question_type="choice",
            state="テスト文脈",
            instructions="指示文",
            criteria={"0": "A", "1": "B"},
            escalation_reason="Top-Margin 不足",
            system1_confidence=0.5,
        )

        result = await verifier.verify_sample(record)

        assert isinstance(result, IsolatedSample)
        assert result.sample_id == "sample_mismatch_02"
        assert "不一致" in result.isolation_reason

    asyncio.run(_run())
