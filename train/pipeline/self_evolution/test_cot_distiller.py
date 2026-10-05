"""CoT 前提抽出 & DAG 射影蒸留モジュールのテスト。"""

from data.schema import QuestionType
from pipeline.self_evolution.cot_distiller import CoTDistiller
from pipeline.self_evolution.verifier import SilverSample


def test_cot_extraction_and_dag_projection() -> None:
    """思考ログから前提命題が抽出され、DAG ノードおよび UnifiedSample へ分解されることを検証する。"""
    distiller = CoTDistiller()

    sample = SilverSample(
        sample_id="cot_test_01",
        question_type="choice",
        state="ユーザーは契約から10日後に返品を希望している。商品は未開封である。",
        instructions="返品可能か判定してください。",
        criteria={"accept": "返品を承認する", "reject": "返品を却下する"},
        consensus_target="accept",
        soft_labels={"accept": 0.95, "reject": 0.05},
        primary_thinking=(
            "ステップ1: 購入から14日以内であるという前提条件に該当する。\n"
            "ステップ2: 商品が未開封であるという要件を満たす。\n"
            "結論: 返品承認基準を満たすため accept を選択する。"
        ),
        secondary_thinking=(
            "- 条件1: 到着後14日以内のクーリングオフ期間内である。\n"
            "- 条件2: 再販売可能な状態である。\n"
            "判定: accept"
        ),
    )

    projection = distiller.project_to_dag(sample)

    assert projection.parent_sample_id == "cot_test_01"
    # 前提マイクロ決定が抽出されていること
    assert len(projection.micro_decisions) >= 1
    for node in projection.micro_decisions:
        assert node.question_type == QuestionType.NOUL
        assert node.target in ("true", "false")

    assert projection.final_decision is not None
    assert projection.final_decision.target == "accept"

    # UnifiedSample への変換
    unified_list = distiller.to_unified_samples(projection, sample.state)
    assert len(unified_list) == len(projection.micro_decisions) + 1

    # スキーマの妥当性確認
    for u in unified_list:
        assert u.state == sample.state
        assert u.sample_id.startswith("cot_test_01")
        if u.question_type == QuestionType.NOUL:
            # Noul プリミティブは criteria が空辞書であること
            assert u.criteria == {}


def test_tagged_premises_extraction() -> None:
    """<premises> 構造化タグから前提条件と真偽値が決定論的に抽出されることを検証する。"""
    distiller = CoTDistiller()

    sample = SilverSample(
        sample_id="cot_test_tagged",
        question_type="choice",
        state="顧客は解約を希望している。",
        instructions="解約可否を判定せよ。",
        criteria={"allow": "許可", "deny": "不許可"},
        consensus_target="allow",
        soft_labels={"allow": 0.9},
        primary_thinking=(
            "推論開始。\n"
            "<premises>\n"
            "- [TRUE] 最低利用期間が満了している\n"
            "- [FALSE] 未払いの請求残高が存在する\n"
            "</premises>\n"
            "以上より allow と判定。"
        ),
        secondary_thinking="同意する。",
    )

    projection = distiller.project_to_dag(sample)

    assert len(projection.micro_decisions) == 2
    # 1つ目: [TRUE] -> target="true"
    assert projection.micro_decisions[0].target == "true"
    assert "最低利用期間" in projection.micro_decisions[0].instruction
    assert projection.micro_decisions[0].criteria == {}

    # 2つ目: [FALSE] -> target="false"
    assert projection.micro_decisions[1].target == "false"
    assert "未払い" in projection.micro_decisions[1].instruction
    assert projection.micro_decisions[1].criteria == {}
