"""数理的複合損失エンジン (JevMultiTaskLoss) の包括的単体テスト。

Choice 型 (Label Smoothing + Focal), Score 型 (EMD / Wasserstein 順序尺度幾何),
Noul 型 (非対称 BCE), InfoNCE 対照損失、パディング不変性、および
異種タスク混在バッチにおける勾配健全性を網羅的に検証する。
"""

import pytest
import torch
from torch import nn

from data.schema import QuestionType
from models.decision_head import JevDecisionModel
from training.loss import (
    AsymmetricBCELoss,
    AsymmetricLoss,
    EarthMoverDistanceLoss,
    InfoNCEContrastiveLoss,
    JevMultiTaskLoss,
    LabelSmoothedFocalLoss,
    RankedProbabilityScoreLoss,
    SymmetryRegularizationLoss,
)


class DummyConfig:
    """テスト用モックバックボーン設定。"""

    def __init__(self, hidden_size: int = 32) -> None:
        """設定を初期化する。

        Args:
            hidden_size (int): 隠れ層次元数。
        """
        self.hidden_size = hidden_size


class DummyBackboneOutputs:
    """テスト用モックバックボーン出力。"""

    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        """出力を初期化する。

        Args:
            last_hidden_state (torch.Tensor): 隠れ状態テンソル。
        """
        self.last_hidden_state = last_hidden_state


class DummyBackbone(nn.Module):
    """テスト用モックバックボーン。"""

    def __init__(self, hidden_size: int = 32) -> None:
        """バックボーンを初期化する。

        Args:
            hidden_size (int): 隠れ層次元数。
        """
        super().__init__()
        self.config = DummyConfig(hidden_size=hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> DummyBackboneOutputs:
        """フォワードパスを実行する。

        Args:
            input_ids (torch.Tensor): 入力ID列。
            attention_mask (torch.Tensor): アテンションマスク。

        Returns:
            DummyBackboneOutputs: 隠れ状態を保持するモックオブジェクト。
        """
        batch_size, seq_len = input_ids.shape
        x = torch.zeros(
            (batch_size, seq_len, self.config.hidden_size),
            dtype=torch.float32,
            device=input_ids.device,
        )
        return DummyBackboneOutputs(last_hidden_state=self.proj(x))


def test_choice_label_smoothed_focal_loss() -> None:
    """Choice 型の動的ラベル平滑化および Focal Loss 変調の検証。"""
    # 3クラス分類のテスト
    loss_fn_ce = LabelSmoothedFocalLoss(label_smoothing=0.0, focal_gamma=0.0)
    loss_fn_ls = LabelSmoothedFocalLoss(label_smoothing=0.1, focal_gamma=0.0)
    loss_fn_focal = LabelSmoothedFocalLoss(label_smoothing=0.0, focal_gamma=2.0)

    logits = torch.tensor([[5.0, 1.0, 0.0]], dtype=torch.float32, requires_grad=True)
    label = torch.tensor([0], dtype=torch.long)

    loss_ce = loss_fn_ce(logits, label)
    loss_ls = loss_fn_ls(logits, label)
    loss_focal = loss_fn_focal(logits, label)

    # 容易な正解予測において、平滑化損失は過信ペナルティにより通常の CE より大きくなる
    assert loss_ls > loss_ce, (
        f"平滑化損失 ({loss_ls.item()}) は純粋な CE ({loss_ce.item()}) より大きくなるべきです。"
    )

    # 高確率で正解しているため、Focal Loss は変調係数 (1 - p)^2 により損失が大幅に抑制される
    assert loss_focal < loss_ce, (
        f"Focal 損失 ({loss_focal.item()}) は通常の CE ({loss_ce.item()}) より小さくなるべきです。"
    )

    # backward の健全性
    loss_ls.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_score_emd_loss_monotonicity() -> None:
    """Score 型 EMD 損失における順序幾何学の単調性検証 (最重要)。

    正解がスコア 2 のとき、予測が 1 の損失よりも 0 の損失の方が厳密に大きくなることを検証する。
    """
    loss_fn = EarthMoverDistanceLoss(power=2)
    label = torch.tensor([2], dtype=torch.long)

    # 5段階評価 (0, 1, 2, 3, 4)
    # 候補ごとのロジット分布を定義
    # 1. 完璧な正解予測 (スコア 2 に集中)
    logits_exact = torch.tensor([-5.0, -5.0, 10.0, -5.0, -5.0], dtype=torch.float32)
    # 2. 隣接した誤答 (スコア 1 に集中)
    logits_near = torch.tensor([-5.0, 10.0, -5.0, -5.0, -5.0], dtype=torch.float32)
    # 3. 離れた誤答 (スコア 0 に集中)
    logits_far = torch.tensor([10.0, -5.0, -5.0, -5.0, -5.0], dtype=torch.float32)

    loss_exact = loss_fn(logits_exact, label)
    loss_near = loss_fn(logits_near, label)
    loss_far = loss_fn(logits_far, label)

    # 厳密な単調性検証: loss_exact < loss_near < loss_far
    assert loss_exact < loss_near, (
        f"完全正解の損失 ({loss_exact.item()}) は隣接誤答 ({loss_near.item()}) より小さくなければなりません。"
    )
    assert loss_near < loss_far, (
        f"隣接誤答の損失 ({loss_near.item()}) は正反対の誤答 ({loss_far.item()}) より小さくなければなりません。"
    )

    # Smooth L1 (power=1) でも同様に単調性が保たれることの確認
    loss_fn_l1 = EarthMoverDistanceLoss(power=1)
    loss_near_l1 = loss_fn_l1(logits_near, label)
    loss_far_l1 = loss_fn_l1(logits_far, label)
    assert loss_near_l1 < loss_far_l1, (
        "Power=1 においても順序幾何学の単調性が成立していません。"
    )


def test_noul_asymmetric_bce_loss() -> None:
    """Noul 型二値クロスエントロピーの正解整合性および pos_weight の検証。

    formatter.py の規約: index 0 が True, index 1 が False。
    """
    loss_fn_symmetric = AsymmetricBCELoss(pos_weight=1.0)
    loss_fn_weighted = AsymmetricBCELoss(pos_weight=3.0)

    # ロジット: [z_true, z_false] = [3.0, -1.0] (True 判定)
    logits_true_pred = torch.tensor([3.0, -1.0], dtype=torch.float32)
    # ロジット: [z_true, z_false] = [-1.0, 3.0] (False 判定)
    logits_false_pred = torch.tensor([-1.0, 3.0], dtype=torch.float32)

    label_true = torch.tensor([0], dtype=torch.long)
    label_false = torch.tensor([1], dtype=torch.long)

    # 正しい判定時の損失は小さく、誤答時の損失は大きい (True 判定)
    loss_correct = loss_fn_symmetric(logits_true_pred, label_true)
    loss_wrong = loss_fn_symmetric(logits_false_pred, label_true)
    assert loss_correct < loss_wrong, (
        "True 予測時の正解損失が誤答損失を上回っています。"
    )

    # False 判定時も同様に正解時の損失が誤答時より小さいことの検証
    loss_false_correct = loss_fn_symmetric(logits_false_pred, label_false)
    loss_false_wrong = loss_fn_symmetric(logits_true_pred, label_false)
    assert loss_false_correct < loss_false_wrong, (
        "False 予測時の正解損失が誤答損失を上回っています。"
    )

    # pos_weight=3.0 の場合、正例 (True) の見逃しに対するペナルティが3倍に増幅されることの検証
    loss_wrong_weighted = loss_fn_weighted(logits_false_pred, label_true)
    assert loss_wrong_weighted > loss_wrong * 2.0, (
        f"重み付き損失 ({loss_wrong_weighted.item()}) は通常損失 ({loss_wrong.item()}) よりも十分に大きくなるべきです。"
    )


def test_infonce_contrastive_loss() -> None:
    """InfoNCE 対照損失の幾何学的分離性能の検証。"""
    loss_fn = InfoNCEContrastiveLoss(temperature=0.1)

    # 文脈ベクトル
    state_repr = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

    # 候補ベクトル: 候補0は state に極めて類似、候補1は直交、候補2は逆方向
    option_repr_aligned = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    label = torch.tensor([0], dtype=torch.long)
    loss_good = loss_fn(state_repr, option_repr_aligned, label)

    # 正解が候補2 (逆方向) の場合、対照損失は非常に大きくなる
    label_bad = torch.tensor([2], dtype=torch.long)
    loss_bad = loss_fn(state_repr, option_repr_aligned, label_bad)

    assert loss_good < loss_bad, (
        f"アラインした正解候補の対照損失 ({loss_good.item()}) は逆方向候補 ({loss_bad.item()}) より小さくなければなりません。"
    )


def test_padding_immunity() -> None:
    """パディングスロットのロジット値が損失値に一切影響を与えないことの検証 (完全遮断)。"""
    multitask_loss = JevMultiTaskLoss(
        label_smoothing=0.05,
        score_loss_power=2,
        noul_pos_weight=1.0,
    )

    # バッチサイズ 3: Choice (3択), Score (4段階), Noul (2択)
    # 最大候補数 6 でパディング
    op_mask = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
            [True, True, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    labels = torch.tensor([1, 2, 0], dtype=torch.long)
    question_types = [
        QuestionType.CHOICE.value,
        QuestionType.SCORE.value,
        QuestionType.NOUL.value,
    ]

    # 基準ロジット
    logits_base = torch.tensor(
        [
            [1.0, 2.0, 0.5, -100.0, -100.0, -100.0],
            [0.5, 1.0, 3.0, 0.2, -100.0, -100.0],
            [2.5, -0.5, -100.0, -100.0, -100.0, -100.0],
        ],
        dtype=torch.float32,
    )

    # パディングスロットに極端な正の値 (+999.0) や負の値 (-9999.0) を代入した汚染テンソル
    logits_polluted = logits_base.clone()
    logits_polluted[0, 3:] = 999.0
    logits_polluted[1, 4:] = -9999.0
    logits_polluted[2, 2:] = 1234.0

    loss_base, dict_base = multitask_loss(
        logits_base, labels, op_mask, question_types=question_types, return_dict=True
    )
    loss_polluted, dict_polluted = multitask_loss(
        logits_polluted,
        labels,
        op_mask,
        question_types=question_types,
        return_dict=True,
    )

    assert torch.isclose(loss_base, loss_polluted, atol=1e-5), (
        f"パディング汚染により総損失が変化しました: base={loss_base.item()}, polluted={loss_polluted.item()}。"
    )

    # 各タスク別損失も完全に一致することの確認
    for key in ["loss_choice", "loss_score", "loss_noul"]:
        assert pytest.approx(dict_base[key], abs=1e-5) == dict_polluted[key], (
            f"タスク別損失 {key} がパディング汚染の影響を受けています。"
        )


def test_heterogeneous_batch_multitask_backward() -> None:
    """異種タスク混在バッチにおける逆伝播と勾配の健全性検証。"""
    multitask_loss = JevMultiTaskLoss(
        label_smoothing=0.05,
        score_loss_power=2,
        noul_pos_weight=1.2,
        contrastive_weight=0.1,
    )

    logits = torch.randn((3, 5), dtype=torch.float32, requires_grad=True)
    labels = torch.tensor([0, 1, 0], dtype=torch.long)
    op_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, False, False, False],
        ],
        dtype=torch.bool,
    )
    question_types = ["choice", "score", "noul"]

    state_repr = torch.randn((3, 16), dtype=torch.float32, requires_grad=True)
    option_repr = torch.randn((3, 5, 16), dtype=torch.float32, requires_grad=True)

    loss, loss_dict = multitask_loss(
        logits=logits,
        labels=labels,
        op_mask=op_mask,
        question_types=question_types,
        state_repr=state_repr,
        option_repr=option_repr,
        return_dict=True,
    )

    assert loss_dict["loss_choice"] > 0.0
    assert loss_dict["loss_score"] > 0.0
    assert loss_dict["loss_noul"] > 0.0
    assert loss_dict["loss_contrast"] > 0.0

    loss.backward()

    # 勾配が正常に逆伝播し、NaN や Inf がないことの確認
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
    assert not torch.isinf(logits.grad).any()

    # パディングスロットの勾配は厳密に 0.0 であること (勾配漏洩の防止)
    assert (logits.grad[0, 3:] == 0.0).all()
    assert (logits.grad[1, 4:] == 0.0).all()
    assert (logits.grad[2, 2:] == 0.0).all()


def test_missing_task_batch_zero_division_guard() -> None:
    """バッチ内に特定のタスクしか含まれない場合のゼロ除算ガード検証。"""
    multitask_loss = JevMultiTaskLoss(
        choice_weight=1.0,
        score_weight=1.0,
        noul_weight=1.0,
    )

    # Choice のみを含むバッチ
    logits = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([1, 0], dtype=torch.long)
    op_mask = torch.tensor([[True, True], [True, True]], dtype=torch.bool)
    question_types = ["choice", "choice"]

    loss, loss_dict = multitask_loss(
        logits=logits,
        labels=labels,
        op_mask=op_mask,
        question_types=question_types,
        return_dict=True,
    )

    assert not torch.isnan(loss)
    assert loss_dict["loss_score"] == 0.0
    assert loss_dict["loss_noul"] == 0.0
    assert loss_dict["loss_choice"] > 0.0


def test_jev_decision_model_features_return() -> None:
    """JevDecisionModel.forward における return_features の動作検証。"""
    backbone = DummyBackbone(hidden_size=16)
    model = JevDecisionModel(backbone=backbone, mlp_hidden_size=16)

    input_ids = torch.randint(0, 100, (2, 8), dtype=torch.long)
    attention_mask = torch.ones((2, 8), dtype=torch.long)
    op_indices = torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.long)

    # 1. 通常呼び出し (logits のみ)
    logits = model(input_ids, attention_mask, op_indices)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 3)

    # 2. return_features=True 呼び出し
    outputs = model(input_ids, attention_mask, op_indices, return_features=True)
    assert isinstance(outputs, tuple)
    assert len(outputs) == 3
    logits_feat, state_repr, option_vectors = outputs

    assert logits_feat.shape == (2, 3)
    assert state_repr.shape == (2, 16)
    assert option_vectors.shape == (2, 3, 16)


def test_ranked_probability_score_loss() -> None:
    """RankedProbabilityScoreLoss (RPS) の順序距離単調性および計算整合性を検証する。"""
    loss_fn = RankedProbabilityScoreLoss()
    label = torch.tensor([2], dtype=torch.long)

    # 5段階評価: 完全一致、隣接誤答、遠隔誤答
    logits_exact = torch.tensor([-5.0, -5.0, 10.0, -5.0, -5.0], dtype=torch.float32)
    logits_near = torch.tensor([-5.0, 10.0, -5.0, -5.0, -5.0], dtype=torch.float32)
    logits_far = torch.tensor([10.0, -5.0, -5.0, -5.0, -5.0], dtype=torch.float32)

    loss_exact = loss_fn(logits_exact, label)
    loss_near = loss_fn(logits_near, label)
    loss_far = loss_fn(logits_far, label)

    # 厳密な単調性: loss_exact < loss_near < loss_far
    assert loss_exact < loss_near < loss_far

    # 確率入力 (is_probs=True) のテスト
    probs_exact = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    loss_prob_exact = loss_fn(probs_exact, label, is_probs=True)
    assert torch.isclose(loss_prob_exact, torch.tensor(0.0), atol=1e-6)


def test_asymmetric_loss_hard_thresholding() -> None:
    """AsymmetricLoss (ASL) における容易な負例ハードマージン切り捨ての検証。"""
    loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip_margin=0.05, eps=1e-8)

    label_negative = torch.tensor([1], dtype=torch.long)  # 負例 (False: target=0.0)

    # 容易な負例: delta_z = -10.0 -> p = sigmoid(-10) ≈ 4.5e-5 < clip_margin (0.05)
    # p_m = clamp(p - 0.05, min=0.0) = 0.0 となり、負例損失が完全に 0 になるはず
    logits_easy_neg = torch.tensor([[-10.0, 0.0]], dtype=torch.float32)
    loss_easy_neg = loss_fn(logits_easy_neg, label_negative)
    assert torch.isclose(loss_easy_neg, torch.tensor(0.0), atol=1e-6)

    # 境界付近の負例: delta_z = 0.0 -> p = 0.5 > 0.05 -> 損失が発生する
    logits_hard_neg = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    loss_hard_neg = loss_fn(logits_hard_neg, label_negative)
    assert loss_hard_neg > 0.0

    # 正例 (True: target=1.0) の損失
    label_positive = torch.tensor([0], dtype=torch.long)
    loss_pos = loss_fn(logits_hard_neg, label_positive)
    assert loss_pos > 0.0


def test_symmetry_regularization_loss() -> None:
    """SymmetryRegularizationLoss における置換不変性 KL 損失の計算整合性を検証する。"""
    loss_fn = SymmetryRegularizationLoss()

    orig_logits = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)
    perm_indices = torch.tensor([[2, 0, 1]], dtype=torch.long)
    op_mask = torch.tensor([[True, True, True]], dtype=torch.bool)

    # 完全に置換同変なロジットの場合、KL 損失は 0.0
    perm_logits_ideal = torch.tensor([[0.0, 2.0, 1.0]], dtype=torch.float32)
    loss_zero = loss_fn(orig_logits, perm_logits_ideal, perm_indices, op_mask)
    assert torch.isclose(loss_zero, torch.tensor(0.0), atol=1e-5)

    # 偏りのあるロジットの場合、KL 損失は正値
    perm_logits_biased = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float32)
    loss_biased = loss_fn(orig_logits, perm_logits_biased, perm_indices, op_mask)
    assert loss_biased > 0.0


def test_jev_multitask_loss_with_insent() -> None:
    """JevMultiTaskLoss に InSeNT 対照損失を統合した場合の複合損失計算を検証する。"""
    loss_fn = JevMultiTaskLoss(
        choice_weight=1.0,
        insent_weight=0.15,
        insent_temperature=0.05,
        insent_lambda_seq=0.2,
    )

    batch_size = 2
    max_options = 3
    num_chunks = 4
    hidden_size = 16

    logits = torch.randn(batch_size, max_options)
    labels = torch.tensor([0, 1], dtype=torch.long)
    op_mask = torch.ones(batch_size, max_options, dtype=torch.bool)

    query_embeddings = torch.randn(batch_size, hidden_size)
    chunk_embeddings = torch.randn(batch_size, num_chunks, hidden_size)
    target_chunk_indices = torch.tensor([1, 2], dtype=torch.long)
    chunk_mask = torch.ones(batch_size, num_chunks, dtype=torch.bool)

    # 質問表現未指定時は ValueError が発生することの検証 (ガード動作)
    with pytest.raises(ValueError, match="質問側の表現"):
        loss_fn(
            logits=logits,
            labels=labels,
            op_mask=op_mask,
            chunk_embeddings=chunk_embeddings,
            target_chunk_indices=target_chunk_indices,
            chunk_mask=chunk_mask,
        )

    # query_embeddings を指定して正常計算。
    total_loss, loss_dict = loss_fn(
        logits=logits,
        labels=labels,
        op_mask=op_mask,
        query_embeddings=query_embeddings,
        chunk_embeddings=chunk_embeddings,
        target_chunk_indices=target_chunk_indices,
        chunk_mask=chunk_mask,
        return_dict=True,
    )

    assert total_loss.item() > 0.0
    assert "loss_choice" in loss_dict
    assert "loss_insent" in loss_dict
    assert loss_dict["loss_insent"] > 0.0
    # InSeNT 損失が 0.15 倍されて加算されていることを確認
    assert torch.isclose(
        total_loss,
        torch.tensor(loss_dict["loss_choice"] + 0.15 * loss_dict["loss_insent"]),
        atol=1e-4,
    )
