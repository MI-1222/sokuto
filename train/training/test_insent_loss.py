"""InSeNT (In-Sequence Negative Training) 対照損失の単体テスト。

In-Sequence 対照損失 (同一文書内 Hard Negatives)、In-Batch 対照損失 (他文書チャンク)、
線形結合 (lambda_seq = 0.2)、マスク処理、勾配逆伝播、および境界エッジケースを検証する。
"""

import torch

from training.insent_loss import InSeNTLoss


def test_insent_loss_basic_and_gradient() -> None:
    """InSeNTLoss の基本順伝播、メトリクス辞書返却、および勾配逆伝播を検証する。"""
    batch_size = 3
    num_chunks = 5
    hidden_size = 16

    loss_fn = InSeNTLoss(temperature=0.05, lambda_seq=0.2)

    query_embeddings = torch.randn(batch_size, hidden_size, requires_grad=True)
    chunk_embeddings = torch.randn(
        batch_size, num_chunks, hidden_size, requires_grad=True
    )
    target_chunk_indices = torch.tensor([1, 0, 3], dtype=torch.long)
    chunk_mask = torch.ones(batch_size, num_chunks, dtype=torch.bool)
    chunk_mask[0, 4] = False  # サンプル0の末尾チャンクを無効化

    loss, metrics = loss_fn(
        query_embeddings=query_embeddings,
        chunk_embeddings=chunk_embeddings,
        target_chunk_indices=target_chunk_indices,
        chunk_mask=chunk_mask,
        return_dict=True,
    )

    # 損失が正の有限値であること
    assert loss.item() > 0.0
    assert torch.isfinite(loss)
    assert "loss_insent" in metrics
    assert "loss_seq" in metrics
    assert "loss_batch" in metrics

    # 勾配逆伝播の検証
    loss.backward()
    assert query_embeddings.grad is not None
    assert chunk_embeddings.grad is not None
    assert torch.isfinite(query_embeddings.grad).all()
    assert torch.isfinite(chunk_embeddings.grad).all()


def test_insent_loss_minimization_on_identical_vectors() -> None:
    """質問と正解チャンクが同一で、負例が直交している場合に損失が極小化することを検証する。"""
    loss_fn = InSeNTLoss(temperature=0.05, lambda_seq=0.2)

    # 直交基底ベクトルを作成
    e0 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    e1 = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    e2 = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    e3 = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    # 質問 0 は e0, 質問 1 は e2
    query_embeddings = torch.stack([e0, e2])  # [2, 8]

    # サンプル0のチャンク: [e0 (正解), e1 (負例)]
    # サンプル1のチャンク: [e2 (正解), e3 (負例)]
    chunk_embeddings = torch.stack(
        [
            torch.stack([e0, e1]),
            torch.stack([e2, e3]),
        ]
    )  # [2, 2, 8]

    target_chunk_indices = torch.tensor([0, 0], dtype=torch.long)

    loss_optimal = loss_fn(
        query_embeddings=query_embeddings,
        chunk_embeddings=chunk_embeddings,
        target_chunk_indices=target_chunk_indices,
    )

    # 逆転させた(不正解の)配置の場合の損失を計算
    wrong_targets = torch.tensor([1, 1], dtype=torch.long)
    loss_wrong = loss_fn(
        query_embeddings=query_embeddings,
        chunk_embeddings=chunk_embeddings,
        target_chunk_indices=wrong_targets,
    )

    # 正解ベクトルの損失は不正解ベクトルよりも劇的に小さいはず
    assert loss_optimal.item() < loss_wrong.item()


def test_insent_loss_single_batch_edge_case() -> None:
    """バッチサイズ 1 の場合、In-Batch 損失なしで In-Sequence 損失のみで正常動作することを検証する。"""
    loss_fn = InSeNTLoss(temperature=0.05, lambda_seq=0.2)
    query = torch.randn(1, 16)
    chunks = torch.randn(1, 4, 16)
    targets = torch.tensor([2], dtype=torch.long)

    loss, metrics = loss_fn(
        query_embeddings=query,
        chunk_embeddings=chunks,
        target_chunk_indices=targets,
        return_dict=True,
    )

    assert loss.item() > 0.0
    assert metrics["loss_seq"] > 0.0
    assert metrics["loss_batch"] == 0.0


def test_insent_loss_single_chunk_edge_case() -> None:
    """同一文書内に 1 チャンクしかない場合、In-Sequence 損失なしで In-Batch 損失のみで正常動作することを検証する。"""
    loss_fn = InSeNTLoss(temperature=0.05, lambda_seq=0.2)
    query = torch.randn(2, 16)
    chunks = torch.randn(2, 1, 16)
    targets = torch.tensor([0, 0], dtype=torch.long)

    loss, metrics = loss_fn(
        query_embeddings=query,
        chunk_embeddings=chunks,
        target_chunk_indices=targets,
        return_dict=True,
    )

    assert loss.item() > 0.0
    assert metrics["loss_seq"] == 0.0
    assert metrics["loss_batch"] > 0.0
