"""長文実務ドキュメント条項依存関係 DAG 抽出モジュール。

契約書や就業規則、法令ガイドラインの長文テキストから条項 (Clause) を分解し、
原則、例外、但し書き、手続要件などの条項間依存関係を有向非巡回グラフ (DAG) として抽出する。
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ClauseRelationType(str, Enum):
    """条項間関係性列挙型。

    PRINCIPLE: 原則規定。
    EXCEPTION: 例外・但書規定。
    PROCEDURE: 手続・通知要件。
    EXCLUSION: 適用除外。
    REFERENCE: 単純参照・準用。
    """

    PRINCIPLE = "principle"
    EXCEPTION = "exception"
    PROCEDURE = "procedure"
    EXCLUSION = "exclusion"
    REFERENCE = "reference"


@dataclass
class ClauseNode:
    """条項ノード。

    Attributes:
        clause_id (str): 条項識別子 (例: '第3条', '第6条第2項')。
        title (str): 条項見出し。
        body (str): 条文本文。
        char_span: (start_char, end_char) のテキスト範囲。
        metadata (dict[str, Any]): 追加メタデータ。
    """

    clause_id: str
    title: str
    body: str
    char_span: tuple[int, int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClauseEdge:
    """条項間依存関係エッジ。

    Attributes:
        source_id (str): 依存元条項 ID。
        target_id (str): 依存先条項 ID。
        relation_type (ClauseRelationType): 関係種別。
        condition_text (str): 例外や要件を規定する文言抜粋。
    """

    source_id: str
    target_id: str
    relation_type: ClauseRelationType
    condition_text: str = ""


@dataclass
class ClauseDAG:
    """条項依存関係有向非巡回グラフ。

    Attributes:
        doc_id (str): 対象ドキュメント ID。
        nodes (dict[str, ClauseNode]): 条項ノード辞書。
        edges (list[ClauseEdge]): 依存関係エッジリスト。
    """

    doc_id: str
    nodes: dict[str, ClauseNode] = field(default_factory=dict)
    edges: list[ClauseEdge] = field(default_factory=list)

    def get_related_clauses(self, clause_id: str) -> list[str]:
        """指定条項に直接または間接に関係する条項 ID リストを取得する。

        Args:
            clause_id (str): 基準条項 ID。

        Returns:
            list[str]: 関連条項 ID のリスト。
        """
        related = set()
        for edge in self.edges:
            if edge.source_id == clause_id:
                related.add(edge.target_id)
            elif edge.target_id == clause_id:
                related.add(edge.source_id)
        return list(related)

    def find_multi_hop_paths(self, min_hops: int = 2) -> list[list[str]]:
        """複数条項が連鎖する Multi-hop 経路 (例: 原則 -> 例外 -> 除外) を探索する。

        Args:
            min_hops (int): 最小ホップ数 (条項数 = ホップ数 + 1)。

        Returns:
            list[list[str]]: 条項 ID 列のリスト。
        """
        adj: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            adj[edge.source_id].append(edge.target_id)

        paths: list[list[str]] = []

        def dfs(current: str, path: list[str]) -> None:
            if len(path) >= min_hops + 1:
                paths.append(list(path))
            if len(path) > 4:  # 深すぎる探索を打ち切り
                return

            for nxt in adj.get(current, []):
                if nxt not in path:
                    path.append(nxt)
                    dfs(nxt, path)
                    path.pop()

        for start_node in self.nodes:
            dfs(start_node, [start_node])

        return paths


class ClauseDAGExtractor:
    """契約書・規程テキストから条項依存関係 DAG を抽出する抽出器。

    ルールベースの構文解析パターン(「第○条」「ただし」「前項にかかわらず」等のキーワード抽出)と
    構造グラフ構築を提供する。
    """

    def extract_dag(
        self,
        document_text: str,
        doc_id: str = "doc_default",
    ) -> ClauseDAG:
        """長文テキストから条項ノードおよび依存関係エッジを抽出して DAG を構築する。

        Args:
            document_text (str): 契約書・規程等の長文テキスト。
            doc_id (str): ドキュメント ID。

        Returns:
            ClauseDAG: 抽出された条項依存関係グラフ。
        """
        dag = ClauseDAG(doc_id=doc_id)

        # 1. 条項見出しノードの抽出
        # 「第○条(見出し)」「第○条の○(見出し)」または「第○条　本文」などの見出し行のみにマッチさせ、
        # 「第○条の規定にかかわらず」などの本文内条文引用 (助詞が続くもの) を除外する。
        clause_pattern = re.compile(
            r"(?:^|\n)[ \t]*(第[0-9一二三四五六七八九十百]+条(?:の[0-9一二三四五六七八九十百]+)?(?:\s*（([^）\n]+)）|[\s　]+|$))"
        )
        matches = list(clause_pattern.finditer(document_text))

        if not matches:
            # 条項ヘッダーが見つからない場合は単一ノードとして登録
            dag.nodes["第1条"] = ClauseNode(
                clause_id="第1条",
                title="本文",
                body=document_text,
                char_span=(0, len(document_text)),
            )
            return dag

        for idx, m in enumerate(matches):
            clause_full = m.group(1).strip()
            id_match = re.search(
                r"第[0-9一二三四五六七八九十百]+条(?:の[0-9一二三四五六七八九十百]+)?",
                clause_full,
            )
            clause_id = (
                id_match.group(0)
                if id_match is not None
                else clause_full.split("（")[0].strip()
            )
            title = m.group(2) or ""
            # 条文ヘッダー文字自体の開始位置 (改行や空白を除いた位置)
            start = document_text.find(clause_id, m.start())
            if start == -1:
                start = m.start()

            if idx + 1 < len(matches):
                next_full = matches[idx + 1].group(1).strip()
                next_match = re.search(
                    r"第[0-9一二三四五六七八九十百]+条(?:の[0-9一二三四五六七八九十百]+)?",
                    next_full,
                )
                next_id = (
                    next_match.group(0)
                    if next_match is not None
                    else next_full.split("（")[0].strip()
                )
                end = document_text.find(next_id, matches[idx + 1].start())
                if end == -1:
                    end = matches[idx + 1].start()
            else:
                end = len(document_text)

            body = document_text[start:end].strip()

            dag.nodes[clause_id] = ClauseNode(
                clause_id=clause_id,
                title=title,
                body=body,
                char_span=(start, end),
            )

        # 2. 条項間依存エッジの抽出 (ルールベース)
        # 他条項への参照(例: 「第○条の規定にかかわらず」「第○条の2第△項に準じ」など)
        ref_pattern = re.compile(
            r"(第[0-9一二三四五六七八九十百]+条(?:の[0-9一二三四五六七八九十百]+)?)"
        )

        for source_id, node in dag.nodes.items():
            # 本文内の参照条項を探索
            body = node.body
            for ref_match in ref_pattern.finditer(body):
                target_id = ref_match.group(1)
                if target_id == source_id or target_id not in dag.nodes:
                    continue

                # 参照箇所の文脈から関係性を推定
                ref_pos = ref_match.start()
                context_snippet = body[
                    max(0, ref_pos - 20) : min(len(body), ref_pos + 60)
                ]

                if "かかわらず" in context_snippet or "ただし" in context_snippet:
                    rel = ClauseRelationType.EXCEPTION
                elif "除き" in context_snippet or "適用しない" in context_snippet:
                    rel = ClauseRelationType.EXCLUSION
                elif "通知" in context_snippet or "手続" in context_snippet:
                    rel = ClauseRelationType.PROCEDURE
                else:
                    rel = ClauseRelationType.REFERENCE

                dag.edges.append(
                    ClauseEdge(
                        source_id=source_id,
                        target_id=target_id,
                        relation_type=rel,
                        condition_text=context_snippet,
                    )
                )

        logger.info(
            "ドキュメント '%s' から %d 条項, %d エッジの DAG を抽出しました。",
            doc_id,
            len(dag.nodes),
            len(dag.edges),
        )
        return dag
