"""長文意思決定特化型合成データ統合オーケストレーションパイプラインモジュール。

長文実務ドキュメント (契約書・規程) から
1. 条項依存関係 DAG 抽出 (ClauseDAGExtractor)
2. Multi-hop 意思決定シナリオ合成 (MultiHopSynthesizer)
3. 最小反事実ペア生成 (ContrastSetGenerator)
4. 最適比率 (12.5%〜15.0%) OOD 混入 (LongContextOODInjector)
5. 統一スキーマ (UnifiedSample) 射影
を段階的に実行し、高品質な長文意思決定学習データを生成・永続化する。
"""

import json
import logging
from pathlib import Path
from typing import Any

from data.schema import UnifiedSample
from data.synthetic.long_context.contrast_generator import (
    ContrastSetGenerator,
)
from data.synthetic.long_context.dag_extractor import (
    ClauseDAGExtractor,
)
from data.synthetic.long_context.multihop_synthesizer import (
    MultiHopSynthesizer,
)
from data.synthetic.long_context.ood_injector import (
    LongContextOODInjector,
)

logger = logging.getLogger(__name__)


class LongContextSyntheticPipeline:
    """長文意思決定合成データパイプラインオーケストレータ。

    長文ドキュメントから条項 DAG、Multi-hop シナリオ、反事実対照ペア、
    および 12.5%〜15.0% 比率の OOD サンプルを統合生成し、`UnifiedSample` 列を出力する。
    """

    def __init__(
        self,
        target_ood_ratio: float = 0.135,
        include_contrast_sets: bool = True,
        seed: int = 42,
    ) -> None:
        """パイプラインを初期化する。

        Args:
            target_ood_ratio (float): 目標 OOD 比率 (デフォルト: 13.5%)。
            include_contrast_sets (bool): 最小反事実対照ペアを生成するか (デフォルト: True)。
            seed (int): 乱数シード。
        """
        self.dag_extractor = ClauseDAGExtractor()
        self.multihop_synthesizer = MultiHopSynthesizer(seed=seed)
        self.contrast_generator = ContrastSetGenerator(seed=seed)
        self.ood_injector = LongContextOODInjector(
            target_ood_ratio=target_ood_ratio, seed=seed
        )
        self.include_contrast_sets = include_contrast_sets
        self.seed = seed

    def process_document(
        self,
        doc_id: str,
        document_text: str,
        dataset_name: str = "synthetic_long_context",
    ) -> list[UnifiedSample]:
        """単一の長文ドキュメントから全合成ステップを実行し、UnifiedSample リストを生成する。

        Args:
            doc_id (str): ドキュメント ID。
            document_text (str): 長文ドキュメント本文 (State)。
            dataset_name (str): データセット名。

        Returns:
            list[UnifiedSample]: 生成された統一サンプルリスト。
        """
        # Step 1: 条項 DAG 抽出
        dag = self.dag_extractor.extract_dag(document_text, doc_id=doc_id)

        # Step 2: Multi-hop シナリオ合成
        base_scenarios = self.multihop_synthesizer.synthesize_scenarios(
            dag, max_scenarios=6
        )

        # Step 3: 最小反事実ペア (Contrast Sets) 生成
        all_scenarios = []
        for scen in base_scenarios:
            all_scenarios.append(scen)
            if self.include_contrast_sets:
                pair = self.contrast_generator.generate_contrast_pair(scen)
                all_scenarios.append(pair.perturbed_scenario)

        # Step 4: OOD (12.5%〜15.0%) 注入
        final_scenarios = self.ood_injector.inject_ood_samples(
            all_scenarios, doc_id=doc_id
        )

        # Step 5: UnifiedSample への型付き射影
        clause_spans = [node.char_span for node in dag.nodes.values()]
        unified_samples: list[UnifiedSample] = []

        for scen in final_scenarios:
            sample_id = f"{doc_id}_{scen.scenario_id}"
            metadata: dict[str, Any] = {
                "doc_id": doc_id,
                "state_id": doc_id,
                "path_clause_ids": scen.path_clause_ids,
                "chunk_spans": clause_spans,
                "evidence_spans": scen.evidence_spans,
                "is_ood": scen.metadata.get("is_ood", False),
            }
            metadata.update(scen.metadata)

            unified_sample = UnifiedSample(
                dataset_name=dataset_name,
                sample_id=sample_id,
                question_type=scen.question_type,
                state=document_text,
                instructions=scen.question_text,
                criteria=scen.criteria,
                target=scen.target,
                metadata=metadata,
            )
            unified_samples.append(unified_sample)

        logger.info(
            "ドキュメント '%s' から UnifiedSample %d 件を生成しました。",
            doc_id,
            len(unified_samples),
        )
        return unified_samples

    def process_corpus(
        self,
        documents: list[dict[str, str]],
        dataset_name: str = "synthetic_long_context",
    ) -> list[UnifiedSample]:
        """複数ドキュメントからなるコーパスを一括処理する。

        Args:
            documents (list[dict[str, str]]): [{'doc_id': ..., 'text': ...}, ...] 形式のリスト。
            dataset_name (str): データセット名。

        Returns:
            list[UnifiedSample]: 全ドキュメントから生成された統一サンプル結合リスト。
        """
        results: list[UnifiedSample] = []
        for doc in documents:
            doc_id = doc.get("doc_id", "doc_unknown")
            text = doc.get("text", "")
            if not text:
                continue
            samples = self.process_document(
                doc_id=doc_id, document_text=text, dataset_name=dataset_name
            )
            results.extend(samples)
        return results

    def export_to_jsonl(
        self,
        samples: list[UnifiedSample],
        output_path: str | Path,
    ) -> Path:
        """生成サンプルリストを JSONL ファイルへ永続化する。

        Args:
            samples (list[UnifiedSample]): サンプルリスト。
            output_path (str | Path): 出力先ファイルパス。

        Returns:
            Path: 書き込み完了後の絶対パス。
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(
                json.dumps(s.to_dict(), ensure_ascii=False) + "\n" for s in samples
            )

        logger.info(
            "JSONL ファイルに %d 件を書き込みました: %s",
            len(samples),
            path,
        )
        return path
