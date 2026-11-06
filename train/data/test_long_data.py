"""長文実務データセットおよび意思決定特化型合成データパイプラインの統合テストモジュール。

8.3.1 JBE-QA, LegalRikai, ComplianceJA 実務ドキュメントコンバータ
8.3.2 日本語 NIAH (Needle In A Haystack) 長文ベンチマーク
8.3.3 長文意思決定合成パイプライン (DAG 抽出、Multi-hop 合成、Contrast Sets、OOD 混入)
および UnifiedDatasetBuilder への統合を網羅的に検証する。
"""

from data.benchmarks.niah_long import (
    DEFAULT_NEEDLE_BANK,
    NIAHBenchmarkGenerator,
    find_safe_insertion_point,
)
from data.builders import UnifiedDatasetBuilder
from data.converters.compliance_ja import ComplianceJAConverter
from data.converters.jbe_qa import JBEQAConverter
from data.converters.legal_rikai import LegalRikaiConverter
from data.schema import QuestionType, UnifiedSample
from data.synthetic.long_context.contrast_generator import (
    ContrastSetGenerator,
)
from data.synthetic.long_context.dag_extractor import (
    ClauseDAGExtractor,
    ClauseRelationType,
)
from data.synthetic.long_context.multihop_synthesizer import (
    MultiHopScenario,
    MultiHopSynthesizer,
)
from data.synthetic.long_context.ood_injector import (
    LongContextOODInjector,
)
from data.synthetic.long_context.pipeline import (
    LongContextSyntheticPipeline,
)


class TestLongContextConverters:
    """8.3.1 実務ドキュメントコンバータ群のテストスイート。"""

    def test_jbe_qa_converter(self) -> None:
        """JBE-QA コンバータが Noul および Choice サンプルを正しく生成し、メタデータを保持することを検証する。"""
        converter = JBEQAConverter(mode="both")
        samples = list(converter.convert_split("train"))
        assert len(samples) > 0

        noul_samples = [s for s in samples if s.question_type == QuestionType.NOUL]
        choice_samples = [s for s in samples if s.question_type == QuestionType.CHOICE]

        assert len(noul_samples) >= 3
        assert len(choice_samples) >= 1

        # Noul サンプルの検証
        sample_noul = noul_samples[0]
        assert sample_noul.target in ("true", "false")
        assert "state_id" in sample_noul.metadata
        assert "chunk_spans" in sample_noul.metadata
        assert len(sample_noul.state) > 300

        # Choice サンプルの検証
        sample_choice = choice_samples[0]
        assert sample_choice.target in sample_choice.criteria
        assert sample_choice.metadata["state_id"] == sample_noul.metadata["state_id"]

    def test_legal_rikai_converter(self) -> None:
        """LegalRikai コンバータが契約書条項パース、chunk_spans、および Choice/Noul を生成することを検証する。"""
        converter = LegalRikaiConverter(mode="both")
        samples = list(converter.convert_split("train"))
        assert len(samples) > 0

        choice_samples = [s for s in samples if s.question_type == QuestionType.CHOICE]
        noul_samples = [s for s in samples if s.question_type == QuestionType.NOUL]

        assert len(choice_samples) >= 2
        assert len(noul_samples) >= 2

        for sample in samples:
            assert "state_id" in sample.metadata
            assert "chunk_spans" in sample.metadata
            assert len(sample.metadata["chunk_spans"]) >= 2
            assert "第" in sample.state

    def test_compliance_ja_converter(self) -> None:
        """ComplianceJA コンバータが就業規則から Score/Choice/Noul サンプルを規格通り生成することを検証する。"""
        converter = ComplianceJAConverter(mode="all")
        samples = list(converter.convert_split("train"))
        assert len(samples) >= 3

        types = {s.question_type for s in samples}
        assert QuestionType.SCORE in types
        assert QuestionType.CHOICE in types
        assert QuestionType.NOUL in types

        # Score サンプルの検証 (0〜4 の昇順連番キー)
        score_sample = next(s for s in samples if s.question_type == QuestionType.SCORE)
        assert len(score_sample.criteria) == 5
        assert list(score_sample.criteria.keys()) == [
            "0",
            "1",
            "2",
            "3",
            "4",
        ]
        assert score_sample.target in score_sample.criteria


class TestNIAHBenchmark:
    """8.3.2 日本語 NIAH (Needle In A Haystack) ベンチマークのテストスイート。"""

    def test_find_safe_insertion_point(self) -> None:
        """安全な挿入位置探索が句点または改行の直後を選択することを検証する。"""
        text = "第1文です。第2文です。第3文です。\n第4文です。"
        # 50% 付近 (第2文の句点後)
        idx = find_safe_insertion_point(text, 0.5)
        assert idx > 0
        # 挿入直前の文字が句点または改行であること
        assert text[idx - 1] in ("。", "\n")

    def test_generate_benchmark_samples(self) -> None:
        """NIAHBenchmarkGenerator が全深度および OOD サンプルを生成することを検証する。"""
        gen = NIAHBenchmarkGenerator(
            target_char_length=2000,
            depth_steps=[0.0, 0.5, 1.0],
        )
        samples = gen.generate_benchmark_samples(include_ood=True)
        # 4 ルール × (3深度 + 1 OOD) = 16 サンプル
        assert len(samples) == len(DEFAULT_NEEDLE_BANK) * (3 + 1)

        # 針が存在するサンプルの確認
        in_needle_sample = next(s for s in samples if not s.metadata["is_ood"])
        assert in_needle_sample.target == "opt_needle"
        assert in_needle_sample.metadata["needle_depth"] in (0.0, 0.5, 1.0)
        assert "特約条項" in in_needle_sample.state

        # OOD (針なし) サンプルの確認
        ood_sample = next(s for s in samples if s.metadata["is_ood"])
        assert ood_sample.target == "none"
        assert ood_sample.metadata["needle_depth"] == -1.0

    def test_evaluate_predictions_and_exit_criteria(self) -> None:
        """NIAH 評価器が全深度正解率および Exit Criteria 合否判定を正しく算出することを検証する。"""
        gen = NIAHBenchmarkGenerator(
            target_char_length=1000,
            depth_steps=[0.0, 0.5, 1.0],
        )
        samples = gen.generate_benchmark_samples(include_ood=True)

        # 1. 全問正解時の評価
        perfect_preds = [s.target for s in samples]
        result = gen.evaluate_predictions(samples, perfect_preds)
        assert result.accuracy == 1.0
        assert result.passed_exit_criteria is True
        assert result.lost_in_the_middle_gap == 0.0

        # 2. 中央部 (0.5) のみが誤答した場合の Lost in the middle ギャップ検出
        imperfect_preds = []
        for s in samples:
            if s.metadata.get("needle_depth") == 0.5:
                imperfect_preds.append("wrong_key")
            else:
                imperfect_preds.append(s.target)

        result_imperfect = gen.evaluate_predictions(samples, imperfect_preds)
        assert result_imperfect.depth_accuracies[0.5] == 0.0
        assert result_imperfect.lost_in_the_middle_gap > 0.0
        assert result_imperfect.passed_exit_criteria is False


class TestLongContextSyntheticPipeline:
    """8.3.3 長文意思決定合成データパイプラインのテストスイート。"""

    def test_dag_extractor(self) -> None:
        """ClauseDAGExtractor が条項ノードおよび依存関係エッジを正しく抽出することを検証する。"""
        contract_text = (
            "第1条（基本原則）\n乙は甲に対して本業務を誠実に履行する。\n\n"
            "第2条（例外免責）\n第1条の規定にかかわらず、天災等の不可抗力による遅延については免責される。\n\n"
            "第3条（通知手続）\n第2条の免責を受ける場合、第1条の期限までに書面で通知しなければならない。\n"
        )
        extractor = ClauseDAGExtractor()
        dag = extractor.extract_dag(contract_text, doc_id="test_contract")

        assert len(dag.nodes) == 3
        assert "第1条" in dag.nodes
        assert "第2条" in dag.nodes
        assert "第3条" in dag.nodes
        assert len(dag.edges) >= 2

        # 関係性エッジの検証
        edge_types = {e.relation_type for e in dag.edges}
        assert ClauseRelationType.EXCEPTION in edge_types

    def test_multihop_and_contrast_generator(self) -> None:
        """MultiHopSynthesizer および ContrastSetGenerator が最小反事実ペアを生成することを検証する。"""
        contract_text = (
            "第1条（基本合意）\n本契約は甲乙間の取引基本事項を定める。\n\n"
            "第2条（損害賠償）\n乙が本契約に違反した場合、生じた通常損害を賠償する。"
            "ただし、第1条の協議事項に基づき事前に合意された特別事由が発生している場合はこの限りでない。\n"
        )
        extractor = ClauseDAGExtractor()
        dag = extractor.extract_dag(contract_text, doc_id="c_test")

        synthesizer = MultiHopSynthesizer()
        scenarios = synthesizer.synthesize_scenarios(dag, max_scenarios=2)
        assert len(scenarios) >= 1

        contrast_gen = ContrastSetGenerator()
        pair = contrast_gen.generate_contrast_pair(scenarios[0])
        assert pair.original_scenario.target != pair.perturbed_scenario.target
        assert pair.inverted_attribute != ""

    def test_ood_injector_ratio_control(self) -> None:
        """LongContextOODInjector が目標比率 (12.5%〜15.0%) で OOD サンプルを混合することを検証する。"""
        injector = LongContextOODInjector(target_ood_ratio=0.135)
        extractor = ClauseDAGExtractor()
        contract_text = (
            "第1条（目的）\n業務支援を行う。\n\n"
            "第2条（例外）\n第1条にかかわらず緊急時は事後承認とする。\n\n"
            "第3条（通知）\n第2条の適用には3日以内の通知を要する。\n"
        )
        dag = extractor.extract_dag(contract_text, doc_id="ood_test")
        synthesizer = MultiHopSynthesizer()
        scenarios = synthesizer.synthesize_scenarios(dag, max_scenarios=10)

        # 7件のイン・ドメインシナリオに対して OOD 注入 (7件に対し1件のOODで 1/8=12.5%)
        in_domain = (scenarios * 7)[:7]
        assert len(in_domain) == 7
        mixed = injector.inject_ood_samples(in_domain, doc_id="ood_test")

        ood_samples = [s for s in mixed if s.metadata.get("is_ood")]
        assert len(ood_samples) == 1
        ood_ratio = len(ood_samples) / len(mixed)
        # 1 / 8 = 12.5%
        assert 0.12 <= ood_ratio <= 0.15

    def test_full_pipeline_to_unified_samples(self) -> None:
        """LongContextSyntheticPipeline が一連の処理を実行し UnifiedSample を生成することを検証する。"""
        pipeline = LongContextSyntheticPipeline(
            target_ood_ratio=0.135,
            include_contrast_sets=True,
        )
        contract_text = (
            "第1条（契約の趣旨）\n甲乙間の秘密保持を定める。\n\n"
            "第2条（開示禁止）\n秘密情報を第三者に開示してはならない。"
            "ただし、第1条の目的範囲内において弁護士等へ開示する場合は第1条の例外とする。\n"
        )
        samples = pipeline.process_document(
            doc_id="pipe_test",
            document_text=contract_text,
        )
        assert len(samples) > 0
        for sample in samples:
            assert isinstance(sample, UnifiedSample)
            assert sample.state == contract_text
            assert "state_id" in sample.metadata
            assert "chunk_spans" in sample.metadata


class TestUnifiedDatasetBuilderIntegration:
    """UnifiedDatasetBuilder への長文コンバータ統合テスト。"""

    def test_builder_with_long_context_converters(self) -> None:
        """UnifiedDatasetBuilder から jbe_qa, legal_rikai, compliance_ja, niah_long を抽出できることを検証する。"""
        builder = UnifiedDatasetBuilder(seed=42)

        long_datasets = [
            "jbe_qa",
            "legal_rikai",
            "compliance_ja",
            "niah_long",
        ]

        for ds_name in long_datasets:
            samples = list(
                builder.stream_samples(
                    dataset_names=[ds_name],
                    split="train",
                    max_samples_per_dataset=5,
                )
            )
            assert len(samples) > 0, (
                f"データセット '{ds_name}' からサンプルが取得できませんでした。"
            )
            sample = samples[0]
            assert isinstance(sample, UnifiedSample)
            assert "state_id" in sample.metadata
            assert (
                "char_chunk_spans" in sample.metadata
                or "chunk_spans" in sample.metadata
            )


class TestActionableImprovements:
    """実運用・境界保護のための 4 つの改善項目の検証テストスイート。"""

    def test_branch_clause_numbers(self) -> None:
        """第○条の○ (枝番条項) が正しく抽出・パースされることを検証する。"""
        from data.converters.compliance_ja import parse_rules_clause_spans
        from data.converters.legal_rikai import parse_contract_clause_spans

        text = (
            "第5条（服務規律）\n労働者は規律を守る。\n\n"
            "第5条の2（テレワーク勤務特約）\n在宅勤務時は事前申請を要する。\n\n"
            "第14条の3（秘密保持）\n退職後も秘密を漏洩してはならない。\n"
        )

        # 1. LegalRikai パース
        spans_legal = parse_contract_clause_spans(text)
        assert len(spans_legal) == 3

        # 2. ComplianceJA パース
        spans_compliance = parse_rules_clause_spans(text)
        assert len(spans_compliance) == 3

        # 3. ClauseDAGExtractor
        extractor = ClauseDAGExtractor()
        dag = extractor.extract_dag(text, doc_id="branch_test")
        assert "第5条" in dag.nodes
        assert "第5条の2" in dag.nodes
        assert "第14条の3" in dag.nodes

    def test_char_spans_to_token_spans_mapping(self) -> None:
        """char_spans_to_token_spans が文字スパンをトークン境界へ正確に射影することを検証する。"""
        from data.formatter import char_spans_to_token_spans

        # [CLS](0,0), "甲"(0,1), "は"(1,2), "乙"(2,3), "に"(3,4), "通知"(4,6), [SEP](0,0)
        offset_mapping = [
            (0, 0),
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 6),
            (0, 0),
        ]
        # "通知" (4〜6文字目) の文字スパン
        char_spans = [(4, 6)]
        token_spans = char_spans_to_token_spans(char_spans, offset_mapping)

        # トークンインデックス 5 ("通知") に該当
        assert token_spans == [(5, 5)]

        # "乙に通知" (2〜6文字目)
        char_spans_multi = [(2, 6)]
        token_spans_multi = char_spans_to_token_spans(char_spans_multi, offset_mapping)
        assert token_spans_multi == [(3, 5)]

    def test_contrast_generator_miss_guard(self) -> None:
        """置換キーワードが存在しない場合でも空振りガードが作動し、事実文が反転されることを検証する。"""
        scenario = MultiHopScenario(
            scenario_id="scen_guard_test",
            doc_id="doc_test",
            path_clause_ids=["第1条"],
            fact_situation="【状況】未定義の独自文脈において事象が発生した。",
            question_text="【状況】未定義の独自文脈において事象が発生した。\n【指示】判定せよ。",
            question_type=QuestionType.NOUL,
            criteria={},
            target="true",
        )
        generator = ContrastSetGenerator()
        pair = generator.generate_contrast_pair(scenario)

        # 事実文が元のままでなく更新されていること
        assert (
            pair.original_scenario.fact_situation
            != pair.perturbed_scenario.fact_situation
        )
        # ターゲットラベルが反転していること
        assert pair.original_scenario.target != pair.perturbed_scenario.target
        assert pair.perturbed_scenario.target == "false"

    def test_niah_safe_length_and_spans(self) -> None:
        """NIAH 生成器が安全な文字数帯域で Needle を挿入し、スパンが完全包含されることを検証する。"""
        gen = NIAHBenchmarkGenerator()
        samples = gen.generate_benchmark_samples(include_ood=True)

        for s in samples:
            if not s.metadata["is_ood"]:
                span = s.metadata["evidence_span"]
                assert span is not None
                start, end = span
                assert 0 <= start < end <= len(s.state)
                # 挿入された Needle が実際にスパン内に存在すること
                needle_in_state = s.state[start:end]
                assert "特約条項" in needle_in_state
