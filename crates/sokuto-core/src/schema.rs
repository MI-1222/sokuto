//! # Jev 互換 API スキーマおよび型定義モジュール
//!
//! TypeSafe AI の `POST /v1/systemone` リクエストおよびレスポンスと
//! 完全な互換性を持つデータ構造を定義する。

use std::fmt;

use indexmap::IndexMap;
use serde::de::Deserializer;
use serde::ser::{SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};

use crate::error::{CoreError, Result};
use crate::gating::{DecisionRoute, GatingConfig, GatingMetadata, SystemRoutingSummary};

/// 質問の決定プリミティブ種別。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[cfg_attr(feature = "openapi", schema(rename_all = "lowercase"))]
pub enum QuestionType {
    /// 候補選択(離散選択肢から最適な1つを選択)。
    Choice,
    /// 順序尺度評価(2〜10段階の評価尺度から期待実数値を算出)。
    Score,
    /// 真偽確率判定(言明が真である確率を直接算出)。
    Noul,
}

impl fmt::Display for QuestionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Choice => write!(f, "choice"),
            Self::Score => write!(f, "score"),
            Self::Noul => write!(f, "noul"),
        }
    }
}

/// 柔軟な評価基準(Criteria)データ構造。
///
/// 質問タイプに応じて Map、List、または未指定(None)のいずれかをとる。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum Criteria {
    /// Choice 型向け: 候補識別子(キー)と説明文(値)の順序付きマップ。
    Map(IndexMap<String, String>),
    /// Score 型向け: 順序付けられた評価段階基準の配列(2〜10段階)。
    List(Vec<String>),
    /// Noul 型向け、または評価基準が指定されていない状態。
    None,
}

impl Criteria {
    /// Map 形式の参照を取得する。
    pub fn as_map(&self) -> Option<&IndexMap<String, String>> {
        match self {
            Self::Map(map) => Some(map),
            _ => None,
        }
    }

    /// List 形式の参照を取得する。
    pub fn as_list(&self) -> Option<&[String]> {
        match self {
            Self::List(list) => Some(list),
            _ => None,
        }
    }

    /// 未指定(None)であるかどうかを判定する。
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// 候補数または段階数を取得する。
    pub fn len(&self) -> usize {
        match self {
            Self::Map(map) => map.len(),
            Self::List(list) => list.len(),
            Self::None => 0,
        }
    }

    /// 要素が空であるかどうかを判定する。
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// 単一の質問定義。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Question {
    /// 質問プリミティブ種別(`choice`, `score`, `noul`)。
    #[serde(rename = "type")]
    #[cfg_attr(feature = "openapi", schema(rename = "type"))]
    pub question_type: QuestionType,

    /// 評価内容に関する自然言語の指示文。
    pub instructions: String,

    /// 型依存の評価基準(Choice 時は Map、Score 時は List、Noul 時は省略可能)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<Criteria>,
}

impl Question {
    /// 新しい Choice 質問を作成する。
    pub fn new_choice(instructions: impl Into<String>, criteria: IndexMap<String, String>) -> Self {
        Self {
            question_type: QuestionType::Choice,
            instructions: instructions.into(),
            criteria: Some(Criteria::Map(criteria)),
        }
    }

    /// 新しい Score 質問を作成する。
    pub fn new_score(instructions: impl Into<String>, criteria: Vec<String>) -> Self {
        Self {
            question_type: QuestionType::Score,
            instructions: instructions.into(),
            criteria: Some(Criteria::List(criteria)),
        }
    }

    /// 新しい Noul 質問を作成する。
    pub fn new_noul(instructions: impl Into<String>) -> Self {
        Self {
            question_type: QuestionType::Noul,
            instructions: instructions.into(),
            criteria: None,
        }
    }

    /// 質問スキーマの整合性を検証する。
    ///
    /// # 引数
    /// - `question_id`: 質問識別子 (任意)。エラー発生時に診断メッセージへ付与される。
    pub fn validate(&self, question_id: Option<&str>) -> Result<()> {
        let qid = question_id.map(|s| s.to_string());
        match self.question_type {
            QuestionType::Choice => {
                let criteria =
                    self.criteria
                        .as_ref()
                        .ok_or_else(|| CoreError::MissingCriteria {
                            question_id: qid.clone(),
                            question_type: "choice".to_string(),
                        })?;
                match criteria {
                    Criteria::Map(map) => {
                        let count = map.len();
                        if count == 0 || count > 255 {
                            return Err(CoreError::InvalidChoiceCount {
                                question_id: qid,
                                count,
                            });
                        }
                    }
                    Criteria::List(_) => {
                        return Err(CoreError::InvalidCriteriaType {
                            question_id: qid,
                            question_type: "choice".to_string(),
                            expected: "Map (オブジェクト)",
                            actual: "List (配列)",
                        });
                    }
                    Criteria::None => {
                        return Err(CoreError::MissingCriteria {
                            question_id: qid,
                            question_type: "choice".to_string(),
                        });
                    }
                }
            }
            QuestionType::Score => {
                let criteria =
                    self.criteria
                        .as_ref()
                        .ok_or_else(|| CoreError::MissingCriteria {
                            question_id: qid.clone(),
                            question_type: "score".to_string(),
                        })?;
                match criteria {
                    Criteria::List(list) => {
                        let count = list.len();
                        if !(2..=10).contains(&count) {
                            return Err(CoreError::InvalidScoreLevelCount {
                                question_id: qid,
                                count,
                            });
                        }
                    }
                    Criteria::Map(_) => {
                        return Err(CoreError::InvalidCriteriaType {
                            question_id: qid,
                            question_type: "score".to_string(),
                            expected: "List (配列)",
                            actual: "Map (オブジェクト)",
                        });
                    }
                    Criteria::None => {
                        return Err(CoreError::MissingCriteria {
                            question_id: qid,
                            question_type: "score".to_string(),
                        });
                    }
                }
            }
            QuestionType::Noul => {
                // Noul 型は criteria を要求しない。指定されている場合でも無視または許容する。
            }
        }
        Ok(())
    }
}

/// 質問群マップおよび任意のゲーティング設定の整合性を検証する共通バリデータ。
///
/// # 概要
/// - 質問群が空配列の場合は [`CoreError::EmptyQuestions`] を返却する。
/// - 含まれる各質問について、種別ごとの基準定義 ([`Question::validate`]) を検証する。
/// - 確信度ゲーティング設定が存在する場合は、各閾値の妥当性 ([`GatingConfig::validate`]) を検証する。
///
/// # 引数
/// - `questions`: 質問群マップ。
/// - `gating`: 任意のゲーティング設定。
///
/// # 戻り値
/// - 成功時は `Ok(())`、不整合検出時は [`CoreError`]。
pub fn validate_request_payload(
    questions: &IndexMap<String, Question>,
    gating: Option<&GatingConfig>,
) -> Result<()> {
    if questions.is_empty() {
        return Err(CoreError::EmptyQuestions);
    }
    for (question_id, question) in questions {
        question.validate(Some(question_id))?;
    }
    if let Some(gating) = gating {
        gating.validate()?;
    }
    Ok(())
}

/// 決定プリミティブ質問群とゲーティング設定を保持するリクエストペイロードの共通トレイト。
pub trait DecisionRequest {
    /// 評価対象となる質問群のマップへの参照を取得する。
    fn questions(&self) -> &IndexMap<String, Question>;

    /// 確信度ゲーティング処理の設定パラメータへの参照を取得する。
    fn gating(&self) -> Option<&GatingConfig>;

    /// リクエスト全体の整合性を検証する。
    fn validate(&self) -> Result<()> {
        validate_request_payload(self.questions(), self.gating())
    }
}

/// Jev 互換 `POST /v1/systemone` リクエストペイロード。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct SystemOneRequest {
    /// 使用するローカルモデル識別子(任意)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// 判断の材料となる非構造化コンテキストデータ(文字列、オブジェクト、配列など)。
    ///
    /// # 備考
    /// サーバー側の前処理ガードレールパイプライン通過後は、推論バックボーンへ供給するために
    /// 常に自然言語文字列 (`Value::String`) として標準化および注釈付加が行われる。
    #[cfg_attr(feature = "openapi", schema(value_type = Object))]
    pub state: serde_json::Value,

    /// 評価対象となる質問群のマップ(キーは質問識別子)。
    pub questions: IndexMap<String, Question>,

    /// 確信度ゲーティング処理の設定パラメータ(任意)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gating: Option<GatingConfig>,

    /// 候補数 K > 16 の Choice 質問に対する透過的粗密階層ルーティングの制御フラグ (任意)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_hierarchical: Option<bool>,

    /// `State` 中間特徴量キャッシュのキー識別子 (`xxHash3` 16進文字列、任意)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_hash: Option<String>,
}

impl SystemOneRequest {
    /// 新規リクエストを作成する。
    pub fn new(state: impl Into<serde_json::Value>, questions: IndexMap<String, Question>) -> Self {
        Self {
            model: None,
            state: state.into(),
            questions,
            gating: None,
            auto_hierarchical: None,
            state_hash: None,
        }
    }

    /// モデル名を指定してリクエストを作成する。
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// 透過的粗密階層ルーティングの有効/無効を指定してリクエストを作成する。
    pub fn with_auto_hierarchical(mut self, auto_hierarchical: bool) -> Self {
        self.auto_hierarchical = Some(auto_hierarchical);
        self
    }

    /// ゲーティング設定を指定してリクエストを作成する。
    pub fn with_gating(mut self, gating: GatingConfig) -> Self {
        self.gating = Some(gating);
        self
    }

    /// `State` キャッシュキーを指定してリクエストを作成する。
    pub fn with_state_hash(mut self, state_hash: impl Into<String>) -> Self {
        self.state_hash = Some(state_hash.into());
        self
    }

    /// リクエスト全体の整合性を検証する。
    pub fn validate(&self) -> Result<()> {
        DecisionRequest::validate(self)
    }
}

impl DecisionRequest for SystemOneRequest {
    fn questions(&self) -> &IndexMap<String, Question> {
        &self.questions
    }

    fn gating(&self) -> Option<&GatingConfig> {
        self.gating.as_ref()
    }
}

/// 単一の質問に対する判定結果ペイロード。
///
/// Jev の3大プリミティブ(Choice / Score / Noul)に対応する排他的な直和型である。
/// 不正な状態(排他フィールドの競合や必須データの欠落)を型レベルで排除する。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnswerPayload {
    /// Choice 型: 採択候補ラベル、確率分布マップ、および正規化確信度。
    Choice {
        /// 採択候補ラベル。
        selected: String,
        /// 候補ごとの確率分布マップ。
        probabilities: IndexMap<String, f64>,
        /// 分布の尖り度に基づく正規化確信度(0.0 <= C <= 1.0)。
        confidence: f64,
    },
    /// Score 型: 加重平均スコア値、各評価段階の確率分布マップ、および正規化確信度。
    Score {
        /// 加重平均スコア値。
        value: f64,
        /// 各評価段階の確率分布マップ。
        probabilities: IndexMap<String, f64>,
        /// 分布の尖り度に基づく正規化確信度(0.0 <= C <= 1.0)。
        confidence: f64,
    },
    /// Noul 型: 言明真実確率値。
    Noul {
        /// 言明真実確率値(0.0 <= P <= 1.0)。
        probability: f64,
    },
}

impl AnswerPayload {
    /// 対応する質問種別 (`QuestionType`) を取得する。
    pub fn question_type(&self) -> QuestionType {
        match self {
            Self::Choice { .. } => QuestionType::Choice,
            Self::Score { .. } => QuestionType::Score,
            Self::Noul { .. } => QuestionType::Noul,
        }
    }

    /// Choice の採択候補ラベルを取得する。
    pub fn as_choice(&self) -> Option<&str> {
        match self {
            Self::Choice { selected, .. } => Some(selected.as_str()),
            _ => None,
        }
    }

    /// Choice の採択候補ラベルを取得する (`as_choice` の別名・非推奨)。
    #[deprecated(since = "0.4.0", note = "代わりに `as_choice` を使用してください。")]
    pub fn choice(&self) -> Option<&str> {
        self.as_choice()
    }

    /// Choice の採択候補ラベルを取得する (`as_choice` の別名・非推奨)。
    #[deprecated(since = "0.4.0", note = "代わりに `as_choice` を使用してください。")]
    pub fn selected(&self) -> Option<&str> {
        self.as_choice()
    }

    /// Score の加重平均値を取得する。
    pub fn as_score(&self) -> Option<f64> {
        match self {
            Self::Score { value, .. } => Some(*value),
            _ => None,
        }
    }

    /// Score の加重平均値を取得する (`as_score` の別名・非推奨)。
    #[deprecated(since = "0.4.0", note = "代わりに `as_score` を使用してください。")]
    pub fn score(&self) -> Option<f64> {
        self.as_score()
    }

    /// Score の加重平均スコア値を取得する (`as_score` の別名・非推奨)。
    #[deprecated(since = "0.4.0", note = "代わりに `as_score` を使用してください。")]
    pub fn score_value(&self) -> Option<f64> {
        self.as_score()
    }

    /// Noul の真実確率値を取得する。
    pub fn as_noul(&self) -> Option<f64> {
        match self {
            Self::Noul { probability } => Some(*probability),
            _ => None,
        }
    }

    /// Noul の真実確率値を取得する (`as_noul` の別名・非推奨)。
    #[deprecated(since = "0.4.0", note = "代わりに `as_noul` を使用してください。")]
    pub fn noul(&self) -> Option<f64> {
        self.as_noul()
    }

    /// Noul の真実確率値を取得する (`as_noul` の別名・非推奨)。
    #[deprecated(since = "0.4.0", note = "代わりに `as_noul` を使用してください。")]
    pub fn noul_probability(&self) -> Option<f64> {
        self.as_noul()
    }

    /// Choice 判定であるかどうか。
    pub fn is_choice(&self) -> bool {
        matches!(self, Self::Choice { .. })
    }

    /// Score 判定であるかどうか。
    pub fn is_score(&self) -> bool {
        matches!(self, Self::Score { .. })
    }

    /// Noul 判定であるかどうか。
    pub fn is_noul(&self) -> bool {
        matches!(self, Self::Noul { .. })
    }

    /// 確率分布マップの参照を取得する。
    pub fn probabilities(&self) -> Option<&IndexMap<String, f64>> {
        match self {
            Self::Choice { probabilities, .. } | Self::Score { probabilities, .. } => {
                Some(probabilities)
            }
            Self::Noul { .. } => None,
        }
    }

    /// 確率分布マップの可変参照を取得する。
    pub fn probabilities_mut(&mut self) -> Option<&mut IndexMap<String, f64>> {
        match self {
            Self::Choice { probabilities, .. } | Self::Score { probabilities, .. } => {
                Some(probabilities)
            }
            Self::Noul { .. } => None,
        }
    }

    /// 確率分布マップを更新する。Choice / Score 型の場合に適用される。
    pub fn set_probabilities(&mut self, new_probabilities: IndexMap<String, f64>) {
        match self {
            Self::Choice { probabilities, .. } | Self::Score { probabilities, .. } => {
                *probabilities = new_probabilities;
            }
            Self::Noul { .. } => {}
        }
    }

    /// 正規化確信度を取得する。
    pub fn confidence(&self) -> Option<f64> {
        match self {
            Self::Choice { confidence, .. } | Self::Score { confidence, .. } => Some(*confidence),
            Self::Noul { .. } => None,
        }
    }

    /// 実効確信度(Effective Confidence)を取得する。
    ///
    /// Choice / Score は `confidence` をそのまま返し、Noul は真偽の尖り度 $|2P - 1.0|$ を算出する。
    pub fn effective_confidence(&self) -> f64 {
        match self {
            Self::Choice { confidence, .. } | Self::Score { confidence, .. } => *confidence,
            Self::Noul { probability } => (2.0 * probability - 1.0).abs().clamp(0.0, 1.0),
        }
    }
}

/// 単一の質問に対する判定結果。
///
/// Jev 公式互換のフラット JSON 表現(`choice`, `score`, `noul`, `probabilities`, `confidence`, `gating`)と
/// 型安全な直和型表現(`AnswerPayload`)を透過的に相互変換する。
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    /// 判定ペイロード(Choice / Score / Noul の直和型)。
    pub payload: AnswerPayload,

    /// 確信度ゲーティングによるルーティングおよび監査メタデータ(任意)。
    pub gating: Option<GatingMetadata>,
}

/// OpenAPI 仕様書向け `Answer` フラットスキーマ定義ヘルパー。
///
/// Jev 公式互換のフラット JSON レスポンス表現(`choice`, `score`, `noul`, `probabilities`, `confidence`, `gating`)を
/// OpenAPI ドキュメントへ正確に反映するための内部スキーマ構造体である。
#[cfg(feature = "openapi")]
#[derive(utoipa::ToSchema)]
#[schema(as = Answer)]
#[allow(dead_code)]
struct AnswerFlatSchema {
    /// Choice 型: 採択候補ラベル。
    #[schema(example = "candidate_a")]
    choice: Option<String>,

    /// Score 型: 加重平均スコア値。
    #[schema(example = 4.2)]
    score: Option<f64>,

    /// Noul 型: 言明真実確率値(0.0 <= P <= 1.0)。
    #[schema(example = 0.95)]
    noul: Option<f64>,

    /// Choice / Score 型: 候補または各評価段階の確率分布マップ。
    probabilities: Option<IndexMap<String, f64>>,

    /// Choice / Score 型: 分布の尖り度に基づく正規化確信度(0.0 <= C <= 1.0)。
    #[schema(example = 0.88)]
    confidence: Option<f64>,

    /// 確信度ゲーティングによるルーティングおよび監査メタデータ(任意)。
    gating: Option<GatingMetadata>,
}

#[cfg(feature = "openapi")]
impl utoipa::__dev::ComposeSchema for Answer {
    fn compose(
        generics: Vec<utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>>,
    ) -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        AnswerFlatSchema::compose(generics)
    }
}

#[cfg(feature = "openapi")]
impl utoipa::ToSchema for Answer {
    fn name() -> std::borrow::Cow<'static, str> {
        AnswerFlatSchema::name()
    }

    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        AnswerFlatSchema::schemas(schemas);
    }
}

impl Serialize for Answer {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Answer", 6)?;
        match &self.payload {
            AnswerPayload::Choice {
                selected,
                probabilities,
                confidence,
            } => {
                state.serialize_field("choice", selected)?;
                state.serialize_field("probabilities", probabilities)?;
                state.serialize_field("confidence", confidence)?;
            }
            AnswerPayload::Score {
                value,
                probabilities,
                confidence,
            } => {
                state.serialize_field("score", value)?;
                state.serialize_field("probabilities", probabilities)?;
                state.serialize_field("confidence", confidence)?;
            }
            AnswerPayload::Noul { probability } => {
                state.serialize_field("noul", probability)?;
            }
        }
        if let Some(ref gating) = self.gating {
            state.serialize_field("gating", gating)?;
        }
        state.end()
    }
}

#[derive(Deserialize)]
struct AnswerFlatHelper {
    #[serde(default)]
    choice: Option<String>,
    #[serde(default)]
    score: Option<f64>,
    #[serde(default)]
    noul: Option<f64>,
    #[serde(default)]
    probabilities: Option<IndexMap<String, f64>>,
    #[serde(default)]
    confidence: Option<f64>,
    #[serde(default)]
    gating: Option<GatingMetadata>,
}

impl<'de> Deserialize<'de> for Answer {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let helper = AnswerFlatHelper::deserialize(deserializer)?;
        let payload = match (helper.choice, helper.score, helper.noul) {
            (Some(selected), None, None) => AnswerPayload::Choice {
                selected,
                probabilities: helper.probabilities.unwrap_or_default(),
                confidence: helper.confidence.unwrap_or(0.0),
            },
            (None, Some(value), None) => AnswerPayload::Score {
                value,
                probabilities: helper.probabilities.unwrap_or_default(),
                confidence: helper.confidence.unwrap_or(0.0),
            },
            (None, None, Some(probability)) => AnswerPayload::Noul { probability },
            (None, None, None) => {
                return Err(serde::de::Error::custom(
                    "Answer must contain exactly one of 'choice', 'score', or 'noul'.",
                ));
            }
            _ => {
                return Err(serde::de::Error::custom(
                    "Answer contains conflicting fields: 'choice', 'score', and 'noul' are mutually exclusive.",
                ));
            }
        };

        Ok(Answer {
            payload,
            gating: helper.gating,
        })
    }
}

impl Answer {
    /// Choice 判定結果を生成する。
    pub fn choice(
        selected: impl Into<String>,
        probabilities: IndexMap<String, f64>,
        confidence: f64,
    ) -> Self {
        Self {
            payload: AnswerPayload::Choice {
                selected: selected.into(),
                probabilities,
                confidence,
            },
            gating: None,
        }
    }

    /// Score 判定結果を生成する。
    pub fn score(value: f64, probabilities: IndexMap<String, f64>, confidence: f64) -> Self {
        Self {
            payload: AnswerPayload::Score {
                value,
                probabilities,
                confidence,
            },
            gating: None,
        }
    }

    /// Noul 判定結果を生成する。
    pub fn noul(probability: f64) -> Self {
        Self {
            payload: AnswerPayload::Noul { probability },
            gating: None,
        }
    }

    /// ゲーティングメタデータを付与して自身を返却する。
    pub fn with_gating(mut self, gating: GatingMetadata) -> Self {
        self.gating = Some(gating);
        self
    }

    /// 割り当てられたルーティング種別を取得する。
    pub fn route(&self) -> Option<DecisionRoute> {
        self.gating.as_ref().map(|g| g.route)
    }

    /// ゲーティング処理等に向けた実効確信度(Effective Confidence)を取得する。
    ///
    /// # 概要
    /// - Choice / Score 型の場合は、構造体に保持されている `confidence` 値をそのまま返却する。
    /// - Noul 型の場合は、真実確率 $P(\text{true})$ から尖り度 $|2P - 1.0|$ を即座に算出して返却する。
    ///
    /// # 戻り値
    /// - 判定結果に対応する 0.0〜1.0 の確信度実数値。
    pub fn effective_confidence(&self) -> Option<f64> {
        Some(self.payload.effective_confidence())
    }
}

impl std::ops::Deref for Answer {
    type Target = AnswerPayload;

    fn deref(&self) -> &Self::Target {
        &self.payload
    }
}

impl std::ops::DerefMut for Answer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.payload
    }
}

/// トークン消費量メタデータ。
///
/// System One モデルは文章生成を行わないため、`completion_tokens` は常に 0 となる。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Usage {
    /// 入力コンテキスト(State + Questions)のトークン総数。
    pub prompt_tokens: usize,

    /// 生成完了トークン数(常に 0)。
    pub completion_tokens: usize,

    /// 処理に要した総トークン数(prompt_tokens と一致)。
    pub total_tokens: usize,
}

impl Usage {
    /// 新規 Usage メタデータを生成する。
    pub fn new(prompt_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens: 0,
            total_tokens: prompt_tokens,
        }
    }
}

/// Jev 互換 `POST /v1/systemone` レスポンスペイロード。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct SystemOneResponse {
    /// 各 question_id に対応する判定結果マップ。
    pub answers: IndexMap<String, Answer>,

    /// トークン使用統計情報。
    pub usage: Usage,

    /// リクエスト全体の集約ルーティングサマリー(任意)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing: Option<SystemRoutingSummary>,
}

impl SystemOneResponse {
    /// 新規レスポンスを生成する。
    pub fn new(answers: IndexMap<String, Answer>, usage: Usage) -> Self {
        Self {
            answers,
            usage,
            routing: None,
        }
    }

    /// 集約ルーティングサマリーを付与して新規レスポンスを生成する。
    pub fn with_routing(mut self, routing: SystemRoutingSummary) -> Self {
        self.routing = Some(routing);
        self
    }
}

/// 単一 `State` に対する一括バッチ質問リクエストペイロード (`POST /v1/systemone/state-batch`)。
///
/// `SystemOneRequest` と同一の構造・振る舞いを持つ型エイリアスである。
pub type StateBatchRequest = SystemOneRequest;

/// 単一 `State` に対する一括バッチ質問レスポンスペイロード (`POST /v1/systemone/state-batch`)。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct StateBatchResponse {
    /// 各 `question_id` に対応する判定結果マップ。
    pub answers: IndexMap<String, Answer>,

    /// トークン使用統計情報。
    pub usage: Usage,

    /// リクエスト全体の集約ルーティングサマリー (任意)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing: Option<SystemRoutingSummary>,

    /// `State` 特徴量キャッシュがヒットしたかどうかのフラグ。
    pub cache_hit: bool,

    /// 処理対象となった `State` のキャッシュハッシュ値 (16進文字列、任意)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_hash: Option<String>,
}

impl StateBatchResponse {
    /// 新規バッチレスポンスを生成する。
    pub fn new(answers: IndexMap<String, Answer>, usage: Usage, cache_hit: bool) -> Self {
        Self {
            answers,
            usage,
            routing: None,
            cache_hit,
            state_hash: None,
        }
    }

    /// 集約ルーティングサマリーを付与する。
    pub fn with_routing(mut self, routing: SystemRoutingSummary) -> Self {
        self.routing = Some(routing);
        self
    }

    /// `State` ハッシュを付与する。
    pub fn with_state_hash(mut self, state_hash: impl Into<String>) -> Self {
        self.state_hash = Some(state_hash.into());
        self
    }
}
