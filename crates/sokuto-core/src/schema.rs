//! # Jev 互換 API スキーマおよび型定義モジュール
//!
//! TypeSafe AI の `POST /v1/systemone` リクエストおよびレスポンスと
//! 完全な互換性を持つデータ構造を定義する。

use std::fmt;

use indexmap::IndexMap;
use serde::de::Deserializer;
use serde::ser::{SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};

use crate::contract::model_spec::{
    MAX_NUM_OPTIONS, MAX_SCORE_LEVELS, MIN_NUM_OPTIONS, MIN_SCORE_LEVELS,
};
use crate::error::{CoreError, Result};
use crate::gating::{DecisionRoute, GatingMetadata, GatingPatch, SystemRoutingSummary};

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
/// 質問タイプに応じて Map または List のいずれかをとる。
/// 評価基準が存在しない状態(Noul 型など)は `Option<Criteria>::None` で表現する。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum Criteria {
    /// Choice 型向け: 候補識別子(キー)と説明文(値)の順序付きマップ。
    Map(IndexMap<String, String>),
    /// Score 型向け: 順序付けられた評価段階基準の配列(2〜10段階)。
    List(Vec<String>),
}

impl Criteria {
    /// Map 形式の参照を取得する。
    pub fn as_map(&self) -> Option<&IndexMap<String, String>> {
        match self {
            Self::Map(map) => Some(map),
            Self::List(_) => None,
        }
    }

    /// Map 形式の参照を取得する。不一致時は質問種別に応じたエラーを返却する。
    ///
    /// # 引数
    /// - `question_type`: 対象の質問タイプ。
    /// - `question_id`: 質問識別子(任意)。
    ///
    /// # エラー
    /// - `Criteria::List` の場合は `CoreError::InvalidCriteriaType` を返却する。
    pub fn as_map_or_err(
        &self,
        question_type: QuestionType,
        question_id: Option<&str>,
    ) -> Result<&IndexMap<String, String>> {
        match self {
            Self::Map(map) => Ok(map),
            Self::List(_) => Err(CoreError::InvalidCriteriaType {
                question_id: question_id.map(str::to_string),
                question_type,
                expected: "Map (オブジェクト)",
                actual: "List (配列)",
            }),
        }
    }

    /// List 形式の参照を取得する。
    pub fn as_list(&self) -> Option<&[String]> {
        match self {
            Self::List(list) => Some(list),
            Self::Map(_) => None,
        }
    }

    /// List 形式の参照を取得する。不一致時は質問種別に応じたエラーを返却する。
    ///
    /// # 引数
    /// - `question_type`: 対象の質問タイプ。
    /// - `question_id`: 質問識別子(任意)。
    ///
    /// # エラー
    /// - `Criteria::Map` の場合は `CoreError::InvalidCriteriaType` を返却する。
    pub fn as_list_or_err(
        &self,
        question_type: QuestionType,
        question_id: Option<&str>,
    ) -> Result<&[String]> {
        match self {
            Self::List(list) => Ok(list),
            Self::Map(_) => Err(CoreError::InvalidCriteriaType {
                question_id: question_id.map(str::to_string),
                question_type,
                expected: "List (配列)",
                actual: "Map (オブジェクト)",
            }),
        }
    }

    /// Map 形式の参照を取得し、Choice 型の候補数境界 (`MIN_NUM_OPTIONS..=MAX_NUM_OPTIONS`)、
    /// 空文字・空白のみラベルの禁止、および重複検査を検証する。
    ///
    /// # 引数
    /// - `question_id`: 質問識別子(任意)。エラー発生時に診断メッセージへ反映される。
    ///
    /// # エラー
    /// - `Criteria::List` の場合は [`CoreError::InvalidCriteriaType`] を返却する。
    /// - 候補数が仕様範囲外の場合は [`CoreError::InvalidChoiceCount`] を返却する。
    /// - キーまたは説明文が空文字・空白のみの場合は [`CoreError::EmptyCriteriaLabel`] を返却する。
    /// - キーまたは説明文に重複が存在する場合は [`CoreError::DuplicateCriteriaLabel`] を返却する。
    pub fn validate_choice_map(
        &self,
        question_id: Option<&str>,
    ) -> Result<&IndexMap<String, String>> {
        let map = self.as_map_or_err(QuestionType::Choice, question_id)?;
        let count = map.len();
        if !(MIN_NUM_OPTIONS..=MAX_NUM_OPTIONS).contains(&count) {
            return Err(CoreError::InvalidChoiceCount {
                question_id: question_id.map(str::to_string),
                count,
            });
        }
        for (i, (key, desc)) in map.iter().enumerate() {
            let trimmed_key = key.trim();
            if trimmed_key.is_empty() {
                return Err(CoreError::EmptyCriteriaLabel {
                    question_id: question_id.map(str::to_string),
                });
            }
            let trimmed_desc = desc.trim();
            if trimmed_desc.is_empty() {
                return Err(CoreError::EmptyCriteriaLabel {
                    question_id: question_id.map(str::to_string),
                });
            }
            for (other_key, other_desc) in map.iter().skip(i + 1) {
                if trimmed_key == other_key.trim() {
                    return Err(CoreError::DuplicateCriteriaLabel {
                        question_id: question_id.map(str::to_string),
                        label: key.clone(),
                    });
                }
                if trimmed_desc == other_desc.trim() {
                    return Err(CoreError::DuplicateCriteriaLabel {
                        question_id: question_id.map(str::to_string),
                        label: desc.clone(),
                    });
                }
            }
        }
        Ok(map)
    }

    /// List 形式の参照を取得し、Score 型の段階数境界 (`MIN_SCORE_LEVELS..=MAX_SCORE_LEVELS`)、
    /// 空文字・空白のみラベルの禁止、および重複検査を検証する。
    ///
    /// # 引数
    /// - `question_id`: 質問識別子(任意)。エラー発生時に診断メッセージへ反映される。
    ///
    /// # エラー
    /// - `Criteria::Map` の場合は [`CoreError::InvalidCriteriaType`] を返却する。
    /// - 段階数が仕様範囲外の場合は [`CoreError::InvalidScoreLevelCount`] を返却する。
    /// - ラベルが空文字・空白のみの場合は [`CoreError::EmptyCriteriaLabel`] を返却する。
    /// - ラベルに重複が存在する場合は [`CoreError::DuplicateCriteriaLabel`] を返却する。
    pub fn validate_score_list(&self, question_id: Option<&str>) -> Result<&[String]> {
        let list = self.as_list_or_err(QuestionType::Score, question_id)?;
        let count = list.len();
        if !(MIN_SCORE_LEVELS..=MAX_SCORE_LEVELS).contains(&count) {
            return Err(CoreError::InvalidScoreLevelCount {
                question_id: question_id.map(str::to_string),
                count,
            });
        }
        for (i, label) in list.iter().enumerate() {
            let trimmed = label.trim();
            if trimmed.is_empty() {
                return Err(CoreError::EmptyCriteriaLabel {
                    question_id: question_id.map(str::to_string),
                });
            }
            if list[i + 1..].iter().any(|other| trimmed == other.trim()) {
                return Err(CoreError::DuplicateCriteriaLabel {
                    question_id: question_id.map(str::to_string),
                    label: label.clone(),
                });
            }
        }
        Ok(list)
    }

    /// 候補数または段階数を取得する。
    pub fn len(&self) -> usize {
        match self {
            Self::Map(map) => map.len(),
            Self::List(list) => list.len(),
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

    /// Choice 型質問の `criteria` 存在、型整合性、および候補数境界を検証し、候補マップへの参照を取得する。
    ///
    /// # 引数
    /// - `question_id`: 質問識別子(任意)。エラー発生時に診断メッセージへ反映される。
    pub fn validate_choice_map(
        &self,
        question_id: Option<&str>,
    ) -> Result<&IndexMap<String, String>> {
        let criteria = self
            .criteria
            .as_ref()
            .ok_or_else(|| CoreError::MissingCriteria {
                question_id: question_id.map(str::to_string),
                question_type: QuestionType::Choice,
            })?;
        criteria.validate_choice_map(question_id)
    }

    /// Score 型質問の `criteria` 存在、型整合性、および段階数境界を検証し、段階配列への参照を取得する。
    ///
    /// # 引数
    /// - `question_id`: 質問識別子(任意)。エラー発生時に診断メッセージへ反映される。
    pub fn validate_score_list(&self, question_id: Option<&str>) -> Result<&[String]> {
        let criteria = self
            .criteria
            .as_ref()
            .ok_or_else(|| CoreError::MissingCriteria {
                question_id: question_id.map(str::to_string),
                question_type: QuestionType::Score,
            })?;
        criteria.validate_score_list(question_id)
    }

    /// 質問スキーマの整合性を検証する。
    ///
    /// # 引数
    /// - `question_id`: 質問識別子 (任意)。エラー発生時に診断メッセージへ付与される。
    pub fn validate(&self, question_id: Option<&str>) -> Result<()> {
        match self.question_type {
            QuestionType::Choice => {
                self.validate_choice_map(question_id)?;
            }
            QuestionType::Score => {
                self.validate_score_list(question_id)?;
            }
            QuestionType::Noul => {
                // Noul 型は criteria を要求しない。指定されている場合でも無視または許容する。
            }
        }
        Ok(())
    }
}

/// 質問群マップおよび任意のゲーティング設定パッチの整合性を検証する共通バリデータ。
///
/// # 概要
/// - 質問群が空配列の場合は [`CoreError::EmptyQuestions`] を返却する。
/// - 含まれる各質問について、種別ごとの基準定義 ([`Question::validate`]) を検証する。
/// - 確信度ゲーティングパッチが存在する場合は、指定項目の妥当性 ([`GatingPatch::validate_partial`]) を検証する。
///
/// # 引数
/// - `questions`: 質問群マップ。
/// - `gating`: 任意のゲーティング設定パッチ。
///
/// # 戻り値
/// - 成功時は `Ok(())`、不整合検出時は [`CoreError`]。
pub fn validate_request_payload(
    questions: &IndexMap<String, Question>,
    gating: Option<&GatingPatch>,
) -> Result<()> {
    if questions.is_empty() {
        return Err(CoreError::EmptyQuestions);
    }
    for (question_id, question) in questions {
        question.validate(Some(question_id))?;
    }
    if let Some(gating) = gating {
        gating.validate_partial()?;
    }
    Ok(())
}

/// 決定プリミティブ質問群とゲーティング設定パッチを保持するリクエストペイロードの共通トレイト。
pub trait DecisionRequest {
    /// 評価対象となる質問群のマップへの参照を取得する。
    fn questions(&self) -> &IndexMap<String, Question>;

    /// 確信度ゲーティング処理の設定パッチへの参照を取得する。
    fn gating(&self) -> Option<&GatingPatch>;

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

    /// 確信度ゲーティング処理の設定パラメータパッチ(任意)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gating: Option<GatingPatch>,

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
    pub fn with_gating(mut self, gating: impl Into<GatingPatch>) -> Self {
        self.gating = Some(gating.into());
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

    fn gating(&self) -> Option<&GatingPatch> {
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

    /// Score の加重平均値を取得する。
    pub fn as_score(&self) -> Option<f64> {
        match self {
            Self::Score { value, .. } => Some(*value),
            _ => None,
        }
    }

    /// Noul の真実確率値を取得する。
    pub fn as_noul(&self) -> Option<f64> {
        match self {
            Self::Noul { probability } => Some(*probability),
            _ => None,
        }
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
    pub fn effective_confidence(&self) -> f64 {
        self.payload.effective_confidence()
    }

    /// 判定ペイロードへの不変参照を取得する。
    pub fn payload(&self) -> &AnswerPayload {
        &self.payload
    }

    /// 判定ペイロードへの可変参照を取得する。
    pub fn payload_mut(&mut self) -> &mut AnswerPayload {
        &mut self.payload
    }

    /// 対応する質問種別 (`QuestionType`) を取得する。
    pub fn question_type(&self) -> QuestionType {
        self.payload.question_type()
    }

    /// Choice 型の採択候補ラベルを取得する。
    pub fn as_choice(&self) -> Option<&str> {
        self.payload.as_choice()
    }

    /// Score 型の加重平均値を取得する。
    pub fn as_score(&self) -> Option<f64> {
        self.payload.as_score()
    }

    /// Noul 型の真実確率値を取得する。
    pub fn as_noul(&self) -> Option<f64> {
        self.payload.as_noul()
    }

    /// Choice 判定であるかどうかを判定する。
    pub fn is_choice(&self) -> bool {
        self.payload.is_choice()
    }

    /// Score 判定であるかどうかを判定する。
    pub fn is_score(&self) -> bool {
        self.payload.is_score()
    }

    /// Noul 判定であるかどうかを判定する。
    pub fn is_noul(&self) -> bool {
        self.payload.is_noul()
    }

    /// 確率分布マップの参照を取得する。
    pub fn probabilities(&self) -> Option<&IndexMap<String, f64>> {
        self.payload.probabilities()
    }

    /// 確率分布マップの可変参照を取得する。
    pub fn probabilities_mut(&mut self) -> Option<&mut IndexMap<String, f64>> {
        self.payload.probabilities_mut()
    }

    /// 確率分布マップを更新する。Choice / Score 型の場合に適用される。
    pub fn set_probabilities(&mut self, new_probabilities: IndexMap<String, f64>) {
        self.payload.set_probabilities(new_probabilities);
    }

    /// 正規化確信度を取得する。
    pub fn confidence(&self) -> Option<f64> {
        self.payload.confidence()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_criteria_as_map_or_err() {
        let mut map = IndexMap::new();
        map.insert("k1".to_string(), "v1".to_string());
        let criteria_map = Criteria::Map(map);
        let criteria_list = Criteria::List(vec!["item1".to_string()]);

        assert!(
            criteria_map
                .as_map_or_err(QuestionType::Choice, Some("q1"))
                .is_ok()
        );

        let err_list = criteria_list
            .as_map_or_err(QuestionType::Choice, Some("q1"))
            .unwrap_err();
        match err_list {
            CoreError::InvalidCriteriaType {
                question_id,
                question_type,
                expected,
                actual,
            } => {
                assert_eq!(question_id, Some("q1".to_string()));
                assert_eq!(question_type, QuestionType::Choice);
                assert_eq!(expected, "Map (オブジェクト)");
                assert_eq!(actual, "List (配列)");
            }
            _ => panic!("予期しないエラー型です: {err_list:?}。"),
        }
    }

    #[test]
    fn test_criteria_as_list_or_err() {
        let mut map = IndexMap::new();
        map.insert("k1".to_string(), "v1".to_string());
        let criteria_map = Criteria::Map(map);
        let criteria_list = Criteria::List(vec!["item1".to_string()]);

        assert!(
            criteria_list
                .as_list_or_err(QuestionType::Score, Some("q2"))
                .is_ok()
        );

        let err_map = criteria_map
            .as_list_or_err(QuestionType::Score, Some("q2"))
            .unwrap_err();
        match err_map {
            CoreError::InvalidCriteriaType {
                question_id,
                question_type,
                expected,
                actual,
            } => {
                assert_eq!(question_id, Some("q2".to_string()));
                assert_eq!(question_type, QuestionType::Score);
                assert_eq!(expected, "List (配列)");
                assert_eq!(actual, "Map (オブジェクト)");
            }
            _ => panic!("予期しないエラー型です: {err_map:?}。"),
        }
    }

    #[test]
    fn test_validate_choice_map_empty_and_duplicate() {
        // 1. 正常系
        let valid_map = IndexMap::from([
            ("opt1".to_string(), "説明 1".to_string()),
            ("opt2".to_string(), "説明 2".to_string()),
        ]);
        let crit_valid = Criteria::Map(valid_map);
        assert!(crit_valid.validate_choice_map(Some("q")).is_ok());

        // 2. 空白のみのキー
        let ws_key_map = IndexMap::from([
            ("   ".to_string(), "説明 1".to_string()),
            ("opt2".to_string(), "説明 2".to_string()),
        ]);
        let crit_ws_key = Criteria::Map(ws_key_map);
        assert!(matches!(
            crit_ws_key.validate_choice_map(Some("q")),
            Err(CoreError::EmptyCriteriaLabel { .. })
        ));

        // 3. 空白のみの説明文
        let ws_desc_map = IndexMap::from([
            ("opt1".to_string(), "  \t  ".to_string()),
            ("opt2".to_string(), "説明 2".to_string()),
        ]);
        let crit_ws_desc = Criteria::Map(ws_desc_map);
        assert!(matches!(
            crit_ws_desc.validate_choice_map(Some("q")),
            Err(CoreError::EmptyCriteriaLabel { .. })
        ));

        // 4. 重複キー (トリム後一致)
        let dup_key_map = IndexMap::from([
            ("opt1".to_string(), "説明 1".to_string()),
            (" opt1 ".to_string(), "説明 2".to_string()),
        ]);
        let crit_dup_key = Criteria::Map(dup_key_map);
        assert!(matches!(
            crit_dup_key.validate_choice_map(Some("q")),
            Err(CoreError::DuplicateCriteriaLabel { .. })
        ));

        // 5. 重複説明文
        let dup_desc_map = IndexMap::from([
            ("opt1".to_string(), "同じ説明".to_string()),
            ("opt2".to_string(), " 同じ説明 ".to_string()),
        ]);
        let crit_dup_desc = Criteria::Map(dup_desc_map);
        assert!(matches!(
            crit_dup_desc.validate_choice_map(Some("q")),
            Err(CoreError::DuplicateCriteriaLabel { ref label, .. }) if label == "同じ説明"
        ));
    }

    #[test]
    fn test_validate_score_list_empty_and_duplicate() {
        // 1. 正常系
        let valid_list = vec!["低".to_string(), "中".to_string(), "高".to_string()];
        let crit_valid = Criteria::List(valid_list);
        assert!(crit_valid.validate_score_list(Some("q")).is_ok());

        // 2. 空白ラベル
        let ws_list = vec!["低".to_string(), "  \n  ".to_string(), "高".to_string()];
        let crit_ws = Criteria::List(ws_list);
        assert!(matches!(
            crit_ws.validate_score_list(Some("q")),
            Err(CoreError::EmptyCriteriaLabel { .. })
        ));

        // 3. 重複ラベル (トリム後一致)
        let dup_list = vec!["低".to_string(), "中".to_string(), " 低 ".to_string()];
        let crit_dup = Criteria::List(dup_list);
        assert!(matches!(
            crit_dup.validate_score_list(Some("q")),
            Err(CoreError::DuplicateCriteriaLabel { ref label, .. }) if label == "低"
        ));
    }
}
