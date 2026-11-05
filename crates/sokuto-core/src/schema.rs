//! # Jev 互換 API スキーマおよび型定義モジュール
//!
//! TypeSafe AI の `POST /v1/systemone` リクエストおよびレスポンスと
//! 完全な互換性を持つデータ構造を定義する。

use std::fmt;

use indexmap::IndexMap;
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
    pub fn validate(&self, _question_id: &str) -> Result<()> {
        match self.question_type {
            QuestionType::Choice => {
                let criteria =
                    self.criteria
                        .as_ref()
                        .ok_or_else(|| CoreError::MissingCriteria {
                            question_type: "choice".to_string(),
                        })?;
                match criteria {
                    Criteria::Map(map) => {
                        let count = map.len();
                        if count == 0 || count > 255 {
                            return Err(CoreError::InvalidChoiceCount { count });
                        }
                    }
                    Criteria::List(_) => {
                        return Err(CoreError::InvalidCriteriaType {
                            question_type: "choice".to_string(),
                            expected: "Map (オブジェクト)",
                            actual: "List (配列)",
                        });
                    }
                    Criteria::None => {
                        return Err(CoreError::MissingCriteria {
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
                            question_type: "score".to_string(),
                        })?;
                match criteria {
                    Criteria::List(list) => {
                        let count = list.len();
                        if !(2..=10).contains(&count) {
                            return Err(CoreError::InvalidScoreLevelCount { count });
                        }
                    }
                    Criteria::Map(_) => {
                        return Err(CoreError::InvalidCriteriaType {
                            question_type: "score".to_string(),
                            expected: "List (配列)",
                            actual: "Map (オブジェクト)",
                        });
                    }
                    Criteria::None => {
                        return Err(CoreError::MissingCriteria {
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
    pub fn with_model(
        model: impl Into<String>,
        state: impl Into<serde_json::Value>,
        questions: IndexMap<String, Question>,
    ) -> Self {
        Self {
            model: Some(model.into()),
            state: state.into(),
            questions,
            gating: None,
            auto_hierarchical: None,
            state_hash: None,
        }
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
        if self.questions.is_empty() {
            return Err(CoreError::EmptyQuestions);
        }
        for (question_id, question) in &self.questions {
            question.validate(question_id)?;
        }
        if let Some(ref gating) = self.gating {
            gating.validate()?;
        }
        Ok(())
    }
}

/// 単一の質問に対する判定結果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Answer {
    /// Choice 型の採択候補ラベル。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub choice: Option<String>,

    /// Score 型の加重平均スコア値。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,

    /// Noul 型の言明真実確率値(0.0 <= P <= 1.0)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noul: Option<f64>,

    /// 各候補または各段階レベルの確率分布マップ。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub probabilities: Option<IndexMap<String, f64>>,

    /// 分布の尖り度に基づく正規化確信度(0.0 <= C <= 1.0)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,

    /// 確信度ゲーティングによるルーティングおよび監査メタデータ(任意)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gating: Option<GatingMetadata>,
}

impl Answer {
    /// Choice 判定結果を生成する。
    pub fn choice(
        selected: impl Into<String>,
        probabilities: IndexMap<String, f64>,
        confidence: f64,
    ) -> Self {
        Self {
            choice: Some(selected.into()),
            score: None,
            noul: None,
            probabilities: Some(probabilities),
            confidence: Some(confidence),
            gating: None,
        }
    }

    /// Score 判定結果を生成する。
    pub fn score(value: f64, probabilities: IndexMap<String, f64>, confidence: f64) -> Self {
        Self {
            choice: None,
            score: Some(value),
            noul: None,
            probabilities: Some(probabilities),
            confidence: Some(confidence),
            gating: None,
        }
    }

    /// Noul 判定結果を生成する。
    pub fn noul(probability: f64) -> Self {
        Self {
            choice: None,
            score: None,
            noul: Some(probability),
            probabilities: None,
            confidence: None,
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
    /// - Noul 型の場合は、Jev 公式スキーマでは `confidence` フィールドが `None` となる規約があるため、
    ///   真実確率 $P(\text{true})$ から尖り度 $|2P - 1.0|$ を即座に算出して返却する。
    ///
    /// # 戻り値
    /// - 判定結果に対応する 0.0〜1.0 の確信度実数値。判定結果が存在しない場合は `None`。
    pub fn effective_confidence(&self) -> Option<f64> {
        if let Some(conf) = self.confidence {
            return Some(conf);
        }
        if let Some(p) = self.noul {
            return Some((2.0 * p - 1.0).abs().clamp(0.0, 1.0));
        }
        None
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct StateBatchRequest {
    /// 使用するローカルモデル識別子 (任意)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,

    /// 判断の材料となる単一の長文コンテキストデータ。
    #[cfg_attr(feature = "openapi", schema(value_type = Object))]
    pub state: serde_json::Value,

    /// 評価対象となる複数質問群のマップ (キーは質問識別子)。
    pub questions: IndexMap<String, Question>,

    /// 確信度ゲーティング処理の設定パラメータ (任意)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gating: Option<GatingConfig>,

    /// 候補数 $K > 16$ の `Choice` 質問に対する透過的粗密階層ルーティングの制御フラグ (任意)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auto_hierarchical: Option<bool>,

    /// クライアント側事前計算または以前取得した `State` キャッシュキー (任意)。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_hash: Option<String>,
}

impl StateBatchRequest {
    /// 新規バッチリクエストを作成する。
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

    /// `State` キャッシュキーを指定してバッチリクエストを作成する。
    pub fn with_state_hash(mut self, state_hash: impl Into<String>) -> Self {
        self.state_hash = Some(state_hash.into());
        self
    }

    /// モデル名を指定してバッチリクエストを作成する。
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// ゲーティング設定を指定してバッチリクエストを作成する。
    pub fn with_gating(mut self, gating: GatingConfig) -> Self {
        self.gating = Some(gating);
        self
    }

    /// 粗密階層ルーティング制御フラグを指定してバッチリクエストを作成する。
    pub fn with_auto_hierarchical(mut self, auto_hierarchical: bool) -> Self {
        self.auto_hierarchical = Some(auto_hierarchical);
        self
    }

    /// リクエスト全体の整合性を検証する。
    pub fn validate(&self) -> Result<()> {
        if self.questions.is_empty() {
            return Err(CoreError::EmptyQuestions);
        }
        for (question_id, question) in &self.questions {
            question.validate(question_id)?;
        }
        if let Some(ref gating) = self.gating {
            gating.validate()?;
        }
        Ok(())
    }
}

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
