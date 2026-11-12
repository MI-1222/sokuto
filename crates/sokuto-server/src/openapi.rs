//! # OpenAPI 3.1 仕様統合モジュール
//!
//! `utoipa` によるコンパイル時スキーマ自動導出、Swagger UI マウント、
//! および CLI/CI 用の静的スキーマエクスポート機能を提供する。

use axum::Router;
use axum::response::IntoResponse;
use sokuto_core::gating::{
    CandidateProbability, DecisionRoute, EscalationContext, GatingConfig, GatingMetadata,
    GatingPatch, GatingThresholdsPatch, SystemRoutingSummary,
};
use sokuto_core::schema::{
    Answer, Criteria, Question, QuestionType, StateBatchRequest, StateBatchResponse,
    SystemOneRequest, SystemOneResponse, Usage,
};
use utoipa::OpenApi;

use crate::error::{ErrorDetail, ErrorResponse};
use crate::handlers::ops::HealthStatusResponse;
use crate::schema::dag::{
    DagDefinitionDto, DagEscalationDetailDto, DagNodeDto, DagNodeTypeDto, DagRequest, DagResponse,
    DagStepResultDto, DagUsageDto, NodeConditionDto, NodeConditionOpDto,
};

/// Jev 互換 HTTP API サーバーの OpenAPI 仕様定義。
#[derive(OpenApi)]
#[openapi(
    paths(
        crate::handlers::systemone::system_one_handler,
        crate::handlers::state_batch::state_batch_handler,
        crate::handlers::dag::dag_handler,
        crate::handlers::ops::health_handler,
        crate::handlers::ops::ready_handler,
        crate::handlers::ops::metrics_handler,
    ),
    components(
        schemas(
            SystemOneRequest,
            SystemOneResponse,
            StateBatchRequest,
            StateBatchResponse,
            Question,
            QuestionType,
            Criteria,
            Answer,
            Usage,
            DecisionRoute,
            GatingConfig,
            GatingPatch,
            GatingThresholdsPatch,
            GatingMetadata,
            CandidateProbability,
            EscalationContext,
            SystemRoutingSummary,
            ErrorResponse,
            ErrorDetail,
            HealthStatusResponse,
            DagRequest,
            DagResponse,
            DagDefinitionDto,
            DagNodeDto,
            DagNodeTypeDto,
            NodeConditionDto,
            NodeConditionOpDto,
            DagStepResultDto,
            DagEscalationDetailDto,
            DagUsageDto,
        )
    ),
    tags(
        (name = "Inference", description = "TypeSafe AI Jev 互換の非自己回帰型判断推論 API"),
        (name = "Operations", description = "ヘルスチェックおよび Prometheus 監視運用 API")
    ),
    info(
        title = "Sokuto HTTP API",
        version = "0.0.0",
        description = "TypeSafe AI Jev 互換の非自己回帰型判断特化モデル向け超低遅延 REST API 仕様書。"
    )
)]
pub struct ApiDoc;

/// OpenAPI 3.1 仕様ドキュメント構造体を生成する。
pub fn generate_openapi_spec() -> utoipa::openapi::OpenApi {
    let mut spec = ApiDoc::openapi();
    spec.info.version = env!("CARGO_PKG_VERSION").to_string();
    spec
}

/// 整形済み OpenAPI 3.1 JSON 文字列をエクスポートする。
pub fn export_openapi_json() -> String {
    generate_openapi_spec()
        .to_pretty_json()
        .expect("OpenAPI 仕様の JSON シリアライズに失敗しました。")
}

/// OpenAPI JSON 仕様エンドポイントのハンドラ。
pub async fn openapi_json_handler() -> impl IntoResponse {
    (
        [("content-type", "application/json; charset=utf-8")],
        export_openapi_json(),
    )
}

/// Swagger UI をルーターにマウントする。
///
/// `swagger` フィーチャーが無効化されている場合は何も追加せずルーターをそのまま返却する。
pub fn mount_swagger_ui<S>(router: Router<S>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    #[cfg(feature = "swagger")]
    {
        use utoipa_swagger_ui::SwaggerUi;
        router.merge(
            SwaggerUi::new("/swagger-ui").url("/api-docs/openapi.json", generate_openapi_spec()),
        )
    }

    #[cfg(not(feature = "swagger"))]
    {
        router.route(
            "/api-docs/openapi.json",
            axum::routing::get(openapi_json_handler),
        )
    }
}
