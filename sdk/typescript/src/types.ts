/**
 * カスケード統合 SDK 向け共通型定義モジュール。
 *
 * System 1 (sokuto) と System 2 (フロンティア自己回帰 LLM) のハイブリッド実行における
 * 3軸ゲーティング判定結果、サーキットブレーカー状態、対比プロンプト定義、
 * および統合実行結果インターフェースを提供する。
 */

/**
 * ゲーティング判定に基づく意思決定ルート種別。
 */
export type DecisionRoute = "auto_execute" | "confirm_or_escalate" | "fallback";

/**
 * 判定を出力した実行主体。
 */
export type CascadeSource = "system1" | "system2";

/**
 * 3軸 Pareto 最適ゲーティングの閾値設定。
 */
export interface GatingConfig {
  /** Top-Margin 判定閾値 (デフォルト: 0.15)。 */
  marginThreshold?: number;
  /** 正規化シャノンエントロピー上限値 (デフォルト: 0.65)。 */
  entropyThreshold?: number;
  /** ヘルムホルツ自由エネルギー上限値 (デフォルト: -1.0)。 */
  energyThreshold?: number;
  /** Score 型における許容正規化分散上限値 (デフォルト: 0.50)。 */
  scoreVarThreshold?: number;
  /** Score 型における双峰性係数限界値 (デフォルト: 0.555)。 */
  bimodalityThreshold?: number;
  /** 自由エネルギー計算用温度パラメータ。 */
  energyTemperature?: number;
}

/**
 * 3軸 Pareto ゲーティングの判定結果および診断メトリクス。
 */
export interface GatingDecision {
  /** 確定されたルーティング種別。 */
  route: DecisionRoute;
  /** エスカレーション要否フラグ。 */
  escalate: boolean;
  /** 実効確信度 (0.0 <= conf <= 1.0)。 */
  confidence: number;
  /** 上位2候補差分 (Top-Margin)。 */
  topMargin?: number;
  /** 正規化シャノンエントロピー。 */
  normalizedEntropy?: number;
  /** ヘルムホルツ自由エネルギー。 */
  freeEnergy?: number;
  /** 未知ドメイン (OOD) 検知フラグ。 */
  isOod: boolean;
  /** 判定根拠の説明文。 */
  reason: string;
  /** 確率上位候補と確率のペア配列。 */
  topCandidates: Array<[string, number]>;
}

/**
 * アンカリング思考汚染防止ニュートラル対比型プロンプト。
 */
export interface TriagePrompt {
  /** 中立検証を促すシステムプロンプト。 */
  systemPrompt: string;
  /** 対照形式のユーザープロンプト。 */
  userPrompt: string;
  /** 要求する構造化出力 JSON スキーマ。 */
  structuredSchema: Record<string, unknown>;
}

/**
 * カスケード推論の最終確定結果コンテナ。
 */
export interface CascadeResult<T = unknown> {
  /** 確定された判定値。 */
  decision: T;
  /** 判定元システム ("system1" または "system2")。 */
  source: CascadeSource;
  /** 判定時の 3 軸ゲーティング詳細。 */
  gating: GatingDecision;
  /** 処理所要時間 (ミリ秒)。 */
  latencyMs: number;
  /** System 2 への委任根拠理由。 */
  escalationReason?: string;
  /** System 2 が出力した思考連鎖 (CoT)。 */
  system2Thinking?: string;
  /** 推論を担当したモデル識別子。 */
  modelName?: string;
  /** System 1 の生レスポンスデータ。 */
  rawS1Response?: Record<string, unknown>;
  /** System 2 の生レスポンスデータ。 */
  rawS2Response?: Record<string, unknown>;
}

/**
 * sokuto 質問定義スキーマ。
 */
export interface QuestionSpec {
  /** 質問の指示文 (Prompt)。 */
  instructions: string;
  /** プリミティブ種別 (choice, score, noul)。 */
  type: "choice" | "score" | "noul" | string;
  /** 評価基準または選択肢候補。 */
  criteria?: Record<string, string> | string[] | null;
}

/**
 * sokuto 推論リクエストペイロード。
 */
export interface SokutoRequest {
  state: unknown;
  questions: Record<string, QuestionSpec>;
  model?: string;
}

/**
 * sokuto 推論レスポンスペイロード。
 */
export interface SokutoResponse {
  model_name?: string;
  answers: Record<
    string,
    {
      choice?: string;
      score?: number;
      noul?: number;
      confidence?: number;
      probabilities?: Record<string, number> | number[];
      gating?: {
        route?: string;
        confidence?: number;
        energy?: number;
        free_energy?: number;
        margin?: number;
        entropy?: number;
        is_ood?: boolean;
        reason?: string;
      };
    }
  >;
}
