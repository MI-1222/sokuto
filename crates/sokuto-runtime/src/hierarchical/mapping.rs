//! # 粗密階層マッピングモジュール
//!
//! 大分類 (Coarse) と細分類 (Fine) のオントロジー定義、双方向インデックス、
//! および MECE (相互排他的かつ網羅的) 性のバリデーションを提供する。

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::engine::coarse::{DEFAULT_NEGATIVE_KEYS, is_negative_candidate};
use crate::error::{Result, RuntimeError};

/// 大分類と細分類の対応関係を保持する階層マッピング構造体。
///
/// 大分類 (4〜8 クラスタ) と細分類 (各 8〜12 候補) の双方向高速参照を実現する。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "RawHierarchicalMapping")]
pub struct HierarchicalMapping {
    /// マッピング全体の識別名。
    pub name: String,
    /// 大分類の識別子と説明文のマッピング。
    pub coarse_categories: IndexMap<String, String>,
    /// 全細分類の識別子と説明文のマッピング。
    pub fine_criteria: IndexMap<String, String>,
    /// 大分類識別子から配下の細分類識別子リストへのマッピング。
    pub coarse_to_fine: IndexMap<String, Vec<String>>,
    /// 細分類識別子から所属する大分類識別子への逆引きマッピング。
    pub fine_to_coarse: IndexMap<String, String>,
    /// 事前キャッシュされた全大分類キーの順序保持リスト。
    pub all_coarse_keys: Vec<String>,
    /// 事前キャッシュされた全細分類キーの順序保持リスト。
    pub all_fine_keys: Vec<String>,
    /// 受け皿 (None / その他) 候補として判定するキー一覧。
    pub negative_keys: Vec<String>,
}

#[derive(Deserialize)]
struct RawHierarchicalMapping {
    name: String,
    coarse_categories: IndexMap<String, String>,
    fine_criteria: IndexMap<String, String>,
    coarse_to_fine: IndexMap<String, Vec<String>>,
    fine_to_coarse: IndexMap<String, String>,
    #[serde(default)]
    all_coarse_keys: Vec<String>,
    #[serde(default)]
    all_fine_keys: Vec<String>,
    #[serde(default)]
    negative_keys: Vec<String>,
}

impl From<RawHierarchicalMapping> for HierarchicalMapping {
    fn from(raw: RawHierarchicalMapping) -> Self {
        let all_coarse_keys = if raw.all_coarse_keys.is_empty() {
            raw.coarse_categories.keys().cloned().collect()
        } else {
            raw.all_coarse_keys
        };
        let all_fine_keys = if raw.all_fine_keys.is_empty() {
            raw.fine_criteria.keys().cloned().collect()
        } else {
            raw.all_fine_keys
        };
        Self {
            name: raw.name,
            coarse_categories: raw.coarse_categories,
            fine_criteria: raw.fine_criteria,
            coarse_to_fine: raw.coarse_to_fine,
            fine_to_coarse: raw.fine_to_coarse,
            all_coarse_keys,
            all_fine_keys,
            negative_keys: raw.negative_keys,
        }
    }
}

impl HierarchicalMapping {
    /// 新しいビルダーを生成する。
    pub fn builder(name: impl Into<String>) -> HierarchicalMappingBuilder {
        HierarchicalMappingBuilder::new(name)
    }

    /// マッピングの MECE 性および整合性を検証する。
    ///
    /// # 検証項目
    /// - 大分類が 1 つ以上定義されていること。
    /// - 各大分類に少なくとも 1 つ以上の細分類が割り当てられていること。
    /// - すべての細分類が `fine_criteria` に定義されていること。
    /// - すべての細分類がいずれか 1 つの大分類にのみ重複なく所属していること。
    /// - `coarse_to_fine` と `fine_to_coarse` が厳密に双方向一致していること。
    /// - 事前キャッシュ `all_coarse_keys` が `coarse_categories` のキー順序および一覧と完全一致していること。
    /// - 事前キャッシュ `all_fine_keys` が `fine_criteria` のキー順序および一覧と完全一致していること。
    pub fn validate(&self) -> Result<()> {
        if self.coarse_categories.is_empty() {
            return Err(RuntimeError::InvalidQuestion(
                "大分類 (coarse_categories) が 1 つも定義されていません。".to_string(),
            ));
        }

        if self.fine_criteria.is_empty() {
            return Err(RuntimeError::InvalidQuestion(
                "細分類 (fine_criteria) が空です。".to_string(),
            ));
        }

        // 1. 各大分類の重複および MECE 性検証
        let mut seen_fine_keys = indexmap::IndexSet::new();
        for fine_list in self.coarse_to_fine.values() {
            for fine_key in fine_list {
                if !seen_fine_keys.insert(fine_key.clone()) {
                    return Err(RuntimeError::InvalidQuestion(format!(
                        "細分類 `{}` が複数の大分類に重複して割り当てられています (MECE 違反)。",
                        fine_key
                    )));
                }
            }
        }

        // 2. 大分類配下の細分類検証
        for (coarse_key, coarse_desc) in &self.coarse_categories {
            if coarse_key.trim().is_empty() {
                return Err(RuntimeError::InvalidQuestion(
                    "空の大分類キーが存在します。".to_string(),
                ));
            }
            if coarse_desc.trim().is_empty() {
                return Err(RuntimeError::InvalidQuestion(format!(
                    "大分類 `{}` の説明文が空です。",
                    coarse_key
                )));
            }

            let fine_list = self.coarse_to_fine.get(coarse_key).ok_or_else(|| {
                RuntimeError::InvalidQuestion(format!(
                    "大分類 `{}` に対する細分類リストが定義されていません。",
                    coarse_key
                ))
            })?;

            if fine_list.is_empty() {
                return Err(RuntimeError::InvalidQuestion(format!(
                    "大分類 `{}` 配下の細分類リストが空です。",
                    coarse_key
                )));
            }

            for fine_key in fine_list {
                if !self.fine_criteria.contains_key(fine_key) {
                    return Err(RuntimeError::InvalidQuestion(format!(
                        "大分類 `{}` 配下の細分類 `{}` が fine_criteria に存在しません。",
                        coarse_key, fine_key
                    )));
                }

                let mapped_coarse = self.fine_to_coarse.get(fine_key).ok_or_else(|| {
                    RuntimeError::InvalidQuestion(format!(
                        "細分類 `{}` から大分類への逆引きエントリが存在しません。",
                        fine_key
                    ))
                })?;

                if mapped_coarse != coarse_key {
                    return Err(RuntimeError::InvalidQuestion(format!(
                        "細分類 `{}` の逆引き大分類 (`{}`) が所属大分類 (`{}`) と一致しません。",
                        fine_key, mapped_coarse, coarse_key
                    )));
                }
            }
        }

        // 定義された全細分類がカバーされているかを検証
        if seen_fine_keys.len() != self.fine_criteria.len() {
            let missing_keys: Vec<_> = self
                .fine_criteria
                .keys()
                .filter(|k| !seen_fine_keys.contains(*k))
                .cloned()
                .collect();
            return Err(RuntimeError::InvalidQuestion(format!(
                "いずれの大分類にも属していない孤立細分類が存在します: {:?}",
                missing_keys
            )));
        }

        // 3. 事前キャッシュキーの完全一致・整合性検証
        if self.all_coarse_keys.len() != self.coarse_categories.len()
            || self
                .all_coarse_keys
                .iter()
                .zip(self.coarse_categories.keys())
                .any(|(a, b)| a != b)
        {
            return Err(RuntimeError::InvalidQuestion(
                "事前キャッシュ `all_coarse_keys` が `coarse_categories` のキー順序および一覧と一致していません。".to_string(),
            ));
        }

        if self.all_fine_keys.len() != self.fine_criteria.len()
            || self
                .all_fine_keys
                .iter()
                .zip(self.fine_criteria.keys())
                .any(|(a, b)| a != b)
        {
            return Err(RuntimeError::InvalidQuestion(
                "事前キャッシュ `all_fine_keys` が `fine_criteria` のキー順序および一覧と一致していません。".to_string(),
            ));
        }

        Ok(())
    }

    /// 事前キャッシュされた全大分類キーおよび全細分類キーを、マップ定義から再構築する。
    pub fn rebuild_key_caches(&mut self) {
        self.all_coarse_keys = self.coarse_categories.keys().cloned().collect();
        self.all_fine_keys = self.fine_criteria.keys().cloned().collect();
    }

    /// 大分類のキーおよび説明文マップを取得する。
    pub fn coarse_criteria(&self) -> &IndexMap<String, String> {
        &self.coarse_categories
    }

    /// 指定された大分類キー配下の細分類識別子スライスを取得する。
    pub fn fine_keys_for_coarse(&self, coarse_key: &str) -> Option<&[String]> {
        self.coarse_to_fine.get(coarse_key).map(|v| v.as_slice())
    }

    /// 指定された細分類が属する大分類キーを取得する。
    pub fn coarse_for_fine(&self, fine_key: &str) -> Option<&str> {
        self.fine_to_coarse.get(fine_key).map(|s| s.as_str())
    }

    /// 登録されている細分類の総数を取得する。
    pub fn total_fine_count(&self) -> usize {
        self.fine_criteria.len()
    }

    /// 登録されている大分類の総数を取得する。
    pub fn total_coarse_count(&self) -> usize {
        self.coarse_categories.len()
    }

    /// 事前キャッシュされた全細分類キーの参照スライスを取得する。
    pub fn all_fine_keys(&self) -> &[String] {
        &self.all_fine_keys
    }

    /// 事前キャッシュされた全大分類キーの参照スライスを取得する。
    pub fn all_coarse_keys(&self) -> &[String] {
        &self.all_coarse_keys
    }

    /// 指定された候補が受け皿 (None / その他) 候補であるかを判定する。
    pub fn is_negative_candidate(&self, key: &str, description: &str) -> bool {
        is_negative_candidate(key, description, &self.negative_keys)
    }

    /// Banking77 向け 7 大分類プリセットを構築する。
    ///
    /// 77 候補の金融インテントを 7 個の大分類クラスタへ網羅的に分類する。
    pub fn banking77() -> Self {
        let mut builder = Self::builder("Banking77-7Clusters");

        // 1. カード発行・利用管理 (card_ops: 12 件)
        builder.add_category(
            "card_ops",
            "カード発行・配送・更新・有効化および利用設定",
            vec![
                (
                    "activate_my_card",
                    "新しいカードを有効化・利用開始登録したい。",
                ),
                (
                    "card_about_to_expire",
                    "カードの有効期限が近づいているため更新したい。",
                ),
                (
                    "card_acceptance",
                    "カードが加盟店や決済端末で利用可能か確認したい。",
                ),
                (
                    "card_arrival",
                    "申し込んだカードがいつ届くか、配送状況を確認したい。",
                ),
                (
                    "card_delivery_estimate",
                    "カードの発送時期や到着予定日を知りたい。",
                ),
                (
                    "card_linking",
                    "カードをアプリや外部アカウントに連携・登録したい。",
                ),
                (
                    "card_not_working",
                    "カード決済や読み取りが正常に動作しない。",
                ),
                (
                    "card_swallowed",
                    "ATM や端末にカードが吸い込まれて戻らない。",
                ),
                (
                    "getting_spare_card",
                    "予備カードや追加カードの発行を申請したい。",
                ),
                (
                    "getting_virtual_card",
                    "即時利用可能なバーチャルカードを発行したい。",
                ),
                (
                    "order_physical_card",
                    "プラスチック製の物理カードを新規発行したい。",
                ),
                (
                    "pin_blocked",
                    "PIN コードの入力間違い等で暗証番号がロックされた。",
                ),
            ],
        );

        // 2. 口座間送金・振込 (transfers: 11 件)
        builder.add_category(
            "transfers",
            "口座間送金・銀行振込・送金エラーおよび送金取消",
            vec![
                (
                    "balance_not_updated_after_bank_transfer",
                    "銀行振込を行ったが残高に反映されない。",
                ),
                (
                    "beneficiary_not_allowed",
                    "振込先口座が登録できない、または送金不可と表示される。",
                ),
                (
                    "cancel_transfer",
                    "実行済みまたは予約中の送金・振込を取り消したい。",
                ),
                (
                    "failed_transfer",
                    "送金手続きそのものが失敗・エラーになった。",
                ),
                (
                    "pending_transfer",
                    "送金処理が保留中 (Pending) のまま完了しない。",
                ),
                (
                    "transfer_fee_charged",
                    "送金時に想定外の手数料が差し引かれた。",
                ),
                (
                    "transfer_into_account",
                    "外部口座から本口座への振込方法や着金を知りたい。",
                ),
                (
                    "transfer_not_received_by_recipient",
                    "送金完了通知が出たが受取人に届いていない。",
                ),
                (
                    "transfer_timing",
                    "送金が相手に届くまでの所要時間やスケジュールを知りたい。",
                ),
                (
                    "unable_to_verify_identity_for_transfer",
                    "送金時の本人確認ステップを通過できない。",
                ),
                (
                    "wrong_amount_sent",
                    "振込金額を誤って多く（または少なく）送金してしまった。",
                ),
            ],
        );

        // 3. 店舗・オンライン決済・手数料 (payments: 11 件)
        builder.add_category(
            "payments",
            "店舗・オンライン決済・カード決済手数料および二重請求",
            vec![
                (
                    "card_payment_fee_charged",
                    "カード決済時に余分な手数料が請求された。",
                ),
                (
                    "card_payment_not_recognised",
                    "カードの利用明細に身に覚えのない請求がある。",
                ),
                (
                    "card_payment_wrong_exchange_rate",
                    "海外決済時の為替レートが想定と異なる。",
                ),
                (
                    "declined_card_payment",
                    "店舗またはオンラインでのカード決済が拒否された。",
                ),
                (
                    "direct_debit_payment_problem",
                    "口座引き落とし・自動引き落としの支払いに失敗した。",
                ),
                (
                    "disposable_card_limits",
                    "使い捨てバーチャルカードの利用限度額や制限を知りたい。",
                ),
                (
                    "extra_charge_on_statement",
                    "請求明細に身に覚えのない追加料金が加算されている。",
                ),
                (
                    "pending_card_payment",
                    "カード決済の引き落としが保留中になっている。",
                ),
                (
                    "refund_not_showing_up",
                    "返品・キャンセルの返金が口座に反映されない。",
                ),
                (
                    "transaction_charged_twice",
                    "1 回の買い物で同じ金額が二重に決済・請求された。",
                ),
                (
                    "unable_to_refund",
                    "アプリ上から返金処理・キャンセル処理を実行できない。",
                ),
            ],
        );

        // 4. ATM 出金・現金取扱 (cash_atm: 9 件)
        builder.add_category(
            "cash_atm",
            "ATM 出金・現金引き出し・ATM 手数料トラブル",
            vec![
                (
                    "atm_support",
                    "利用可能な提携 ATM や利用方法について知りたい。",
                ),
                (
                    "cash_withdrawal_charge",
                    "ATM 現金引き出し時に想定外の手数料が引かれた。",
                ),
                (
                    "cash_withdrawal_not_recognised",
                    "利用明細に身に覚えのない ATM 現金引き出しがある。",
                ),
                (
                    "declined_cash_withdrawal",
                    "ATM での現金引き出しが拒否・失敗した。",
                ),
                (
                    "supported_cards_currencies_at_atm",
                    "ATM で利用可能なカード種別や対応通貨を知りたい。",
                ),
                (
                    "wrong_amount_of_cash_received",
                    "ATM から指定と異なる金額の現金が出てきた。",
                ),
                (
                    "wrong_exchange_rate_for_cash_withdrawal",
                    "海外 ATM で出金した際の為替レートが不当に高い。",
                ),
                (
                    "contactless_not_working",
                    "非接触 (タッチ) 決済がリーダーで反応しない。",
                ),
                (
                    "declined_transfer",
                    "システム規制や不正検知により送金が強制拒否された。",
                ),
            ],
        );

        // 5. 残高チャージ・入金 (topup: 11 件)
        builder.add_category(
            "topup",
            "残高チャージ・入金反映・チャージ失敗および限度額",
            vec![
                (
                    "automatic_top_up",
                    "残高連動のオートチャージ (自動入金) を設定・解除したい。",
                ),
                (
                    "balance_not_updated_after_cheque_or_cash_deposit",
                    "小切手や現金で入金したが口座残高に反映されない。",
                ),
                (
                    "pending_top_up",
                    "チャージ処理が処理中・保留のまま残高に加算されない。",
                ),
                (
                    "reverted_card_payment",
                    "カード決済が店舗側により取り消し・差し戻された。",
                ),
                (
                    "top_up_by_bank_transfer_charge",
                    "銀行振込による残高チャージで手数料が発生した。",
                ),
                (
                    "top_up_by_card_charge",
                    "クレジットカード等からのチャージで手数料が発生した。",
                ),
                (
                    "top_up_by_cash_or_cheque",
                    "現金や小切手による口座残高のチャージ方法を知りたい。",
                ),
                (
                    "top_up_failed",
                    "クレジットカード等からの残高チャージ操作が失敗した。",
                ),
                (
                    "top_up_limits",
                    "残高チャージの上限額や回数制限を確認したい。",
                ),
                (
                    "top_up_reverted",
                    "チャージした金額が取り消され元に戻ってしまった。",
                ),
                (
                    "verify_top_up",
                    "残高チャージ時に追加の認証や確認を求められた。",
                ),
            ],
        );

        // 6. 口座設定・認証・セキュリティ (account_sec: 13 件)
        builder.add_category(
            "account_sec",
            "口座設定・認証情報・紛失盗難対応・セキュリティ",
            vec![
                ("change_pin", "カードの暗証番号 (PIN) を変更したい。"),
                (
                    "compromised_card",
                    "カード番号が漏洩・不正利用された疑いがある。",
                ),
                (
                    "edit_personal_details",
                    "氏名、住所、電話番号などの登録個人情報を更新したい。",
                ),
                (
                    "lost_or_stolen_card",
                    "カードを紛失した、または盗難に遭ったため即時停止したい。",
                ),
                (
                    "lost_or_stolen_phone",
                    "アプリが登録されている携帯電話を紛失・盗難された。",
                ),
                (
                    "passcode_forgotten",
                    "アプリのログインパスコードやパスワードを忘れた。",
                ),
                (
                    "terminate_account",
                    "サービスを退会し、口座・アカウントを解約したい。",
                ),
                (
                    "verify_my_identity",
                    "口座開設や取引制限解除のための本人確認を行いたい。",
                ),
                (
                    "verify_source_of_funds",
                    "資金源や入金根拠に関する書類確認を求められた。",
                ),
                (
                    "why_verify_identity",
                    "なぜ本人確認書類の提出が必要なのか理由を知りたい。",
                ),
                (
                    "age_limit",
                    "サービスの利用年齢制限や未成年者の利用条件を知りたい。",
                ),
                (
                    "country_support",
                    "本サービスが利用可能な対応国・地域を確認したい。",
                ),
                (
                    "visa_or_mastercard",
                    "発行されるカードの国際ブランド (Visa/Mastercard) を知りたい。",
                ),
            ],
        );

        // 7. 外貨両替・付帯サービス (exchange_fee: 10 件)
        builder.add_category(
            "exchange_fee",
            "外貨両替・為替手数料・アプリ内両替および付帯サービス",
            vec![
                (
                    "exchange_charge",
                    "通貨両替時に適用される手数料体系について知りたい。",
                ),
                (
                    "exchange_rate",
                    "現在の適用為替レートやリアルタイム換算レートを確認したい。",
                ),
                (
                    "exchange_via_app",
                    "アプリ内で複数の法定通貨・外貨を両替・保有したい。",
                ),
                (
                    "fiat_currency_support",
                    "サポートされている法定通貨の種類や制限を知りたい。",
                ),
                (
                    "pending_cash_withdrawal",
                    "ATM 現金引き出しの処理が完了待ちになっている。",
                ),
                (
                    "request_refund",
                    "店舗やサービス提供元に対して返金請求を行いたい。",
                ),
                (
                    "receiving_money",
                    "国内外からのお金の受け取り手順や受取口座情報を確認したい。",
                ),
                (
                    "transfer_fee_refund",
                    "誤って差し引かれた送金手数料の返還を求めたい。",
                ),
                (
                    "wrong_amount_refunded",
                    "店舗から返金された金額が購入金額と異なっている。",
                ),
                (
                    "other_or_unsupported",
                    "上記のいずれにも該当しないその他の問い合わせ。",
                ),
            ],
        );

        builder
            .build()
            .expect("Banking77 プリセットのバリデーションに合格する。")
    }
}

/// `HierarchicalMapping` を構築するためのビルダー構造体。
#[derive(Debug, Clone)]
pub struct HierarchicalMappingBuilder {
    name: String,
    coarse_categories: IndexMap<String, String>,
    fine_criteria: IndexMap<String, String>,
    coarse_to_fine: IndexMap<String, Vec<String>>,
    fine_to_coarse: IndexMap<String, String>,
    negative_keys: Vec<String>,
}

impl HierarchicalMappingBuilder {
    /// 新しいビルダーを初期化する。
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            coarse_categories: IndexMap::new(),
            fine_criteria: IndexMap::new(),
            coarse_to_fine: IndexMap::new(),
            fine_to_coarse: IndexMap::new(),
            negative_keys: DEFAULT_NEGATIVE_KEYS
                .iter()
                .map(|&s| s.to_string())
                .collect(),
        }
    }

    /// ネガティブキーのリストを設定する。
    pub fn negative_keys(mut self, keys: Vec<String>) -> Self {
        self.negative_keys = keys;
        self
    }

    /// ネガティブキーを 1 件追加する。
    pub fn add_negative_key(mut self, key: impl Into<String>) -> Self {
        self.negative_keys.push(key.into());
        self
    }

    /// 大分類と、その配下の細分類群を一括追加する。
    ///
    /// # 引数
    /// - `coarse_key`: 大分類キー。
    /// - `coarse_desc`: 大分類の説明文。
    /// - `fine_items`: (細分類キー, 細分類説明文) のリスト。
    pub fn add_category(
        &mut self,
        coarse_key: impl Into<String>,
        coarse_desc: impl Into<String>,
        fine_items: Vec<(impl Into<String>, impl Into<String>)>,
    ) -> &mut Self {
        let coarse_k = coarse_key.into();
        let coarse_d = coarse_desc.into();

        let mut fine_keys = Vec::with_capacity(fine_items.len());
        for (f_key, f_desc) in fine_items {
            let fk = f_key.into();
            let fd = f_desc.into();
            self.fine_criteria.insert(fk.clone(), fd);
            self.fine_to_coarse.insert(fk.clone(), coarse_k.clone());
            fine_keys.push(fk);
        }

        self.coarse_categories.insert(coarse_k.clone(), coarse_d);
        self.coarse_to_fine.insert(coarse_k, fine_keys);
        self
    }

    /// マッピングを検証し、構築する。
    pub fn build(self) -> Result<HierarchicalMapping> {
        let all_coarse_keys: Vec<String> = self.coarse_categories.keys().cloned().collect();
        let all_fine_keys: Vec<String> = self.fine_criteria.keys().cloned().collect();

        let mapping = HierarchicalMapping {
            name: self.name,
            coarse_categories: self.coarse_categories,
            fine_criteria: self.fine_criteria,
            coarse_to_fine: self.coarse_to_fine,
            fine_to_coarse: self.fine_to_coarse,
            all_coarse_keys,
            all_fine_keys,
            negative_keys: self.negative_keys,
        };
        mapping.validate()?;
        Ok(mapping)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_banking77_preset_validity() {
        let mapping = HierarchicalMapping::banking77();
        assert_eq!(mapping.total_coarse_count(), 7);
        assert_eq!(mapping.total_fine_count(), 77);
        assert!(mapping.validate().is_ok());

        // 大分類引き
        let card_fines = mapping.fine_keys_for_coarse("card_ops").unwrap();
        assert_eq!(card_fines.len(), 12);
        assert!(card_fines.contains(&"activate_my_card".to_string()));

        // 逆引き
        assert_eq!(
            mapping.coarse_for_fine("activate_my_card"),
            Some("card_ops")
        );
        assert_eq!(
            mapping.coarse_for_fine("transfer_timing"),
            Some("transfers")
        );
    }

    #[test]
    fn test_validation_mece_violation() {
        // 重複細分類キーがある場合
        let mut builder = HierarchicalMapping::builder("InvalidMapping");
        builder.add_category(
            "cat1",
            "カテゴリ 1",
            vec![("item1", "アイテム 1"), ("dup_item", "重複アイテム")],
        );
        builder.add_category(
            "cat2",
            "カテゴリ 2",
            vec![("item2", "アイテム 2"), ("dup_item", "重複アイテム")],
        );
        let res = builder.build();
        assert!(res.is_err());
        let err_msg = res.unwrap_err().to_string();
        assert!(err_msg.contains("MECE 違反"));
    }

    #[test]
    fn test_validation_empty_category() {
        let mut builder = HierarchicalMapping::builder("EmptyCategory");
        builder.add_category("cat1", "カテゴリ 1", vec![("item1", "アイテム 1")]);
        builder.add_category("cat2", "カテゴリ 2", Vec::<(&str, &str)>::new());
        let res = builder.build();
        assert!(res.is_err());
        let err_msg = res.unwrap_err().to_string();
        assert!(err_msg.contains("空です"));
    }

    #[test]
    fn test_negative_candidate_detection() {
        let mapping = HierarchicalMapping::banking77();
        assert!(mapping.is_negative_candidate(
            "other_or_unsupported",
            "上記のいずれにも該当しないその他の問い合わせ。"
        ));
        assert!(!mapping.is_negative_candidate("activate_my_card", "カード有効化。"));
    }

    #[test]
    fn test_deserialization_auto_reconstructs_cached_keys() {
        let json_data = r#"{
            "name": "MinimalMapping",
            "coarse_categories": {
                "cat_a": "カテゴリAの説明。"
            },
            "fine_criteria": {
                "item_1": "アイテム1の説明。"
            },
            "coarse_to_fine": {
                "cat_a": ["item_1"]
            },
            "fine_to_coarse": {
                "item_1": "cat_a"
            },
            "negative_keys": []
        }"#;

        let mapping: HierarchicalMapping =
            serde_json::from_str(json_data).expect("JSON からのデシリアライズに成功。");
        assert_eq!(mapping.all_coarse_keys(), &["cat_a".to_string()]);
        assert_eq!(mapping.all_fine_keys(), &["item_1".to_string()]);
        assert!(
            mapping.validate().is_ok(),
            "自動再構築されたキーキャッシュにより検証に合格。"
        );
    }

    #[test]
    fn test_validation_detects_inconsistent_cached_keys() {
        let mut mapping = HierarchicalMapping::banking77();

        // 1. all_coarse_keys の欠落
        let original_coarse = mapping.all_coarse_keys.clone();
        mapping.all_coarse_keys.pop();
        let res = mapping.validate();
        assert!(res.is_err());
        assert!(
            res.unwrap_err()
                .to_string()
                .contains("事前キャッシュ `all_coarse_keys`")
        );

        // 復元
        mapping.all_coarse_keys = original_coarse;
        assert!(mapping.validate().is_ok());

        // 2. all_fine_keys の不一致
        let original_fine = mapping.all_fine_keys.clone();
        mapping.all_fine_keys[0] = "unknown_fine_key".to_string();
        let res = mapping.validate();
        assert!(res.is_err());
        assert!(
            res.unwrap_err()
                .to_string()
                .contains("事前キャッシュ `all_fine_keys`")
        );

        // 3. rebuild_key_caches による修復
        mapping.rebuild_key_caches();
        assert_eq!(mapping.all_fine_keys, original_fine);
        assert!(
            mapping.validate().is_ok(),
            "rebuild_key_caches 呼び出し後の検証に成功。"
        );
    }
}
