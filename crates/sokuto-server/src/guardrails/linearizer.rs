//! # 構造化 JSON State の正準テキスト線形化モジュール
//!
//! ネストした JSON オブジェクトや配列 (JSON 文字列を含む) を階層パス付きのキー・バリュー行形式へ決定論的に変換する。

use serde_json::Value;

/// JSON 線形化の設定構造体。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearizerConfig {
    /// 線形化処理を有効化するかどうかのフラグ。
    pub enabled: bool,
    /// 探索する最大ネスト深度。
    pub max_depth: usize,
}

impl Default for LinearizerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_depth: 16,
        }
    }
}

/// 構造化 JSON (`Object` / `Array` または JSON 文字列) を階層パス付き行形式へ決定論的に線形化する。
///
/// # 引数
/// - `val`: 対象となる JSON 値参照。
/// - `config`: 線形化設定。
///
/// # 戻り値
/// 線形化された行テキスト (失敗時や非対象時は `None`)。
pub fn linearize_state_value(val: &Value, config: &LinearizerConfig) -> Option<String> {
    if !config.enabled {
        return None;
    }

    let parsed_owned: Value;
    let target = match val {
        Value::Object(_) | Value::Array(_) => val,
        Value::String(s) => {
            let trimmed = s.trim();
            if !(trimmed.starts_with('{') || trimmed.starts_with('[')) {
                return None;
            }
            parsed_owned = serde_json::from_str(trimmed).ok()?;
            if !(parsed_owned.is_object() || parsed_owned.is_array()) {
                return None;
            }
            &parsed_owned
        }
        _ => return None,
    };

    let mut lines = Vec::new();
    flatten_json(target, "", 0, config.max_depth, &mut lines);
    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

/// JSON 値を再帰的に走査し、階層パス付き行リストに追加する。
fn flatten_json(val: &Value, prefix: &str, depth: usize, max_depth: usize, out: &mut Vec<String>) {
    if depth > max_depth {
        out.push(format!("- {}: {}", path_label(prefix), val));
        return;
    }

    match val {
        Value::Object(map) => {
            if map.is_empty() {
                out.push(format!("- {}: {{}}", path_label(prefix)));
                return;
            }
            for (k, v) in map {
                let next_path = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                flatten_json(v, &next_path, depth + 1, max_depth, out);
            }
        }
        Value::Array(arr) => {
            if arr.is_empty() {
                out.push(format!("- {}: []", path_label(prefix)));
                return;
            }
            let all_scalars = arr.iter().all(|item| !item.is_object() && !item.is_array());
            if all_scalars {
                let joined: Vec<String> = arr.iter().map(format_scalar).collect();
                out.push(format!("- {}: [{}]", path_label(prefix), joined.join(", ")));
            } else {
                for (idx, item) in arr.iter().enumerate() {
                    let next_path = if prefix.is_empty() {
                        format!("[{idx}]")
                    } else {
                        format!("{prefix}[{idx}]")
                    };
                    flatten_json(item, &next_path, depth + 1, max_depth, out);
                }
            }
        }
        scalar => {
            out.push(format!(
                "- {}: {}",
                path_label(prefix),
                format_scalar(scalar)
            ));
        }
    }
}

/// パスラベルを取得する。空文字の場合は `"root"` を返却する。
fn path_label(prefix: &str) -> &str {
    if prefix.is_empty() { "root" } else { prefix }
}

/// スカラー値を人間可読な文字列表現に整形する。
fn format_scalar(val: &Value) -> String {
    match val {
        Value::String(s) => s.clone(),
        Value::Null => "null".to_string(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_linearize_nested_object() {
        let val = json!({
            "customer": {
                "id": "c123",
                "tier": "VIP"
            },
            "status": "active"
        });
        let config = LinearizerConfig::default();
        let res = linearize_state_value(&val, &config).unwrap();

        assert_eq!(
            res,
            "- customer.id: c123\n- customer.tier: VIP\n- status: active"
        );
    }

    #[test]
    fn test_linearize_json_string() {
        let raw = r#"{"customer": {"tier": "VIP"}}"#;
        let val = Value::String(raw.to_string());
        let config = LinearizerConfig::default();
        let res = linearize_state_value(&val, &config).unwrap();

        assert_eq!(res, "- customer.tier: VIP");
    }

    #[test]
    fn test_linearize_scalar_array() {
        let val = json!({
            "tags": ["urgent", "payment"]
        });
        let config = LinearizerConfig::default();
        let res = linearize_state_value(&val, &config).unwrap();

        assert_eq!(res, "- tags: [urgent, payment]");
    }

    #[test]
    fn test_linearize_object_array() {
        let val = json!({
            "items": [
                {"name": "apple", "price": 100},
                {"name": "banana", "price": 200}
            ]
        });
        let config = LinearizerConfig::default();
        let res = linearize_state_value(&val, &config).unwrap();

        assert_eq!(
            res,
            "- items[0].name: apple\n- items[0].price: 100\n- items[1].name: banana\n- items[1].price: 200"
        );
    }

    #[test]
    fn test_linearize_empty_and_plain_string() {
        let config = LinearizerConfig::default();
        // 通常の文字列は対象外
        let val1 = Value::String("通常のテキスト".to_string());
        assert_eq!(linearize_state_value(&val1, &config), None);

        // 空オブジェクト
        let val2 = json!({});
        assert_eq!(
            linearize_state_value(&val2, &config),
            Some("- root: {}".to_string())
        );

        // 空配列
        let val3 = json!([]);
        assert_eq!(
            linearize_state_value(&val3, &config),
            Some("- root: []".to_string())
        );
    }

    #[test]
    fn test_linearize_max_depth_cutoff() {
        let val = json!({
            "a": {
                "b": {
                    "c": "val"
                }
            }
        });
        let config = LinearizerConfig {
            enabled: true,
            max_depth: 1,
        };
        let res = linearize_state_value(&val, &config).unwrap();
        assert_eq!(res, "- a.b: {\"c\":\"val\"}");
    }
}
