# sokuto Python SDK

日本語特化・TypeSafe AI「sokuto (System One)」の Python クライアント SDK です。
文章生成を行わず、Pydantic v2 スキーマからミリ秒単位で型安全な決定（Choice, Score, Noul）を取得します。

## インストール

```bash
pip install sokuto
```

## クイックスタート

```python
from typing import Annotated, Literal
from pydantic import BaseModel, Field
from sokuto.schema import SokutoClient, Uncertain


class TriageDecision(BaseModel):
    category: Literal["tech", "billing", "fraud"] = Field(description="問い合わせカテゴリ。")
    is_urgent: bool = Field(description="至急対応が必要か。")
    severity: Uncertain[Annotated[int, Field(ge=1, le=5, description="重要度 (1〜5)。")]]


client = SokutoClient(base_url="http://localhost:8080")
result = client.predict(TriageDecision, state="ログインできずに困っています。")

print(result.category)  # "tech"
print(result.is_urgent)  # True
print(result.severity.value)  # 4
print(result.severity.meta.confidence)  # 0.94
```
