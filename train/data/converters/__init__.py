"""Jev データセットコンバータパッケージ。

Banking77, CLINC150, MNLI, AG News などの公開 NLP コーパスを
統一スキーマ `UnifiedSample` へ変換するコンバータ群を提供する。
"""

from data.converters.ag_news import AGNewsConverter
from data.converters.banking77 import Banking77Converter
from data.converters.base import BaseDatasetConverter
from data.converters.clinc150 import Clinc150Converter
from data.converters.compliance_ja import ComplianceJAConverter
from data.converters.jbe_qa import JBEQAConverter
from data.converters.jglue_jcommonsenseqa import JGlueJCommonsenseQAConverter
from data.converters.jglue_jnli import JGlueJNLIConverter
from data.converters.jglue_jsts import JGlueJSTSConverter
from data.converters.jglue_marc_ja import JGlueMarcJaConverter
from data.converters.legal_rikai import LegalRikaiConverter
from data.converters.massive import MASSIVEConverter, build_massive_hierarchical_mapping
from data.converters.mnli import MNLIConverter
from data.converters.niah import NIAHBenchmarkConverter
from data.converters.sst5 import SST5Converter
from data.converters.synthetic import SyntheticDatasetConverter
from data.converters.synthetic_long import LongContextSyntheticConverter

__all__ = [
    "AGNewsConverter",
    "Banking77Converter",
    "BaseDatasetConverter",
    "Clinc150Converter",
    "ComplianceJAConverter",
    "JBEQAConverter",
    "JGlueJCommonsenseQAConverter",
    "JGlueJNLIConverter",
    "JGlueJSTSConverter",
    "JGlueMarcJaConverter",
    "LegalRikaiConverter",
    "LongContextSyntheticConverter",
    "MASSIVEConverter",
    "MNLIConverter",
    "NIAHBenchmarkConverter",
    "SST5Converter",
    "SyntheticDatasetConverter",
    "build_massive_hierarchical_mapping",
]
