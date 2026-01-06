import pandas as pd
import numpy as np
import json
import re
from tqdm.auto import tqdm
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os


# --- CẤU HÌNH ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=GEMINI_API_KEY)

# --------------------------------------------------------------------------
# PHẦN 1: LLM GENERATION - TẠO ALPHA BẰNG GEMINI (THEO PROMPT PAPER)
# --------------------------------------------------------------------------

class GeminiAlphaGenerator:
    def __init__(self, gemini_api_key):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def generate_alpha_formulas(self, data_features: list, sample_df: pd.DataFrame = None) -> dict:
        # Tách các loại features
        price_cols = [c for c in data_features if c.lower() in ['close', 'open', 'high', 'low', 'volume']]
        technical_cols = [c for c in data_features if any(x in c for x in ['SMA', 'EMA', 'RSI', 'MACD', 'BB', 'Momentum', 'OBV'])]

        # Tạo DataFrame sample nếu có (để pass vào prompt như paper)
        df_sample_json = ""
        if sample_df is not None:
            # Lấy 5 rows đầu tiên làm sample
            sample_subset = sample_df[data_features].head(5)
            df_sample_json = f"\n\nDataFrame Input (Sample):\n{sample_subset.to_json(orient='split', indent=2)}"

        # === PROMPT 100% THEO BÀI BÁO GỐC (BỎ PHẦN SENTIMENT) ===
        STRUCTURED_PROMPT = f"""
Task Prompt: Generating Predictive Alphas for Stock Prices

Objective: Generate formulaic alpha signals to predict stock prices using:
1. Stock features (e.g., close, open, high, low, volume),
2. Technical indicators (e.g., RSI, moving averages, MACD).

Input Data: A single pandas.DataFrame with rows representing trading days and columns including:
- Stock Features: {', '.join(price_cols) if price_cols else 'Close, Open, High, Low, Volume'}
- Technical Indicators: {', '.join(technical_cols[:10]) if technical_cols else 'e.g., RSI, SMA, EMA, MACD, Bollinger Bands'}
{df_sample_json}

Requirements:
1. Alpha Formulation: Propose 5 formulaic alphas combining stock features and technical indicators.
2. Feature Engineering: Normalize inputs (e.g., Z-scores), handle missing data.

Example Alpha:
α1 = (Close - SMA_20) / SMA_20
α2 = RSI_14 - 50

Additional Technical Notes for Implementation:
- Use np.where(condition, if_true, if_false) for conditional logic
- Use column.shift(1) to access previous values
- Available functions: np.log(), np.sqrt(), np.abs(), np.sign(), np.tanh()
- Add 1e-9 to denominators to prevent division by zero

Return ONLY valid JSON format:
{{
  "Alpha_1": "your formula here",
  "Alpha_2": "your formula here",
  "Alpha_3": "your formula here",
  "Alpha_4": "your formula here",
  "Alpha_5": "your formula here"
}}
"""

        try:
            response = self.model.generate_content(
                STRUCTURED_PROMPT,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.5  # Paper không chỉ định - dùng 0.5 cho balance
                )
            )
            formulas = json.loads(response.text.strip())
            final = {f"Alpha_{i+1}": formulas.get(f"Alpha_{i+1}", list(formulas.values())[i] if i < len(formulas) else "")
                     for i in range(5)}
            final = {k: v for k, v in final.items() if v}

            # === VALIDATION ===
            print(f"\n✓ Generated {len(final)} Alpha formulas:")
            for name, formula in final.items():
                # Đếm technical indicators được dùng
                tech_count = sum(1 for tech in technical_cols if tech in formula)
                print(f"  {name}: {tech_count} indicator(s) | {formula[:80]}{'...' if len(formula) > 80 else ''}")

            return final
        except Exception as e:
            print(f"✗ Error generating alphas: {e}")
            return {}

# --------------------------------------------------------------------------
# PHẦN 2: CÔNG CỤ CONVERT IF-ELSE THÀNH NP.WHERE
# --------------------------------------------------------------------------

def convert_ternary_to_npwhere(formula: str) -> str:
    """
    Chuyển đổi Python ternary operator thành np.where()

    Ví dụ:
    "(1 if RSI_14 < 30 else 0)" → "np.where(RSI_14 < 30, 1, 0)"
    """
    # Pattern: (value_true if condition else value_false)
    pattern = r'\(([^)]+)\s+if\s+([^)]+)\s+else\s+([^)]+)\)'

    def replacer(match):
        val_true = match.group(1).strip()
        condition = match.group(2).strip()
        val_false = match.group(3).strip()
        return f"np.where({condition}, {val_true}, {val_false})"

    converted = re.sub(pattern, replacer, formula)

    # Pattern without parentheses: value_true if condition else value_false
    pattern2 = r'(\w+)\s+if\s+([^)]+)\s+else\s+(\w+)'
    converted = re.sub(pattern2, lambda m: f"np.where({m.group(2)}, {m.group(1)}, {m.group(3)})", converted)

    return converted

# --------------------------------------------------------------------------
# PHẦN 3: TRIỂN KHAI ALPHA ĐỘNG (HỖ TRỢ CONDITIONAL + SHIFT)
# --------------------------------------------------------------------------

def calculate_generated_alphas_dynamic(df_group, generated_formulas: dict):
    """
    Thực thi các công thức alpha với hỗ trợ:
    - NumPy functions
    - np.where() cho conditional logic
    - .shift() cho lagged values
    """
    df_copy = df_group.copy().reset_index(drop=True)  # Reset index để shift() hoạt động đúng

    # Tạo namespace an toàn
    safe_namespace = {
        'np': np,
        '__builtins__': {}
    }

    # Thêm tất cả columns vào namespace dưới dạng pandas Series (để hỗ trợ .shift())
    for col in df_copy.columns:
        safe_namespace[col] = df_copy[col]

    for name, formula in generated_formulas.items():
        try:
            # Chuyển đổi ternary operator nếu có
            formula_converted = convert_ternary_to_npwhere(formula)

            # Thực thi công thức
            result = eval(formula_converted, safe_namespace)

            # Convert về numpy array nếu cần
            if hasattr(result, 'values'):
                result = result.values

            df_copy[name] = result

            if formula != formula_converted:
                print(f"✓ Thành công áp dụng {name}")
                print(f"  Original: {formula}")
                print(f"  Converted: {formula_converted}")
            else:
                print(f"✓ Thành công áp dụng {name}: {formula}")

        except ZeroDivisionError:
            print(f" Cảnh báo: {name} có division by zero - thay thế bằng NaN")
            df_copy[name] = np.nan

        except Exception as e:
            print(f"✗ Lỗi thực thi {name}: {e}")
            print(f"  Formula: {formula}")
            df_copy[name] = np.nan

    return df_copy