"""
AST-based validation of LLM-generated alpha expressions.

Validation is purely syntactic - no code is executed.
"""
import ast
from typing import Set


class AlphaValidationError(Exception):
    pass


ALLOWED_OPERATORS: Set[str] = {
    'abs', 'sign', 'log', 'rank', 'delay', 'delta',
    'ts_rank', 'ts_mean', 'ts_std', 'ts_min', 'ts_max',
    'ts_argmax', 'ts_argmin', 'correlation', 'covariance',
    'decay_linear', 'scale', 'if_else',
}

# Paper's 11 indicators (new names) + legacy aliases + OHLCV + sentiment.
ALLOWED_VARIABLES: Set[str] = {
    # OHLCV
    'open', 'high', 'low', 'close', 'volume', 'vwap',
    # Derived
    'returns', 'log_return',
    # Technical indicators (full set used in experiments)
    'SMA_5', 'SMA_10', 'SMA_20', 'SMA_60',
    'EMA_10', 'EMA_12', 'EMA_26',
    'Momentum_3', 'Momentum_10', 'momentum_5', 'momentum_10',
    'RSI_14',
    'MACD', 'MACD_Signal', 'MACD_diff', 'MACD_signal',
    'BB_Upper', 'BB_Lower', 'BB_upper', 'BB_lower', 'BB_middle', 'BB_pct_b', 'BB_width',
    'OBV',
    'ATR_14', 'volume_ratio', 'Volume_ratio',
    'realized_vol_5d', 'realized_vol_20d',
    # Per-company polarity (from sentiment_features/<TICKER>.csv)
    'polarity_avg',
    # AAPL related
    'AAPL_polarity', 'GOOGL_polarity', 'AMZN_polarity', 'MSFT_polarity',
    'TSLA_polarity', 'NVDA_polarity', 'META_polarity', 'INTC_polarity',
    'AMD_polarity', 'IBM_polarity', 'SSNLF_polarity',
    # HSBC related
    'HSBC_polarity', 'JPM_polarity', 'SCBFF_polarity', 'MS_polarity',
    'UBS_polarity', 'C_polarity', 'GS_polarity', 'CS_polarity', 'NWG_polarity',
    # PEP related
    'PEP_polarity', 'KO_polarity', 'WMT_polarity', 'COST_polarity',
    'CL_polarity', 'CVX_polarity', 'PG_polarity', 'JNJ_polarity',
    # TM related
    'TM_polarity', 'NSANY_polarity', 'HMC_polarity', 'MZDAY_polarity',
    'F_polarity', 'GM_polarity', 'VWAPY_polarity', 'BMWYY_polarity',
    'HYMTF_polarity', 'BYDDY_polarity',
    # TCEHY related
    'TCEHY_polarity', 'NTES_polarity', 'BIDU_polarity', 'JD_polarity',
    'BABA_polarity',
}

ALLOWED_NAMES: Set[str] = ALLOWED_OPERATORS | ALLOWED_VARIABLES

ALLOWED_LOOKBACKS: Set[int] = {2, 3, 5, 10, 15, 20, 30, 60}

# Only these operators accept a lookback-window integer argument.
LOOKBACK_OPERATORS: Set[str] = {
    'delay', 'delta', 'ts_rank', 'ts_mean', 'ts_std',
    'ts_min', 'ts_max', 'ts_argmax', 'ts_argmin',
    'correlation', 'covariance', 'decay_linear',
}

# Per-company *_polarity is matched by suffix in check_sentiment_requirement.
SENTIMENT_VARIABLE_NAMES: Set[str] = set()

MAX_NESTING_DEPTH = 4


class _DepthVisitor(ast.NodeVisitor):
    """Track call-nesting depth and validate names/lookbacks."""

    def __init__(self):
        self.max_depth = 0
        self._depth = 0
        self.errors = []
        self.names_seen = set()

    def visit_Name(self, node: ast.Name):
        if node.id not in ALLOWED_NAMES:
            self.errors.append(f"Name '{node.id}' is not allowed.")
        self.names_seen.add(node.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname not in ALLOWED_OPERATORS:
                self.errors.append(f"Function '{fname}' is not allowed.")
        elif isinstance(node.func, ast.Attribute):
            self.errors.append(
                f"Attribute access '{ast.unparse(node.func)}' is not allowed."
            )

        fname = node.func.id if isinstance(node.func, ast.Name) else None
        if fname in LOOKBACK_OPERATORS:
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                    if arg.value not in ALLOWED_LOOKBACKS:
                        self.errors.append(
                            f"Lookback window {arg.value} is not in the allowed set "
                            f"{sorted(ALLOWED_LOOKBACKS)}."
                        )
                elif (
                    isinstance(arg, ast.UnaryOp)
                    and isinstance(arg.op, ast.USub)
                    and isinstance(arg.operand, ast.Constant)
                    and isinstance(arg.operand.value, int)
                ):
                    self.errors.append(
                        f"Negative lookback window -{arg.operand.value} is not allowed "
                        f"(would be forward-looking)."
                    )

        self._depth += 1
        if self._depth > self.max_depth:
            self.max_depth = self._depth
        self.generic_visit(node)
        self._depth -= 1

    def visit_BinOp(self, node: ast.BinOp):
        if isinstance(node.op, ast.Sub) and isinstance(node.right, ast.BinOp) \
                and isinstance(node.right.op, ast.Mult):
            for child in (node.right.left, node.right.right):
                if isinstance(child, ast.Constant) and child.value == 1:
                    self.errors.append(
                        "Operator precedence trap: '... - 1 * X' is parsed as "
                        "'... - X'. If you meant '(... - 1) * X', add explicit parentheses."
                    )
                    break

        if isinstance(node.op, ast.Mult):
            for child in (node.left, node.right):
                if isinstance(child, ast.Compare):
                    self.errors.append(
                        "Bare boolean comparison used as numeric multiplier - "
                        "wrap with if_else(cond, 1, -1) or if_else(cond, value, 0)."
                    )
                    break

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        self.errors.append(
            f"Attribute access '.{node.attr}' is not allowed."
        )

    def visit_Import(self, node):
        self.errors.append("import statements are not allowed.")

    def visit_ImportFrom(self, node):
        self.errors.append("import statements are not allowed.")


def validate_alpha_expression(expression: str) -> None:
    """
    Validate an alpha expression string against the allowed whitelist.

    Raises AlphaValidationError with a descriptive message on the first
    category of violation found.  Does not execute any code.
    """
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as e:
        raise AlphaValidationError(f"Syntax error: {e}") from e

    visitor = _DepthVisitor()
    visitor.visit(tree)

    if visitor.errors:
        raise AlphaValidationError(
            f"Expression contains {len(visitor.errors)} violation(s): "
            + "; ".join(visitor.errors[:3])
            + ("..." if len(visitor.errors) > 3 else "")
        )

    if visitor.max_depth > MAX_NESTING_DEPTH:
        raise AlphaValidationError(
            f"Nesting depth {visitor.max_depth} exceeds maximum {MAX_NESTING_DEPTH}."
        )


def check_sentiment_requirement(expression: str) -> bool:
    """Return True if the expression contains at least one sentiment variable."""
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and (
            node.id in SENTIMENT_VARIABLE_NAMES or node.id.endswith('_polarity')
        ):
            return True
    return False
