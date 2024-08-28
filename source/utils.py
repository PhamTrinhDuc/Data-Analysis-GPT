import pandas as pd
import matplotlib.pyplot as plt
from logs.logger import set_logging_error

log_error = set_logging_error()


def excutor_code(scripts: str, df: pd.DataFrame):
    """
    Execute code in the context of the current dataframe
    """
    try:
        local_vars = {"plt": plt, "df": df}
        compiled_code = compile(scripts, "<string>", "exec")
        exec(compiled_code, globals(), local_vars)
    except Exception as e:
        log_error.error(f"ERROR: {e}")
