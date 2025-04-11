from typing import List
import pandas as pd
import statsmodels.formula.api as smf


def fit_multi(df: pd.DataFrame, x: str, y: str, groupby: List[str]):
    def fit_inner(dfi):
        res = smf.ols(f"{y} ~ {x}", data=dfi).fit()
        upper, lower = res.conf_int(0.05).loc[x]
        return pd.Series(
            {
                "slope": res.params.loc[x],
                "intercept": res.params.loc["Intercept"],
                "rvalue": res.rsquared,
                "lower": upper,
                "upper": lower,
            }
        )

    return df.groupby(groupby).apply(fit_inner)
