import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(page_title="Equation Finder", page_icon="🧮", layout="wide")
st.title("🧮 Find the Best-Fit Equation")
st.write("Upload your Excel data, choose the variables you want to use, and let the app compare polynomial equations. You can then compare the measured target values with the values calculated by the best model.")

DEGREES = (1, 2, 3)

def num_cols(d):
    return [c for c in d.columns if pd.to_numeric(d[c], errors="coerce").notna().mean() > 0.5]

def fmt(x):
    return f"{x:.10g}"

def term_strings(features, powers):
    py = []
    for f, p in zip(features, powers):
        if p == 0:
            continue
        py.append(f"[{f}]" + (f"**{p}" if p > 1 else ""))
    return "*".join(py)

class Terms(BaseEstimator, TransformerMixin):
    def __init__(self, powers=None):
        self.powers = powers
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.prod(X[:, None, :] ** np.asarray(self.powers)[None, :, :], axis=2)

def make_powers(n_vars, degree, interactions):
    powers = PolynomialFeatures(degree, include_bias=False).fit(np.zeros((1, n_vars))).powers_
    if not interactions:
        powers = powers[(powers > 0).sum(axis=1) <= 1]
    return powers

def find_equations(df, features, target, degrees, cv_mode, interactions):
    features = list(features)
    if not features or target in features:
        raise ValueError("Choose at least one input variable, and do not include the target among inputs.")
    d = df[features + [target]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) < 4:
        raise ValueError("Not enough complete numeric rows. At least 4 rows are needed for cross-validation.")
    # Choose K automatically. Keep at least two observations in each test fold
    # so R² can be calculated, and use five folds for larger datasets.
    n_splits = min(5, len(d) // 2)
    X, y = d[features], d[target]
    clip0 = y.min() >= 0
    cv = KFold(n_splits=n_splits, shuffle=(cv_mode == "Random rows"),
               random_state=0 if cv_mode == "Random rows" else None)
    results = []
    for deg in degrees:
        powers = make_powers(len(features), deg, interactions)
        model = make_pipeline(Terms(powers), LinearRegression())
        scores = cross_validate(model, X, y, cv=cv, scoring=("r2", "neg_root_mean_squared_error"))
        model.fit(X, y)
        lr = model[-1]
        py_eq = f"[{target}] = {fmt(lr.intercept_)}"
        for coef, pw in zip(lr.coef_, powers):
            py_term = term_strings(features, pw)
            sign = "+" if coef >= 0 else "-"
            py_eq += f" {sign} {fmt(abs(coef))}*{py_term}"
        results.append(dict(degree=deg, terms=len(lr.coef_) + 1,
                            CV_R2=scores["test_r2"].mean(),
                            CV_RMSE=-scores["test_neg_root_mean_squared_error"].mean(),
                            python_equation=py_eq, model=model))
    return pd.DataFrame(results).sort_values("CV_R2", ascending=False).reset_index(drop=True), d

uploaded = st.file_uploader("Upload Excel file (.xlsx or .xls)", type=["xlsx", "xls"])
if uploaded:
    try:
        xls = pd.ExcelFile(uploaded)
        sheet_stats = []
        for sheet in xls.sheet_names:
            temp = pd.read_excel(xls, sheet_name=sheet)
            numeric_count = sum(
                not str(c).startswith("Unnamed") and
                pd.to_numeric(temp[c], errors="coerce").notna().mean() > 0.5
                for c in temp.columns
            )
            sheet_stats.append({"Sheet": sheet, "Rows": len(temp), "Named numeric columns": numeric_count})
        stats = pd.DataFrame(sheet_stats)
        st.subheader("Your workbook")
        st.dataframe(stats, use_container_width=True, hide_index=True)
        default_sheet = stats.sort_values(["Named numeric columns", "Rows"], ascending=False).iloc[0]["Sheet"]
        sheet = st.selectbox("Choose worksheet", xls.sheet_names, index=xls.sheet_names.index(default_sheet))
        df = pd.read_excel(xls, sheet_name=sheet)
        cols = num_cols(df)
        if len(cols) < 2:
            st.error("This sheet needs at least two columns containing mostly numeric values.")
            st.stop()
        st.caption(f"Selected sheet: {sheet} · {len(df):,} rows · {len(df.columns)} columns")
        with st.expander("Preview data and numeric-column detection"):
            st.dataframe(df.head(20), use_container_width=True)
            st.dataframe(pd.DataFrame({
                "Column": df.columns,
                "Numeric %": [round(100 * pd.to_numeric(df[c], errors="coerce").notna().mean()) for c in df.columns]
            }), use_container_width=True, hide_index=True)

        pref_x = [c for c in ["GR", "NPHI", "RHOB"] if c in cols] or cols[:min(3, len(cols))]
        default_target = "PHIE" if "PHIE" in cols else next((c for c in cols if c not in pref_x), cols[-1])
        st.subheader("Choose your variables and model settings")
        c1, c2 = st.columns(2)
        with c1:
            features = st.multiselect("Input variables (X)", cols, default=pref_x)
        with c2:
            target = st.selectbox("Target variable (Y)", cols, index=cols.index(default_target))
        c3, c4 = st.columns(2)
        with c3:
            interactions = st.checkbox("Include interaction terms (e.g. GR × NPHI)", value=True)
        with c4:
            cv_mode = st.selectbox("Cross-validation", ["Continuous blocks", "Random rows"])
        degrees = st.multiselect("Polynomial degrees to test", [1, 2, 3], default=[1, 2, 3])
        if st.button("Generate equations", type="primary", use_container_width=True):
            try:
                if not degrees:
                    st.error("Select at least one polynomial degree.")
                    st.stop()
                with st.spinner("Fitting models and validating equations..."):
                    res, used = find_equations(df, features, target, sorted(degrees), cv_mode, interactions)
                st.session_state["equation_results"] = res
                st.session_state["equation_used"] = used
                st.session_state["equation_df"] = df
                st.session_state["equation_target"] = target
                st.session_state["equation_features"] = features
                st.session_state["equation_sheet"] = sheet
            except Exception as e:
                st.error(f"Could not generate equations: {e}")

        if "equation_results" in st.session_state and st.session_state.get("equation_sheet") == sheet:
            res = st.session_state["equation_results"]
            used = st.session_state["equation_used"]
            current_target = st.session_state["equation_target"]
            st.subheader("How did the models perform?")
            auto_folds = min(5, len(used) // 2)
            st.caption(f"Rows used: {len(used):,} of {len(df):,}; incomplete/non-numeric rows excluded: {len(df)-len(used):,} · Automatic cross-validation: {auto_folds}-fold")
            st.dataframe(res[["degree", "terms", "CV_R2", "CV_RMSE"]].round(4), use_container_width=True, hide_index=True)
            st.subheader("Equations found")
            for _, r in res.iterrows():
                with st.expander(f"Degree {int(r.degree)} · CV R² {r.CV_R2:.4f} · CV RMSE {r.CV_RMSE:.5g}", expanded=(r.name == 0)):
                    st.markdown("**Python equation**")
                    st.code(r.python_equation, language="python")
            best = res.iloc[0]
            st.success(f"Best-performing model in this comparison: degree {int(best.degree)} (cross-validated R² = {best.CV_R2:.4f}).")
            out = df.copy()
            out[f"{current_target}_pred"] = np.nan
            out.loc[used.index, f"{current_target}_pred"] = best.model.predict(used[st.session_state["equation_features"]])
            if used[current_target].min() >= 0:
                out[f"{current_target}_pred"] = out[f"{current_target}_pred"].clip(lower=0)

            # Compare measured target values with values calculated by the best model.
            actual = used[current_target].to_numpy()
            calculated = out.loc[used.index, f"{current_target}_pred"].to_numpy()
            plot_data = pd.DataFrame({"Actual": actual, "Calculated": calculated}).dropna()
            st.subheader("Actual target vs. calculated target")
            st.write(
                "Each dot represents one row in your dataset. The closer the dots are to the "
                "diagonal line, the closer the calculated values are to the measured values."
            )
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(plot_data["Actual"], plot_data["Calculated"], alpha=0.65)
            low = min(plot_data["Actual"].min(), plot_data["Calculated"].min())
            high = max(plot_data["Actual"].max(), plot_data["Calculated"].max())
            if low == high:
                padding = max(abs(low) * 0.05, 1e-6)
                low, high = low - padding, high + padding
            ax.plot([low, high], [low, high], linestyle="--", linewidth=1.5, label="Perfect match")
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_xlabel(f"Actual {current_target}")
            ax.set_ylabel(f"Calculated {current_target}")
            ax.set_title(f"Actual vs. calculated {current_target}")
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                res.drop(columns="model").to_excel(writer, sheet_name="Equations", index=False)
                out.to_excel(writer, sheet_name="Data + prediction", index=False)
            st.download_button("Download equations and calculated values (Excel)", data=output.getvalue(),
                               file_name="equation_results.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)
    except Exception as e:
        st.error(f"Unable to read this workbook: {e}")
else:
    st.info("Start by uploading your Excel workbook. You can then choose your input variables and target variable.")
