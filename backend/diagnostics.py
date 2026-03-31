import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def scatter_plot(y_test, y_pred):
    """Scatter plot of predicted vs actual with y=x reference line."""
    st.subheader("Scatter Plot: Predicted vs Actual")

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.7, color="blue", label="Predictions")
    # reference line y=x
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal (y=x)")
    ax.set_xlabel("Actual values")
    ax.set_ylabel("Predicted values")
    ax.legend()
    st.pyplot(fig)

def feature_importance(model, df, target_col):
    """Feature importance bar chart."""
    st.subheader("Feature Importance")
    try:
        # scikit-learn linear SVR
        if hasattr(model, "coef_") and model.coef_ is not None:
            importance = model.coef_.flatten()
        # scratch linear SVR
        elif hasattr(model, "w") and model.w is not None:
            importance = model.w
        else:
            # fallback: correlation with target
            X = df.drop(columns=[target_col]).values
            y = df[target_col].values
            corr = np.corrcoef(X.T, y)[0:len(df.columns)-1, -1]
            importance = np.abs(corr)

        features = df.drop(columns=[target_col]).columns
        imp_df = pd.DataFrame({"Feature": features, "Importance": importance})
        st.bar_chart(imp_df.set_index("Feature"))
    except Exception as e:
        st.error(f"Feature importance not available: {e}")

def run_diagnostics(model, data, df, target_col):
    """Run diagnostics: scatter + feature importance."""
    st.header("🔍 Diagnostics")
    y_test = data["y_test"]
    y_pred = model.predict(data["X_test_scaled"])
    scatter_plot(y_test, y_pred)
    feature_importance(model, df, target_col)