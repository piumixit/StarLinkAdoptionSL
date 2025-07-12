# pages/modeling_page.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import joblib
import numpy as np
import shap
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    ConfusionMatrixDisplay
)

@st.cache_data # Cache data loading
def load_data(file_path):
    """Loads the dataset."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Error: Dataset not found at {file_path}")
        return None

@st.cache_resource # Cache the model training
def train_models(X_train, y_train, preprocess):
    """Trains candidate models."""
    models = {
        "Logistic": Pipeline([
            ("prep", preprocess),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))
        ]),
        "RandomForest": Pipeline([
            ("prep", preprocess),
            ("model", RandomForestClassifier(
                n_estimators=400, max_depth=None, random_state=42, n_jobs=-1
            ))
        ]),
        "ExtraTrees": Pipeline([
            ("prep", preprocess),
            ("model", ExtraTreesClassifier(
                n_estimators=400, max_depth=None, random_state=42, n_jobs=-1
            ))
        ]),
        "GradientBoost": Pipeline([
            ("prep", preprocess),
            ("model", GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05, random_state=42
            ))
        ]),
        "AdaBoost": Pipeline([
            ("prep", preprocess),
            ("model", AdaBoostClassifier(
                n_estimators=300, learning_rate=0.1, random_state=42
            ))
        ]),
        "XGBoost": Pipeline([
            ("prep", preprocess),
            ("model", xgb.XGBClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                objective="binary:logistic", eval_metric="logloss",
                random_state=42, n_jobs=-1
            ))
        ]),
        "LightGBM": Pipeline([
            ("prep", preprocess),
            ("model", LGBMClassifier(
                n_estimators=500, learning_rate=0.05, max_depth=-1,
                objective="binary", random_state=42, n_jobs=-1
            ))
        ]),
        "SVC": Pipeline([
            ("prep", preprocess),
            ("model", SVC(kernel="rbf", C=5, probability=True, gamma="scale"))
        ]),
        "KNN": Pipeline([
            ("prep", preprocess),
            ("model", KNeighborsClassifier(n_neighbors=15, weights="distance"))
        ]),
        "MLP": Pipeline([
            ("prep", preprocess),
            ("model", MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=500,
                alpha=1e-3, random_state=42
            ))
        ])
    }

    fitted_models = {}
    for name, pipe in models.items():
        st.write(f"Training {name}...")
        pipe.fit(X_train, y_train)
        fitted_models[name] = pipe
    return fitted_models

def evaluate(model, X, y, name="model"):
    """Evaluates a fitted classifier and returns metrics and confusion matrix."""
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = model.decision_function(X)
        # Scale to [0, 1] for AUC calculation if needed
        y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())


    report = classification_report(y, y_pred, digits=3, output_dict=True)
    auc = roc_auc_score(y, y_prob).round(3)

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y, y_pred, cmap="Blues", ax=ax)
    ax.set_title(f"{name}: Confusion Matrix")

    return report, auc, fig

@st.cache_resource # Cache the tuning process
def tune_models(X_train, y_train, preprocess, models_for_tune, param_grids):
    """Tunes specified models using RandomizedSearchCV."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_models = {}

    for name, pipe in models_for_tune.items():
        st.write(f"\n Tuning {name} …")
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grids[name],
            n_iter=30,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
        search.fit(X_train, y_train)
        st.write(f" Best mean CV AUC: {search.best_score_:.4f}")
        st.write(" Best params :", search.best_params_)
        best_models[name] = search.best_estimator_
    return best_models


def run_modeling_page():
    """Code for the Modeling page."""
    st.title("Modeling and Evaluation")

    file_path = 'starlink_household_synthetic.csv'
    df1 = load_data(file_path)

    if df1 is None:
        return

    target_col = "starlink_proxy_adoption"

    # Drop the 'roof_persons' column if it exists
    if 'roof_persons' in df1.columns:
        df1 = df1.drop(columns=['roof_persons'])


    X = df1.drop(columns=[target_col])
    y = df1[target_col].map({"No":0, "Yes":1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    st.subheader("Train-Test Split")
    st.write(f"Training set shape: {X_train.shape}")
    st.write(f"Testing set shape: {X_test.shape}")


    st.subheader("Pre-processing Pipeline")
    numeric_features   = X.select_dtypes(include=["int64","float64"]).columns
    categorical_features = X.select_dtypes(include=["object","category"]).columns

    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("num",  numeric_transformer, numeric_features),
        ("cat",  categorical_transformer, categorical_features)
    ])
    st.write("Preprocessing pipeline defined: Standard Scaler for numeric, OneHotEncoder for categorical features.")


    st.subheader("Candidate Models")
    st.write("Training various classification models...")
    fitted_models = train_models(X_train, y_train, preprocess)


    st.subheader("Model Evaluation on Test Set")
    records = []
    for name, mdl in fitted_models.items():
        report, auc, fig = evaluate(mdl, X_test, y_test, name=name)
        st.write(f"\n**=== {name} ===**")
        st.text("Test-set metrics:\n" + classification_report(y_test, mdl.predict(X_test), digits=3))
        st.write("ROC-AUC:", auc)
        st.pyplot(fig)
        plt.close(fig)

        # For ranking
        y_pred = mdl.predict(X_test)
        y_prob = (mdl.predict_proba(X_test)[:, 1]
                  if hasattr(mdl, "predict_proba")
                  else mdl.decision_function(X_test))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        acc = accuracy_score(y_test, y_pred)
        records.append({
            "model": name,
            "auc": roc_auc_score(y_test, y_prob),
            "f1_macro": f1,
            "precision_macro": prec,
            "recall_macro": rec,
            "accuracy": acc
        })


    st.subheader("Model Performance Ranking")
    results_df = pd.DataFrame(records) \
               .sort_values(["auc", "f1_macro"], ascending=False) \
               .reset_index(drop=True)

    st.dataframe(results_df.style.format({"auc": "{:.3f}",
                                         "f1_macro": "{:.3f}",
                                         "accuracy": "{:.3f}",
                                          "precision_macro": "{:.3f}",
                                          "recall_macro": "{:.3f}"}))

    best_name   = results_df.iloc[0, 0]
    st.write(f"\nChampion model based on ROC-AUC (then F1): **{best_name}**")

    st.subheader("Model Tuning (Top 3)")
    st.write("Initiating randomized cross-validated search for the top 3 models...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grids = {
        "RandomForest": {
            "model__n_estimators":  [300, 450, 600, 750],
            "model__max_depth":     [None, 10, 20, 30],
            "model__min_samples_leaf": [1, 2, 4]
        },
        "LightGBM": {
            "model__n_estimators":   [300, 400, 500, 600, 700],
            "model__learning_rate":  [0.01, 0.05, 0.1, 0.15, 0.2],
            "model__num_leaves":     [15, 31, 63, 127],
            "model__max_depth":      [-1, 5, 10, 15],
            "model__subsample":      [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree":[0.6, 0.8, 1.0],
            "model__reg_lambda":     [0, 0.5, 1.0]
        },
        "XGBoost": {
            "model__n_estimators":   [300, 400, 500, 600, 700],
            "model__learning_rate":  [0.01, 0.05, 0.1, 0.15, 0.2],
            "model__max_depth":      [3, 4, 5, 6],
            "model__subsample":      [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree":[0.6, 0.8, 1.0],
            "model__gamma":          [0, 0.5, 1]
        }
    }

    rf_base = Pipeline([
        ("prep", preprocess),
        ("model", RandomForestClassifier(
            random_state=42, n_jobs=-1
        ))
    ])

    lgbm_base = Pipeline([
        ("prep", preprocess),
        ("model", LGBMClassifier(
            objective="binary",
            random_state=42,
            n_jobs=-1
        ))
    ])

    xgb_base = Pipeline([
        ("prep", preprocess),
        ("model", xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42, n_jobs=-1
        ))
    ])

    models_for_tune = {
        "RandomForest": rf_base,
        "LightGBM":     lgbm_base,
        "XGBoost":      xgb_base
    }

    best_tuned_models = tune_models(X_train, y_train, preprocess, models_for_tune, param_grids)


    st.subheader("Evaluate Tuned Models")
    tuned_records = []
    for name, model in best_tuned_models.items():
        report, auc, fig = evaluate(model, X_test, y_test, name=f"{name} (tuned)")
        st.write(f"\n**=== Tuned {name} ===**")
        st.text("Test-set metrics:\n" + classification_report(y_test, model.predict(X_test), digits=3, zero_division=0))
        st.write("ROC-AUC:", auc)
        st.pyplot(fig)
        plt.close(fig)

        # For ranking tuned models
        y_pred = model.predict(X_test)
        y_prob = (model.predict_proba(X_test)[:, 1]
                  if hasattr(model, "predict_proba")
                  else model.decision_function(X_test))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        acc = accuracy_score(y_test, y_pred)
        tuned_records.append({
            "model": name,
            "auc": roc_auc_score(y_test, y_prob),
            "f1_macro": f1,
            "precision_macro": prec,
            "recall_macro": rec,
            "accuracy": acc
        })

    st.subheader("Tuned Model Performance Ranking")
    tuned_results_df = pd.DataFrame(tuned_records) \
                       .sort_values(["auc", "f1_macro"], ascending=False) \
                       .reset_index(drop=True)

    st.dataframe(tuned_results_df.style.format({"auc": "{:.3f}",
                                               "f1_macro": "{:.3f}",
                                               "accuracy": "{:.3f}",
                                                "precision_macro": "{:.3f}",
                                                "recall_macro": "{:.3f}"}))

    best_tuned_name = tuned_results_df.iloc[0, 0]
    final_model = best_tuned_models[best_tuned_name]
    st.write(f"\nFinal selected model based on ROC-AUC (then F1): **{best_tuned_name}**")

    st.subheader("Final Model Evaluation (on Test Set)")
    report, auc, fig = evaluate(final_model, X_test, y_test, name="Final Model")
    st.text("Test-set metrics:\n" + classification_report(y_test, final_model.predict(X_test), digits=3, zero_division=0))
    st.write("ROC-AUC:", auc)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Model Interpretability with SHAP")

    # Need to refit the final model on the full training data before SHAP
    # This is often done for the final model deployed
    st.write("Refitting the final model on the entire training data for SHAP explanation...")
    final_model.fit(X_train, y_train)


    # 1. Pulling out the fitted booster and the pre-processed features
    # This part depends on the specific model type.
    # For tree-based models inside a pipeline, the model is usually in the last step.
    # We need to handle different model types if they are not tree-based.
    # For simplicity, assuming the final model is tree-based or supports feature_importances_
    # or we use a general explainer like KernelExplainer (slower).

    # Let's check the type of the final model's model step
    model_step = final_model.named_steps["model"]

    try:
        # Use TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model_step)
        st.write("Using TreeExplainer.")
    except Exception as e:
        st.warning(f"Could not use TreeExplainer: {e}. Using KernelExplainer (may be slower)...")
        # Need a background dataset for KernelExplainer
        # Preprocess a subset of the training data to use as the background
        X_train_prep_subset = preprocess.transform(X_train.sample(100, random_state=42))
        explainer = shap.KernelExplainer(model_step.predict_proba, X_train_prep_subset)
        st.write("Using KernelExplainer.")


    X_test_prep  = final_model.named_steps["prep"].transform(X_test)

    # Convert to a dense numeric array if sparse
    if isinstance(X_test_prep, (np.ndarray, pd.DataFrame)):
        X_test_prep_dense = X_test_prep.astype(np.float32)
    elif hasattr(X_test_prep, 'toarray'):
         X_test_prep_dense = X_test_prep.toarray().astype(np.float32)
    else:
         st.error("Unsupported data type after preprocessing for SHAP.")
         return


    st.write("Calculating SHAP values (this may take some time)...")
    if isinstance(explainer, shap.TreeExplainer):
         # TreeExplainer can take the original data or preprocessed data
         # If the pipeline preprocesses, it's better to pass preprocessed data
         # shap_values = explainer.shap_values(X_test_prep_dense)
         # TreeExplainer expects the raw data if the model inside the pipeline
         # was fitted on the raw data. Since we refitted the pipeline on X_train,
         # the model inside is fitted on preprocessed data. So we pass preprocessed.
          if hasattr(model_step, "predict_proba"):
              # For classification models with predict_proba, shap_values will be a list of arrays
              shap_values = explainer.shap_values(X_test_prep_dense)
              # We need SHAP values for the positive class (index 1)
              shap_values = shap_values[1]
          else:
              # For models without predict_proba, shap_values will be a single array
              shap_values = explainer.shap_values(X_test_prep_dense)

    elif isinstance(explainer, shap.KernelExplainer):
         # KernelExplainer expects preprocessed data
         shap_values = explainer.shap_values(X_test_prep_dense)
         # KernelExplainer for classification returns a list of arrays
         shap_values = shap_values[1] # Get values for the positive class (index 1)


    st.subheader("Global Feature Importance (SHAP Summary Plot)")
    # Get feature names after preprocessing
    feature_names = final_model.named_steps["prep"].get_feature_names_out()

    # Ensure feature_names match the columns of X_test_prep_dense if it's a DataFrame
    if isinstance(X_test_prep_dense, pd.DataFrame):
         X_test_plot = X_test_prep_dense
    else:
         X_test_plot = pd.DataFrame(X_test_prep_dense, columns=feature_names)

    fig_summary, ax_summary = plt.subplots() # Create a figure and an axes
    shap.summary_plot(shap_values, X_test_plot,
                      feature_names=feature_names,
                      show=False,  # Important: don't show here, let Streamlit show
                      # Set plotting_k to limit the number of features shown
                      plot_size = (10, min(20, len(feature_names) * 0.5 + 2)) # Adjust plot size based on number of features
                     )
    plt.tight_layout()
    st.pyplot(fig_summary)
    plt.close(fig_summary)


    st.subheader("SHAP Dependence Plots")
    st.write("Select a feature to see its dependence plot.")

    # Get actual feature names from the preprocessor
    preprocessed_feature_names = final_model.named_steps["prep"].get_feature_names_out()
    feature_to_plot = st.selectbox("Select a feature:", preprocessed_feature_names)

    # Find the index of the selected feature
    try:
        feat_idx = list(preprocessed_feature_names).index(feature_to_plot)
        fig_dep, ax_dep = plt.subplots()
        shap.dependence_plot(
            feat_idx,
            shap_values,
            X_test_plot, # Use the DataFrame with correct column names
            feature_names=preprocessed_feature_names,
            interaction_index=None,
            show=False # Let Streamlit handle showing the plot
        )
        plt.tight_layout()
        st.pyplot(fig_dep)
        plt.close(fig_dep)
    except ValueError:
        st.error(f"Feature '{feature_to_plot}' not found.")


    st.subheader("Single-Household Force Plot")
    st.write("This plot shows how individual features contribute to the prediction for a specific household.")

    # Select an index for the force plot
    # Ensure the index is within the bounds of the test set
    max_idx = X_test.shape[0] - 1
    idx = st.slider("Select a household index (from the test set):", 0, max_idx, 42)

    row_shap   = shap_values[idx]
    row_feats  = X_test_plot.iloc[idx] # Use the DataFrame for feature names

    # For force_plot, explainer.expected_value should match the base value of the SHAP plot
    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

    st.write(f"Force plot for household index {idx}:")
    # Use shap.initjs() and components.html to render the force plot in Streamlit
    # Reference: https://docs.streamlit.io/library/components/built-in/html
    # shap.force_plot returns a matplotlib figure if matplotlib=True
    # If matplotlib=False, it returns an Explanation object which is not directly plottable by st.pyplot
    # We need to render the HTML output if matplotlib=False

    # To get the HTML, we can save the force plot to an HTML file and then read it
    # Or use shap.plot.force which directly generates the HTML
    # Using shap.plot.force requires the latest shap version and might need extra handling

    # Let's try generating the HTML directly from the plot object
    # shap.force_plot(expected_value, row_shap, row_feats, matplotlib=False) does not return HTML directly.

    # A common workaround is to use shap.save_html or similar
    # Or use the `shap.display.force_plot` which is intended for display environments like notebooks/colab
    # Streamlit can display HTML using st.components.v1.html

    # Generate the HTML representation of the force plot
    # This requires the latest SHAP version and might behave differently
    # depending on the environment.
    # Let's try generating the plot and see if it works in Streamlit via st.pyplot or other means.

    # Revert to using matplotlib=True and display the matplotlib figure
    # This seems more reliable in Streamlit with st.pyplot
    fig_force, ax_force = plt.subplots(figsize=(10, 3)) # Adjust figure size
    shap.force_plot(expected_value, row_shap, row_feats, matplotlib=True, show=False, ax=ax_force)
    plt.tight_layout()
    st.pyplot(fig_force)
    plt.close(fig_force)

    # Note: The interactive JavaScript force plot is harder to embed directly in Streamlit
    # without using custom components or saving to HTML and embedding.
    # The matplotlib version provides a static representation.

    # Save the final model and explainer (optional in the Streamlit app itself, but useful for deployment)
    # st.subheader("Save Model and Explainer")
    # if st.button("Save Final Model and Explainer"):
    #     try:
    #         joblib.dump(final_model, "starlink_final_model.pkl")
    #         # explainer object might not be directly serializable depending on its type and content
    #         # A safer approach might be to save the model and regenerate the explainer when loading
    #         # For TreeExplainer, saving the model should be sufficient to regenerate it
    #         # joblib.dump(explainer, "starlink_explainer.pkl") # This might fail
    #         # np.save("shap_global_values.npy", shap_values) # This saves the values, not the explainer
    #         st.success("Final model saved as starlink_final_model.pkl")
    #     except Exception as e:
    #         st.error(f"Error saving model: {e}")


if __name__ == "__main__":
    run_modeling_page()
