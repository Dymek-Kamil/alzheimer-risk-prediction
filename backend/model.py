import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import umap
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

dane = pd.read_csv("data/alzheimers_disease_data.csv")

# Usunięcie zbędnych kolumn
columns_to_delete = ["PatientID", "DoctorInCharge", "Diagnosis"]
X = dane.drop(columns=[col for col in columns_to_delete if col in dane.columns])
y_raw = dane["Diagnosis"]  # Liczbowe 0 i 1 (bezpieczniejsze dla ROC-AUC)
y_labels = y_raw.map({0: "Zdrowy (0)", 1: "Alzheimer (1)"})  # Tekstowe do wykresów


# 1 - PCA
# 2 - tSNE
# 3 - UMAP
# 4 - Klasyfikacja waznosci modelem regresji liniowej. (domyślne)
switch = 4

X = pd.get_dummies(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if switch == 1:
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    columns_pca = [f"PC{i+1}" for i in range(pca.n_components_)]
    X = pd.DataFrame(X_pca, columns=columns_pca)

    df_plot = X.copy()
    df_plot["Diagnosis"] = y_labels.values

    var_pc1 = pca.explained_variance_ratio_[0] * 100
    var_pc2 = pca.explained_variance_ratio_[1] * 100
    var_pc3 = pca.explained_variance_ratio_[2] * 100

    fig = px.scatter_3d(
        df_plot,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Diagnosis",
        color_discrete_map={"Zdrowy (0)": "#2b5c8f", "Alzheimer (1)": "#d95f02"},
        opacity=0.6,
        title=f"PCA 3D (Wyjaśniona wariancja: {var_pc1+var_pc2+var_pc3:.2f}%)",
        labels={
            "PC1": f"PC1 ({var_pc1:.2f}%)",
            "PC2": f"PC2 ({var_pc2:.2f}%)",
            "PC3": f"PC3 ({var_pc3:.2f}%)",
        },
    )
    fig.update_traces(marker=dict(size=3))
    fig.show()

elif switch == 2:
    tsne = TSNE(
        n_components=3,
        perplexity=30,
        random_state=42,
        init="pca",
        learning_rate="auto",
    )
    X_tsne = tsne.fit_transform(X_scaled)
    X = pd.DataFrame(X_tsne, columns=["t-SNE 1", "t-SNE 2", "t-SNE 3"])
    df_plot = X.copy()
    df_plot["Diagnosis"] = y_labels.values

    fig = px.scatter_3d(
        df_plot,
        x="t-SNE 1",
        y="t-SNE 2",
        z="t-SNE 3",
        color="Diagnosis",
        color_discrete_map={"Zdrowy (0)": "#2b5c8f", "Alzheimer (1)": "#d95f02"},
        opacity=0.65,
        title="Wizualizacja t-SNE 3D (perplexity=30)",
    )
    fig.update_traces(marker=dict(size=3))
    fig.show()

elif switch == 3:
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
    )
    X_umap = reducer.fit_transform(X_scaled)
    X = pd.DataFrame(X_umap, columns=["UMAP 1", "UMAP 2", "UMAP 3"])

    df_plot = X.copy()
    df_plot["Diagnosis"] = y_labels.values

    fig = px.scatter_3d(
        df_plot,
        x="UMAP 1",
        y="UMAP 2",
        z="UMAP 3",
        color="Diagnosis",
        color_discrete_map={"Zdrowy (0)": "#2b5c8f", "Alzheimer (1)": "#d95f02"},
        opacity=0.65,
        title="Wizualizacja UMAP 3D (n_neighbors=15, min_dist=0.1)",
    )
    fig.update_traces(marker=dict(size=3))
    fig.show()

elif switch == 4:
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
        X, y_raw, test_size=0.2, random_state=42, stratify=y_raw
    )

    scaler_lr = StandardScaler()
    X_train_scaled = scaler_lr.fit_transform(X_train_lr)
    X_test_scaled = scaler_lr.transform(X_test_lr)

    model_lr = LogisticRegression(max_iter=1000, random_state=42)
    model_lr.fit(X_train_scaled, y_train_lr)

    wplyw_cech = pd.DataFrame(
        {
            "Cecha": X.columns,
            "Wspolczynnik": model_lr.coef_[0],
            "Odds_Ratio": np.exp(model_lr.coef_[0]),
            "Sila_wplywu": np.abs(model_lr.coef_[0]),
        }
    ).sort_values(by="Wspolczynnik", ascending=False)

    plt.figure(figsize=(11, 9))
    kolory = ["#d95f02" if c > 0 else "#2b5c8f" for c in wplyw_cech["Wspolczynnik"]]

    sns.barplot(
        data=wplyw_cech,
        x="Wspolczynnik",
        y="Cecha",
        palette=kolory,
        hue="Cecha",
        legend=False,
    )
    plt.axvline(x=0, color="black", linewidth=1)
    plt.title("Wpływ cech na diagnozę Alzheimera (Regresja Logistyczna)", fontsize=14)
    plt.xlabel(
        "Współczynnik ( < 0 zmniejsza ryzyko | > 0 zwiększa ryzyko )", fontsize=11
    )
    plt.ylabel("Cecha", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

    y_pred_lr = model_lr.predict(X_test_scaled)
    y_prob_lr = model_lr.predict_proba(X_test_scaled)[:, 1]



    wybrane_kolumny_lr = wplyw_cech[wplyw_cech["Sila_wplywu"] > 0.5]["Cecha"].tolist()
    X = X[wybrane_kolumny_lr]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
y_try = final_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_try):.4f}")