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

dane = pd.read_csv("data/alzheimers_disease_data.csv")

# plt.figure(figsize=(10, 10))
# sns.countplot(data=dane, x="Diagnosis", hue="Diagnosis")
# plt.title("liczebność grup badanych")
# plt.tight_layout()
# plt.savefig(f'plots/grupyBadane.png', dpi=300, bbox_inches='tight')
# plt.close()
#
# columns = ['Gender', 'Ethnicity', 'EducationLevel', 'Smoking', 'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression', 'HeadInjury', 'Hypertension', 'MemoryComplaints', 'BehavioralProblems', 'Confusion', 'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness']
# for i, col in enumerate(columns):
#     data = pd.crosstab(dane['Diagnosis'], dane[col])
#     ax = data.plot(kind='bar', figsize=(10, 10))
#     plt.title(f'{col} distribution')
#     plt.xlabel('Diagnosis')
#     plt.ylabel('Count')
#     plt.xticks(rotation=0)
#     plt.tight_layout()
#     plt.savefig(f'plots/{col}.png', dpi=300, bbox_inches='tight')
#     plt.close()

# columns = ['BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE', 'FunctionalAssessment', 'ADL']
# for i, col in enumerate(columns):
#     plt.figure(figsize=(10, 10))
#     sns.boxplot(data=dane, x="Diagnosis", y=col)
#     plt.savefig(f'plots/{col}.png')
#     plt.close()
#
# # Usunięcie zbędnych kolumn
# columns_to_delete = ["PatientID", "DoctorInCharge", "Diagnosis"]
# X = dane.drop(columns=[col for col in columns_to_delete if col in dane.columns])
# y_raw = dane["Diagnosis"]  # Liczbowe 0 i 1 (bezpieczniejsze dla ROC-AUC)
# y_labels = y_raw.map({0: "Zdrowy (0)", 1: "Alzheimer (1)"})  # Tekstowe do wykresów


switch = 3

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
y_try = final_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_try):.4f}")