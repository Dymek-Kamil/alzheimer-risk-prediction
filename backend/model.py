import seaborn as sns
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, chi2, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dane = pd.read_csv("data/alzheimers_disease_data.csv")
# Deleting columns with no use
columns_to_delete = ["PatientID", "DoctorInCharge"]
dane = dane.drop(columns = [col for col in columns_to_delete if col in dane.columns])
corr_dane = dane.corr(numeric_only=True)

# Heatmap

plt.figure(figsize=(20,20))
sns.heatmap(corr_dane, annot=False, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
# ForrestClassifier
y = dane["Diagnosis"]
X = dane.drop(columns = ["Diagnosis"])

X = pd.get_dummies(X)

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

columns_pca = [f"PC{i+1}" for i in range(pca.n_components_)]
X = pd.DataFrame(X_pca, columns=columns_pca)

df_plot = pd.DataFrame(X, columns=["PC1", "PC2","PC3"])
df_plot["Diagnosis"] = y.values
df_plot["Status"] = df_plot["Diagnosis"].map({0: "Zdrowy (0)", 1: "Alzheimer (1)"})

# Procent wariancji wyjaśnianej przez każdą składową
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
        "PC3": f"PC3 ({var_pc3:.2f}%)"
    }
)

fig.update_traces(marker=dict(size=3))
fig.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
y_try = final_model.predict_proba(X_test)[:,1]

print(classification_report(y_test,y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_try):.4f}")





