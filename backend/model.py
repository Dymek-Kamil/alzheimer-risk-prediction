import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold, chi2, SelectKBest
from sklearn.ensemble import RandomForestClassifier

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

trait_importance = pd.Series(model_rf.feature_importances_, index=X.columns)
trait_importance = trait_importance.sort_values(ascending = False)

print(trait_importance)





