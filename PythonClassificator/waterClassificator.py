# IMPORTAR LAS LIBRERIAS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score

# 1. CARGAR EL CONJUNTO DE DATOS
df = pd.read_csv("https://raw.githubusercontent.com/Majojojo1/Use-dataset/main/Dataset/water_potability.csv", sep=",")
df.head(5)

# 2. LIMPIEZA DE DATOS
# Verifica si hay valores nulos en todo el conjunto de datos
print(df.isnull().sum())

df['ph'].fillna(df['ph'].mean(), inplace=True)
df['Sulfate'].fillna(df['Sulfate'].min(), inplace=True)
df['Trihalomethanes'].fillna(method='ffill', inplace=True)

# 3. TRANSFORMACIÓN DE DATOS
def typeCarbon(organic_carbon):
    if organic_carbon <= 13.000000:
        return "A"
    else:
        return "Z"
    
def turbihardsolid(turbidity, hardness, solids):
   return (solids - hardness) / turbidity

# Aplicar la función a toda la columna 'type'
df['TypeCarbon'] = df['Organic_carbon'].apply(typeCarbon)
df['Turbihardsolid'] = df.apply(lambda newCol: turbihardsolid(newCol['Turbidity'], newCol['Hardness'], newCol['Solids']), axis=1)

features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Turbihardsolid']
X = df[features]
y = df['Potability']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4.1. ALGORITMOS DE CLASIFICACIÓN
clfHGBC = HistGradientBoostingClassifier(max_iter=100)
clfHGBC = clfHGBC.fit(X_train, y_train)
y_predHGBC = clfHGBC.predict(X_test)

clfSVC =  AdaBoostClassifier(n_estimators=100)
clfSVC = clfSVC.fit(X_train, y_train)
y_predSVC = clfSVC.predict(X_test)

print(classification_report(y_test, y_predHGBC))
print(classification_report(y_test, y_predSVC))

print("Confusion Matrix for HistGradientBoostingClassifier:")
print(confusion_matrix(y_test, y_predHGBC))

print("Confusion Matrix for AdaBoostClassifier:")
print(confusion_matrix(y_test, y_predSVC))

# 4.2. ALGORITMOS DE REGRECIÓN
kernel = DotProduct() + WhiteKernel()
modeloGPR = GaussianProcessRegressor(kernel=kernel, random_state=0)
modeloGPR.fit(X_train, y_train)
y_predGPR = modeloGPR.predict(X_test)

modeloR = Ridge(alpha=1.0)
modeloR.fit(X_train, y_train)
y_predR = modeloR.predict(X_test)

# PRIMER MODELO
r2GPR = r2_score(y_test, y_predGPR)
rmseGPR = mean_squared_error(y_test, y_predGPR, squared=True)
maeGPR = mean_absolute_error(y_test, y_predGPR)
mapeGPR = mean_absolute_percentage_error(y_test, y_predGPR)

# Calcular métricas para el modelo CLF
r2R = r2_score(y_test, y_predR)
rmseR = mean_squared_error(y_test, y_predR, squared=True)
maeR = mean_absolute_error(y_test, y_predR)
mapeR = mean_absolute_percentage_error(y_test, y_predR)

# Imprimir métricas para el modelo GaussianProcessRegressor
print("Metrics for GaussianProcessRegressor:")
print(f"R2: {r2GPR}")
print(f"RMSE: {rmseGPR}")
print(f"MAE: {maeGPR}")
print(f"MAPE: {mapeGPR}")

# Imprimir métricas para el modelo Ridge
print("Metrics for Ridge:")
print(f"R2: {r2R}")
print(f"RMSE: {rmseR}")
print(f"MAE: {maeR}")
print(f"MAPE: {mapeR}")

# 4.3. TABLA COMPARATIVA
# Calcular métricas para cada modelo
accuracy_HGBC = accuracy_score(y_test, y_predHGBC)
accuracy_SVC = accuracy_score(y_test, y_predSVC)

f1_HGBC = f1_score(y_test, y_predHGBC)
f1_SVC = f1_score(y_test, y_predSVC)

mse_KNR = mean_squared_error(y_test, y_predGPR)
mse_R = mean_squared_error(y_test, y_predR)

r2_KNR = r2_score(y_test, y_predGPR)
r2_R = r2_score(y_test, y_predR)

data = {
    'Modelo': ['HistGradientBoostingClassifier', 'AdaBoostClassifier', 'GaussianProcessRegressor', 'Ridge'],
    'Precisión': [accuracy_HGBC, accuracy_SVC, "-", "-"],
    'F1-score': [f1_HGBC, f1_SVC, "-", "-"],
    'Error cuadrático medio': ["-", "-", mse_KNR, mse_R],
    'R2': ["-", "-", r2_KNR, r2_R],
}

datadf = pd.DataFrame(data)
print(datadf)
