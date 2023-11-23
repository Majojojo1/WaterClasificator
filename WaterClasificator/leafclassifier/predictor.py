from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

def predict_with_model(features, target):

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    clfHGBC = HistGradientBoostingClassifier(max_iter=100)
    clfHGBC.fit(X_train, y_train)

    predictions = clfHGBC.predict(X_test)

    return predictions
