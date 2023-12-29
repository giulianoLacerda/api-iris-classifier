import mlflow
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

iris = fetch_ucirepo(id=53)
remote_server_uri = "http://localhost:8081/"
mlflow.set_tracking_uri(remote_server_uri)
mlflow.set_experiment(experiment_id="1")
mlflow.autolog()
grid = [
    {
        "clf__C": np.logspace(-3, 3, 7),
        "clf__solver": ["liblinear"],
        "clf__penalty": ["l1", "l2"],
        "clf__max_iter": [200]
    },
    {
        "clf__C": np.logspace(-3, 3, 7),
        "clf__solver": ["lbfgs"],
        "clf__penalty": ["l2"],
        "clf__max_iter": [200]
    },
    {
        "clf__C": np.logspace(-3, 3, 7),
        "clf__solver": ["newton-cg"],
        "clf__penalty": ["l2"],
        "clf__max_iter": [200]
    }
]

logreg = LogisticRegression()
scaler = MinMaxScaler()
x_train, x_test, y_train, y_test = train_test_split(iris.data.features, iris.data.targets,
                                                    test_size=0.2, random_state=3)

model = Pipeline(steps=[('scaler', scaler), ('clf', logreg)])
logreg_cv = GridSearchCV(model, grid, cv=5)
logreg_cv.fit(x_train.values, y_train.values.ravel())
print("tuned hpyerparameters :(best parameters) ", logreg_cv.best_params_)
print("accuracy :", logreg_cv.best_score_)

y_predict = logreg_cv.predict(x_test.values)
f1_test_score = f1_score(y_test, y_predict, average="macro")
print(f"f1 score {f1_test_score}")
