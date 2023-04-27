import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
import joblib


def load_data(file_path):

    file = pd.read_csv(file_path)

    for name in ['JPK_A', 'JPK_B', 'JPK_C', 'Feeder_A', 'Feeder_B', 'Feeder_C',]:
        name_new = name +"_diff"
        file[name_new] = file[name] - file["Ambient"]

    return file


def data_split(file):

    X = file[[
        'Envelop-Feeder_Top',
        'Envelop-Feeder_Bot',
        'Envelop-Feeder_L',
        'Envelop-Feeder_R',
        'Envelop-JPK Cover_Top',
        'Envelop-JPK Cover_Bot',
        'Envelop-JPK Cover_ZN_L',
        'Envelop-JPK Cover_ZN_R',
        'Envelop-JPK Cover_ZP_L',
        'Envelop-JPK Cover_ZP_R',]]

    target = ['JPK_A_diff', 'JPK_B_diff', 'JPK_C_diff', 'Feeder_A_diff', 'Feeder_B_diff', 'Feeder_C_diff',]
    y = file[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

    return X_train, X_test, y_train, y_test, target

def train(X_train, X_test, y_train, y_test, target):

    enet = ElasticNet(alpha=0, l1_ratio=0)
    enet.fit(X_train, y_train)
    joblib.dump(enet, "model.pkl")

    prediction = pd.DataFrame(enet.predict(X_test), columns=target)
    r2 = r2_score(y_test, prediction)

    print("r2 is : ", r2)
    print("-rmse is : ", -mean_squared_error(y_test, prediction))
    return r2

if __name__ == "__main__":
    file = load_data(r"C:\Users\sesa704291\Desktop\母线场景\负样本.csv")
    X_train, X_test, y_train, y_test, target = data_split(file)
    train(X_train, X_test, y_train, y_test, target)
