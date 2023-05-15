from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the PMML file and create a prediction pipeline
# pmml_pipeline = PMMLPipeline.from_pmml(open('model.pmml', 'r'))
model = joblib.load('model.pkl')


# Define a predict endpoint

def load_data():
    '''
    此处为加载鸢尾花数据集，并划分数据集
    return:训练数据、测试数据
    '''

    """-----------------加载数据----------------------------"""
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    """-------------------划分数据集---  -------------------"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    print("数据准备完毕！！")
    return X_train, X_test, y_train, y_test


def modelTrain(X_train, X_test, y_train, y_test):
    '''
    定义逻辑回归模型
    return ：模型评估和模型
    '''
    print("---------------开始训练-------------------------")

    """-------------------模型训练---  -------------------"""
    lr = LogisticRegression(C=5, class_weight=None, dual=False, fit_intercept=True,
                            intercept_scaling=1, max_iter=100, multi_class='multinomial',
                            n_jobs=None, penalty='l2', random_state=None, solver='lbfgs',
                            tol=0.0001, verbose=0, warm_start=False)

    lr.fit(X_train, y_train)

    return lr


def modelSave(lr):
    '''
    path：保存路径
    pipmodel：需要保存的模型
    '''
    joblib.dump(lr, "model.pkl")

    # sklearn2pmml(pipmodel, path, with_repr=True)
    print("模型保存成功")

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data as a dictionary
    input_data = request.get_json()
    # Make a prediction using the loaded model
    prediction = model.predict(input_data)
    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})

# %%
if "__main__" == __name__:

    # 加载数据
    X_train, X_test, y_train, y_test = load_data()
    # 模型训练
    lr = modelTrain(X_train, X_test, y_train, y_test)
    # 模型保存
    modelSave(lr)

    app.run(debug=True)