import torch
from tf_model.dataset import X_test, y_test
from utils import *

np_model =  NumpyModel(NumpyModel_path)

tf_model = models.load_model(TFmodel_path)

pyt_model = TorchModel()
pyt_model.load_state_dict(torch.load(TorchModel_path))
pyt_model.eval()

n = len(y_test)

correct_np = 0
correct_tf = 0
correct_pyt = 0


for i in range(100):

    X = X_test[i]

    X = X.reshape(1,28,28,1)

    correct_np += ((np_model.predict(X)).argmax() == y_test[i])*1
    correct_tf += ((tf_model.predict(X)).argmax() == y_test[i])*1
    correct_pyt += ((pyt_model.predict(X)).argmax() == y_test[i])*1

print((correct_np/n)*100, (correct_tf/n)*100, (correct_pyt/n)*100)