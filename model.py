from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

df_=pd.read_csv("C:\\Users\\berat\\pythonEğitimleri\\python\\Turkcell Makine Öğrenmesi\\sınıflandırma\\diabetes.csv")
y=df_["Outcome"]
df=df_.drop(["Outcome"],axis=1)
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.3,random_state=200)


svm=SVC()
svm_params={
    "C":np.arange(1,10),#1 ile 10 arasında 9 sayı atar
    "kernel":["linear","rbf"]
}

svm_cv=GridSearchCV(svm,svm_params,cv=5,n_jobs=-1,verbose=2)
svm_cv.fit(x_train,y_train)
C=svm_cv.best_params_["C"]
kernel=svm_cv.best_params_["kernel"]
svm_tuned=SVC(C=C,kernel=kernel)
svm_tuned.fit(x_train,y_train)
predict=svm_tuned.predict(x_test)
acscroe=accuracy_score(y_test,predict)
print(acscroe)











