import pandas as pd
import numpy as np
import os
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split,GridSearchCV
import pickle
import warnings
from skopt.space import Space
from skopt.sampler import Grid
from ProductDesignFunction import RMSE,index_of_train, index_of_test, GP_tuning, RMSE, input_modi, error_analysis, \
    validation_loss, find_min_index, GP, plot_quartile, hyper_tuning
warnings.filterwarnings("ignore")


def fun_SVR(train_input,train_output,test_input,test_output): # Train the model using SVR method
    param_grid={
        'C':[1,10,100,1000],
        'epsilon':[0.0001,0.001,0.01,0.1],
                }
    print("-----------------simple1----------------")
    grid_search = GridSearchCV(SVR(kernel="linear",gamma="auto"),param_grid,cv=5,verbose=1)
    grid_search.fit(train_input, train_output)
    model_simple =grid_search.best_estimator_
    print(f"best param: {grid_search.best_params_}")
    print("-----------------stop training------------------")
    test_predict_simple = model_simple.predict(test_input)
    train_predict_simple = model_simple.predict(train_input)
    return  train_predict_simple, test_predict_simple


# proposed framework (using a part of molecules for training).....................................
property_list=["vc", "pc", "ait", "gf", "hf", "hsolp","hv", "lmv","tc"] # the test properties

for property in property_list:
    df = pd.read_excel(fr"the location of dataset files\{property}.xlsx",index_col=0)
    train_num = max(int(0.8 * len(df.index)),len(df.index)-200)
    molecule=[df.loc[j, [f"Group {i + 1}" for i in range(424)]].values for j in range(1,train_num+1)]
    n = 50 # the number of training molecules
    def MSC(list1,list2): # calculate the molecular similarity
        intersection = [min(list1[i],list2[i])+1 for i in range(len(list1))]
        union = [max(list1[i],list2[i])+1 for i in range(len(list1))]
        res=(np.product(intersection)-1)/(np.product(union)-1)
        return res

    def Model(molecule_new,molecule):
        MSC_dict={i+1:MSC(molecule_new,molecule[i]) for i in range(len(molecule))}
        sorted_MSC = {key: MSC_dict[key] for key in sorted(MSC_dict.keys(), key=MSC_dict.get,  reverse=True)}
        molecule_training=[df.loc[list(sorted_MSC.keys())[i], [f"Group {j + 1}" for j in range(424)]].values for i in range(n)]
        property_training=[df.loc[list(sorted_MSC.keys())[i], f"{property}"] for i in range(n)]
        _,pro_pre=fun_SVR(np.array(molecule_training).astype(float),np.array(property_training).astype(float),np.array([molecule_new]).astype(float),np.array([0]).astype(float))
        return pro_pre[0],list(sorted_MSC.values())[0]

    df_output=pd.DataFrame()
    for i in range(train_num+1,len(df.index)+1):
        print(i)
        molecule_new = df.loc[i, [f"Group {j + 1}" for j in range(424)]].values
        property_new=df.loc[i, f"{property}"]
        property_pre,MSC_new=Model(molecule_new, molecule)
        print(property_pre,property_new,MSC_new)
        df_output.loc[i,f"{property}_pre"]=property_pre
        df_output.loc[i, f"{property}_real"] = property_new
        df_output.loc[i, "MSC"] = MSC_new
        df_output.loc[i, "error"] = abs((property_pre-property_new)/property_new)
    df_output.to_excel(fr"the location of result files\SVR\{property}\{property}_output_{n}.xlsx")








# using all molecules for training.....................................
property_list=["vc", "pc", "ait", "gf", "hf", "hsolp","hv", "lmv","tc"] # the test properties

def MSC(list1,list2): # calculate the molecular similarity
    intersection = [min(list1[i],list2[i])+1 for i in range(len(list1))]
    union = [max(list1[i],list2[i])+1 for i in range(len(list1))]
    res=(np.product(intersection)-1)/(np.product(union)-1)
    return res
for property in property_list:
    df = pd.read_excel(fr"the location of dataset files\{property}.xlsx",index_col=0)
    train_num = max(int(0.8 * len(df.index)),len(df.index)-200)
    molecule_training=[df.loc[j, [f"Group {i + 1}" for i in range(424)]].values for j in range(1,train_num+1)]
    molecule_testing=[df.loc[j, [f"Group {i + 1}" for i in range(424)]].values for j in range(train_num+1,len(df.index)+1)]
    property_training=[df.loc[j, f"{property}"] for j in range(1,train_num+1)]
    MaxMSC_dict={i:max(MSC(molecule_testing[i],molecule_training[j]) for j in range(len(molecule_training))) for i in range(len(molecule_testing))}
    _,pro_pre=fun_SVR(np.array(molecule_training).astype(float),np.array(property_training).astype(float),np.array(molecule_testing).astype(float),np.array([0 for i in range(len(df.index)-train_num)]).astype(float))

    df_output=pd.DataFrame()
    for i in range(train_num+1,len(df.index)+1):
        print(i)
        property_new=df.loc[i, f"{property}"]
        print(pro_pre[i-train_num-1],property_new,MaxMSC_dict[i-train_num-1])
        df_output.loc[i,f"{property}_pre"]=pro_pre[i-train_num-1]
        df_output.loc[i, f"{property}_real"] = property_new
        df_output.loc[i, "MSC"] = MaxMSC_dict[i-train_num-1]
        df_output.loc[i, "error"] = abs((pro_pre[i-train_num-1]-property_new)/property_new)
    df_output.to_excel(fr"the location of result files\SVR\{property}\{property}_output_{train_num}.xlsx")








