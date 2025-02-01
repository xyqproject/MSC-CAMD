import math
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import RBF
import numpy as np
from skopt.space import Space
from skopt.sampler import Grid
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.gaussian_process import GaussianProcessRegressor


def R_square(y_test,y_predict):
    ybar = np.sum(y_test) / len (y_test)
    SSE=np.sum((y_test - y_predict)**2)
    SSR = np.sum((ybar-y_predict)**2)
    SST = np.sum((y_test - ybar)**2)
    return 1-SSE/SST


# kernel accelerate
def my_kernal(x,y,sigma=1,l=1):
    kernel_package=sigma**2 * RBF(length_scale=l)
    return  kernel_package(x,y)

    
def GP_tuning(train_input,train_output,test_input,test_output,l):
    K_=[]
    K_trans_=[]
    if type(l)==float or type(l)==int or type(l)==np.float64:
        K_trans=my_kernal(test_input,train_input,1,l)
        K=my_kernal(train_input,train_input,1,l)
        K_.append(K)
        K_trans_.append(K_trans)
    else:
        for i in range(len(l)):
            K_trans_.append(my_kernal(test_input,train_input,1,l[i]))
            K_.append(my_kernal(train_input,train_input,1,l[i]))
    return K_,K_trans_
   
 
# normal GP
def GP(train_input,train_output,test_input,test_output,l,sigma_e=1e-5,return_y=False):
    train_size=len(train_output)
    test_size=len(test_output)
    K_trans=np.zeros((test_size,train_size))
    K=np.zeros((train_size,train_size))
    if type(l)==float or type(l)==int or type(l)==np.float64:
        K_trans=K_trans+my_kernal(test_input,train_input,1,l)
        K=K+my_kernal(train_input,train_input,1,l)
    else:
        for i in range(len(l)):
            K_trans=K_trans+my_kernal(test_input,train_input,1,l[i])
            K=K+my_kernal(train_input,train_input,1,l[i])
    K2=K+sigma_e*np.eye(train_size)
    K_inv = np.linalg.lstsq(K2,np.eye(train_size),rcond=None)[0]
    y_predict=K_trans@K_inv@train_output
    rmse=RMSE(y_predict,test_output)
    if return_y:
        return rmse,y_predict
    else:
        return rmse


def hyper_tuning(train_input,train_output,grid_number=200,fold_number=10):
    space = Space([(0.1,1),(5.0,10.0),(0.1,10.0),(10.0,100.0)]) 
    grid = Grid(border="include", use_full_layout=False)  # Grid search
    hyperparameter = grid.generate(space.dimensions, grid_number)
    train_number=train_input.shape[0]
    validation_number=(train_number-train_number%fold_number)/fold_number
    validation_input=[]
    validation_output=[]
    train_except_validation_input=[]
    train_except_validation_output=[]
    for i in range(fold_number):
        begin_index=int(validation_number*i)
        end_index=int(validation_number*(i+1))
        delete_index=np.array([i for i in range(begin_index,end_index)])
        validation_input.append(train_input[begin_index:end_index])
        validation_output.append(train_output[begin_index:end_index])
        train_except_validation_input.append(np.delete(train_input,delete_index, axis=0))
        train_except_validation_output.append(np.delete(train_output,delete_index, axis=0))
    print("------------start hyperparameter tuning------------------")
    loss=[]     
    for i in range(grid_number):
        loss.append(validation_loss(train_except_validation_input, train_except_validation_output,
                            validation_input,validation_output,hyperparameter[i][:4],fold_number))
        print("--------------",i,"length scale:",hyperparameter[i],"average rmse on validation set:",loss[i],"--------------")
    min_loss_index=find_min_index(np.array(loss))[1]
    sigma_e=1e-5
    save={"length_scale":hyperparameter[min_loss_index],"sigma_e":sigma_e,"minimum loss":loss[min_loss_index]}
    return save
    

# calculate hyperparameter
def validation_loss(train_input,train_output,validation_input,validation_output,hyperparameter,fold_number):
    rmse_array=np.zeros(fold_number)
    for i in range(fold_number):
        rmse_vali=GP(train_input[i],train_output[i],validation_input[i],validation_output[i],hyperparameter)
        rmse_array[i]=rmse_vali
        # print("rmse for  "+str(i+1)+"th fold:",rmse_vali)
    rmse_aver=sum(rmse_array)/fold_number
    return rmse_aver


# calculate RMSE
def RMSE(y_test,y_predict):
    y_test=y_test.reshape(-1)
    y_predict=y_predict.reshape(-1)
    rmse=math.sqrt(mean_squared_error(y_test, y_predict))
    return rmse


#filter the sample in test dataset as they are disordered
def index_of_test(train_input,train_output,data_input,data_output,dimention=424):
    train_input=train_input.reshape(-1,dimention)
    train_output=train_output.reshape(-1,1)
    data_input=data_input.reshape((-1,dimention))
    data_output=data_output.reshape(-1,1)
    index_exist=np.zeros(data_output.shape[0])
    for i in range(data_output.shape[0]):
        for j in range (train_output.shape[0]):
            if (data_output[i]==train_output[j]).all() and (data_input[i]==train_input[j]).all():
               index_exist[i]=1
    index_test=[]
    for i in range(len(index_exist)):
        if index_exist[i]==0:
            index_test.append(i)
    return index_test


def index_of_train(train_input,train_output,data_input_atom,data_output_atom,dimention=424):
    train_input=train_input.reshape(-1,dimention)
    train_output=train_output.reshape(-1,1)
    data_input_atom=data_input_atom.reshape((-1,dimention+15))
    data_output_atom=data_output_atom.reshape(-1,1)
    index_exist=np.zeros(data_output_atom.shape[0])
    index_train=np.zeros(train_output.shape[0])
    for j in range(train_output.shape[0]):
        for i in range(data_output_atom.shape[0]):
            if (data_input_atom[i,15:]==train_input[j]).all(): #(data_output_atom[i]==train_output[j]).all() and 
               index_exist[i]=1
               index_train[j]=i
    return index_train


#multiply different coefficients in different dimensions
def input_modi(coef,alfa,dataset):
    dataset_modi=np.zeros_like(dataset)
    dataset=dataset.reshape(-1,dataset.shape[1])
    for i1,element in enumerate(dataset.T):  
        if coef[i1]!=0:
            dataset_modi[:,i1]=element*((np.abs(coef[i1]))**alfa) 
    return dataset_modi


#analyse the percentage of data within 1%,5%,10%
def error_analysis(y_predict,y_true):
    number_sample=len(y_predict)
    within_one_percent=np.zeros(number_sample)
    within_five_percent=np.zeros(number_sample)
    within_ten_percent=np.zeros(number_sample)
    relative_error=np.zeros(number_sample)
    for i in range(number_sample):
        relative_error[i]=abs(y_predict[i]-y_true[i])/y_true[i]
        if relative_error[i]<=0.01:
            within_one_percent[i]=1
        if relative_error[i]<=0.05:
            within_five_percent[i]=1
        if relative_error[i]<0.1:
            within_ten_percent[i]=1
    error_one_percent=sum(within_one_percent)/number_sample
    error_five_percent=sum(within_five_percent)/number_sample
    error_ten_percent=sum(within_ten_percent)/number_sample
    error=[error_one_percent,error_five_percent,error_ten_percent]
    error=np.array(error).reshape(1,3)
    return {"relative_error":relative_error,"error_one_percent":error_one_percent,"error_five_percent":error_five_percent,"error_ten_percent":error_ten_percent,"error":error}

    
#find minimum value and its coefficient in a two-dimentional matrix    
def find_min_index(x):
    if len(x.shape)==2:
        index=[0,0]
        min_value=x.flatten()[0]
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # print(i,j)
                if x[i,j]<min_value:
                    index[0]=i
                    index[1]=j
                    min_value=x[i,j]
    if len(x.shape)==1:
        index=0
        min_value=x[0]
        for i in range(len(x)):
            if x[i]<min_value:
                index=i
                min_value=x[i]      
    return min_value,index


def plot_quartile(data):
    data = pd.DataFrame(data)
    plt.rcParams['savefig.dpi'] = 1000
    plt.show()


    

    
    
