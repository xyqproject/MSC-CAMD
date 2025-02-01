import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pyomo.opt import SolverFactory
from pyomo.environ import *



# proposed framework (using a part of molecules for training).....................................
property_list=["pc", "gf", "hf", "hsolp","hv", "lmv","tc"] # the test properties

for property in property_list:
    df = pd.read_excel(fr"the location of dataset files\{property}.xlsx",index_col=0)
    train_num = max(int(0.8 * len(df.index)),len(df.index)-200)
    molecule=[df.loc[j, [f"Group {i + 1}" for i in range(424)]].values for j in range(1,train_num+1)]
    n=50 # the number of training molecules

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

        model = ConcreteModel()
        model.I = Set(initialize=[i for i in range(424)])
        model.par = Var(model.I, within=Reals, initialize=0)
        model.par0 = Var(within=Reals, initialize=0)
        def obj_rule(model):
            error = 0
            for i in range(len(molecule_training)):
                Tc = model.par0
                for j in range(424):
                    Tc += model.par[j] * molecule_training[i][j]
                error += (Tc - property_training[i]) ** 2
            error = (error / len(molecule_training)) ** 0.5
            return error
        model.obj = Objective(rule=obj_rule, sense=minimize)
        opt = SolverFactory('gams')
        io_options = dict()
        io_options['solver'] = "minos"
        io_options['mtype'] = "NLP"
        result = opt.solve(model, tee=False, keepfiles=False, io_options=io_options)
        pro_pre=np.sum([value(model.par[i])*molecule_new[i] for i in range(424)])+value(model.par0)
        return pro_pre,list(sorted_MSC.values())[0]

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
    df_output.to_excel(fr"the location of result files\GC\{property}\{property}_output_{n}.xlsx")





# using all molecules for training.....................................
property_list=["vc", "pc", "ait", "gf", "hf", "hsolp","hv", "lmv","tc"] # the test properties

for property in property_list:
    df = pd.read_excel(fr"the location of dataset files\{property}.xlsx",index_col=0)
    train_num = max(int(0.8 * len(df.index)),len(df.index)-200)

    def MSC(list1,list2): # calculate the molecular similarity
        intersection = [min(list1[i],list2[i])+1 for i in range(len(list1))]
        union = [max(list1[i],list2[i])+1 for i in range(len(list1))]
        res=(np.product(intersection)-1)/(np.product(union)-1)
        return res

    molecule_training = [df.loc[j, [f"Group {i + 1}" for i in range(424)]].values for j in range(1, train_num + 1)]
    molecule_testing = [df.loc[j, [f"Group {i + 1}" for i in range(424)]].values for j in range(train_num + 1, len(df.index) + 1)]
    property_training = [df.loc[j, f"{property}"] for j in range(1, train_num + 1)]
    MaxMSC_dict = {i: max(MSC(molecule_testing[i], molecule_training[j]) for j in range(len(molecule_training))) for i in range(len(molecule_testing))}

    model = ConcreteModel()
    model.I = Set(initialize=[i for i in range(424)])
    model.par = Var(model.I, within=Reals, initialize=0)
    model.par0 = Var(within=Reals, initialize=0)
    def obj_rule(model):
        error = 0
        for i in range(len(molecule_training)):
            Tc = model.par0
            for j in range(424):
                Tc += model.par[j] * molecule_training[i][j]
            error += (Tc - property_training[i]) ** 2
        error = (error / len(molecule_training)) ** 0.5
        return error
    model.obj = Objective(rule=obj_rule, sense=minimize)
    opt = SolverFactory('gams')
    io_options = dict()
    io_options['solver'] = "minos"
    io_options['mtype'] = "NLP"
    result = opt.solve(model, tee=False, keepfiles=False, io_options=io_options)
    pro_pre=[np.sum([value(model.par[i])*molecule_testing[j][i] for i in range(424)])+value(model.par0) for j in range(len(molecule_testing))]
    df_output=pd.DataFrame()
    for i in range(train_num + 1, len(df.index) + 1):
        print(i)
        property_new = df.loc[i, f"{property}"]
        print(pro_pre[i - train_num - 1], property_new, MaxMSC_dict[i - train_num - 1])
        df_output.loc[i, f"{property}_pre"] = pro_pre[i - train_num - 1]
        df_output.loc[i, f"{property}_real"] = property_new
        df_output.loc[i, "MSC"] = MaxMSC_dict[i - train_num - 1]
        df_output.loc[i, "error"] = abs((pro_pre[i - train_num - 1] - property_new) / property_new)
    df_output.to_excel(fr"the location of result files\GC\{property}\{property}_output_{train_num}.xlsx")





