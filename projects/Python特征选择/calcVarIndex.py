"""
Author: 公众号-算法进阶
卡方分箱，计算PSI、WOE、IV等指标
"""


import random
import numpy as np
import pandas as pd
import pandas.io.sql as sqlio
from sklearn.model_selection import train_test_split
from matplotlib.ticker import FuncFormatter
from collections import Counter
from scipy import stats
from sklearn import metrics
import io
import datetime
from sqlalchemy import create_engine
import matplotlib.pyplot as plt


def get_char_and_num_var_names(df, excluses=None):
    '''
    区分离散、连续变量
    @df:dataframe
    @excluses:不进行区分的列名
    '''
    if excluses is not None:
        df = df.drop(excluses, axis=1)
        
    var_types = df.dtypes
    var_types_index = var_types.index
    char_var_names,num_var_names = [], []
    for i in range(len(var_types)):
        if var_types[i] == np.datetime64:
            break
        elif var_types[i] == np.object:
            char_var_names.append(var_types_index[i])
        else:
            num_var_names.append(var_types_index[i])
    return char_var_names, num_var_names

def transformInterval(bingroup):
    '''连续变量分箱区间处理---str转为pd.Interval/set/list/float'''
    if type(bingroup)!=str:
        return '非字符串'
    
    def f(bingroup, left_side,right_side,closed):
        left =bingroup[bingroup.find(left_side)+1:bingroup.find(right_side)].split(',')[0]
        if left=='-∞':
            left = -np.inf
        else:
            left = float(left)
            
        right=bingroup[bingroup.find(left_side)+1:bingroup.find(right_side)].split(',')[1]
        if right=='+∞':
            right = np.inf
        else:
            right = float(right)
        
        if 'or' not in bingroup: #格式1 正常值--返回pd.Interval
            return pd.Interval(left,right,closed)
        else: # 格式2 正常值（区间）+异常值(集合) --返回list[pd.Interval,set]
            return [pd.Interval(left,right,closed),
                   {float(i) for i in bingroup[bingroup.find('{')+1:bingroup.find('}')].split(',')}]
        
    try:
        bingroup = bingroup.replace(' ','')
        bingroup = bingroup.replace('，',',')
        
        if '(' in bingroup and ']' in bingroup:
            return f(bingroup,'(',']',closed='right')
        elif '(' in bingroup and ')' in bingroup:
            return f(bingroup,'(',')',closed='neither')
        elif '[' in bingroup and ')' in bingroup:
            return f(bingroup,'[',')',closed='left')
        elif '[' in bingroup and ']' in bingroup:
            return f(bingroup,'[',']',closed='both')
        elif '{' in bingroup and '}' in bingroup: #格式3，异常值集合---返回set
            return {float(i) for i in bingroup[bingroup.find('{')+1:bingroup.find('}')].split(',')}
        else:#格式4单个异常值或单个正常值---返回int
            return int(float(bingroup.split('_')[-1]))
    except:
        return '字符串个数有误'


# 离散分箱
def varChi2bin_char(df, gbflag, abnor, char_var_names):
    '''
    离散变量列表内的特征分箱，最少10个特征
   
    '''
    rs = pd.DataFrame()
    t = len(char_var_names)//10
    if t ==0:t=1
    i=0
    for char_var_name in char_var_names:
        if df[char_var_name].unique().size<=150:
            char_bin = CharVarChi2Bin(df, char_var_name, gbflag, abnor, BinMax=12)
            rs = rs.append(char_bin)
        i+=1
        if i%t==0:
            print("* ",end="")
    print("")
    return rs

def CharVarChi2Bin(df,varname,gbflag,abnor=[],BinMax=10,BinPcntMin=0.03):
    '''
    离散特征卡方分箱
    @df:dataframe
    @varname:要分箱的列名列表
    @gbflag:标签
    @abnor:特殊值
    @binMax:最大分箱数
    @BinPctMin:每个分箱的最小样本占比
    '''
    # 正常数据样本
    df_nor = df[~df[varname].isin(abnor)]
    merge_bin = initBin(df_nor,varname,gbflag) # 正常数据样本初始化分箱，按照坏样本占比排序
    merge_bin = matchBin(merge_bin,BinMax,BinPcntMin) # 正常数据样本合并分箱，知道满足条件
    
    # 特殊值样本
    df_abnor = df[df[varname].isin(abnor)]
    result_abnor = initBin(df_abnor,varname,gbflag) # 特殊值数据样本分箱
    
    # 合并
    result = pd.concat([merge_bin,result_abnor],axis=0)
    result.reset_index(inplace=True)
    result.rename(columns={varname:'bingroup'},inplace=True)
    result.insert(0,'varname',varname)
    
    result = idCharBin(result)
    return result

# 离散变量初始化分箱
def initBin(df,varname,gbflag,sort='bad_percent'):
    '''
    初始化分箱
    @df:dataframe
    @varname:要分箱的列名列表
    @gbflag:标签
    @sort:排序的列名
    '''
    print([varname,gbflag])
    print(df[[varname,gbflag]])
    init_bin = df[[varname,gbflag]].groupby(varname).agg(['count','sum'])[gbflag]
    init_bin.columns = ['total_counts','bad_counts']
    init_bin.insert(1,'good_counts',init_bin.total_counts-init_bin.bad_counts)
    init_bin['bad_percent'] = init_bin.bad_counts/init_bin.total_counts
    
    if sort=='bad_percent':
        init_bin.sort_values('bad_percent', inplace=True) # 离散数据初始化分箱，按照坏样本占比排序
    else:# 年龄型数据初始化分箱
        init_bin.sort_index(inplace=True) # 按照数值大小排序
        var_list = list(init_bin.index.insert(0,'-∞'))
        var_list[-1] = '+∞'
        init_bin.index = ['({},{})'.format(var_list[i],var_list[i+1]) for i in range(len(var_list)-1)]
        
    init_bin.drop(columns=['bad_percent'],inplace=True)
    return init_bin

def matchBin(df,BinMax,BinPcntMin):
    '''
    判断正常样本分箱是否符合条件
    @df:dataframe
    @binMax:最大分箱数
    @BinPctMin:每个分箱的最小样本占比
    '''
    # 检查分组是否<=BinMax
    while len(df)>BinMax:
        minChi2=None
        for i in range(len(df.index)-1):
            chi2 = calcChi2(df.iloc[i:i+2]) #分别计算相邻两组的卡方值
            if minChi2 is None or minChi2>chi2:
                minChi2=chi2
                minIndex=i
        df = mergeBin(df, minIndex, minIndex+1)
        
        # 检查每个分组是否包同时含好样本和坏样本
        while len(df)>1 and len(df[(df['good_counts']==0) | (df['bad_counts']==0)])!=0:
            #若存在只包含好样本/坏样本的分组，且分组>1
            to_bin = df[(df['good_counts']==0)|(df['bad_counts']==0)].head(1)
            
            for i in range(len(df.index.tolist())):
                if to_bin.index[0] == df.index.tolist()[i]:
                    if i == 0:#第一个分组需要合并
                        df = mergeBin(df, i, i+1)
                        break
                    elif i == len(df.index.tolist())-1:#最后一个分组需要合并
                        df = mergeBin(df, i-1, i)
                        break
                    else:#中间分组需要合并
                        chi2_f = calcChi2(df.iloc[i-1:i+1])
                        chi2_b = calcChi2(df.iloc[i:i+2])
                        if chi2_f<=chi2_b:#与前一个分组的卡方值更小
                            df = mergeBin(df, i-1, i)
                            break
                            
                            
        # 检查每个分组的样本占比是否>=BinPcntMin
        while len(df)>1 and len(df[df.total_counts/df.total_counts.sum()<BinPcntMin])!=0:
            to_bin = df[df.total_counts/df.total_counts.sum()<BinPcntMin].head(1)
            
            for i in range(len(df.index.tolist())):
                if to_bin.index[0] == df.index.tolist()[i]:
                    if i == 0:#第一个分组需要合并
                        df = mergeBin(df, i , i+1)
                        break
                    elif i ==len(df.index.tolist())-1:#最后一个分组需要合并
                        df = mergeBin(df, i-1, i)
                        break
                    else:#中间分组需要合并
                        chi2_f = calcChi2(df.iloc[i-1:i+1])
                        chi2_b = calcChi2(df.iloc[i:i+2])
                        if chi2_f<chi2_b:#与前一个分组的卡方值更小
                            df = mergeBin(df, i-1, i)
                            break
                        else:#与后一个分组的卡方值更小
                            df = mergeBin(df, i, i+1)
                            break
    return df


def varChi2Bin_num(df, gbflag, abnor, num_var_names):
    '''
    对连续变量列表内的特征进行分箱
   @df:dataframe
   @gbflag:标签
   @abnor:特殊值
   @num_var_names:进行分箱的列名列表
    '''
    rs = pd.DataFrame()
    t = len(num_var_names)//10
    if t==0:t=1
    i=0
    for num_var_name in num_var_names:
        num_bins = NumVarChi2Bin(df,num_var_name, gbflag, abnor, BinMax=12)
        rs = rs.append(num_bins)
        i+=1
        if i%t==0:
            print("*",end="")
    print("")
    return rs


# 连续变量分箱
def NumVarChi2Bin(df,varname,gbflag,abnor=[],InitGroup=100,BinMax=10,BinPcntMin=0.03,check=True):
    '''
    连续变量分箱
    @df:dataframe
    @gbflag:标签
    @abnor:特殊值列表
    @InitGroup:特征个数
    @BinMax:最大分箱数
    @BinPcntMin:每箱的最小占比
    '''
    df_nor = df[~df[varname].isin(abnor)]
    if len(Counter(df[varname]))>100:
        merge_bin = initNumBin(df_nor,varname,gbflag,InitGroup) #等频,按数值大小排序
    else:
        merge_bin = initBin(df_nor,varname,gbflag,sort='num')
    merge_bin = matchBin(merge_bin,BinMax,BinPcntMin) #正常样本合并分箱，知道满足条件
    if check:
        merge_bin = check_bad_ratio_order(merge_bin,varname)
    merge_bin.insert(0,'bingroup',['({},{})'.format(i.split(',')[0].split('(')[1], i.split(',')[-1].split(']')[0]) for i in merge_bin.index])
    merge_bin.reset_index(drop=True, inplace=True)
    
    # 特殊值样本
    df_abnor = df[df[varname].isin(abnor)]
    result_abnor = initBin(df_abnor,varname,gbflag) #特殊值样本分箱
    result_abnor.reset_index(inplace=True)
    result_abnor.rename(columns={'index':'bingroup',varname:'bingroup'}, inplace=True)
    
    # 合并
    result = pd.concat([merge_bin,result_abnor],axis=0)
    result.reset_index(drop=True,inplace=True)
    result.insert(0,'varname',varname)
    
    result = idNumBin(result)
    return result
    
def initNumBin(df,varname,gbflag,InitGroup):
    '''
    连续型数据初始化分箱(等频分箱)
    @df:dataframe
    @varname:变量名
    @gbflag:标签
    @InitGroup:初始化分箱数
    '''
    init_bin = pd.DataFrame(df[[varname,gbflag]])
    print(init_bin[varname])
    print('-------------------')
    print(InitGroup)
    init_bin[varname]=pd.qcut(init_bin[varname],InitGroup,precision=4,duplicates='drop')
    init_bin=init_bin.groupby(varname).agg(['count','sum'])[gbflag].sort_index()
    init_bin.columns=['total_counts','bad_counts']
    init_bin.insert(1,'good_counts',init_bin.total_counts-init_bin.bad_counts)
    init_bin.index=init_bin.index.astype(str)
    
    # 区间处理(前后为∞)
    init_bin.rename(index={
        init_bin.index[0]:'(-∞,{}]'.format(init_bin.index[0].split(',')[-1].split(']')[0]),
        init_bin.index[-1]:'({},+∞]'.format(init_bin.index[-1].split(',')[0].split('(')[1])},inplace=True)
    
    return init_bin

def charVarBinCount(df_bin,dataSet,gbflag,varname='varname',bingroups='bingroups'):
    '''
    给出分箱，计算离散变量样本计数
    @df_bin:分箱后的区间和变量名['varname','groups']
    @dataSet:进行计数的dataframe
    @gbflag:标签
    @varname:变量名
    @bingroups:分箱区间
    '''
    allVarBin = pd.DataFrame()
    for var in pd.unique(df_bin[varname]).tolist():
        print('--------------'+var+'--------------')
        varData = pd.DataFrame(dataSet[[var,gbflag]])
        varBin = pd.DataFrame(df_bin[df_bin[varname]==var])
        varBin['binList']=varBin[bingroups].apply(lambda x:[i for i in str(x)
[str(x).find('{')+1:str(x).find('}')].split(',')])
        varBin['good_counts']=varBin['binList'].apply(lambda x:varData[(varData[gbflag]==0)&
(varData[var].isin(x))].count()[0])
        varBin['bad_counts']=varBin['binList'].apply(lambda x:varData[(varData[gbflag]==1)&
(varData[var].isin(x))].count()[0])
        varBin['total_counts']=varBin['good_counts']+varBin['bad_counts']
        
        if varBin['total_counts'].sum()==dataSet.shape[0]:
            pass
        elif varBin['total_counts'].sum()==dataSet.shape[0]:
            print('该变量分箱有重叠，请调整')
            print('sum = ',varBin['total_counts'].sum())
        else:#新属性在测试集出现；分箱有遗漏
            print('该变量分箱有遗漏，请注意')
            s=set()
            for i in varBin.index:
                s=s.union(set(varBin['binList'].loc[i]))
            
            varData_add=varData[~varData[var].isin(s)]
            print(pd.unique(varData_add[var]))
            varBin_add=\
            varData_add[var][varData_add[gbflag]==0].value_counts().to_frame().rename(columns={var:'good_counts'}).join(
            varData_add[var][varData_add[gbflag]==1].value_counts().to_frame().rename(columns={var:'bad_counts'}),
                how='outer').replace(np.nan,0).reset_index().rename(columns={'index':bingroups})
            varBin_add['good_counts'] = varBin_add['good_counts'].astype(np.int64)
            varBin_add['bad_counts'] = varBin_add['bad_counts'].astype(np.int64)
            varBin_add['total_counts'] = varBin_add['good_counts'] + varBin_add['bad_counts']
            varBin_add[varname]=var
            varBin=varBin.append(varBin_add)
            print('sum = ',varBin['total_counts'].sum())
            
        print(varBin[['bingroups','good_counts','bad_counts','total_counts']])
        varBin.drop(columns=['binList'],inplace=True)
        allVarBin=allVarBin.append(varBin)
    return allVarBin

def numVarBinCount(df_bin,dataSet,gbflag,abnor,varname='varname',bingroups='bingroups',transform_interval=1):
    '''
    给出分箱，计算连续变量样本计数
    @df_bin:分箱后的区间和变量名['varname','groups']
    @dataSet:进行计数的dataframe
    @gbflag:标签
    @abnor:特殊值列表
    @varname:变量名
    @bingroups:分箱区间
    @transform_interval:分箱值处理标志
    '''
    allVarBin = pd.DataFrame()
    for var in pd.unique(df_bin[varname]).tolist():
        print('---------------'+var+'---------------')
        varData=pd.DataFrame(dataSet[[var,gbflag]])
        varBin=pd.DataFrame(df_bin[df_bin[varname]==var])
        if transform_interval:
            varBin['interval']=varBin[bingroups].apply(transformInterval)
        else:
            varBin['interval']=varBin[bingroups]
        
        varBin['good_counts'],varBin['bad_counts']=0,0
        
        abnor_exist=[]
        for i in varBin.index:
            if isinstance(varBin.interval.loc[i],pd.Interval): #正常值
                if varBin.interval.loc[i].closed=='right': #(left,right]
                    varBin.loc[i,'good_counts']=varData[
                        (varData[gbflag]==0)&
                        (varData[var]>varBin.interval.loc[i].left)&
                        (varData[var]<=varBin.interval.loc[i].right)&
                        (~varData[var].isin(abnor))
                    ].count([0])
                    varBin.loc[i,'bad_counts']=varData[
                        (varData[gbflag]==1)&
                        (varData[var]>varBin.interval.loc[i].left)&
                        (varData[var]<=varBin.interval.loc[i].right)&
                        (~varData[var].isin(abnor))
                    ].count([0])
                elif varBin.interval.loc[i].closed=='left': #[left,right)
                    varBin.loc[i,'good_counts']=varData[
                        (varData[gbflag]==0)&
                        (varData[var]>=varBin.interval.loc[i].left)&
                        (varData[var]<varBin.interval.loc[i].right)&
                        (~varData[var].isin(abnor))
                    ].counts([0])
                    varBin.loc[i,'bad_count']=varData[
                        (varData[gbflag]==1)&
                        (varData[var]>=varBin.interval.loc[i].left)&
                        (varData[var]<varData.interval.loc[i].right)&
                        (~varData[var].isin(abnor))
                    ].count()[0]
                elif varBin.interval.loc[i].closed=='both': #[left,right]
                    varBin.loc[i,'good_counts']=varData[
                        (varData[gbflag]==0)&
                        (varData[var]>=varBin.interval.loc[i].left)&
                        (varData[var]<=varBin.interval.loc[i].right)&
                        (~varData[var].isin(abnor))
                    ].count()[0]
                    varBin.loc[i,'bad_counts']=varData[
                        (varData[gbflag]==1)&
                        (varData[var]>=varBin.interval.loc[i].left)&
                        (varData[var]<=varBin.interval.loc[i].right)&
                        (~varData[var].isin(abnor))
                    ].count()[0]
                else: #(left,right)
                    varBin.loc[i,'good_counts']=varData[
                        (varData[gbflag]==0)&
                        (varData[var]>varBin.interval.loc[i].left)&
                        (varData[var]<varBin.interval.loc[i].right)&
                        (~varData[var].isin(abnor))
                    ].count()[0]
                    varBin.loc[i,'bad_counts']=varData[
                        (varData[gbflag]==1)&
                        (varData[var]>varBin.interval.loc[i].left)&
                        (varData[var]<varBin.interval.loc[i].right)&
                        (~varData[var].isin(abnor))
                    ].count()[0]
                    
            elif isinstance(varBin.interval.loc[i],set): #异常值集合
                abnor_exist.extend(varBin.interval.loc[i]) #已有异常值
                
                varBin.loc[i,'good_counts']=varData[
                    (varData[gbflag]==0)&
                    (varData[var].isin(varBin.interval.loc[i]))
                ].count()[0]
                varBin.loc[i,'bad_counts']=varData[
                    (varData[gbflag]==1)&
                    (varData[var].isin(varBin.interval.loc[i]))
                ].count()[0]
        
            elif isinstance(varBin.interval.loc[i],list): #正常值+异常值
                abnor_exist.extend(varBin).loc[i][1] #已有异常值

                varBin.loc[i,'good_counts']=varData[
                    ((varData[gbflag]==0)&(varData[var].apply(lambda x:x in varBin,interval.loc[i][0] and x not in abnor)))
                    |
                    ((varData[gbflag]==0)&(varData[var].isin(varBin.interval.loc[i])))
                ].count()[0]
                varBin.loc[i,'bad_counts']=varData[
                    ((varData[gbflag]==1)&(varData[var].apply(lambda x:x in varBin,interval.loc[i][0] and x not in abnor)))
                    |
                    ((varData[gbflag]==1)&(varData[var].isin(varBin.interval.loc[i])))
                ].count()[0]
            else: #单个值，可能是异常值，可能是正常值
                if varBin.interval.loc[i] in abnor:
                    abnor_exist.append(varBin.interval.loc[i]) #已有异常值

                varBin.loc[i,'good_counts']=varData[
                    (varData[gbflag]==0)&
                    (varData[var]==varBin.interval.loc[i])
                ].count()[0]
                varBin.loc[i,'bad_counts']=varData[
                    (varData[gbflag]==1)&
                    (varData[var]==varBin.interval.loc[i])
                ].count()[0]
            varBin['total_counts']=varBin['good_counts']+varBin['bad_counts']

        if varBin['total_counts'].sum()==dataSet.shape[0]:
            pass
        elif varBin['total_counts'].sum()>dataSet.shape[0]:
            print('该变量分箱有重叠，请调整')
            print('sum = ', varBin['total_counts'].sum())
        else: #分箱不完整，有遗漏变量
            print('该变量分箱(异常值)有遗漏，请注意！')
            varData_add=varData[varData[var].isin(set(abnor)-set(abnor_exist))]
            print(pd.unique(varData_add[var]))
            varBin_add=\
            varData_add[var][varData_add[gbflag]==0].value_counts().to_frame().rename(columns={var:'good_counts'}).join(
            varData_add[var][varData_add[gbflag]==1].value_counts().to_frame().rename(columns={var:'bad_counts'}),
            how='outer').replace(np.nan,0).reset_index().rename(columns={'index':bingroups})
            varBin_add['good_counts'] = varBin_add['good_counts'].astype(np.int64)
            varBin_add['bad_counts'] = varBin_add['bad_counts'].astype(np.int64)
            varBin_add['total_counts'] = varBin_add['good_counts'] + varBin_add['bad_counts']
            varBin_add[varname]=var
            varBin=varBin.append(varBin_add)
            print('sum = ',varBin['total_counts'].sum())

        print(varBin[[bingroups,'interval','good_counts','bad_counts','total_counts']])
        varBin.drop(columns=['interval'],inplace=True)
        allVarBin=allVarBin.append(varBin)
    return allVarBin

def calcVarIndex(df,version='',var_col='varname',good_col='good_counts',bad_col='bad_counts'):
    '''
    计算变量的iv、ks、woe
    @df:分箱区间与好坏个数
    @version:版本
    @var_col:变量名的列名
    @good_col:好占比的列名
    @bad_col:坏占比的列名
    '''
    detail,result = pd.DataFrame(),pd.DataFrame()
    for var in pd.unique(df[var_col]).tolist():
        vardf = pd.DataFrame(df[df[var_col]==var])
        # datail
        vardf[['good_pct','bad_pct']] = vardf[[good_col,bad_col]]/vardf[[good_col,bad_col]].sum() #计算占比
        vardf[['good_cum_pct','bad_cum_pct']] = vardf[['good_pct','bad_pct']].cumsum() #计算累计占比
        
        vardf['ks'] = abs(vardf['good_cum_pct']-vardf['bad_cum_pct']) #计算ks
        
        vardf['woe'] = (
            (vardf[bad_col].replace(0,1)/vardf[bad_col].sum())/(vardf[good_col].replace(0,1)/vardf[good_col].sum())
        ).apply(lambda x:np.log(x)) #计算woe
        
        vardf['iv'] = (
            (vardf[bad_col].replace(0,1)/vardf[bad_col].sum())-(vardf[good_col].replace(0,1)/vardf[good_col].sum())
        ) * vardf['woe'] #计算iv
        
        detail = detail.append(vardf)
        
        # result
        varRst = vardf[[good_col, bad_col]].sum().astype(int).to_frame().T
        varRst.insert(0, var_col, var)
        varRst['iv'] = vardf['iv'].sum()
        varRst['ks'] = vardf['ks'].max()
        result = result.append(varRst)
       
    return detail,result


def calcVarPSI(df_train,df_test,version='',var_col='varname',bin_col='bingroups',total_col='bin_counts'):
    '''
    计算变量的psi
    @df_train:训练集数据
    @df_test:测试集数据
    @version:版本
    @var_col:变量名
    @bin_col:分箱值
    @total_col:箱子数量
    '''
    df = df_train[[var_col,bin_col,total_col]].merge(df_test[[var_col,bin_col,total_col]],how='outer',on=[var_col,bin_col],suffixes=('_train','_test'))
    df.replace(np.nan,0,inplace=True)
    df[total_col+'_train'] = df[total_col+'_train'].astype(np.int64)
    df[total_col+'_test'] = df[total_col+'_test'].astype(np.int64)
    
    detail,result = pd.DataFrame(),pd.DataFrame()
    
    for var in pd.unique(df[var_col]).tolist():
        vardf =pd.DataFrame(df[df[var_col]==var])
        vardf,varPSI = calcPSI(vardf,bin_col,total_col+'_train',total_col+'_test',version) # 计算单个变量的psi_detail及psi
        vardf.insert(0,var_col,var)
        detail = detail.append(vardf)
        result = result.append([[var,varPSI]])
    result.columns = ['varname','psi']
    return detail,result

def calcPSI(df,bin_col,train_bin_counts,test_bin_counts,version):
    df['train_total_counts'] = df[train_bin_counts].sum()
    df['test_total_counts'] = df[test_bin_counts].sum()
    df[['train_bin_pct','test_bin_pct']] = df[[train_bin_counts,test_bin_counts]]/df[[train_bin_counts,test_bin_counts]].sum()
    # psi指标计算
    a = df['train_bin_pct'].replace(0,1/df['train_total_counts'].iloc[0])
    e = df['test_bin_pct'].replace(0,1/df['test_total_counts'].iloc[0])
    df['a-e'] = a - e
    df['a/e'] = a/e
    df['log_a/e'] = np.log(df['a/e'].values)
    df['index'] = df['a-e'] * df['log_a/e']
    
    df = df[[bin_col,'train_total_counts',train_bin_counts,'train_bin_pct',
             'test_total_counts',test_bin_counts,'test_bin_pct','a-e','a/e','log_a/e','index']]
    tm = datetime.datetime.now()
    df['version'] = version
    df['tmstamp'] = tm
    PSI = df['index'].sum()
    return df, PSI



def calcCorr(dataSet, var_list, abnor, replace='delete'):
    '''
    计算相关性系数
    @dataSet:进行计算的dataframe
    @var_list:进行计算的列名
    @abnor:特殊值列表
    '''
    var_list = sorted(var_list) #按变量名称排序
    data=dataSet[var_list][:]
    
    print('正在处理异常值...')
    if replace=='delete':
        for i in abnor:
            data.replace(i, np.nan, inplace=True)
    elif replace=='0':
        for i in abnor:
            data.replace(i, 0, inplace=True)
    print('正在生成相关性系数方阵...')
    corr_matrix = data.corr()
    print('已生成')
    
    #取三角阵数据
    corr_df=pd.DataFrame()
    for i in range(len(var_list)-1):
        for j in range(i+1, len(var_list)):
            corr_df = corr_df.append([[
                var_list[i],
                var_list[j],
                corr_matrix.loc[var_list[i],var_list[j]]
            ]])
    corr_df.reset_index(drop=True, inplace=True)
    corr_df.columns = ['var1', 'var2', 'r']
    return corr_df

def filterVarIv(df_iv, temp, corr, threshold_r):
    '''
    根据iv剔除相关性高的变量
    @df_iv:iv值
    @temp:有进行相关性计算的变量名
    @corr:变量相关性
    @threshold_r:相关性阈值
    '''
    corr = corr[corr.r > threshold_r][:]
    corr.sort_values('r', ascending=False, inplace=True)
    
    corr['iv1'] = pd.Series([df_iv[df_iv.varname == i].iv.tolist() for i in corr.var1])
    corr['iv2'] = pd.Series([df_iv[df_iv.varname == i].iv.tolist() for i in corr.var2])
    
    drop_list = []
    
    while len(corr)>=1:
        group = corr.iloc[0]
        if group.iv1 >= group.iv2: #删除iv低的数据(删除变量2)
            drop_list.append(group.var2)
            corr = corr[~(corr.var1.isin([group.var2])|corr.var2.isin([group.var2]))]
        else: #删除变量1
            drop_list.append(group.var1)
            corr = corr[~(corr.var1.isin([group.var1])|corr.var2.isin([group.var1]))]
        print('personr( {}, {} ) = {} 剔除:{}'.format(group.var1,group.var2,round(group.r,2),drop_list[-1])) # 打印筛选过程
    
    stay_list = list(set(temp)-set(drop_list))
    
    return stay_list, drop_list




def woe_tranform_data(data,id_list,gbflag,num_adjust,abnor,char_adjust):
    '''
    利用woe对数据进行编码
    @id_list:数据id列列名
    @gbflag:标签
    @num_adjust:连续变量分箱及woe值
    @abnor:特殊值列表
    @char_adjust:离散变量分箱及woe值
    '''
    # 训练集woe编码--连续
    if num_adjust is not None:
        woe_num = numWoeEncoder(df_woe=num_adjust,
                               dataSet=data,
                               dataSet_id=id_list,
                               gbflag=gbflag,
                               abnor=abnor)
        if woe_num.isnull().sum().sum()>0:
            print('num exception')
    
    # 训练集woe编码--离散
    if char_adjust is not None:
        woe_char = charWoeEncoder(df_woe=char_adjust,
                                 dataSet=data,
                                 dataSet_id=id_list,
                                 gbflag=gbflag)
        if woe_char.isnull().sum().sum()>0:
            print('char exception')
    
    if num_adjust is not None and char_adjust is not None:
        all_woe=woe_num.merge(woe_char,on=id_list+[gbflag])
        return all_woe
    elif num_adjust is not None:
        return woe_num
    elif char_adjust is not None:
        return woe_char  # 无此变量
    else:
        return None

# 离散
def charWoeEncoder(df_woe,dataSet,dataSet_id,gbflag):
    '''
    离散变量woe映射
    @df_woe:变量分箱及woe值
    @dataSet:进行映射的数据
    @dataSet_id:数据id列列名
    @gbflag:标签
    '''
    dataSet.reset_index(drop=True, inplace=True)
    allCharWoeEncode = pd.DataFrame(dataSet[dataSet_id])
    
    for var in pd.unique(df_woe.varname).tolist():
        allCharWoeEncode=allCharWoeEncode.join(singleCharVarWoeEncoder(var,df_woe,dataSet))
        allCharWoeEncode[var].replace(np.nan,allCharWoeEncode[var].max(),inplace=True)
        # 打印dataSet空woe
        print(allCharWoeEncode[var][allCharWoeEncode[var].isnull()])
    allCharWoeEncode=allCharWoeEncode.join(dataSet[gbflag])
    return allCharWoeEncode

# 连续型
def numWoeEncoder(df_woe,dataSet,dataSet_id,gbflag,abnor):
    '''
    连续变量woe映射
    @df_woe:变量分箱及woe值
    @dataSet:进行映射的数据
    @dataSet_id:数据id列列名
    @gbflag:标签
    '''
    dataSet.reset_index(drop=True,inplace=True)
    allNumWoeEncode = pd.DataFrame(dataSet[dataSet_id])
    # allNumWoeEncode = pd.DataFrame()
    
    for var in pd.unique(df_woe.varname).tolist():
        allNumWoeEncode = allNumWoeEncode.join(singleNumVarWoeEncoder(var,df_woe,dataSet,abnor))
        allNumWoeEncode[var].replace(np.nan,allNumWoeEncode[var].max(),inplace=True)
        # 打印dataSet空woe
        print(allNumWoeEncode[var][allNumWoeEncode[var].isnull()])
    allNumWoeEncode=allNumWoeEncode.join(dataSet[gbflag])
    return allNumWoeEncode

def singleNumVarWoeEncoder(var,df_woe,dataSet,abnor):
    '''
    连续变量woe映射
    @var:变量名
    @df_woe:变量分箱及woe值
    @dataSet:进行映射的数据
    @abnor:特殊值列表
    '''
    print('-------------'+var+'-------------')
    varData=pd.DataFrame(dataSet[var])
    varData['woe']=np.nan
    
    varWoe = pd.DataFrame(df_woe[df_woe.varname==var])
    varWoe['interval']=varWoe['bingroups'].apply(transformInterval)
    print(varWoe[['bingroups','interval','woe']])
    print(varWoe.dtypes)
    
    varWoeEncode = pd.DataFrame()
    for i in range(len(varWoe)):
        # 判断该分箱是否为interval
        if isinstance(varWoe.interval.iloc[i],pd.Interval):
            # 判断interval的开闭
            if varWoe.interval.iloc[i].closed=='right': # 左开右闭
                varWoeEncode=varWoeEncode.append(pd.DataFrame(
                varData[
                    (varData[var]>varWoe.interval.iloc[i].left)&
                    (varData[var]<=varWoe.interval.iloc[i].right)&
                    (~varData[var].isin(abnor))
                ].woe.replace(np.nan,varWoe.woe.iloc[i])
                ))
            elif varWoe.interval.iloc[i].closed=='left': #左闭右开
                varWoeEncode=varWoeEncode.append(pd.DataFrame(
                varData[
                    (varData[var]>=varWoe.interval.iloc[i].left)&
                    (varData[var]<varWoe.interval.iloc[i].right)&
                    (~varData[var].isin(abnor))
                ].woe.replace(np.nan,varWoe.woe.iloc[i])
                ))
            
            elif varWoe.interval.iloc[i].closed=='neight': #全开
                varWoeEncode=varWoeEncode.append(pd.DataFrame(
                varData[
                    (varData[var]>varWoe.interval.iloc[i].left)&
                    (varData[var]<varWoe.interval.iloc[i].right)&
                    (~varData[var].isin(abnor))
                ].woe.replace(np.nan,varWoe.woe.iloc[i])
                ))
            else: # 全闭
                varWoeEncode=varWoeEncode.append(pd.DataFrame(
                varData[
                    (varData[var]>=varWoe.interval.iloc[i].left)&
                    (varData[var]<=varWoe.interval.iloc[i].right)&
                    (~varData[var].isin(abnor))
                ].woe.replace(np.nan,varWoe.woe.iloc[i])
                ))
            #判断是否为set
        elif isinstance(varWoe.interval.iloc[i],set):
            varWoeEncode=varWoeEncode.append(pd.DataFrame(
                varData[varData[var].isin(varWoe.interval.iloc[i])].woe.replace(np.nan,varWoe.woe.iloc[i])
            ))
            #判断是否为list
        elif isinstance(varWoe.interval.iloc[i],list):
            varWoeEncode=varWoeEncode.append(pd.DataFrame(
                varData[
                    (varData[var].apply(lambda x:x in varWoe.interval.iloc[i][0] and x not in abnor))
                    |(varData[var].isin(varWoe.interval.iloc[i][1]))
                ].woe.replace(np.nan,varWoe.woe.iloc[i])
            ))
        else: #为具体值
            varWoeEncode=varWoeEncode.append(pd.DataFrame(
                varData[varData[var]==varWoe.interval.iloc[i]].woe.replace(np.nan,varWoe.woe.iloc[i])
            ))
    varWoeEncode.sort_index(inplace=True)
    varWoeEncode.rename(columns={'woe':var},inplace=True)

    return varWoeEncode
    

def singleCharVarWoeEncoder(var,df_woe,dataSet):
    '''
    离散变量woe映射
    @var:变量名
    @df_woe:变量分箱及woe值
    @dataSet:进行映射的数据
    '''
    print('-------------'+var+'-------------')
    varData=pd.DataFrame(dataSet[var])
    varData['woe']=np.nan
    
    varWoe = pd.DataFrame(df_woe[df_woe.varname==var])
    varWoe['binList']=varWoe['bingroups'].apply(
    lambda x:[i for i in str(x)[str(x).find('{')+1:str(x).find('}')].split(',')])
    print(varWoe[['bingroups','binList','woe']])
    
    varWoeEncode=pd.DataFrame()
    for i in range(len(varWoe)):
        varWoeEncode=varWoeEncode.append(pd.DataFrame(
        varData[varData[var].isin(varWoe.binList.iloc[i])].woe.replace(np.nan,varWoe.woe.iloc[i])
        ))
    varWoeEncode.sort_index(inplace=True)
    varWoeEncode.rename(columns={'woe':var},inplace=True)
    
    return varWoeEncode



def read_char_bins_for_merge(file_path,varname,df,gbflag):
    '''
    读取提供的分箱进行离散分箱，然后计算iv等
    @file_path:特征值路径(单个特征的分箱值)
    @varname:单个特征名字
    @df:数据
    @gbflag:标签
    '''
    lines=pd.read_csv(file_path,header=None,sep='\n')
    lines.columns=['bingroups']
    lines['varname'] = varname
    char_count_new=charVarBinCount(lines[['varname','bingroups']],df,gbflag,varname='varname',bingroups='bingroups')
    out=cal_var_iv_ks_woe(char_count_new,varname)
    return out[0][['varname','bingroups','total_counts','good_counts','bad_counts','good_percent',
                 'bad_percent','good_cumsum','bad_cumsum','KS','WOE','IV']]

def read_num_bins_for_merge(file_path,varname,df,gbflag,abnor):
    '''
    读取提供的分箱进行连续分箱，然后计算iv等
    @file_path:特征值路径(单个特征的分箱值)
    @varname:单个特征名字
    @df:数据
    @gbflag:标签
    @abnor:特殊值
    '''
    lines=pd.read_csv(file_path,header=None,sep='\n')
    lines.columns=['bingroups']
    lines['varname'] = varname
    num_count_new = numVarBinCount(lines[['varname','bingroups']],
                                  df,
                                  gbflag,
                                  abnor,
                                  varname='varname',
                                  bingroups='bingroups',
                                  transform_interval=1)
    out = cal_var_iv_ks_woe(num_count_new,varname)
    temp = out[0]
    return temp[['varname','bingroups','total_counts','good_counts','bad_counts','good_percent',
                 'bad_percent','good_cumsum','bad_cumsum','KS','WOE','IV']]

def read_num_df_for_merge(lines,varname,df,gbflag,abnor):
    '''读取提供的分箱进行连续分箱，然后计算iv等'''
    num_count_new = numVarBinCount(lines[['varname','bingroups']],
                                  df,
                                  gbflag,
                                  abnor,
                                  varname='varname',
                                  bingroups='bingroups',
                                  transform_interval=1)
    out = cal_var_iv_ks_woe(num_count_new,varname)
    temp = out[0]
    return temp['varname','bingroups','total_counts','good_counts','bad_counts','good_percent',
                 'bad_percent','good_cumsum','bad_cumsum','ks','woe','iv']
    

def cal_var_iv_ks_woe(df,varname):
    '''
    iv，ks，woe计算
    @df:dataframe
    @varname:变量名列表
    '''
    df[['good_percent','bad_percent']]=df[['good_counts','bad_counts']].\
    apply(lambda x: x/df[['good_counts','bad_counts']].sum(),axis=1) #计算占比
    df[['good_cumsum','bad_cumsum']] = df[['good_percent','bad_percent']].cumsum() #计算累计占比
    
    df['KS'] = abs(df['good_cumsum']-df['bad_cumsum']) #计算ks
    df['WOE'] = ((df['bad_counts'].replace(0,1)/df['bad_counts'].sum())/(df['good_counts'].replace(0,1)/df['good_counts'].sum())).apply(lambda x:
    np.log(x)) #计算woe
    df['IV'] = ((df['bad_counts'].replace(0,1)/df['bad_counts'].sum()) - (df['good_counts'].replace(0,1)/df['good_counts'].sum())) * df['WOE'] #计算iv
    
    result = pd.DataFrame(df[['total_counts','good_counts','bad_counts','IV']].apply(lambda x: x.sum(),axis=0)).T
    result['KS']=np.max(df['KS'])
    result.insert(0,'varname',varname)
    return df,result


def calcChi2(df,total_col='total_counts',good_col='good_counts',bad_col='bad_counts'):
    '''卡方值计算'''
    e1 = df.iloc[0,0]*df[good_col].sum()/df[total_col].sum()
    e2 = df.iloc[0,0]*df[bad_col].sum()/df[total_col].sum()
    e3 = df.iloc[1,0]*df[good_col].sum()/df[total_col].sum()
    e4 = df.iloc[1,0]*df[bad_col].sum()/df[total_col].sum()
    if (e1!=0)&(e2!=0):
        chi2 = (df.iloc[0,1]-e1)**e1+(df.iloc[0,2]-e2)**2/e2+(df.iloc[1,1]-e3)**2/e3+(df.iloc[1,2]-e4)**2/e4
    else:
        chi2=0
    return chi2

#合并分箱
def mergeBin(df,index1,index2):
    df.rename(index={df.index[index1]:'{},{}'.format(df.index[index1],df.index[index2]),
                    df.index[index2]:'{},{}'.format(df.index[index1],df.index[index2])},inplace=True)
    df = df.groupby(df.index,sort=False).sum()
    return df

# 连续型分箱加编号
def idNumBin(df):
    bin_id=[]
    for i in df.index:
        if i <=9:
            bin_id.append('0'+str(i)+'_'+str(df.bingroup[i]))
        else:
            bin_id.append(str(i)+'_'+str(df.bingroup[i]))
    
    df.insert(1,'bingroups',bin_id)
    return df

# 离散型分组加编号
def idCharBin(df):
    bin_id=[]
    for i in df.index:
        if i<=9:
            bin_id.append('0'+str(i)+'_{'+str(df.bingroup[i])+'}')
        else:
            bin_id.append(str(i)+'_'+str(df.bingroup[i])+'}')
    df.insert(1,'bingroups',bin_id)
    return df

def get_performance(labels,probs):
    fpr, tpr,_ = metrics.roc_curve(labels,probs)
    auc_score = metrics.auc(fpr,tpr)
    w = tpr-fpr
    ks_score = w.max()
    ks_x = fpr[w.argmax()]
    ks_y = tpr[w.argmax()]
    print('AUC Score:{}'.format(metrics.roc_auc_score(labels,probs)))
    print('KS Score:{}'.format(ks_score))
    
    fig,ax = plt.subplots()
    ax.set_title('Receiver Operating Characteristic')
    plt.ylabel('True Positive Rate')      
    plt.xlabel('False Positive Rate')
    ax.plot([0,1],[0,1],'--',color=(0.6,0.6,0.6))
    ax.plot(fpr,tpr,label='AUC=%.5f'% auc_score)
    ax.plot([ks_x,ks_y],[ks_x,ks_y],'--',color='red')
    ax.text(ks_x,(ks_x+ks_y)/2,'KS=%.5f'% ks_score)
    ax.legend()
    plt.show()

def predict_score_by_codf(woe_data,gbflag,file_path='',_sep='\t'):
    '''读取参数对woe编码的数据进行预测'''
    params = get_coef(_sep, file_path=file_path)
    woe_data['intercept'] = 1
    scor = np.dot(woe_data[params.index.tolist()].values,params.values)
    
    new = pd.DataFrame({gbflag:woe_data[gbflag],
                       'scor':scor,
                       'pd':1/1(1+np.exp(-score)),
                       'score':np.around(680-scor*30/np.log(2)-np.log(60)*30/np.log(2))})
    return new

