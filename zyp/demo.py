# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
import json
from plotnine import *
from django.http import JsonResponse, HttpRequest
import pandas as pd
import numpy as npy
import scipy.stats as st
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from sklearn import preprocessing
import re
from scipy.stats.mstats import winsorize
import traceback

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

def g_p_str(pval) -> str:
    """
    内部函数:显著性判断打星号
    """
    if pval < 0.01:
        return "***"
    elif pval >= 0.01 and pval < 0.05:
        return "**"
    elif pval >= 0.05 and pval < 0.1:
        return "*"
    else:
        return ""

def smols2excel(ols_res_dict: dict) -> pd.DataFrame:
    """
    内部函数:把回归结果格式化处理到Excel
    """
    ret_dict = {}
    ArgList = ols_res_dict['ArgeList']
    ArgList.append("const")
    for i in range(0, ols_res_dict['count']):  # 按个数循环
        jiba = []
        jiba.append('(' + str(i + 1) + ')')
        jiba.append(ols_res_dict['OLSList'][i]['argu_i'])

        ret_dict['(' + str(i + 1) + ')'] = {"被解释变量": ols_res_dict['OLSList'][i]['argu_i']}  # 建立索引
        try:
            for key in ArgList:
                if key in ols_res_dict['OLSList'][i]['Result']['coeff']:
                    p_str = g_p_str(ols_res_dict['OLSList'][i]['Result']['pvalue'][key])
                    #temp = '(' + str(round(ols_res_dict['OLSList'][i]['Result']['std_err'][key], 3)) + ')'
                    ret_dict['(' + str(i + 1) + ')'][key] = str(
                        round(ols_res_dict['OLSList'][i]['Result']['coeff'][key], 3)) + p_str
                else:
                    ret_dict['(' + str(i + 1) + ')'][key] = ""

            for key in ArgList:
                if key in ols_res_dict['OLSList'][i]['Result']['coeff']:
                    temp = '(' + str(round(ols_res_dict['OLSList'][i]['Result']['std_err'][key], 3)) + ')'
                    ret_dict['(' + str(i + 1) + ')'][key + "_hello"] = str(temp)
                else:
                    ret_dict['(' + str(i + 1) + ')'][key + "_hello"] = ""

            if 'entity_effect' in ols_res_dict['OLSList'][i]['Result']:  # 把剩下的项目加入这个文件
                ret_dict['(' + str(i + 1) + ')']['时间固定效应'] = ols_res_dict['OLSList'][i]['Result']['entity_effect']
                ret_dict['(' + str(i + 1) + ')']['个体固定效应'] = ols_res_dict['OLSList'][i]['Result']['time_effect']
            ret_dict['(' + str(i + 1) + ')']['观测值'] = str(ols_res_dict['OLSList'][i]['Result']['n'])
            ret_dict['(' + str(i + 1) + ')']['R^2'] = str(round(ols_res_dict['OLSList'][i]['Result']['r2'], 3))
        except Exception as e:
            traceback.print_exc()
    ret_df = pd.DataFrame(convert(ret_dict))
    return ret_df

def smols2excelV2(ols_res_dict: dict) -> pd.DataFrame:
    """
    内部函数:把回归结果格式化处理到Excel
    """
    ret_list = []
    first_column = []

    ArgList = ols_res_dict['ArgeList']
    ArgList.insert(0, "const")

    for i in range(0, ols_res_dict['count']):  # 按个数循环
        column = []
        first_column_append(first_column, i, "")
        column.append('(' + str(i + 1) + ')')
        first_column_append(first_column, i, "被解释变量")
        column.append(ols_res_dict['OLSList'][i]['argu_i'])

        try:
            for key in ArgList:
                if key in ols_res_dict['OLSList'][i]['Result']['coeff']:
                    p_str = g_p_str(ols_res_dict['OLSList'][i]['Result']['pvalue'][key])
                    first_column_append(first_column, i, key)
                    column.append(str(round(ols_res_dict['OLSList'][i]['Result']['coeff'][key], 3)) + p_str)
                    temp = '(' + str(round(ols_res_dict['OLSList'][i]['Result']['std_err'][key], 3)) + ')'
                    first_column_append(first_column, i, "")
                    column.append(temp)
                else:
                    first_column_append(first_column, i, key)
                    column.append("")
                    first_column_append(first_column, i, "")
                    column.append("")
            first_column_append(first_column, i, "观测值")
            column.append(str(ols_res_dict['OLSList'][i]['Result']['n']))
            if i == 0:
                ret_list.append(first_column)
            ret_list.append(column)
        except Exception as e:
            traceback.print_exc()
    return pd.DataFrame(ret_list).T

def first_column_append(ret_fist, i, name):
    if i == 0:
        ret_fist.append(name)

def put_file_excel(df,ind=False,index_label=None)->str:
    """
    封装好的数据表转文件函数
    """
    uid='heyangV4'
    f_name=os.path.join('/Users/heyang/Desktop/',uid+".xlsx")
    df.to_excel(f_name,index=ind,sheet_name="CE-API",index_label=index_label) #分别控制表名、索引导出
    return uid

def put_file_excelV2(df,ind=False,index_label=None)->str:
    """
    封装好的数据表转文件函数
    """
    uid='heyangV3'
    f_name=os.path.join('/Users/heyang/Desktop/',uid+".xlsx")
    df.to_excel(f_name,index=ind,sheet_name="CE-API",index_label=index_label) #分别控制表名、索引导出
    return uid


def convert(ret_dict) -> dict:
    result = {}
    list = ['被解释变量', 'const', '观测值', 'R^2']
    for key in ret_dict:
        map = ret_dict[key]
        v = {}
        v['被解释变量'] = map["被解释变量"]
        v['const'] = map['const']
        for k in map:
            if k not in list:
                v[k] = map[k]
        v['观测值'] = map['观测值']
        v['R2'] = map['R^2']
        result[key] = v
    return result

def logit(dta, argu1, argu2):
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(dta[argu2])
    x = sm.add_constant(dta[argu1])
    model = sm.Logit(y, x)
    results = model.fit()

    pvalue = results.pvalues
    coeff = results.params
    std_err = results.bse
    #r2 = results.rsquared
    res_df = pd.DataFrame({  # 从回归结果中提取需要的结果
        "pvalue": pvalue,
        "coeff": coeff,
        "std_err": std_err,
    })
    res_js = json.loads(res_df.to_json())
    res_js['n'] = dta.shape[0]
    #res_js['r2'] = r2
    # print({"argu_i": argu2, "Result": res_js})
    return {"argu_i": argu2, "Result": res_js}

def logit_repeat():
    dta = pd.read_excel('/Users/heyang/Desktop/居民人均可支配收入.xlsx')
    count = 3
    logit_argus = [
        {'argu_i': '测试变量', 'argu_e': ['年份']},
        {'argu_i': '测试变量', 'argu_e': ['农村居民人均可支配收入/元']},
        {'argu_i': '测试变量', 'argu_e': ['城镇居民人均可支配收入/元']},
    ]
    logit_result = []
    argu_il = set(logit_argus[0]['argu_i'])
    argu_el = set(logit_argus[0]['argu_e'])
    for i in range(0, count):
        logit_result.append(logit(dta, logit_argus[i]['argu_e'],  # 解释变量
                                  logit_argus[i]['argu_i'],  # 被解释变量
                                  ))
        argu_il = argu_il.union(logit_argus[i]['argu_i'])
        argu_el = argu_el.union(logit_argus[i]['argu_e'])
    ret_s = {"count": len(logit_result),  # 计数
             "OLSList": logit_result,  # 回归结果
             "ArgeList": list(argu_el)}  # 参数的并集
    # print(ret_s)
    ret_df = smols2excelV2(ret_s)
    put_file_excelV2(ret_df, False)
    print('----')

def probit(dta, argu1, argu2):
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(dta[argu2])
    x = sm.add_constant(dta[argu1])
    model = sm.Probit(y, x)
    results = model.fit()

    pvalue = results.pvalues
    coeff = results.params
    std_err = results.bse
    # r2 = results.rsquared
    res_df = pd.DataFrame({  # 从回归结果中提取需要的结果
        "pvalue": pvalue,
        "coeff": coeff,
        "std_err": std_err,
    })
    res_js = json.loads(res_df.to_json())
    res_js['n'] = dta.shape[0]
    return {"argu_i": argu2, "Result": res_js}

def probit_repeat():
    dta = pd.read_excel('/Users/heyang/Desktop/居民人均可支配收入.xlsx')
    count = 3
    logit_argus = [
        {'argu_i': '测试变量', 'argu_e': ['年份']},
        {'argu_i': '测试变量', 'argu_e': ['农村居民人均可支配收入/元']},
        {'argu_i': '测试变量', 'argu_e': ['城镇居民人均可支配收入/元']},
    ]
    logit_result = []
    argu_il = set(logit_argus[0]['argu_i'])
    argu_el = set(logit_argus[0]['argu_e'])
    for i in range(0, count):
        logit_result.append(probit(dta, logit_argus[i]['argu_e'],  # 解释变量
                                   logit_argus[i]['argu_i'],  # 被解释变量
                                   ))
        argu_il = argu_il.union(logit_argus[i]['argu_i'])
        argu_el = argu_el.union(logit_argus[i]['argu_e'])
    ret_s = {"count": len(logit_result),  # 计数
             "OLSList": logit_result,  # 回归结果
             "ArgeList": list(argu_el)}  # 参数的并集
    ret_df = smols2excelV2(ret_s)
    put_file_excelV2(ret_df, False)
    print('----')

def smols2excelV3(ols_res_dict: dict) -> pd.DataFrame:
    ret_list = []
    first_column = []

    ArgList = ols_res_dict['ArgeList']
    ArgList.append('const')

    for i in range(0, ols_res_dict['count']):  # 按个数循环
        column = []
        first_column_append(first_column, i, "")
        column.append('(' + str(i + 1) + ')')
        first_column_append(first_column, i, "被解释变量")
        column.append(ols_res_dict['OLSList'][i]['argu_i'])

        try:
            for key in ArgList:
                if key in ols_res_dict['OLSList'][i]['Result']['coeff']:
                    p_str = g_p_str(ols_res_dict['OLSList'][i]['Result']['pvalue'][key])
                    first_column_append(first_column, i, key)
                    column.append(str(round(ols_res_dict['OLSList'][i]['Result']['coeff'][key], 3)) + p_str)
                    temp = '(' + str(round(ols_res_dict['OLSList'][i]['Result']['std_err'][key], 3)) + ')'
                    first_column_append(first_column, i, "")
                    column.append(temp)
                else:
                    first_column_append(first_column, i, key)
                    column.append("")
                    first_column_append(first_column, i, "")
                    column.append("")
            if 'entity_effect' in ols_res_dict['OLSList'][i]['Result']:  # 把剩下的项目加入这个文件
                first_column_append(first_column, i, '时间固定效应')
                column.append(ols_res_dict['OLSList'][i]['Result']['entity_effect'])
                first_column_append(first_column, i, '个体固定效应')
                column.append(ols_res_dict['OLSList'][i]['Result']['time_effect'])
            first_column_append(first_column, i, "观测值")
            column.append(str(ols_res_dict['OLSList'][i]['Result']['n']))
            if i == 0:
                ret_list.append(first_column)
            ret_list.append(column)
        except Exception as e:
            traceback.print_exc()
    return pd.DataFrame(ret_list).T

def effect_probit(dta, argu1, argu2, entity_effects, time_effects):
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(dta[argu2])
    x = sm.add_constant(dta[argu1])
    mod = PanelOLS(dta[argu2], x, entity_effects=entity_effects, time_effects=time_effects)
    results = mod.fit()

    pvalue = results.pvalues
    coeff = results.params
    std_err = results.std_errors
    res_df = pd.DataFrame({  # 从回归结果中提取需要的结果
        "pvalue": pvalue,
        "coeff": coeff,
        "std_err": std_err,
    })
    res_js = json.loads(res_df.to_json())
    res_js['time_effect'] = time_effects
    res_js['entity_effect'] = entity_effects
    res_js['n'] = dta.shape[0]
    return {"argu_i": argu2, "Result": res_js}
def probit_effect_repeat():
    argu1 = {
        'argue': ['地区', '年份'],
        'count': 3,
        'argus': [
            {
                'argu_e': ['农村居民人均可支配收入/元'],
                'argu_i':'测试变量',
                'entity_effect': True,
                'time_effect': True
            },
            {
                'argu_e': ['城镇居民人均可支配收入/元'],
                'argu_i': '测试变量',
                'entity_effect': True,
                'time_effect': True
            },
            {
                'argu_e': ['全体居民人均可支配收入/元'],
                'argu_i': '测试变量',
                'entity_effect': True,
                'time_effect': True
            }
        ]
    }
    dta = pd.read_excel('/Users/heyang/Desktop/居民人均可支配收入.xlsx')
    count = argu1['count']
    probit_args = argu1['argus']
    dta = dta.set_index(argu1['argue'])
    probit_result = []
    argu_il = set(probit_args[0]['argu_i'])
    argu_el = set(probit_args[0]['argu_e'])
    for i in range(0, count):
        probit_result.append(effect_probit(dta,
                                           probit_args[i]['argu_e'],
                                           probit_args[i]['argu_i'],
                                           probit_args[i]['entity_effect'],  # 个体固定效应(Bool)
                                           probit_args[i]['time_effect']  # 时间固定效应
                                           ))
        argu_il = argu_il.union(probit_args[i]['argu_i'])
        argu_el = argu_el.union(probit_args[i]['argu_e'])
    ret_s = {"count": len(probit_result),  # 计数
             "OLSList": probit_result,  # 被解释变量
             "ArgeList": list(argu_el)}  # 参数的并集
    print(ret_s)
    ret_df = smols2excelV3(ret_s)
    ret_uid = put_file_excel(ret_df, False)
    # ret_s['File'] = {"uid": ret_uid, "f_suffix": ".xlsx"}

def winsor(dataList,down=1,up=1): #默认上下均为1%缩尾，dataList是一列，即dataframce中的一列
    """
    将传入的列表数据进行缩尾处理（替换观测值而不是删除）
    """
    winsored = winsorize(dataList,limits=[down/100,up/100])
    return winsored.data


def boxplot(variableName,dataframe):
    """
    返回单变量的箱线图，variableName指的是需要绘制箱线图的变量，dataframe是包含了需要绘制箱线图变量的DataFrame
    将会新生成一个数据表，仅用于绘制单变量的箱线图
    """
    newDf = pd.DataFrame({variableName:[value for value in dataframe[variableName].to_list()],"xlabel":[variableName for value in dataframe[variableName].to_list()]})

    return (ggplot(newDf, aes(y=variableName,x="xlabel")) + geom_boxplot() + ggtitle('测试') + theme(text=element_text(family='SimHei')))


def multiBoxplot(labels, df, title):

    newDict = {"value": [], "labels": []}
    for column in labels:
        newDict["value"] += df[column].to_list()
        newDict["labels"] += [column for line in df[column].to_list()]
    newDf = pd.DataFrame(newDict)

    p = ggplot(newDf, aes(x="labels", y="value")) + geom_boxplot() + ggtitle(title) + theme(text=element_text(family='SimHei'))
    return p

def save(img):
	image_path='/Users/heyang/Desktop'
	f_name='demo.svg'
	width=18
	height=23
	img.save(os.path.join(image_path,f_name),format="svg",width=width,height=height)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dta = pd.read_excel('/Users/heyang/Desktop/居民人均可支配收入.xlsx')
    argu2 = '年份'
    argu1 = '农村居民人均可支配收入/元'
    Label_list = [argu1, argu2]
    title = '我的测试'
    width = 12
    height = 8
    img_density=(multiBoxplot(Label_list,dta, title))
    print(img_density)
    print('-----')
