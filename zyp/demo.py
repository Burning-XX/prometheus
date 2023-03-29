# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
import json
from django.http import JsonResponse, HttpRequest
import pandas as pd
import numpy as npy
import scipy.stats as st
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from sklearn import preprocessing
import re
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
            if 'entity_effect' in ols_res_dict['OLSList'][i]['Result']:  # 把剩下的项目加入这个文件
                first_column_append(first_column, i, '时间固定效应')
                column.append(ols_res_dict['OLSList'][i]['Result']['entity_effect'])
                first_column_append(first_column, i, '个体固定效应')
                column.append(ols_res_dict['OLSList'][i]['Result']['time_effect'])
            first_column_append(first_column, i, "观测值")
            column.append(str(ols_res_dict['OLSList'][i]['Result']['n']))
            first_column_append(first_column, i, "R^2")
            column.append(str(round(ols_res_dict['OLSList'][i]['Result']['r2'], 3)))
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
    uid='heyang'
    f_name=os.path.join('/Users/heyang/Desktop/',uid+".xlsx")
    df.to_excel(f_name,index=ind,sheet_name="CE-API",index_label=index_label) #分别控制表名、索引导出
    return uid

def put_file_excelV2(df,ind=False,index_label=None)->str:
    """
    封装好的数据表转文件函数
    """
    uid='heyangV2'
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

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dict1 = {'argu_i': '城镇居民人均可支配收入/元', 'Result': {'pvalue': {'const': 3.290194342e-59, '农村居民人均可支配收入/元': 7.319795936e-314}, 'coeff': {'const': 4736.2735035495, '农村居民人均可支配收入/元': 2.0747488185}, 'std_err': {'const': 253.8145946131, '农村居民人均可支配收入/元': 0.0224565049}, 'n': 496, 'r2': 0.9452925513571547}}
    dict2 = {'argu_i': '城镇居民人均可支配收入/元', 'Result': {'pvalue': {'const': 3.364180973e-87, '全体居民人均可支配收入/元': 1.6228e-319}, 'coeff': {'const': 5807.5573941295, '全体居民人均可支配收入/元': 1.0552815398}, 'std_err': {'const': 237.3523718599, '全体居民人均可支配收入/元': 0.0111085214}, 'n': 496, 'r2': 0.9481011823723947}}
    dict3 = {'argu_i': '城镇居民人均可支配收入/元', 'Result': {'pvalue': {'const': 6.9241834e-125, '年份': 1.060582995e-125}, 'coeff': {'const': -4459888.037476313, '年份': 2228.4352941177}, 'std_err': {'const': 137116.0471980422, '年份': 68.1320186312}, 'n': 496, 'r2': 0.6841003134912228}}
    ret_s = {'count': 3,
             'OLSList': [dict1, dict2, dict3],
             'ArgeList': ['农村居民人均可支配收入/元', '年份', '全体居民人均可支配收入/元']}
    ret_df = smols2excelV2(ret_s)
    put_file_excelV2(ret_df, False)
    print(ret_df)

    # list1 = ['被解释变量', '常数项', "农村居民人均可支配收入/元", 2000]
    # list2 = [1, 2, 3, 4, 5]
    # list3 = ["a", "b", "c", "d"]
    # list = [list1, list2, list3]
    # ddff = pd.DataFrame(list).T
    # print(pd.DataFrame(list).T)
    # f_name=os.path.join('/Users/heyang/Desktop/',"fuck.xlsx")
    # ddff.to_excel(f_name,index=False,sheet_name="CE-API",index_label=None)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
