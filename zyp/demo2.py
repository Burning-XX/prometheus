
import pandas as pd
from sklearn import preprocessing
import statsmodels.api as sm
import json
import traceback
import copy
import sys
import re

def effect_probit(dta, argu1, argu2, entity_effects, time_effects, entity, time):
    label_encoder = preprocessing.LabelEncoder()

    argu1_copy = copy.deepcopy(argu1)
    if entity_effects != False:
        entityDummy = pd.get_dummies(dta[entity], prefix=entity, drop_first=True)
        dta = pd.concat([dta, entityDummy], axis=1)
        argu1_copy += list(entityDummy.columns)
    if time_effects != False:
        timeDummy = pd.get_dummies(dta[time], prefix=time, drop_first=True)
        dta = pd.concat([dta, timeDummy], axis=1)
        argu1_copy += list(timeDummy.columns)

    y = label_encoder.fit_transform(dta[argu2])
    x = sm.add_constant(dta[argu1_copy])
    model = sm.Probit(y, x)
    results = model.fit()

    # PseudoR2 = re.findall("Pseudo R-squ.*", str(results.summary()))[0].split(" ")[-1]
    # LogLikelihood = re.findall("Log-Likelihood:.*", str(results.summary()))[0].split(" ")[-1]
    pvalue = results.pvalues
    coeff = results.params
    std_err = results.bse
    res_df = pd.DataFrame({
        "pvalue": pvalue,
        "coeff": coeff,
        "std_err": std_err,
    })
    res_js = json.loads(res_df.to_json())
    res_final = {'pvalue': {}, 'coeff': {}, 'std_err': {}}
    target_list = copy.deepcopy(argu1)
    target_list.append('const')
    for key in target_list:
        if not res_js['pvalue'][key] is None:
            res_final['pvalue'][key] = res_js['pvalue'][key]
        if not res_js['coeff'][key] is None:
            res_final['coeff'][key] = res_js['coeff'][key]
        if not res_js['std_err'][key] is None:
            res_final['std_err'][key] = res_js['std_err'][key]

    res_final['n'] = dta.shape[0]
    res_final['time_effect'] = time_effects
    res_final['entity_effect'] = entity_effects
    # res_final['LogLikelihood'] = LogLikelihood
    # res_final['r2'] = PseudoR2

    # 生成解释变量和被解释变量之间的回归文案
    regType = "OLS"
    Y_Name = argu2
    X_Name = argu1[0]
    coef = res_final['coeff'][argu1[0]]
    p = res_final['pvalue'][argu1[0]]

    desc = regResult(argu2, argu1[0], regType, coef, p)
    return {"argu_i": argu2, "Result": res_final, "Desc": desc}

def regResult(Y_Name,X_Name,regType,coef,p):
    base =  f"{regType}的回归结果显示，{X_Name}与{Y_Name}的相关性系数是{coef}，对应的P值是{p}"
    if p>0.1:
        significant = "不显著"
    elif 0.05<p<=0.1:
        significant = "在10%的显著性水平上显著"
    elif 0.01<p<=0.05:
        significant = "在5%的显著性水平上显著"
    else:
        significant = "在1%的显著性水平上显著"

    if coef>0:
        corr = "正相关"
    elif coef<0:
        corr = "负相关"
    else:
        corr = "不存在明显的影响"

    return f"{base},{corr},{significant}，但是，如果没有将可能影响Y的重要变量加以控制，则可能有遗漏变量的问题，导致核心解释变量系数存在内生性，建议参考已有对Y的研究，将相关控制变量加入回归，以解决遗漏变量问题，此外，还应该考虑核心解释变量有没有可能被Y所影响，如果有可能，则可能存在反向因果，可以考虑工具变量回归或者准自然实验的方法来解决这一问题。"


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

def first_column_append(ret_fist, i, name):
    if i == 0:
        ret_fist.append(name)

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

def effect_logit(dta, argu1, argu2, entity_effects, time_effects, entity, time):
    label_encoder = preprocessing.LabelEncoder()

    argu1_copy = copy.deepcopy(argu1)
    if entity_effects != False:
        entityDummy = pd.get_dummies(dta[entity], prefix=entity, drop_first=True)
        dta = pd.concat([dta, entityDummy], axis=1)
        argu1_copy += list(entityDummy.columns)
    if time_effects != False:
        timeDummy = pd.get_dummies(dta[time], prefix=time, drop_first=True)
        dta = pd.concat([dta, timeDummy], axis=1)
        argu1_copy += list(timeDummy.columns)

    y = label_encoder.fit_transform(dta[argu2])
    x = sm.add_constant(dta[argu1])
    model = sm.Logit(y, x)
    results = model.fit()

    PseudoR2 = re.findall("Pseudo R-squ.*", str(results.summary()))[0].split(" ")[-1]
    LogLikelihood = re.findall("Log-Likelihood:.*", str(results.summary()))[0].split(" ")[-1]

    pvalue = results.pvalues
    coeff = results.params
    std_err = results.bse
    res_df = pd.DataFrame({
        "pvalue": pvalue,
        "coeff": coeff,
        "std_err": std_err,
    })
    res_js = json.loads(res_df.to_json())
    res_final = {'pvalue': {}, 'coeff': {}, 'std_err': {}}
    target_list = copy.deepcopy(argu1)
    target_list.append('const')
    for key in target_list:
        if not res_js['pvalue'][key] is None:
            res_final['pvalue'][key] = res_js['pvalue'][key]
        if not res_js['coeff'][key] is None:
            res_final['coeff'][key] = res_js['coeff'][key]
        if not res_js['std_err'][key] is None:
            res_final['std_err'][key] = res_js['std_err'][key]

    res_final['n'] = dta.shape[0]
    res_final['time_effect'] = time_effects
    res_final['entity_effect'] = entity_effects
    res_final['LogLikelihood'] = LogLikelihood
    res_final['r2'] = PseudoR2
    return {"argu_i": argu2, "Result": res_final}

def loss_delete(dta, argues) -> pd.DataFrame:
    data = []
    argues = set(argues)
    for x in dta.index:
        flag = 0
        for y in argues:
            if (pd.isnull(dta.loc[x, y]) or dta.loc[x, y] == ' '):
                flag = flag + 1
                break
        if flag == 0:
            data.append(dta.loc[x])
    result_df = pd.DataFrame(data)
    return result_df

if __name__ == '__main__':
    args = {
        'argu1':{
            'count':2,
            'argue':['地区', '年份'],
            'argus':[
                {
                    'argu_e': ['农村居民人均可支配收入/元'],
                    'argu_i': '测试变量',
                    'entity_effect': False,
                    'time_effect': False
                },
                {
                    'argu_e': ['城镇居民人均可支配收入/元'],
                    'argu_i': '测试变量',
                    'entity_effect': False,
                    'time_effect': False
                }
            ]
            }
    }
    dta = pd.read_excel('/Users/heyang/Desktop/居民人均可支配收入V2.xlsx')
    count = args['argu1']['count']  # 回归个数
    probit_args = args['argu1']['argus']
    probit_result = []
    argu_il = set(probit_args[0]['argu_i'])
    argu_el = set(probit_args[0]['argu_e'])
    entity = args['argu1']['argue'][0]  # 用户选择的个体变量
    time = args['argu1']['argue'][1]  # 用户选择的时间变量

    # 缺失值删减
    temp = [entity, time]
    for i in range(0, count):
        for j in probit_args[i]['argu_e']:
            temp.append(j)
        temp.append(probit_args[i]['argu_i'])
    dta = loss_delete(dta, temp)

    for i in range(0, count):
        probit_result.append(effect_probit(dta,
                                           probit_args[i]['argu_e'],  # 解释变量
                                           probit_args[i]['argu_i'],  # 被解释变量
                                           probit_args[i]['entity_effect'],  # 个体固定效应
                                           probit_args[i]['time_effect'],  # 时间固定效应
                                           entity,
                                           time
                                           ))
        argu_il = argu_il.union(probit_args[i]['argu_i'])
        argu_el = argu_el.union(probit_args[i]['argu_e'])
    ret_s = {"count": len(probit_result),  # 计数
             "OLSList": probit_result,  # 被解释变量
             "ArgeList": list(argu_el)}  # 参数的并集
    desc = ''
    for val in probit_result:
        desc = desc + val['Desc'] + '<br/>'
    print(desc)
    #ret_df = smols2excelV3(ret_s)
    print('---------')


