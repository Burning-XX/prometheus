
import pandas as pd
from sklearn import preprocessing
import statsmodels.api as sm
import json
import traceback
import copy


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
    return {"argu_i": argu2, "Result": res_final}

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
    return {"argu_i": argu2, "Result": res_final}

if __name__ == '__main__':
    args = {
        'argu1':{
            'count':2,
            'argue':['地区', '年份'],
            'argus':[
                {
                    'argu_e': ['农村居民人均可支配收入/元'],
                    'argu_i': '测试变量',
                    'entity_effect': True,
                    'time_effect': True
                },
                {
                    'argu_e': ['城镇居民人均可支配收入/元'],
                    'argu_i': '测试变量',
                    'entity_effect': True,
                    'time_effect': True
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
    for i in range(0, count):
        probit_result.append(effect_logit(dta,
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
    ret_df = smols2excelV3(ret_s)
    print(ret_s)
    print('---------')


