"""
   ___                            _        _   _                   _                                        _
  / __\___  _ __ ___  _ __  _   _| |_ __ _| |_(_) ___  _ __   __ _| |   ___  ___ ___  _ __   ___  _ __ ___ (_) ___ ___
 / /  / _ \| '_ ` _ \| '_ \| | | | __/ _` | __| |/ _ \| '_ \ / _` | |  / _ \/ __/ _ \| '_ \ / _ \| '_ ` _ \| |/ __/ __|
/ /__| (_) | | | | | | |_) | |_| | || (_| | |_| | (_) | | | | (_| | | |  __/ (_| (_) | | | | (_) | | | | | | | (__\__ \
\____/\___/|_| |_| |_| .__/ \__,_|\__\__,_|\__|_|\___/|_| |_|\__,_|_|  \___|\___\___/|_| |_|\___/|_| |_| |_|_|\___|___/
                     |_|

计算经济学数据处理工具箱 API
DATAC.PY
核心数据处理函数,提供纯数据处理类的方法,以单个函数的方式提供
包括但不仅限于:
相关系数矩阵
核心变量描述性统计
三大回归模型(OLS/Probit/Logit)

本页作者:
Aliebc (aliebcx@outlook.com)
Andy (andytsangyuklun@gmail.com)
Jingwei Luo

Copyright(C)2022 All Rights reserved.
"""
import json
from django.http import JsonResponse, HttpRequest
import pandas as pd
import numpy as npy
import scipy.stats as st
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from scipy.stats.mstats import winsorize
from sklearn import preprocessing
import re
import traceback
from .ce import ret2, ret3, request_analyse, ret_error, ret_success
from .filer import get_file_data, put_file_excel
from .filer import argues_loss_delete
from .filer import put_file_all as put_file
import copy

def dcorr(request: HttpRequest) -> JsonResponse:
    """
    HTTP请求处理:获取单个相关系数
    """
    try:
        dta = get_file_data(request)
        argus = request_analyse(request)
        argu1 = argus['argu1']
        argu2 = argus['argu2']
        c2 = st.pearsonr(dta[argu1], dta[argu2])
        cord = dta.corr()
        d = cord.to_json()
    except Exception as e:
        return ret_error(e)
    return ret_success({"CorrMartix": json.loads(d), "Significance": c2})


def xcorr_single(
        dataset: pd.DataFrame,
        cord: list
) -> dict:
    """
    内部函数:获取单个相关系数
    """
    dta = dataset
    ret = {}
    for i in cord:
        ret[i] = {}
        for j in cord:
            cor = st.pearsonr(dta[i], dta[j])
            ret[i][j] = cor
            if i == j:
                ret[i][j] = [1, 0]  # 防止精度问题出现(0.9999...)
    return ret

def correlation(name1, name2, cor, pvalue):  # cor是两个变量之间的相关性系数，pvalue是这个相关性系数是否显著
    if pvalue > 0.1:
        return f"[{name1}]与[{name2}]之间不存在显著的相关性，二者之间的关系可能需要进一步检验。"
    else:
        if abs(cor) >= 0.7:
            return f"[{name1}]与[{name2}]之间存在显著的相关性，相关系数是{'%.3f' % cor}，可以认为这两个变量之间存在强相关性。"
        else:
            return f"[{name1}]与[{name2}]之间存在显著的相关性，相关系数是{'%.3f' % cor}，可以认为二者之间的相关性较低，如果从理论上，这个变量对被解释变量有重要的作用，那么在回归时，可以加入这个变量。"


def describe_correlation(name1, name2, cor, pvalue):
    if name1 == name2:  # 如果是同一个变量，就不用分析了
        return ""
    if cor > 0:
        return f"{correlation(name1, name2, cor, pvalue)}这两个变量是正相关的，但是，二者之间的正相关性，并不能被解释为因果关系,二者的关系可能被遗漏变量、测量误差以及反向因果所干扰，因此，因果关系的检验还需要进一步的研究。"
    else:
        return f"{correlation(name1, name2, cor, pvalue)}这两个变量是正相关的，但是，二者之间的负相关性，并不能被解释为因果关系,二者的关系可能被遗漏变量、测量误差以及反向因果所干扰，因此，因果关系的检验还需要进一步的研究。"


def xcorr_safe(request: HttpRequest) -> JsonResponse:
    """
    HTTP请求处理:获取相关系数矩阵
    """
    try:
        dta = get_file_data(request)
        args = request_analyse(request)
        cord = args['argu1']
        # 针对操作的变量, 进行缺失值删减
        dta = argues_loss_delete(dta, cord)
        ret = xcorr_single(dta, cord)
        # 组织描述性文案
        desc = ''
        for i in range(len(cord)):
            for j in range(i + 1, len(cord)):
                v1 = cord.__getitem__(i)
                v2 = cord.__getitem__(j)
                corr = ret[v1][v2].correlation
                pvalue = ret[v1][v2].pvalue
                desc = desc + describe_correlation(v1, v2, corr, pvalue) + '<br/>'
    except Exception as e:
        traceback.print_exc()
        return ret_error(e)
    return ret2(0, {"CorrMartix": ret, "Desc": desc}, None)


def dtype(request: HttpRequest) -> JsonResponse:
    """
    HTTP请求处理:单个变量的描述性统计
    """
    try:
        dta = get_file_data(request)
        args = request_analyse(request)
        retu = dta[args['argu1']]
        return ret_success(json.loads(retu.value_counts().to_json()))
    except Exception as e:
        return ret_error(e)


def dsummary(request: HttpRequest) -> JsonResponse:
    """
    HTTP请求处理:全变量的描述性统计
    """
    try:
        dta = get_file_data(request)
        argu1 = json.loads(request.body)['argu1']
        retu = dta[argu1]
        return ret2(0, json.loads(retu.describe().to_json()), None)
    except Exception as e:
        return ret_error(e)


def xsummary(request: HttpRequest) -> JsonResponse:
    """
    HTTP请求处理:多个变量的描述性统计
    """
    try:
        dta = get_file_data(request)
        a2 = {}
        for key in dta:
            r2 = dta[key]
            a2[key] = json.loads(r2.describe().to_json())
        return ret2(0, a2, None)
    except Exception as e:
        return ret_error(e)


def xsummary2(request: HttpRequest) -> JsonResponse:
    """
    HTTP请求处理:多个变量的描述性统计-带文件处理
    """
    try:
        args = request_analyse(request)
        argu1 = args['argu1']
        dta = get_file_data(request)
        a2 = {}
        for key in argu1:
            r2 = dta[key]
            if not r2.dtype == 'object':
                a2[key] = json.loads(r2.describe().to_json())
        df2 = pd.read_json(json.dumps(a2), orient="index")
        uid = put_file_excel(df2, True, "Variable")
        return ret_success({'ValueList': a2, 'File': {'uid': uid, 'f_suffix': '.xlsx'}})
    except Exception as e:
        traceback.print_exc()
        return ret_error(e)


def xsummary3(request: HttpRequest) -> JsonResponse:
    try:
        args = request_analyse(request)
        argu1 = args['argu1']
        dta = get_file_data(request)
        a2 = {}
        for key in argu1:
            r2 = dta[key]
            if not r2.dtype == 'object':
                desc = r2.describe()
                skew = r2.skew()
                kurt = r2.kurt()
                list = [desc, pd.Series({"skew": skew}), pd.Series({"kurt": kurt})]
                desc = pd.concat(list)
                a2[key] = json.loads(desc.to_json())
        df2 = pd.read_json(json.dumps(a2), orient="index")
        uid = put_file_excel(df2, True, "Variable")
        result = []
        desc = ''
        result.append(['变量名', 'count', 'mean', 'std', 'min', 'max', '25%', '50%', '75%', 'skew', 'kurt'])
        for key in argu1:
            if a2.__contains__(key):
                count = a2[key]['count']
                mean = a2[key]['mean']
                std = a2[key]['std']
                min = a2[key]['min']
                max = a2[key]['max']
                q1 = a2[key]['25%']
                q2 = a2[key]['50%']
                q3 = a2[key]['75%']
                skew = a2[key]['skew']
                kurt = a2[key]['kurt']
                result.append([key, count, mean, std, min, max, q1, q2, q3, skew, kurt])
                desc = desc + describe_variable(key, count, mean, std, max, min, q1, q2, q3, kurt, skew) + '<br/>'
        return ret_success({'ValueList': result, 'File': {'uid': uid, 'f_suffix': '.xlsx'}, 'Desc':desc})
    except Exception as e:
        traceback.print_exc()
        return ret_error(e)

def mean2median(mean, Q2, Std):  # 返回关于中心趋势的建议，这里判断一个变量是否存在极端值的标准是：均值是否偏离中位数超过两个标准差，这是一个经验数值，可以调整
    if abs((mean - Q2) / Std) >= 2:
        return "均值偏离中位数超过两个标准差，均值可能收到极端值的影响，明显偏离了中位数，需要进一步检查是否存在异常值,包括检查直方图，是否存在偏离正常值较远的异常数据点，或者箱线图，是否存在超过上下四分位数1.5倍以上的异常值，如果存在，可以考虑缩尾处理。"
    else:
        return "均值和中位数相差没有超过两个标准差，变量分布较为对称，没有明显偏斜，可能不存在大量极端值，变量分布偏向于正态分布，可以进一步进行统计分析。"

def skewness(skew):  # 返回关于偏斜的建议，这里判断变量是否有偏斜的标准是偏斜度
    if skew >= 0.5:
        return "如果一个偏斜度越接近于0，说明该变量越接近于正态分布。在本例中，该变量偏斜度超过0.5，存在一定的偏斜，数据分布呈现右偏，即尾部向右侧延伸，右侧存在较多极端值。"
    if skew <= -0.5:
        return "如果一个偏斜度越接近于0，说明该变量越接近于正态分布。在本例中，该变量偏斜度小于-0.5，存在一定的偏斜，数据分布呈现左偏，即尾部向左侧延伸，左侧存在较多极端值。"
    else:
        return "变量的偏斜度绝对值小于0.5，可以认为接近于正态分布。"

# mean2median以及skewness这两个函数将会被descirbe_variable这个函数调用，只需要把上两个函数放在代码中即可，describe_variable将会返回单个变量的描述文本
def describe_variable(varName, N, Mean, Std, Max, Min, Q1, Q2, Q3, kurt, skew):
    return (f"变量[{varName}]的观测数量是{N}个，均值是{'%.3f' % Mean}，标准差是{'%.3f' % Std}，最大值是{'%.3f' % Max}，最小值是{'%.3f' % Min}，25%分位数是{'%.3f' % Q1}，50%分位数是{'%.3f' % Q2}，75%分位数是{'%.3f' % Q3}，峰度是{'%.3f' % kurt},偏斜度是{'%.3f' % skew}。从描述性统计来看，{mean2median(Mean, Q2, Std)}{skewness(skew)}")

def deal_with_pairs(pairs):
    data = dict(pairs)
    list_k = ['count', 'mean', 'std', 'min', 'max', '25%', '50%', '75%']
    list_v = [data['count'], data['mean'], data['std'], data['min'], data['max'], data['25%'], data['50%'], data['75%']]
    # print('----')
    return dict(zip(list_k, list_v))


def dlm3(request) -> JsonResponse:
    """
    HTTP请求处理:参数型回归分析
    """
    try:
        dta = get_file_data(request)
        args = request_analyse(request)
        argu1 = args['argu1']
        argu2 = args['argu2']
        reg = args['reg']
        retu = npy.polyfit(dta[argu1], dta[argu2], reg)
        return ret_success({
            "reg": reg, "RegList": retu.tolist(),
            "DataList": {
                argu1: json.loads(dta[argu1].to_json()),
                argu2: json.loads(dta[argu2].to_json())
            }
        })
    except Exception as e:
        return ret_error(e)


def heter_compare_apply(value, y, c_name) -> str:
    """
    内部函数:大于小于处理
    """
    if value > y:
        return str(c_name) + ">" + str(y)
    else:
        return str(c_name) + "<=" + str(y)


def heter_compare_df(df, col_name, s) -> pd.DataFrame:
    """
    内部函数:比较处理
    """
    df.loc[:, col_name + '_type'] = df[col_name].apply(heter_compare_apply, y=s, c_name=col_name)
    return df


def type_corr(request: HttpRequest) -> JsonResponse:
    """
    HTTP处理函数:异质性分析的分段相关系数
    """
    try:
        dta = get_file_data(request)
        xe = request_analyse(request)
        argu1 = xe['argu1']
        argu_type = xe['argu_type']
        segment = xe['segment']
        dta2_t = dta[dta[argu_type] > segment]
        re1 = xcorr_single(dta2_t, argu1)
        dta3_t = dta[dta[argu_type] <= segment]
        re2 = xcorr_single(dta3_t, argu1)
        return ret_success({'More': re1, 'Less': re2})
    except Exception as e:
        return ret_error(e)


# By Andy at 2022/2/7 17:30
# Modified By Aliebc at 2022/2/7 17:50

def ols(request):
    try:
        dta = get_file_data(request)
        argu1 = json.loads(request.body)['argu1']
        argu2 = json.loads(request.body)['argu2']
        x = sm.add_constant(dta[argu1])
        model = sm.OLS(dta[argu2], x)
        results = model.fit()
        pvals = results.pvalues
        coeff = results.params
        conf_lower = results.conf_int()[0]
        conf_higher = results.conf_int()[1]
        r2 = results.rsquared
        r2adj = results.rsquared_adj
        ll = results.llf
        fvalue = results.fvalue
        y = results.summary()
        results_df = pd.DataFrame({"pvals": pvals,
                                   "coeff": coeff,
                                   "conf_lower": conf_lower,
                                   "conf_higher": conf_higher,
                                   "r_squared": r2,
                                   "r_squared_adj": r2adj,
                                   "log_likelihood": ll,
                                   "f_statistic": fvalue
                                   })
    # Reordering...
    # results_df = results_df[["r_squared","r_squared_adj","coeff","pvals","conf_lower","conf_higher"]]
    except Exception as e:
        return ret_error(e)
    return ret_success({
        "Regression Summary": json.loads(results_df.to_json()),
        "s_text": re.sub(r"Notes(.|\n)*", "", str(y))
    })

#新加——对数化操作
def data2log(data):
    """
    返回一个在非负数值域的数值x的对数：ln(1+x)
    """
    ln = npy.log(1+data)
    if not ln>=0:
        return ''
    else:
        return ln
def data_log(request):
    try:
        dta = get_file_data(request)
        xe = json.loads(request.body)
        argu1 = xe['argu1']
        f_suffix = xe['f_suffix']
        for label in argu1:
            dta['ln_' + label] = dta[label].apply(data2log)

        uid = put_file(dta, f_suffix)
        return ret_success(
            {"uid": uid, "f_suffix": f_suffix, "Datalist": json.loads(dta.to_json(orient='index'))})
    except Exception as e:
        return ret_error(e)

def winsor(dataList,down=1,up=1): #默认上下均为1%缩尾，dataList是一列，即dataframce中的一列
    """
    将传入的列表数据进行缩尾处理（替换观测值而不是删除）
    """
    winsored = winsorize(dataList,limits=[down/100,up/100])
    return winsored.data
def winsor_data(request):
    try:
        dta = get_file_data(request)
        xe = json.loads(request.body)
        argu1 = xe['argu1']
        f_suffix = xe['f_suffix']
        for label in argu1:
            dta['w_' + label] = winsor(npy.array(dta[label]),1,1)

        uid = put_file(dta, f_suffix)
        return ret_success(
            {"uid": uid, "f_suffix": f_suffix, "Datalist": json.loads(dta.to_json(orient='index'))})
    except Exception as e:
        traceback.print_exc()
        return ret_error(e)

def binary_probit(request):
    try:
        label_encoder = preprocessing.LabelEncoder()
        dta = get_file_data(request)
        argu1 = json.loads(request.body)['argu1']
        argu2 = json.loads(request.body)['argu2']
        y = label_encoder.fit_transform(dta[argu2])
        x = sm.add_constant(dta[argu1])
        model = sm.Probit(y, x)
        results = model.fit()

        pvals = results.pvalues
        coeff = results.params
        conf_lower = results.conf_int()[0]
        conf_higher = results.conf_int()[1]
        ll = results.llf
        pseudor2 = results.prsquared
        llnull = results.llnull
        llrpvalue = results.llr_pvalue
        f_test = results.f_test
        y = results.summary()

        results_df = pd.DataFrame({"pvals": pvals,
                                   "coeff": coeff,
                                   "conf_lower": conf_lower,
                                   "conf_higher": conf_higher,
                                   "log_likelihood": ll,
                                   "pseudo_r_squared": pseudor2,
                                   "ll_null": llnull,
                                   "llr_p_value": llrpvalue,
                                   "f_statistic": f_test
                                   })
        results_df = results_df[[
            "pseudo_r_squared",
            "log_likelihood",
            "ll_null",
            "llr_p_value",
            "f_statistic",
            "coeff",
            "pvals",
            "conf_lower",
            "conf_higher"]]
    except Exception as e:
        return ret2(-1, None, "Error(#3:Internal). Check if argu2 is binary.")
    return ret_success({"Regression Summary": json.loads(results_df.to_json()), "s_text": str(y)})


def binary_logit(request):
    try:
        label_encoder = preprocessing.LabelEncoder()
        dta = get_file_data(request)
        argu1 = json.loads(request.body)['argu1']
        argu2 = json.loads(request.body)['argu2']
        y = label_encoder.fit_transform(dta[argu2])
        x = sm.add_constant(dta[argu1])
        model = sm.Logit(y, x)
        results = model.fit()
        pvals = results.pvalues
        coeff = results.params
        conf_lower = results.conf_int()[0]
        conf_higher = results.conf_int()[1]
        ll = results.llf
        pseudor2 = results.prsquared
        llnull = results.llnull
        llrpvalue = results.llr_pvalue
        f_test = results.f_test
        y = results.summary()

        results_df = pd.DataFrame({"pvals": pvals,
                                   "coeff": coeff,
                                   "conf_lower": conf_lower,
                                   "conf_higher": conf_higher,
                                   "log_likelihood": ll,
                                   "pseudo_r_squared": pseudor2,
                                   "ll_null": llnull,
                                   "llr_p_value": llrpvalue,
                                   "f_statistic": f_test
                                   })
        results_df = results_df[
            ["pseudo_r_squared", "log_likelihood", "ll_null", "llr_p_value", "f_statistic", "coeff", "pvals",
             "conf_lower", "conf_higher"]]
    except Exception as e:
        return ret2(-1, None, "Error(#3:Internal). Check if argu2 is binary.")
    return ret2(0, {"Regression Summary": json.loads(results_df.to_json()), "s_text": str(y)}, None)


## This Part is developed by Jingwei Luo

def loss_test(request):
    try:
        dta = get_file_data(request)
        argu1 = json.loads(request.body)['argu1']
        loss = 0
        data = []
        for x in dta.index:
            for y in argu1:
                if (pd.isnull(dta.loc[x, y]) or dta.loc[x, y] == " "):
                    loss = loss + 1
                    data.append(dta.loc[x])
                    break
        result_df = pd.DataFrame(data)
        ob1 = len(dta)
        return ret_success({"loss": loss, "Observed": ob1, "Datalist": json.loads(result_df.to_json(orient='index'))})
    except Exception as e:
        return ret_error(e)


def loss_delete(request):
    try:
        dta = get_file_data(request)
        xe = json.loads(request.body)
        argu1 = xe['argu1']
        f_suffix = xe['f_suffix']
        data = []

        for x in dta.index:
            flag = 0
            for y in argu1:
                if (pd.isnull(dta.loc[x, y]) or dta.loc[x, y] == " "):
                    flag = flag + 1
                    break
            if flag == 0:
                data.append(dta.loc[x])
        result_df = pd.DataFrame(data)
        uid = put_file(result_df, f_suffix)
        return ret_success(
            {"uid": uid, "f_suffix": f_suffix, "Datalist": json.loads(result_df.to_json(orient='index'))})
    except Exception as e:
        return ret_error(e)


def var_filter(request):
    try:
        df = get_file_data(request)
        xe = json.loads(request.body)
        f_suffix = xe['f_suffix']
        params = xe['params']
        for param in params:
            variable = param['variable']
            type = param['type']
            where = param['where']
            if (type == 1):
                label1 = where[0]['condition']
                num1 = where[0]['number']
                if (len(where) == 1):
                    if (label1 == 1):
                        for x in df.index:
                            if (pd.notnull(df.loc[x, variable])):
                                if (df.loc[x, variable] > num1):
                                    df.drop(index=x, inplace=True)
                            else:
                                df.drop(index=x, inplace=True)
                    elif (label1 == 2):
                        for x in df.index:
                            if (pd.notnull(df.loc[x, variable])):
                                if (df.loc[x, variable] >= num1):
                                    df.drop(index=x, inplace=True)
                            else:
                                df.drop(index=x, inplace=True)
                    elif (label1 == 3):
                        for x in df.index:
                            if (pd.notnull(df.loc[x, variable])):
                                if (df.loc[x, variable] == num1):
                                    df.drop(index=x, inplace=True)
                            else:
                                df.drop(index=x, inplace=True)
                    elif (label1 == 4):
                        for x in df.index:
                            if (pd.notnull(df.loc[x, variable])):
                                if (df.loc[x, variable] <= num1):
                                    df.drop(index=x, inplace=True)
                            else:
                                df.drop(index=x, inplace=True)
                    elif (label1 == 5):
                        for x in df.index:
                            if (pd.notnull(df.loc[x, variable])):
                                if (df.loc[x, variable] < num1):
                                    df.drop(index=x, inplace=True)
                            else:
                                df.drop(index=x, inplace=True)
                elif (len(where) == 2):
                    label2 = where[1]['condition']
                    num2 = where[1]['number']
                    if (num1 == num2):
                        return ret2(-1, None, "Error(#:number_choose).")

                    if (label1 == 3):
                        for x in df.index:
                            if (pd.notnull(df.loc[x, variable])):
                                if (df.loc[x, variable] == num1):
                                    df.drop(index=x, inplace=True)
                            else:
                                df.drop(index=x, inplace=True)

                    if label1 == 1 and label2 == 5:
                        if num1 < num2:
                            for x in df.index:
                                if (pd.notnull(df.loc[x, variable])):
                                    if (df.loc[x, variable] > num1 and df.loc[x, variable] < num2):
                                        df.drop(index=x, inplace=True)
                                else:
                                    df.drop(index=x, inplace=True)
                        else:
                            for x in df.index:
                                if (pd.notnull(df.loc[x, variable])):
                                    if (df.loc[x, variable] > num1 or df.loc[x, variable] < num2):
                                        df.drop(index=x, inplace=True)
                                else:
                                    df.drop(index=x, inplace=True)

                    elif label1 == 5 and label2 == 1:
                        if num1 < num2:
                            for x in df.index:
                                if (pd.notnull(df.loc[x, variable])):
                                    if df.loc[x, variable] < num1 or df.loc[x, variable] > num2:
                                        df.drop(index=x, inplace=True)
                                else:
                                    df.drop(index=x, inplace=True)
                        else:
                            for x in df.index:
                                if (pd.notnull(df.loc[x, variable])):
                                    if df.loc[x, variable] < num1 and df.loc[x, variable] > num2:
                                        df.drop(index=x, inplace=True)
                                else:
                                    df.drop(index=x, inplace=True)

                    elif label1 == 2 and label2 == 5:
                        if num1 < num2:
                            for x in df.index:
                                if (pd.notnull(df.loc[x, variable])):
                                    if df.loc[x, variable] >= num1 and df.loc[x, variable] < num2:
                                        df.drop(index=x, inplace=True)
                                else:
                                    df.drop(index=x, inplace=True)
                        else:
                            for x in df.index:
                                if (pd.notnull(df.loc[x, variable])):
                                    if df.loc[x, variable] >= num1 or df.loc[x, variable] < num2:
                                        df.drop(index=x, inplace=True)
                                else:
                                    df.drop(index=x, inplace=True)

                    elif label1 == 5 and label2 == 2:
                        if num1 < num2:
                            for x in df.index:
                                if (pd.notnull(df.loc[x, variable])):
                                    if df.loc[x, variable] < num1 or df.loc[x, variable] >= num2:
                                        df.drop(index=x, inplace=True)
                                else:
                                    df.drop(index=x, inplace=True)
                        else:
                            for x in df.index:
                                if (pd.notnull(df.loc[x, variable])):
                                    if df.loc[x, variable] < num1 and df.loc[x, variable] >= num2:
                                        df.drop(index=x, inplace=True)
                                else:
                                    df.drop(index=x, inplace=True)

                    elif label1 == 1 and label2 == 4:
                        if num1 < num2:
                            for x in df.index:
                                if (pd.notnull(df.loc[x, variable])):
                                    if df.loc[x, variable] > num1 and df.loc[x, variable] <= num2:
                                        df.drop(index=x, inplace=True)
                                else:
                                    df.drop(index=x, inplace=True)
                        else:
                            for x in df.index:
                                if (pd.notnull(df.loc[x, variable])):
                                    if df.loc[x, variable] > num1 or df.loc[x, variable] <= num2:
                                        df.drop(index=x, inplace=True)
                                else:
                                    df.drop(index=x, inplace=True)

                    elif label1 == 4 and label2 == 1:
                        if num1 < num2:
                            for x in df.index:
                                if (pd.notnull(df.loc[x, variable])):
                                    if df.loc[x, variable] <= num1 or df.loc[x, variable] > num2:
                                        df.drop(index=x, inplace=True)
                                else:
                                    df.drop(index=x, inplace=True)
                        else:
                            for x in df.index:
                                if (pd.notnull(df.loc[x, variable])):
                                    if df.loc[x, variable] <= num1 and df.loc[x, variable] > num2:
                                        df.drop(index=x, inplace=True)
                                else:
                                    df.drop(index=x, inplace=True)

                    elif label1 == 2 and label2 == 4:
                        if num1 < num2:
                            for x in df.index:
                                if (pd.notnull(df.loc[x, variable])):
                                    if df.loc[x, variable] >= num1 and df.loc[x, variable] <= num2:
                                        df.drop(index=x, inplace=True)
                                else:
                                    df.drop(index=x, inplace=True)
                        else:
                            for x in df.index:
                                if (pd.notnull(df.loc[x, variable])):
                                    if df.loc[x, variable] >= num1 or df.loc[x, variable] <= num2:
                                        df.drop(index=x, inplace=True)
                                else:
                                    df.drop(index=x, inplace=True)

                    elif label1 == 4 and label2 == 2:
                        if num1 < num2:
                            for x in df.index:
                                if (pd.notnull(df.loc[x, variable])):
                                    if df.loc[x, variable] <= num1 or df.loc[x, variable] >= num2:
                                        df.drop(index=x, inplace=True)
                                else:
                                    df.drop(index=x, inplace=True)
                        else:
                            for x in df.index:
                                if (pd.notnull(df.loc[x, variable])):
                                    if df.loc[x, variable] <= num1 and df.loc[x, variable] >= num2:
                                        df.drop(index=x, inplace=True)
                                else:
                                    df.drop(index=x, inplace=True)

                    else:
                        if label1 and label2:
                            return ret2(-1, None,
                                        "Error(#:label):" + "con1:" + str(label1) + "num1:" + str(num1) + "con2:" + str(
                                            label2) + "num2:" + str(num2))
                        elif not label1:
                            return ret2(-1, None, "Error(#:label1_null).")
                        elif not label2:
                            return ret2(-1, None, "Error(#:label2_null).")
                        else:
                            return ret2(-1, None, "Error(#:label_all_null).")
                else:
                    return ret2(-1, None, "Error(#:num_length).")

            elif (type == 2):
                delete_way = where[0]['way']
                str_select = where[0]['str_select']
                if (delete_way == 1):
                    for x in df.index:
                        if (df.loc[x, variable] == str_select):
                            df.drop(index=x, inplace=True)
                else:
                    df = df[~ df[variable].str.contains(str_select)]

            else:
                return ret2(-1, None, "Error(#:Type).")
        uid = put_file(df, f_suffix)
        return ret_success({"uid": uid, "f_suffix": f_suffix, "Datalist": json.loads(df.to_json(orient='index'))})
    except Exception as e:
        return ret_error(e)


## End this part

def ols_plain_inter(df, argu_i, argu_e):
    argu_e_param = argu_e
    argu_e = sm.add_constant(df[argu_e])
    mod = sm.OLS(df[argu_i], argu_e)
    results = mod.fit()
    pvalue = results.pvalues
    coeff = results.params
    std_err = results.bse
    r2 = results.rsquared
    res_df = pd.DataFrame({  # 从回归结果中提取需要的结果
        "pvalue": pvalue,
        "coeff": coeff,
        "std_err": std_err,
    })
    res_js = json.loads(res_df.to_json())
    res_js['n'] = df.shape[0]
    res_js['r2'] = r2
    # 生成解释变量和被解释变量之间的回归文案
    regType = "OLS"
    coef = res_js['coeff'][argu_e_param[0]]
    p = res_js['pvalue'][argu_e_param[0]]
    desc = regResult(argu_i, argu_e_param[0], regType, coef, p)

    return {"argu_i": argu_i, "Result": res_js, 'desc': desc}


def ols_effect_inter(df, argu_i, argu_e, entity_effects, time_effects):
    argu_e_param = argu_e
    argu_e = sm.add_constant(df[argu_e])
    mod = PanelOLS(df[argu_i], argu_e, entity_effects=entity_effects, time_effects=time_effects)
    results = mod.fit()
    pvalue = results.pvalues
    coeff = results.params
    std_err = results.std_errors
    r2 = results.rsquared
    res_df = pd.DataFrame({  # 从回归结果中提取需要的结果
        "pvalue": pvalue,
        "coeff": coeff,
        "std_err": std_err,
    })
    res_js = json.loads(res_df.to_json())
    res_js['n'] = df.shape[0]
    res_js['time_effect'] = time_effects
    res_js['entity_effect'] = entity_effects
    res_js['r2'] = r2
    # 生成解释变量和被解释变量之间的回归文案
    regType = "OLS"
    coef = res_js['coeff'][argu_e_param[0]]
    p = res_js['pvalue'][argu_e_param[0]]
    desc = regResult(argu_i, argu_e_param[0], regType, coef, p)
    return {"argu_i": argu_i, "Result": res_js, "desc": desc}


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

def convert(ret_dict) -> dict:
    result = {}
    list = ['被解释变量', 'const', '观测值', 'R^2']
    for key in ret_dict:
        map = ret_dict[key]
        v = {}
        v['被解释变量'] = map["被解释变量"]
        v['常数项'] = map['const']
        for k in map:
            if k not in list:
                v[k] = map[k]
        v['观测值'] = map['观测值']
        v['R2'] = map['R^2']
        result[key] = v
    return result

def smols2excel(ols_res_dict: dict) -> pd.DataFrame:
    """
    内部函数:把回归结果格式化处理到Excel
    """
    ret_dict = {}
    ArgList = ols_res_dict['ArgeList']
    ArgList.append("const")
    for i in range(0, ols_res_dict['count']):  # 按个数循环
        ret_dict['(' + str(i + 1) + ')'] = {"被解释变量": ols_res_dict['OLSList'][i]['argu_i']}  # 建立索引
        for key in ArgList:
            if key in ols_res_dict['OLSList'][i]['Result']['coeff']:
                p_str = g_p_str(ols_res_dict['OLSList'][i]['Result']['pvalue'][key])
                temp = '(' + str(round(ols_res_dict['OLSList'][i]['Result']['std_err'][key], 3)) + ')'
                ret_dict['(' + str(i + 1) + ')'][key] = str(
                    round(ols_res_dict['OLSList'][i]['Result']['coeff'][key], 3)) + p_str + '\n' + temp  # 拼接显著性标记和标准误
            else:
                ret_dict['(' + str(i + 1) + ')'][key] = ""
        if 'entity_effect' in ols_res_dict['OLSList'][i]['Result']:  # 把剩下的项目加入这个文件
            ret_dict['(' + str(i + 1) + ')']['时间固定效应'] = ols_res_dict['OLSList'][i]['Result']['entity_effect']
            ret_dict['(' + str(i + 1) + ')']['个体固定效应'] = ols_res_dict['OLSList'][i]['Result']['time_effect']
        ret_dict['(' + str(i + 1) + ')']['观测值'] = str(ols_res_dict['OLSList'][i]['Result']['n'])
        ret_dict['(' + str(i + 1) + ')']['R^2'] = str(round(ols_res_dict['OLSList'][i]['Result']['r2'], 3))
    ret_df = pd.DataFrame(convert(ret_dict))
    return ret_df


def smols2excelV2(ols_res_dict: dict) -> pd.DataFrame:
    """
    内部函数:把回归结果格式化处理到Excel
    """
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
            first_column_append(first_column, i, "R^2")
            column.append(str(round(ols_res_dict['OLSList'][i]['Result']['r2'], 3)))
            if i == 0:
                ret_list.append(first_column)
            ret_list.append(column)
        except Exception as e:
            traceback.print_exc()
    return pd.DataFrame(ret_list).T

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
            first_column_append(first_column, i, "R^2")
            column.append(ols_res_dict['OLSList'][i]['Result']['r2'])
            first_column_append(first_column, i, "Likelihood")
            column.append(ols_res_dict['OLSList'][i]['Result']['LogLikelihood'])

            if i == 0:
                ret_list.append(first_column)
            ret_list.append(column)
        except Exception as e:
            traceback.print_exc()
    return pd.DataFrame(ret_list).T

def first_column_append(ret_fist, i, name):
    if i == 0:
        ret_fist.append(name)

def ols_effect_repeat(request: HttpRequest) -> JsonResponse:
    """
    HTTP处理函数:格式化输出OLS固定效应回归的结果
    """
    try:
        args = request_analyse(request)
        dta = get_file_data(request)
        count = args['argu1']['count']
        olss = args['argu1']['argus']
        dta = dta.set_index(args['argu1']['argue'])
        ols_res = []
        argu_il = set(olss[0]['argu_i'])
        argu_el = set(olss[0]['argu_e'])
        for i in range(0, count):
            ols_res.append(ols_effect_inter(dta,
                                            olss[i]['argu_i'],  # 被解释变量
                                            olss[i]['argu_e'],  # 解释变量
                                            olss[i]['entity_effect'],  # 个体固定效应(Bool)
                                            olss[i]['time_effect']))  # 时间固定效应(Bool)
            argu_il = argu_il.union(olss[i]['argu_i'])
            argu_el = argu_el.union(olss[i]['argu_e'])

        # 解释变量与被解释变量之间的回归分析文案
        desc = ''
        for val in ols_res:
            desc = desc + val['desc'] + '<br/>'

        ret_s = {"count": len(ols_res),  # 计数
                 "OLSList": ols_res,  # 被解释变量
                 "ArgeList": list(argu_el),
                 "Desc": desc}
        ret_df = smols2excelV2(ret_s)
        ret_uid = put_file_excel(ret_df, False)
        ret_s['File'] = {"uid": ret_uid, "f_suffix": ".xlsx"}
        return ret_success(ret_s)
    except Exception as e:
        traceback.print_exc()
        return ret_error(e)


def ols_repeat(request: HttpRequest) -> JsonResponse:
    """
    HTTP处理函数:格式化输出OLS回归的结果
    """
    try:
        args = request_analyse(request)
        dta = get_file_data(request)
        count = args['argu1']['count']
        olss = args['argu1']['argus']
        ols_res = []
        argu_il = set(olss[0]['argu_i'])
        argu_el = set(olss[0]['argu_e'])

        # 缺失值删减
        temp = []
        for i in range(0, count):
            for j in olss[i]['argu_e']:
                temp.append(j)
            temp.append(olss[i]['argu_i'])
        dta = argues_loss_delete(dta, temp)

        for i in range(0, count):
            ols_res.append(ols_plain_inter(dta,
                                           olss[i]['argu_i'],  # 被解释变量
                                           olss[i]['argu_e']))  # 解释变量
            argu_il = argu_il.union(olss[i]['argu_i'])
            argu_el = argu_el.union(olss[i]['argu_e'])

        # 解释变量与被解释变量之间的回归分析文案
        desc = ''
        for val in ols_res:
            desc = desc + val['desc'] + '<br/>'

        ret_s = {"count": len(ols_res),  # 计数
                 "OLSList": ols_res,  # 回归结果
                 "ArgeList": list(argu_el),
                 "Desc": desc}
        ret_df = smols2excelV2(ret_s)
        ret_uid = put_file_excel(ret_df, False)  # 输出索引
        ret_s['File'] = {"uid": ret_uid, "f_suffix": ".xlsx"}  # 文件列表
        return ret_success(ret_s)
    except Exception as e:
        traceback.print_exc()
        return ret_error(e)

def logit(dta, argu1, argu2):
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(dta[argu2])
    x = sm.add_constant(dta[argu1])
    model = sm.Logit(y, x)
    results = model.fit()
    PseudoR2 = re.findall("Pseudo R-squ.*", str(results.summary()))[0].split(" ")[-1]
    LogLikelihood = re.findall("Log-Likelihood:.*", str(results.summary()))[0].split(" ")[-1]

    pvalue = results.pvalues
    coeff = results.params
    std_err = results.bse
    res_df = pd.DataFrame({  # 从回归结果中提取需要的结果
        "pvalue": pvalue,
        "coeff": coeff,
        "std_err": std_err,
    })
    res_js = json.loads(res_df.to_json())
    res_js['n'] = dta.shape[0]
    res_js['LogLikelihood'] = LogLikelihood
    res_js['r2'] = PseudoR2

    # 生成解释变量和被解释变量之间的回归文案
    regType = "Logit"
    coef = res_js['coeff'][argu1[0]]
    p = res_js['pvalue'][argu1[0]]
    desc = regResult(argu2, argu1[0], regType, coef, p)
    return {"argu_i": argu2, "Result": res_js, "desc": desc}

def logit_repeat(request: HttpRequest) -> JsonResponse:
    try:
        args = request_analyse(request)
        dta = get_file_data(request)
        count = args['argu1']['count']
        logit_args = args['argu1']['argus']
        logit_result = []
        argu_il = set(logit_args[0]['argu_i'])
        argu_el = set(logit_args[0]['argu_e'])

        # 针对操作的变量, 进行缺失值删减
        temp = []
        for i in range(0, count):
            for j in logit_args[i]['argu_e']:
                temp.append(j)
            temp.append(logit_args[i]['argu_i'])
        dta = argues_loss_delete(dta, temp)

        for i in range(0, count):
            logit_result.append(logit(dta,
                                           logit_args[i]['argu_e'],
                                           logit_args[i]['argu_i']))
            argu_il = argu_il.union(logit_args[i]['argu_i'])
            argu_el = argu_el.union(logit_args[i]['argu_e'])

        # 解释变量与被解释变量之间的回归分析文案
        desc = ''
        for val in logit_result:
            desc = desc + val['desc'] + '<br/>'

        ret_s = {"count": len(logit_result),  # 计数
                 "OLSList": logit_result,  # 回归结果
                 "ArgeList": list(argu_el),  # 参数的并集
                 "Desc": desc
                 }
        ret_df = smols2excelV3(ret_s)
        ret_uid = put_file_excel(ret_df, False)  # 输出索引
        ret_s['File'] = {"uid": ret_uid, "f_suffix": ".xlsx"}  # 文件列表
        return ret_success(ret_s)
    except Exception as e:
        traceback.print_exc()
        return ret_error(e)

def logit_effect_repeat(request: HttpRequest) -> JsonResponse:
    try:
        args = request_analyse(request)
        dta = get_file_data(request)
        count = args['argu1']['count']
        logit_args = args['argu1']['argus']
        logit_result = []
        argu_il = set(logit_args[0]['argu_i'])
        argu_el = set(logit_args[0]['argu_e'])
        entity = args['argu1']['argue'][0]  # 用户选择的个体变量
        time = args['argu1']['argue'][1]  # 用户选择的时间变量

        # 针对操作的变量, 进行缺失值删减
        temp = [entity, time]
        for i in range(0, count):
            for j in logit_args[i]['argu_e']:
                temp.append(j)
            temp.append(logit_args[i]['argu_i'])
        dta = argues_loss_delete(dta, temp)

        for i in range(0, count):
            logit_result.append(effect_logit(dta,
                                logit_args[i]['argu_e'],  # 解释变量
                                logit_args[i]['argu_i'],  # 被解释变量
                                logit_args[i]['entity_effect'],  # 个体固定效应(Bool)
                                logit_args[i]['time_effect'],   # 时间固定效应
                                entity,
                                time
                                ))
            argu_il = argu_il.union(logit_args[i]['argu_i'])
            argu_el = argu_el.union(logit_args[i]['argu_e'])

        # 解释变量与被解释变量之间的回归分析文案
        desc = ''
        for val in logit_result:
            desc = desc + val['desc'] + '<br/>'

        ret_s = {"count": len(logit_result),  # 计数
                 "OLSList": logit_result,  # 被解释变量
                 "ArgeList": list(argu_el),  # 参数的并集
                 "Desc": desc
                 }
        ret_df = smols2excelV3(ret_s)
        ret_uid = put_file_excel(ret_df, False)
        ret_s['File'] = {"uid": ret_uid, "f_suffix": ".xlsx"}
        return ret_success(ret_s)
    except Exception as e:
        traceback.print_exc()
        return ret_error(e)

def probit(dta, argu1, argu2):
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(dta[argu2])
    x = sm.add_constant(dta[argu1])
    model = sm.Probit(y, x)
    results = model.fit()
    PseudoR2 = re.findall("Pseudo R-squ.*", str(results.summary()))[0].split(" ")[-1]
    LogLikelihood = re.findall("Log-Likelihood:.*", str(results.summary()))[0].split(" ")[-1]

    pvalue = results.pvalues
    coeff = results.params
    std_err = results.bse
    res_df = pd.DataFrame({  # 从回归结果中提取需要的结果
        "pvalue": pvalue,
        "coeff": coeff,
        "std_err": std_err,
    })
    res_js = json.loads(res_df.to_json())
    res_js['n'] = dta.shape[0]
    res_js['LogLikelihood'] = LogLikelihood
    res_js['r2'] = PseudoR2

    # 生成解释变量和被解释变量之间的回归文案
    regType = "Probit"
    coef = res_js['coeff'][argu1[0]]
    p = res_js['pvalue'][argu1[0]]
    desc = regResult(argu2, argu1[0], regType, coef, p)
    return {"argu_i": argu2, "Result": res_js, "desc": desc}

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

    # 生成解释变量和被解释变量之间的回归文案
    regType = "Probit"
    coef = res_js['coeff'][argu1[0]]
    p = res_js['pvalue'][argu1[0]]
    desc = regResult(argu2, argu1[0], regType, coef, p)
    return {"argu_i": argu2, "Result": res_final, "desc": desc}

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

    # 生成解释变量和被解释变量之间的回归文案
    regType = "Logit"
    coef = res_js['coeff'][argu1[0]]
    p = res_js['pvalue'][argu1[0]]
    desc = regResult(argu2, argu1[0], regType, coef, p)
    return {"argu_i": argu2, "Result": res_final, "desc": desc}


def probit_repeat(request: HttpRequest) -> JsonResponse:
    try:
        args = request_analyse(request)
        dta = get_file_data(request)
        count = args['argu1']['count']
        probit_args = args['argu1']['argus']
        probit_result = []
        argu_il = set(probit_args[0]['argu_i'])
        argu_el = set(probit_args[0]['argu_e'])

        # 针对操作的变量, 进行缺失值删减
        temp = []
        for i in range(0, count):
            for j in probit_args[i]['argu_e']:
                temp.append(j)
            temp.append(probit_args[i]['argu_i'])
        dta = argues_loss_delete(dta, temp)

        for i in range(0, count):
            probit_result.append(probit(dta,
                                           probit_args[i]['argu_e'],
                                           probit_args[i]['argu_i']))
            argu_il = argu_il.union(probit_args[i]['argu_i'])
            argu_el = argu_el.union(probit_args[i]['argu_e'])

        # 解释变量与被解释变量之间的回归分析文案
        desc = ''
        for val in probit_result:
            desc = desc + val['desc'] + '<br/>'

        ret_s = {"count": len(probit_result),  # 计数
                 "OLSList": probit_result,  # 回归结果
                 "ArgeList": list(argu_el),  # 参数的并集
                 "Desc": desc
                 }
        ret_df = smols2excelV3(ret_s)
        ret_uid = put_file_excel(ret_df, False)  # 输出索引
        ret_s['File'] = {"uid": ret_uid, "f_suffix": ".xlsx"}  # 文件列表
        return ret_success(ret_s)
    except Exception as e:
        traceback.print_exc()
        return ret_error(e)
def probit_effect_repeat(request: HttpRequest) -> JsonResponse:
    try:
        args = request_analyse(request)
        dta = get_file_data(request)
        count = args['argu1']['count']
        probit_args = args['argu1']['argus']
        probit_result = []
        argu_il = set(probit_args[0]['argu_i'])
        argu_el = set(probit_args[0]['argu_e'])
        entity = args['argu1']['argue'][0]  # 用户选择的个体变量
        time = args['argu1']['argue'][1]  # 用户选择的时间变量

        # 针对操作的变量, 进行缺失值删减
        temp = [entity, time]
        for i in range(0, count):
            for j in probit_args[i]['argu_e']:
                temp.append(j)
            temp.append(probit_args[i]['argu_i'])
        dta = argues_loss_delete(dta, temp)

        for i in range(0, count):
            probit_result.append(effect_probit(dta,
                                probit_args[i]['argu_e'],  # 解释变量
                                probit_args[i]['argu_i'],  # 被解释变量
                                probit_args[i]['entity_effect'],  # 个体固定效应(Bool)
                                probit_args[i]['time_effect'],   # 时间固定效应
                                entity,
                                time
                                ))
            argu_il = argu_il.union(probit_args[i]['argu_i'])
            argu_el = argu_el.union(probit_args[i]['argu_e'])

        # 解释变量与被解释变量之间的回归分析文案
        desc = ''
        for val in probit_result:
            desc = desc + val['desc'] + '<br/>'
        ret_s = {"count": len(probit_result),  # 计数
                 "OLSList": probit_result,  # 被解释变量
                 "ArgeList": list(argu_el),  # 参数的并集
                 "Desc": desc}
        ret_df = smols2excelV3(ret_s)
        ret_uid = put_file_excel(ret_df, False)
        ret_s['File'] = {"uid": ret_uid, "f_suffix": ".xlsx"}
        return ret_success(ret_s)
    except Exception as e:
        traceback.print_exc()
        return ret_error(e)

def regResult(Y_Name, X_Name, regType, coef, p):
    if ((coef == None) or (p == None)) :
        return ""
    base =  f"{regType}的回归结果显示，[{X_Name}]与[{Y_Name}]的相关性系数是{'%.3f' % coef}，对应的P值是{'%.3f' % p}。"
    if p > 0.1:
        significant = "不显著"
    elif 0.05 < p <= 0.1:
        significant = "在10%的显著性水平上显著"
    elif 0.01 < p <= 0.05:
        significant = "在5%的显著性水平上显著"
    else:
        significant = "在1%的显著性水平上显著"

    if coef > 0:
        corr = "正相关"
    elif coef < 0:
        corr = "负相关"
    else:
        corr = "不存在明显的影响"

    return f"{base},{corr},{significant}，但是，需要考虑是否有一些同时和X以及Y相关的变量没有被纳入回归方程，此时可能存在遗漏变量问题，建议参考有关遗漏变量问题的解决方法（如工具变量，准自然实验，面板数据使用固定效应等）。此外，还应该考虑核心解释变量有没有可能被Y所影响，如果有可能，则可能存在反向因果，可以考虑工具变量回归或者准自然实验的方法来解决这一问题。"
