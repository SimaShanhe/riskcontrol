import pandas as pd
from tqdm import tqdm
import warnings
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import logging
from numpy import inf
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def is_numeric_dtype(lst):
    if len(lst) == 0:
        return True
    else:
        s = list(lst)[0]
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def bins_sorted(bins):
    value_type = 'char'
    for item in bins:
        if item != 'null' and '(' in item:
            value_type = 'numerical'
        break
    points = []
    if value_type == 'numerical' and len(bins) > 1:
        for idx, item in enumerate(bins):
            if item != 'null':
                item = item.split(',')
                left = item[0][1:]
                if 'inf' in left:
                    points.append(-np.inf)
                else:
                    points.append(float(left))


            else:
                points.append(np.inf)
        df = pd.DataFrame()
        df['cut_points'] = bins
        df['points'] = points
    else:
        return []
    return df


def to_percent(temp, position):
    return '%2d' % (100 * temp) + '%'


def check_data_y(data):
    '''
    检查数据结构，数据预测变量为0,1，并以“y”命名
    '''

    if 'y' not in data.columns:
        logging.ERROR('未检测到"y"变量，请将预测变量命名改为"y"')


def get_descison_tree_cut_point(data, col, criterion='entropy', max_depth=None, max_leaf_nodes=4,
                                min_samples_leaf=0.05):
    data_notnull = data[[col, 'y']][data[col].notnull()]  # 删除空值
    cut_point = []
    if len(np.unique(data_notnull[col])) > 1:
        x = data_notnull[col].values.reshape(-1, 1)
        y = data_notnull['y'].values

        clf = DecisionTreeClassifier(criterion=criterion,  # “信息熵”最小化准则划分
            max_depth=max_depth,  # 树的深度
            max_leaf_nodes=max_leaf_nodes,  # 最大叶子节点数
            min_samples_leaf=min_samples_leaf)  # 叶子节点样本数量最小占比
        clf.fit(x, y)  # 训练决策树

        threshold = np.unique(clf.tree_.threshold)
        x_num = np.unique(x)

        for i in threshold:
            if i != -2:
                point = np.round(max(x_num[x_num < i]), 2)  # 取切分点左边的数
                if point not in cut_point:
                    cut_point.extend([point])
        cut_point = [float(str(i)) for i in cut_point]
        cut_point = [-inf] + cut_point + [inf]
    return cut_point


def get_col_continuous_cut_points(data, col, criterion='entropy', max_depth=None, max_leaf_nodes=8,
                                  min_samples_leaf=0.05):
    if col != 'y':
        point = get_descison_tree_cut_point(data[[col, 'y']], col, criterion=criterion, max_depth=max_depth,
            max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf)
        return point
    else:
        warnings.warn("变量名称为{}与目标变量冲突".format(col))
        return


def feature_iv(data, col, criterion='entropy', max_depth=8, max_leaf_nodes=8, min_samples_leaf=0.05):
    """
    输出一个基于决策树的最优分箱统计结果，包括各个分箱的iv值
    :param data:
    :param col:
    :return:
    """
    df = pd.DataFrame()
    df['y'] = data['y']

    col_cut_points = set(data[data[col].notnull()][col].values)

    if is_numeric_dtype(col_cut_points):
        data[col] = data[col].astype(float)

    if not is_numeric_dtype(col_cut_points) and len(col_cut_points) > 100:
        raise ValueError('{} is non-numeric variables with many kinds of value,Please check it!'.format(col))

    if is_numeric_dtype(col_cut_points) and len(col_cut_points) > 2:
        cut_points = get_col_continuous_cut_points(data, col, criterion, max_depth, max_leaf_nodes, min_samples_leaf)
        df[col] = pd.cut(data[col], cut_points).astype("str")
    elif is_numeric_dtype(col_cut_points) and len(col_cut_points) <= 2:
        df[col] = data[col].astype("str")
    else:
        df[col] = data[col]
    df = df.fillna('null')
    df.replace('nan', 'null', inplace=True)

    result = df.groupby(col)['y'].agg([('1_num', lambda y: (y == 1).sum()),
                                       ('0_num', lambda y: (y == 0).sum()),
                                       ('total_num', 'count')]).reset_index()
    result['1_pct'] = result['1_num'] / result['1_num'].sum()
    result['0_pct'] = result['0_num'] / result['0_num'].sum()
    result['total_pct'] = result['total_num'] / result['total_num'].sum()
    result['1_rate'] = result['1_num'] / result['total_num']
    result['woe'] = np.log(result['1_pct'] / result['0_pct'])  # WOE
    result['iv'] = (result['1_pct'] - result['0_pct']) * result['woe']  # IV
    result.replace([-inf, inf], [0, 0], inplace=True)
    result['total_iv'] = result['iv'].sum()
    # result.replace([-inf, inf], [0, 0], inplace=True)
    result = result.rename(columns={col: "cut_points"})

    # sorted
    sort = bins_sorted(result['cut_points'])
    if len(sort):
        result = pd.merge(result, sort, on='cut_points', how='left')
        result = result.sort_values(by="points", ascending=True)
        del result['points']

    result = result.reset_index()
    del result['index']

    return result


def split_box_plot_new(data: pd.DataFrame, col: str, save_path: str = '', criterion='entropy', max_depth=8,
                       max_leaf_nodes=8, min_samples_leaf=0.05):
    result = feature_iv(data, col, criterion, max_depth, max_leaf_nodes, min_samples_leaf)

    # risk corr statistic
    woe_table_no_null = result[result['cut_points'] != 'null']
    woe_table_no_null['order'] = [i for i in range(len(woe_table_no_null))]
    col_woe_bad_rate = woe_table_no_null[['order', '1_rate']]
    risk_corr = round(col_woe_bad_rate.corr('spearman').values[0][1], 4)
    if risk_corr < 0:
        factor_impact = '正因子'
    else:
        factor_impact = '负因子'

    if abs(risk_corr) >= 0.8:
        risk_monotonicity = '强'
    elif abs(risk_corr) >= 0.6:
        risk_monotonicity = '弱'
    else:
        risk_monotonicity = '无'
        factor_impact = '无'

    xlabels = result['cut_points'].values

    badrate = [a / b for a, b in zip(result['1_num'].values, result['total_num'].values)]
    people_rate = result['total_pct'].values

    fig, ax1 = plt.subplots(figsize=(16, 9), facecolor='white')
    plt.title("【{0}】逾期统计直方图".format(col), fontsize=20, fontweight='heavy')
    plt.title('iv = {0},{1}线性, {2}'.format(round(result['iv'].sum(), 4), risk_monotonicity, factor_impact),
        loc='right', bbox=dict(facecolor='y', edgecolor='blue', alpha=0.65), fontsize=14)
    ax1.set_xlabel("特征【{}】的最优划分区间".format(col), fontsize=14)
    ax1.set_ylabel("会员占比", fontsize=14)

    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))

    plt.rcParams['font.sans-serif'] = ['SimHei']
    ax1.bar(xlabels, people_rate, color='steelblue')
    for x, p in zip(xlabels, people_rate):  # 数据标签
        ax1.text(x, p + 0.001, '{:.2f}%'.format(round(p * 100, 2)), ha='center', va='bottom', fontsize=12)

    ax2 = ax1.twinx()
    ax2.plot(xlabels, badrate, '-or', label='首逾30+率', linestyle='dashdot', linewidth=2)
    ax2.set_ylabel('逾期率', fontsize=14)
    ax2.axhline(y=len(data[data['y'] == 1]) / len(data), color='b', linestyle='-.', linewidth=2)

    plt.text(0.78, 0.97, '{}条样本平均y值率：{:.2f}%'.format(len(data), round(len(data[data['y'] == 1]) / len(data) * 100, 2)),
        transform=ax2.transAxes, fontsize=14, color='blue')

    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    for x, b in zip(xlabels, badrate):  # 数据标签
        plt.text(x, b + 0.001, '{:.2f}%'.format(round(b * 100, 2)), ha='center', va='bottom', fontsize=14, color='red')

    if len(save_path) > 0:
        plt.savefig(save_path + r'\pic\{}_iv_bar.png'.format(col))
    else:
        plt.show()


def feature_miss_ana(df, criterion='entropy', max_depth=8, max_leaf_nodes=8, min_samples_leaf=0.05):
    overdue_r = len(df[df['y'] == 1]) / len(df)
    describe = []

    for col in tqdm(list(df)):
        col_type = df.dtypes[col]
        n_all = len(df)
        n_miss = len(df[df[col].isnull()])
        n_notnull = n_all - n_miss
        n_bad_notnull = len(df[(df[col].notnull()) & (df['y'] == 1)])
        n_bad = len(df[(df[col].isnull()) & (df['y'] == 1)])

        try:
            woe_table = feature_iv(df, col, criterion, max_depth, max_leaf_nodes, min_samples_leaf)
            iv = round(woe_table['total_iv'].values[0], 4)
            max_risk_value_id = woe_table['1_rate'].idxmax()
            # max bad rate cut point
            if max_risk_value_id == 0:
                max_risk_value = '小'
            elif max_risk_value_id == len(woe_table) - 1:
                if woe_table.loc[max_risk_value_id]['cut_points'] == 'null':
                    max_risk_value = 'null'
                else:
                    max_risk_value = '大'
            else:
                if 'null' in woe_table['cut_points'] and max_risk_value_id == len(woe_table) - 2:
                    max_risk_value = '大'
                else:
                    max_risk_value = '中间'

            # Spear-man corr
            woe_table_no_null = woe_table[woe_table['cut_points'] != 'null']
            woe_table_no_null['order'] = [i for i in range(len(woe_table_no_null))]
            col_woe_bad_rate = woe_table_no_null[['order', '1_rate']]
            risk_corr = col_woe_bad_rate.corr('spearman').values[0][1]

            if risk_corr < 0:
                factor_impact = '正因子'
            else:
                factor_impact = '负因子'

            if abs(risk_corr) >= 0.8:
                risk_monotonicity = '强'
            elif abs(risk_corr) >= 0.6:
                risk_monotonicity = '弱'
            else:
                risk_monotonicity = '无'
                factor_impact = '无'

        except Exception as e:
            print(e)
            iv = None
            risk_monotonicity = None
            factor_impact = None
            max_risk_value = None

        if n_miss == 0:
            miss_overdue_r = None
        else:
            miss_overdue_r = round(n_bad / n_miss, 4)

        if n_notnull == 0:
            notnull_overdue_r = None
        else:
            notnull_overdue_r = round(n_bad_notnull / n_notnull, 4)

        describe.append([col, col_type, iv, n_all, n_miss, round(n_miss / n_all, 4), n_bad, miss_overdue_r, n_notnull,
                         n_bad_notnull, notnull_overdue_r, overdue_r, risk_monotonicity, factor_impact, max_risk_value])

    describe_df = pd.DataFrame(describe,
        columns=['col', 'col_type', 'iv', 'sample_n', 'miss_n', 'miss_r', 'miss_n_bad', 'miss_overdue_r',
                 'n_notnull', 'n_bad_notnull', 'notnull_overdue_r', 'sample_overdue_r', 'risk_monotonicity',
                 'factor_impact', 'max_risk_value'])
    return describe_df


