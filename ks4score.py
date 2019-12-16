import pandas as pd
from tqdm import tqdm
import time
from matplotlib import pyplot as plt
import warnings
import seaborn as sns
import numpy as np

warnings.filterwarnings('ignore')


def ks_compute(df, feature_name, ks_plot=True, file_path='', savefig=True):
    """
    计算ks值，并可选择画图
    :param df: 输入一个data-frame类型
    :param ks_plot: 默认情况下展示ks曲线
    :return: 返回ks值
    """
    if 'y' not in df.columns.tolist():
        print("error:no variable called 'y'~~ ")
        return None

    # abnormal value processing
    df = df.dropna()
    for item in set(df[feature_name].values):
        try:
            complex(item)  # for int, long, float and complex
        except ValueError:
            df[feature_name] = df[feature_name].replace(item, np.nan)
    df = df.dropna()

    # value type translation
    df[feature_name] = df[feature_name].astype('float')
    df['y'] = df['y'].astype('float')

    # Extreme value processing
    abnormalval_big = np.percentile(np.array(df[feature_name].values), 99.9)
    abnormalval_litter = np.percentile(np.array(df[feature_name].values), 0.1)
    scorelst = sorted(
        set(df[(df[feature_name] < abnormalval_big) & (df[feature_name] > abnormalval_litter)][feature_name].values))
    kslst = []
    goodcum = []
    badcum = []

    print("ks值在计算中...")
    time.sleep(0.01)
    for item in tqdm(scorelst):
        goodrate = len(df[(df[feature_name] <= item) & (df['y'] == 0)]) / len(df[df['y'] == 0])
        badrate = len(df[(df[feature_name] <= item) & (df['y'] == 1)]) / len(df[df['y'] == 1])
        kslst.append(badrate - goodrate)
        if ks_plot:
            goodcum.append(goodrate)
            badcum.append(badrate)
    time.sleep(0.01)
    print("ks值为{:.6f}".format(max(kslst)))

    if ks_plot:
        plt.figure(figsize=(16, 9))
        plt.title("{}-KS cure".format(feature_name), fontsize=24)
        plt.xlabel("{}".format(feature_name), fontsize=14)
        plt.ylabel("Cumulative ratio", fontsize=14)
        plt.grid(color='r', linestyle='--', linewidth=1, alpha=0.3)
        plt.plot(scorelst, goodcum, label='Cumulative Good Rate')
        plt.plot(scorelst, badcum, label='Cumulative Bad Rate')
        plt.plot(scorelst, kslst, label='ks cure')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2,
            ncol=3, mode="expand", borderaxespad=0.)

        ind = kslst.index(max(kslst))
        plt.vlines(scorelst[ind], 0, max(kslst), color="red", linestyles='--')
        plt.text(scorelst[ind], max(kslst) + 0.2, r'ks={:.6}'.format(max(kslst)), fontsize=14)
        if savefig:
            try:
                plt.savefig(file_path + r'\pic\{}_ks_cure.png'.format(feature_name))
            except Exception as e:
                raise ("please make a dir in current path named 'pic'!!!")
        else:
            plt.show()

    return max(kslst)


def score_interval_describe(df, feature_name, file_path, bins=None):
    """
    用于统计模型分数分段指标和展示
    :param df: 包含y值的data-frame数据结构
    :param file_path: 结果保存路径
    :param bins: score切分点列表，默认为等距十分
    :return: 返回结果并且保存
    """

    if 'y' not in df.columns.tolist():
        print("error:no variable called 'y'~~ ")
        return

    # abnormal value processing
    df = df.dropna()
    for item in set(df[feature_name].values):
        try:
            complex(item)  # for int, long, float and complex
        except ValueError:
            df[feature_name] = df[feature_name].replace(item, np.nan)
    df = df.dropna()

    # value type translation
    df[feature_name] = df[feature_name].astype('float')
    df['y'] = df['y'].astype('float')

    if not bins:
        left = np.percentile(np.array(df[feature_name].values), 1)
        right = np.percentile(np.array(df[feature_name].values), 99)
        step = (right - left) / 8
        bins = [int(left + i * step) for i in range(9)]
    if max(df[feature_name].values) > bins[-1]:
        bins = bins + [int(max(df[feature_name].values)) + 1]

    interval_table = pd.DataFrame()
    score_interval = []
    member_num_interval = []
    overdue_num_interval = []
    member_rate_interval = []
    overdue_rate_interval = []
    good_mem_cum = []
    bad_mem_cum = []
    good_rate_cum = []
    bad_rate_cum = []
    ks_cum = []
    kpilst = []
    for ind, cutoff in enumerate(bins):
        if ind == 0:
            interval_left = int(min(df[feature_name].values)) - 1
        else:
            interval_left = bins[ind - 1]
        interval_right = cutoff

        interval = str(interval_left) + '-' + str(interval_right)
        member_num = len(df[(df[feature_name] <= interval_right) & (df[feature_name] > interval_left)])
        overdue_num = len(
            df[(df[feature_name] <= interval_right) & (df[feature_name] > interval_left) & (df['y'] == 1)])
        member_rate = member_num / len(df) if len(df) > 0 else 0
        overdue_rate = overdue_num / member_num if member_num > 0 else 0
        good_mem = len(df[(df[feature_name] <= interval_right) & (df['y'] == 0)])
        bad_mem = len(df[(df[feature_name] <= interval_right) & (df['y'] == 1)])
        good_rate = good_mem / len(df[df['y'] == 0]) if len(df[df['y'] == 0]) > 0 else 0
        bad_rate = bad_mem / len(df[df['y'] == 1]) if len(df[df['y'] == 1]) > 0 else 0
        ks_value = bad_rate - good_rate
        if len(df[df[feature_name] > interval_right]) > 0:
            kpi = len(df[(df[feature_name] > interval_right) & (df['y'] == 1)]) / len(
                df[df[feature_name] > interval_right])
        else:
            kpi = 0

        score_interval.append(interval)
        member_num_interval.append(member_num)
        overdue_num_interval.append(overdue_num)
        member_rate_interval.append(member_rate)
        overdue_rate_interval.append(overdue_rate)
        good_mem_cum.append(good_mem)
        bad_mem_cum.append(bad_mem)
        good_rate_cum.append(good_rate)
        bad_rate_cum.append(bad_rate)
        ks_cum.append(ks_value)
        kpilst.append(kpi)

    interval_table['分数区间'] = score_interval
    interval_table['会员数'] = member_num_interval
    interval_table['逾期会员数'] = overdue_num_interval
    interval_table['会员占比'] = member_rate_interval
    interval_table['区间逾期率'] = overdue_rate_interval
    interval_table['非逾期会员累计数'] = good_mem_cum
    interval_table['逾期会员累计数'] = bad_mem_cum
    interval_table['非逾期会员累计占比'] = good_rate_cum
    interval_table['逾期会员累计占比'] = bad_rate_cum
    interval_table['ks值'] = ks_cum
    interval_table['切分KPI'] = kpilst

    interval_table.to_csv(file_path + r'score_interval_statistics.csv', encoding='gbk')

    plt.figure(figsize=(16, 9))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title("{}区间逾期率统计直方图".format(feature_name), fontsize=24)
    sns.barplot(x=u'分数区间', y=u'区间逾期率', data=interval_table)
    plt.show()

    return interval_table
