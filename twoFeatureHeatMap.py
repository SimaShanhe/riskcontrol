import seaborn as sns
from .compute_iv import *
import datetime
from prettytable import PrettyTable



def two_feature_pivot_table(data, col1, col2, col1_binnum=10, col2_binnum=10):
    """
    数据离散化（默认10等分）计算各个离散后组合的目标指标均值
    :param data: 原始数据data-frame
    :param col1_binnum: 特征1的分箱数
    :param col2_binnum: 特征2的分箱数
    :return:未排序的数据透视表data-frame
    """
    if 'y' not in data.columns.tolist():
        raise ValueError('columns do not contain y!')
    col1_cut_points = set(data[data[col1].notnull()][col1].values)
    col2_cut_points = set(data[data[col2].notnull()][col2].values)
    df = pd.DataFrame()
    df['y'] = data['y']

    col1_dtype = is_numeric_dtype(data[col1].values.tolist())
    col2_dtype = is_numeric_dtype(data[col2].values.tolist())

    if col1_dtype and len(col1_cut_points) > 10:
        col1_values = data[data[col1].notnull()][col1].values
        cut1_points = [round(np.percentile(col1_values, i * 100 / col1_binnum), 1) for i in range(col1_binnum)]
        cut1_points = sorted(list(set(cut1_points)))
        cut1_points[0] = -np.inf
        cut1_points.append(np.inf)
        df[col1] = pd.cut(data[col1], cut1_points).astype("str")
    elif col1_dtype and len(col1_cut_points) <= 10:
        df[col1] = data[col1].astype("str")
    else:
        df[col1] = data[col1]

    if col2_dtype and len(col2_cut_points) > 10:
        col2_values = data[data[col2].notnull()][col2].values
        cut2_points = [round(np.percentile(col2_values, i * 100 / col2_binnum), 1) for i in range(col2_binnum)]
        cut2_points = sorted(list(set(cut2_points)))
        cut2_points[0] = -np.inf
        cut2_points.append(np.inf)
        df[col2] = pd.cut(data[col2], cut2_points).astype("str")
    elif col2_dtype and len(col2_cut_points) <= 10:
        df[col2] = data[col2].astype("str")
    else:
        df[col2] = data[col2]

    df = df.fillna('null')
    df.replace('nan', 'null', inplace=True)
    df = df.pivot_table(index=col1, columns=col2, aggfunc=['count', np.mean])
    df = df.fillna(0)
    return df


def pivot_table_sorted(df):
    # df = df.fillna(0)

    index = df.index.tolist()
    columns = df.columns.get_level_values(1).tolist()
    df = pd.DataFrame(df.values, columns=columns)
    df['index'] = index
    df = df.set_index('index')

    for i in range(2):

        if i == 1:
            columns = df.index.tolist()
            index = df.columns.tolist()
            arr = np.transpose(df.values)
            df = pd.DataFrame(arr, columns=columns, index=index)
        index_lst = df.index.tolist()

        df['cut_points'] = index_lst

        flag = 0
        binsdf = bins_sorted(index_lst)
        for item in index_lst:
            if '(' in item:
                flag = 1
                break
        if flag and len(binsdf)>0:
            df = pd.merge(df, binsdf, on='cut_points', how='left')
            df = df.sort_values(by="points", ascending=True)
            index_lst = df['cut_points'].values
            del df['points']
        else:
            df = df.sort_values(by="cut_points", ascending=True)
            index_lst = df['cut_points'].values
        del df['cut_points']
        df['index'] = index_lst
        df = df.set_index('index')
    return df


def top3_of_matrix(df):
    df_count = df['count']
    df_mean = df['mean']
    mat_mean = df_mean.as_matrix()
    mat_count = df_count.as_matrix()
    v = []
    for i in range(3):
        raw, column = mat_mean.shape
        positon = np.argmax(mat_mean)
        m, n = divmod(positon, column)
        v.append([mat_count[m,n],mat_mean[m,n]])
        mat_mean[m,n]=0
    return v


def interval_counts(df):
    df = df['count']
    df = pivot_table_sorted(df)
    df = df.fillna(0)
    x = [sum(df[x]) for x in list(df)]
    datasum = sum(x)
    xv = [round(sum(df[x])/datasum,3) for x in list(df)]
    yv = list(df.apply(lambda x:round( x.sum()/datasum,3), axis=1).values)
    return xv,yv



def matrix_heatmap(df, xlabels=[], ylabels=[], col1='', col2='', data_ori=pd.DataFrame(),top3=[],interval_counts=[]):
    if len(ylabels) > 0:
        df['idx'] = df.index.tolist()
        df['idx'] = df['idx'].replace(ylabels, list(range(len(ylabels))))
        df.sort_values('idx', inplace=True)
        del df['idx']

    result = np.array(df)
    fig = plt.figure(figsize=(16,9))
    left, bottom, width, height = 0.03, 0.05, 0.93, 0.91
    ax = fig.add_axes([left, bottom, width, height])
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.title("{}&{}交叉矩阵热力图".format(col1,col2), fontsize=16)
    sns.heatmap(result, annot=True, annot_kws={'size': 12, 'weight': 'bold', 'color': 'red'}, linewidths=.5,
                cmap=plt.cm.Blues)
    if len(ylabels) == 0:
        ylabels = df.index.tolist()
    if len(xlabels) == 0:
        xlabels = df.columns.tolist()
    plt.yticks([i + 0.5 for i in range(len(ylabels))], labels=ylabels, fontsize=8)
    plt.xticks([i + 0.5 for i in range(len(xlabels))], labels=xlabels, fontsize=8)
    plt.xlabel(col1,fontsize=14)
    plt.ylabel(col2,fontsize=14)
    if len(data_ori):
        df1 = data_ori[data_ori[col1].notna()]
        df2 = data_ori[data_ori[col2].notna()]
        col1_missing = 1-len(df1)/len(data_ori)
        col2_missing = 1-len(df2) / len(data_ori)
        text = '{}条样本\n总体y值率：{:.2f}%\n横轴NAN:{:.2f}%\n纵轴NAN:{:.2f}%'.format(len(data_ori),
                           round(len(data_ori[data_ori['y'] == 1]) / len(data_ori) * 100, 2),
                           round(col1_missing * 100, 2),
                           round(col2_missing * 100, 2))
        ax.text(1.15, 0.74, text,alpha=0.8, fontsize=14,transform=ax.transAxes,color='black',
                bbox = dict(facecolor = "r", alpha = 0.4))


    if len(top3)>0:
        x = PrettyTable(["Top", "Yr", "Pr"])
        x.add_row(["1", "{:.3f}%".format(round(top3[0][1]*100, 3)), "{:.2f}%".format(round(top3[0][0]*100/len(data_ori), 2))])
        x.add_row(["2", "{:.3f}%".format(round(top3[1][1]*100, 3)), "{:.2f}%".format(round(top3[1][0]*100/len(data_ori), 2))])
        x.add_row(["3", "{:.3f}%".format(round(top3[2][1]*100, 3)), "{:.2f}%".format(round(top3[2][0]*100/len(data_ori), 2))])
        y = str(x)
        ax.text(1.15, 0.58, y,alpha=0.8, fontsize=9,transform=ax.transAxes,color='black',
                bbox = dict(facecolor = "b", alpha = 0.4))


    cols = []
    for col in [col1,col2]:
        cols.append(col[:11])
    infodesc = 'Date: {}\n横轴:{}\n纵轴:{}'\
        .format(datetime.datetime.now().strftime('%Y-%m-%d'),cols[0],cols[1])
    ax.text(1.15, 0.89, infodesc,
            alpha=0.8, fontsize=14, transform=ax.transAxes, color='black',
            bbox=dict(facecolor="gray", alpha=0.4))

    ax.text(1.18, 1, '仪表盘',alpha=0.8, fontsize=14, transform=ax.transAxes,va='center')
    # ax.text(1.15, 0.02, '@猪油教缪晓峰', alpha=0.2, fontsize=14, transform=ax.transAxes, va='center')

    # 添加子图1
    left, bottom, width, height = 0.891, 0.38, 0.09, 0.15
    ax2 = fig.add_axes([left, bottom, width, height])
    y2 = interval_counts[0]
    x2 = list(range(len(y2)))
    plt.plot(x2,y2)
    ax2.set_title(col1[:18])


    # 添加子图2
    left, bottom, width, height = 0.891, 0.16, 0.09, 0.15
    ax3 = fig.add_axes([left, bottom, width, height])
    y3 = interval_counts[1]
    x3 = list(range(len(y3)))
    plt.plot(x3,y3)
    ax3.set_title(col2[:18])


    plt.show()


def twoFeatureHeatmap(data, col1, col2, xlabels=[], ylabels=[],data_ori=pd.DataFrame()):
    df1 = two_feature_pivot_table(data, col1, col2)
    df = pivot_table_sorted(df1['mean'])
    v = top3_of_matrix(df1)
    xv,yv = interval_counts(df1)
    matrix_heatmap(df, col1=col1, col2=col2, xlabels=xlabels, ylabels=ylabels,data_ori=data_ori,top3=v
                   ,interval_counts=[xv,yv])








