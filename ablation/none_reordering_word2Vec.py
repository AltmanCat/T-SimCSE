# -*- codeing = utf-8 -*-
# 作者——wwq
# 时间： 2025/3/25 11:13
import operator
import math
import metrics
import file_tool_other
from gensim.models.keyedvectors import KeyedVectors
from WocSim import WocSim
import pandas as pd
import numpy as np
import time
from itertools import combinations
import multiprocessing as mp
# 无重排Word2Vec
# RESULT_XML = "../EasyClinic/oracle/UC_TC.txt"
# RESULT_XML = "../GANNT/AnswerSetHighToLow.txt"
RESULT_XML = "../IceBreaker/Requirements2ClassMatrix.txt"
# RESULT_XML = "../dronology/req-dd.txt"


# TA_ADDRESS = "../EasyClinic/3 - test cases/"
# TA_ADDRESS = "../GANNT/low/"
TA_ADDRESS = "../IceBreaker/requirements.txt"
# TA_ADDRESS = "../dronology/design_definition/"

# SA_ADDRESS = "../EasyClinic/1 - use cases/"
# SA_ADDRESS = "../GANNT/high/"
SA_ADDRESS = "../IceBreaker/ClassDiagram.txt"
# SA_ADDRESS = "../dronology/req/"


GENERATOR_EXCEL = "word2Vec_IceBreaker_none_reordering.xlsx"
MODEL_PATH = '../Word2Vec/data/GoogleNews-vectors-negative300.bin'
STOPWORDS_PATH = "../Word2Vec/data/stopwords_en.txt"
PROCRSSES_NUM = 1  # 进程数

# 参数配置
start = 0.01
end = 1.0
step1 = 0.01
step2 = 0.01

# 初始化NLP模型
model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)
stopwords = []
with open(STOPWORDS_PATH, 'r') as fh:
    stopwords = fh.read().split(",")
ds = WocSim(model, stopwords=stopwords)

# 数据准备
answer = file_tool_other.parse_file(RESULT_XML)
print(answer)
targets, targetsName = file_tool_other.parse_file_SorT(TA_ADDRESS)
sources, sourcesName = file_tool_other.parse_file_SorT(SA_ADDRESS)


def save_to_excel(data):
    df = pd.DataFrame(data, columns=["map", "Precision", "Recall", "f1", "f2","PR"]) # 创建 DataFrame
    df.to_excel(GENERATOR_EXCEL, index=False)
    print(f"数据已成功保存到 {GENERATOR_EXCEL}")

def calculate_ta_sim(args):
    i, j, targets = args
    return i, j, ds.calculate_similarity(targets[i], targets[j])

k1set = [round(start + i*step1, 2) for i in range(int((end-start)/step1) + 1)]
k2set = [round(start + i*step2, 2) for i in range(int((end-start)/step2) + 1)]

dict_idf = {}
data = []

def precompute_ta_sim_matrix(targets):
    print("预计算TA相似度矩阵...")
    indices = list(combinations(range(len(targets)), 2))
    with mp.Pool(processes=PROCRSSES_NUM) as pool:
        results = pool.map(
            calculate_ta_sim,
            [(i, j, targets) for i, j in indices]
        )
    ta_sim_matrix = np.eye(len(targets))
    for i, j, sim in results:
        ta_sim_matrix[i, j] = sim
        ta_sim_matrix[j, i] = sim
    return ta_sim_matrix;

def vectorized_sim(source_text):
    return np.array([ds.calculate_similarity(source_text, ta) for ta in targets])

def precompute_sa_ta_sim_matrix(sources):
    print("预计算SA-TA相似度矩阵...")
    with mp.Pool(processes=PROCRSSES_NUM) as pool:
        sa_ta_sim_rows = pool.map(vectorized_sim, sources)
    sa_ta_sim_matrix = np.vstack(sa_ta_sim_rows)
    return sa_ta_sim_matrix


# 创建名称到索引的映射
ta_name_to_idx = {name: idx for idx, name in enumerate(targetsName)}
sa_name_to_idx = {name: idx for idx, name in enumerate(sourcesName)}
def main_none_reordering(ta_sim_matrix,sa_ta_sim_matrix,answer,sources,targets,sourcesName,targetsName,ta_name_to_idx,sa_name_to_idx):
    ap, f1, f2 = 0, 0, 0
    averagePrecision, averageRecall = 0, 0
    averagePrecision_Recall = [0 for i in range(0, 10)]
    recall = [i / 10 for i in range(1, 11)]
    ans = 0
    for i in range(0, len(sources)):
        sa_idx = sa_name_to_idx[sourcesName[i]]
        dict = {targetsName[j]: sa_ta_sim_matrix[sa_idx][j] for j in range(len(targets))}
        s_t_sim_list = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
        if answer.get(sourcesName[i]):
            ans += 1
            ap = metrics.map(sourcesName, answer, i, ap, s_t_sim_list)
            p1 = metrics.precision(answer, sourcesName, i, s_t_sim_list)
            averagePrecision += p1
            r1 = metrics.recall(answer, sourcesName, i, s_t_sim_list)
            averageRecall += r1
            averagePrecision_Recall = metrics.precision_recall(answer, sourcesName, recall, i, averagePrecision_Recall,
                                                               s_t_sim_list)
    for i in range(0, 10):
        averagePrecision_Recall[i] = averagePrecision_Recall[i] / ans
    data_f = [
        ap / ans,
        averagePrecision / ans,
        averageRecall / ans,
        metrics.fn(1, averagePrecision / ans, averageRecall / ans),
        metrics.fn(2, averagePrecision / ans, averageRecall / ans),
        averagePrecision_Recall
    ]
    return data_f

if __name__ == '__main__':
    # 记录程序开始时间
    start_time = time.perf_counter()
    # 生成所有参数组合
    ta_sim_matrix = precompute_ta_sim_matrix(targets)
    sa_ta_sim_matrix = precompute_sa_ta_sim_matrix(sources)
    # 添加检查
    if ta_sim_matrix is None:
        raise ValueError("ta_sim_matrix 未被正确计算")
    if sa_ta_sim_matrix is None:
        raise ValueError("sa_ta_sim_matrix 未被正确计算")

    print(f"ta_sim_matrix 类型: {type(ta_sim_matrix)}, 形状: {ta_sim_matrix.shape}")
    print(f"sa_ta_sim_matrix 类型: {type(sa_ta_sim_matrix)}, 形状: {sa_ta_sim_matrix.shape}")

    results = main_none_reordering(ta_sim_matrix, sa_ta_sim_matrix, answer, sources, targets, sourcesName, targetsName,
                                   ta_name_to_idx, sa_name_to_idx)
    data.append(results)
    save_to_excel(data)
    # 计算总耗时
    total_time = time.perf_counter() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTotal execution time: {int(hours):0>2}h {int(minutes):0>2}m {seconds:05.2f}s")