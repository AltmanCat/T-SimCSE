# -*- codeing = utf-8 -*-
# 作者——wwq
# 时间： 2025/3/25 9:46
# -*- codeing = utf-8 -*-
# 作者——wwq
# 时间： 2025/3/24 20:08
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import time
from itertools import combinations
import multiprocessing as mp
import operator
import math
import metrics
import numpy as np
import file_tool
import os
import re
import torch
from typing import List



# LSI 完整版，选择最佳K1 K2适用
# RESULT_XML = "../EasyClinic/oracle/UC_TC.txt"
RESULT_XML = "../GANNT/AnswerSetHighToLow.txt"
# RESULT_XML = "../IceBreaker/Requirements2ClassMatrix.txt"
# RESULT_XML = "../dronology/req-dd.txt"


# TA_ADDRESS = "../EasyClinic/3 - test cases/"
TA_ADDRESS = "../GANNT/low/"
# TA_ADDRESS = "../IceBreaker/requirements.txt"
# TA_ADDRESS = "../dronology/design_definition/"

# SA_ADDRESS = "../EasyClinic/1 - use cases/"
SA_ADDRESS = "../GANNT/high/"
# SA_ADDRESS = "../IceBreaker/ClassDiagram.txt"
# SA_ADDRESS = "../dronology/req/"

PROCRSSES_NUM = 1  # 进程数

GENERATOR_EXCEL = "LSI_GANNT_k1_k2.xlsx"

# 参数配置
start = 0.01
end = 1.0
step1 = 0.01
step2 = 0.01

dict_idf = {}

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_lsi_similarities(
    target_req: str,
    source_reqs: list[str],
    n_components: int = 100
) -> list[float]:
    corpus = [target_req] + source_reqs
    vectorizer = TfidfVectorizer(
        stop_words='english',
        token_pattern=r"(?u)\b\w+(?:'\w+)?\b"
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    lsi_model = TruncatedSVD(n_components=n_components, random_state=42)
    lsi_vectors = lsi_model.fit_transform(tfidf_matrix)
    target_vector = lsi_vectors[0].reshape(1, -1)
    source_vectors = lsi_vectors[1:]
    similarities = cosine_similarity(target_vector, source_vectors)[0]
    return [round(s, 4) for s in similarities]

def save_to_excel(data):
    df = pd.DataFrame(data, columns=["k1","k2", "map", "Precision", "Recall", "f1", "f2","PR"]) # 创建 DataFrame
    df.to_excel(GENERATOR_EXCEL, index=False)
    print(f"数据已成功保存到 {GENERATOR_EXCEL}")

def calculate_ta_sim(args):
    i, targets = args
    return i, calculate_lsi_similarities(targets[i], targets)


def precompute_ta_sim_matrix(targets: list[str]) -> np.ndarray:
    print("预计算TA-TA相似度矩阵...")
    corpus = targets.copy()
    # TF-IDF向量化
    vectorizer = TfidfVectorizer(
        stop_words='english',
        token_pattern=r"(?u)\b\w+(?:'\w+)?\b"
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    n_components = min(100, len(corpus) - 1) if len(corpus) > 1 else 1
    lsi_model = TruncatedSVD(n_components=n_components, random_state=42)
    lsi_vectors = lsi_model.fit_transform(tfidf_matrix)
    similarity_matrix = cosine_similarity(lsi_vectors)
    np.fill_diagonal(similarity_matrix, 1.0)

    return similarity_matrix


def calculate_lsi_batch(args: tuple) -> List[float]:
    target, sources = args
    return calculate_lsi_similarities(target, sources)


def precompute_sa_ta_sim_matrix(
        sources: list[str],
        targets: list[str]
) -> np.ndarray:
    print("预计算SA-TA相似度矩阵...")
    full_corpus = sources + targets
    # TF-IDF向量化
    vectorizer = TfidfVectorizer(
        stop_words='english',
        token_pattern=r"(?u)\b\w+(?:'\w+)?\b"
    )
    tfidf_matrix = vectorizer.fit_transform(full_corpus)

    n_components = min(100, len(full_corpus) - 1) if len(full_corpus) > 1 else 1
    lsi_model = TruncatedSVD(n_components=n_components, random_state=42)
    lsi_vectors = lsi_model.fit_transform(tfidf_matrix)

    sa_vectors = lsi_vectors[:len(sources)]
    ta_vectors = lsi_vectors[len(sources):]

    return cosine_similarity(sa_vectors, ta_vectors)
def main_k1_k2(ta_sim_matrix,sa_ta_sim_matrix,k1,k2,answer,sources,targets,sourcesName,targetsName,ta_name_to_idx,sa_name_to_idx):
    print(f"k1: {k1}, k2: {k2}")
    idf_map = {}
    for i in range(0, len(targets)):
        i_dict = {targetsName[j]: ta_sim_matrix[i][j] for j in range(len(targets))}
        sim_list = sorted(i_dict.items(), key=operator.itemgetter(1), reverse=True)
        max_index = min(math.ceil(len(targets) * k1) + 1, len(sim_list))
        for j in range(0, max_index):
            if not idf_map.__contains__(sim_list[j][0]):
                idf_map[sim_list[j][0]] = 1
            else:
                idf_map[sim_list[j][0]] = idf_map.get(sim_list[j][0]) + 1
    for key in idf_map.keys():
        dict_idf[key] = math.log((len(targets)-1) / idf_map.get(key))
    ap, f1, f2 = 0, 0, 0
    averagePrecision, averageRecall = 0, 0
    averagePrecision_Recall = [0 for i in range(0, 10)]
    recall = [i / 10 for i in range(1, 11)]
    ans = 0
    for i in range(0, len(sources)):
        dont_sort = []
        label_text = {name: idx for idx, name in enumerate(targetsName)}
        sa_idx = sa_name_to_idx[sourcesName[i]]
        dict = {targetsName[j]: sa_ta_sim_matrix[sa_idx][j] for j in range(len(targets))}
        s_t_sim_list = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
        max_index_1 = min(math.ceil(len(targets) * k2), len(s_t_sim_list))
        for o in range(0, max_index_1):
            label = s_t_sim_list[o][0]
            dont_sort.append(label)
        new_dict = {}
        for o in range(0, max_index_1):
            sum_new = 0  # HPTA的Spec累加
            label = s_t_sim_list[o][0]
            index = label_text.get(label)
            for p in range(0, len(targets)):
                if index == p:
                    continue
                cosine_sim = ta_sim_matrix[index][p]
                new_dict[targetsName[p]] = cosine_sim
            new_list = sorted(new_dict.items(), key=operator.itemgetter(1), reverse=True)

            max_index_2 = min(round(len(new_list) * k1), len(new_list))
            for q in range(0, max_index_2):
                name = new_list[q][0]
                if dont_sort.__contains__(name) or dict.get(label) < dict.get(
                        name):
                    continue
                if dict_idf.get(name):
                    sum_new = sum_new + dict_idf.get(name)

            for q in range(0, max_index_2):
                name = new_list[q][0]
                if dont_sort.__contains__(name) or dict.get(label) < dict.get(name):
                    continue
                if sum_new != 0 and dict_idf.get(name):
                    k = dict_idf.get(name) / sum_new
                else:
                    k = 0
                dict[name] = dict.get(name) + ((dict.get(label) - dict.get(name)) * k)
        p_list = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)

        if answer.get(sourcesName[i]):
            ans += 1
            ap = metrics.map(sourcesName, answer, i, ap, p_list)
            p1 = metrics.precision(answer, sourcesName, i, p_list)
            averagePrecision += p1
            r1 = metrics.recall(answer, sourcesName, i, p_list)
            averageRecall += r1
            averagePrecision_Recall = metrics.precision_recall(answer, sourcesName, recall, i, averagePrecision_Recall,
                                                               p_list)
    for i in range(0, 10):
        averagePrecision_Recall[i] = averagePrecision_Recall[i] / ans
    data_f = [
        k1, k2,
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

    k1set = [round(start + i * step1, 2) for i in range(int((end - start) / step1) + 1)]
    k2set = [round(start + i * step2, 2) for i in range(int((end - start) / step2) + 1)]

    dict_idf = {}
    data = []
    # 数据准备
    answer = file_tool.parse_file(RESULT_XML)
    print(answer)
    targets, targetsName = file_tool.parse_file_SorT(TA_ADDRESS)
    sources, sourcesName = file_tool.parse_file_SorT(SA_ADDRESS)

    # 创建名称到索引的映射
    ta_name_to_idx = {name: idx for idx, name in enumerate(targetsName)}
    sa_name_to_idx = {name: idx for idx, name in enumerate(sourcesName)}
    # 生成所有参数组合
    params = [(k1, k2) for k1 in k1set for k2 in k2set]
    ta_sim_matrix = precompute_ta_sim_matrix(targets)
    sa_ta_sim_matrix = precompute_sa_ta_sim_matrix(sources,targets)
    # 添加检查
    if ta_sim_matrix is None:
        raise ValueError("ta_sim_matrix 未被正确计算")
    if sa_ta_sim_matrix is None:
        raise ValueError("sa_ta_sim_matrix 未被正确计算")

    print(f"ta_sim_matrix 类型: {type(ta_sim_matrix)}, 形状: {ta_sim_matrix.shape}")
    print(f"sa_ta_sim_matrix 类型: {type(sa_ta_sim_matrix)}, 形状: {sa_ta_sim_matrix.shape}")

    with mp.Pool(processes=PROCRSSES_NUM) as pool:
        results = pool.starmap(main_k1_k2, [(ta_sim_matrix, sa_ta_sim_matrix, p[0], p[1], answer,sources,targets,sourcesName,targetsName,ta_name_to_idx,sa_name_to_idx) for p in params])

    data = results
    save_to_excel(data)
    # 计算总耗时
    total_time = time.perf_counter() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTotal execution time: {int(hours):0>2}h {int(minutes):0>2}m {seconds:05.2f}s")
