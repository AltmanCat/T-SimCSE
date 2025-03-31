# -*- codeing = utf-8 -*-
# 作者——wwq
# 时间： 2025/3/25 11:12
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
import file_tool_other
import os
import re
import torch
from typing import List



# 无重排LSI
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

GENERATOR_EXCEL = "LSI_IceBreaker_none_reordering.xlsx"
PROCRSSES_NUM = 1  # 进程数

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
    # TF-IDF向量化
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
    df = pd.DataFrame(data, columns=["map", "Precision", "Recall", "f1", "f2","PR"])
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
    dict_idf = {}
    data = []
    # 数据准备
    answer = file_tool_other.parse_file(RESULT_XML)
    print(answer)
    targets, targetsName = file_tool_other.parse_file_SorT(TA_ADDRESS)
    sources, sourcesName = file_tool_other.parse_file_SorT(SA_ADDRESS)

    # 创建名称到索引的映射
    ta_name_to_idx = {name: idx for idx, name in enumerate(targetsName)}
    sa_name_to_idx = {name: idx for idx, name in enumerate(sourcesName)}
    # 生成所有参数组合
    ta_sim_matrix = precompute_ta_sim_matrix(targets)
    sa_ta_sim_matrix = precompute_sa_ta_sim_matrix(sources,targets)
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
