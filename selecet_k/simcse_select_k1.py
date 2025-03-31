# -*- codeing = utf-8 -*-
# 作者——wwq
# 时间： 2025/3/24 16:28
# -*- codeing = utf-8 -*-
# 作者——wwq
# 时间： 2025/3/24 15:43
# -*- codeing = utf-8 -*-
# 作者——wwq
# 时间： 2025/3/24 11:13

from gensim.models.keyedvectors import KeyedVectors
from WocSim import WocSim
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
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

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

GENERATOR_EXCEL = "T-simCSE_GANNT_k1_k2.xlsx"
MODEL_PATH = '../simcse/'
PROCRSSES_NUM = 2  # 进程数

# 参数配置
start = 0.01
end = 1.0
step1 = 0.01
step2 = 0.01

dict_idf = {}

def save_to_excel(data):
    df = pd.DataFrame(data, columns=["k1","k2", "map", "Precision", "Recall", "f1", "f2","PR"])
    df.to_excel(GENERATOR_EXCEL, index=False)
    print(f"数据已成功保存到 {GENERATOR_EXCEL}")

def calculate_ta_sim(args):
    i, j, targets = args
    return i, j, sentence_similarity(targets[i], targets[j])


def precompute_ta_sim_matrix(targets_embeddings):
    print("预计算TA相似度矩阵...")
    indices = list(combinations(range(len(targets_embeddings)), 2))
    with mp.Pool(processes=PROCRSSES_NUM) as pool:
        results = pool.map(
            calculate_ta_sim,
            [(i, j, targets_embeddings) for i, j in indices]
        )
    ta_sim_matrix = np.eye(len(targets_embeddings))
    for i, j, sim in results:
        ta_sim_matrix[i, j] = sim
        ta_sim_matrix[j, i] = sim
    return ta_sim_matrix;

def sentence_similarity(s_embedding, t_embedding):
    return 1 - cosine(s_embedding, t_embedding)
def vectorized_sim(sources_embedding,targets_embeddings):
    return np.array([sentence_similarity(sources_embedding, ta) for ta in targets_embeddings])

def precompute_sa_ta_sim_matrix(sources_embeddings,targets_embeddings):
    print("预计算SA-TA相似度矩阵...")
    with mp.Pool(processes=PROCRSSES_NUM) as pool:
        sa_ta_sim_rows = pool.starmap(vectorized_sim, [(sa, targets_embeddings) for sa in sources_embeddings])
    sa_ta_sim_matrix = np.vstack(sa_ta_sim_rows)
    return sa_ta_sim_matrix

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

    # 初始化NLP模型
    try:
        # 从本地目录加载 tokenizer 和 model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModel.from_pretrained(MODEL_PATH)
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
    if model is not None:
        print("模型名称:", model.name_or_path)
    else:
        print("模型加载失败")

    k1set = [round(start + i * step1, 2) for i in range(int((end - start) / step1) + 1)]
    k2set = [round(start + i * step2, 2) for i in range(int((end - start) / step2) + 1)]

    dict_idf = {}
    data = []
    # 数据准备
    answer = file_tool.parse_file(RESULT_XML)
    print(answer)
    targets, targetsName = file_tool.parse_file_SorT(TA_ADDRESS)
    sources, sourcesName = file_tool.parse_file_SorT(SA_ADDRESS)

    targets_inputs = tokenizer(targets, padding=True, truncation=True, return_tensors="pt")
    sources_inputs = tokenizer(sources, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        targets_embeddings = model(**targets_inputs, output_hidden_states=True, return_dict=True).pooler_output
        sources_embeddings = model(**sources_inputs, output_hidden_states=True, return_dict=True).pooler_output

    # 创建名称到索引的映射
    ta_name_to_idx = {name: idx for idx, name in enumerate(targetsName)}
    sa_name_to_idx = {name: idx for idx, name in enumerate(sourcesName)}
    # 生成所有参数组合
    params = [(k1, k2) for k1 in k1set for k2 in k2set]
    ta_sim_matrix = precompute_ta_sim_matrix(targets_embeddings)
    sa_ta_sim_matrix = precompute_sa_ta_sim_matrix(sources_embeddings,targets_embeddings)
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