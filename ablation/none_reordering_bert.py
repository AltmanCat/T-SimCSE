# -*- codeing = utf-8 -*-
# 作者——wwq
# 时间： 2025/3/25 11:12
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
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

# sentence-transformers/bert-base-nli-mean-tokens无重排
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


GENERATOR_EXCEL = "bert_IceBreaker_none_reordering.xlsx"
MODEL_PATH = '../bert/'
PROCRSSES_NUM = 1  # 进程数

# 参数配置
start = 0.01
end = 1.0
step1 = 0.01
step2 = 0.01

dict_idf = {}

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
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

def save_to_excel(data):
    df = pd.DataFrame(data, columns=["map", "Precision", "Recall", "f1", "f2","PR"])
    df.to_excel(GENERATOR_EXCEL, index=False)
    print(f"数据已成功保存到 {GENERATOR_EXCEL}")
def main_none_reordering(ta_sim_matrix,sa_ta_sim_matrix,answer,sources,targets,sourcesName,targetsName,ta_name_to_idx,sa_name_to_idx):
    ap, f1, f2 = 0, 0, 0
    averagePrecision, averageRecall = 0, 0
    averagePrecision_Recall = [0 for i in range(0, 10)]
    recall = [i / 10 for i in range(1, 11)]
    ans = 0
    for i in range(0, len(sources)):
        label_text = {name: idx for idx, name in enumerate(targetsName)}
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

    # 初始化NLP模型
    try:
        # 从本地目录加载 tokenizer 和 model
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModel.from_pretrained(MODEL_PATH)
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
    # 检查模型的一些属性
    if model is not None:
        print("模型名称:", model.name_or_path)
    else:
        print("模型加载失败")
    dict_idf = {}
    data = []
    # 数据准备
    answer = file_tool_other.parse_file(RESULT_XML)
    print(answer)
    targets, targetsName = file_tool_other.parse_file_SorT(TA_ADDRESS)
    sources, sourcesName = file_tool_other.parse_file_SorT(SA_ADDRESS)

    tokenizer.model_max_length = 512  # 🛠️ 更新默认最大长度
    targets_inputs = tokenizer(targets, padding=True, truncation=True, return_tensors="pt")
    sources_inputs = tokenizer(sources, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        targets_output = model(**targets_inputs)
        targets_embeddings = mean_pooling(targets_output, targets_inputs['attention_mask'])
        sources_output = model(**sources_inputs)
        sources_embeddings = mean_pooling(sources_output, sources_inputs['attention_mask'])
    # 创建名称到索引的映射
    ta_name_to_idx = {name: idx for idx, name in enumerate(targetsName)}
    sa_name_to_idx = {name: idx for idx, name in enumerate(sourcesName)}
    # 生成所有参数组合
    ta_sim_matrix = precompute_ta_sim_matrix(targets_embeddings)
    sa_ta_sim_matrix = precompute_sa_ta_sim_matrix(sources_embeddings,targets_embeddings)
    # 添加检查
    if ta_sim_matrix is None:
        raise ValueError("ta_sim_matrix 未被正确计算")
    if sa_ta_sim_matrix is None:
        raise ValueError("sa_ta_sim_matrix 未被正确计算")

    print(f"ta_sim_matrix 类型: {type(ta_sim_matrix)}, 形状: {ta_sim_matrix.shape}")
    print(f"sa_ta_sim_matrix 类型: {type(sa_ta_sim_matrix)}, 形状: {sa_ta_sim_matrix.shape}")

    result = main_none_reordering(ta_sim_matrix, sa_ta_sim_matrix, answer,sources,targets,sourcesName,targetsName,ta_name_to_idx,sa_name_to_idx)
    data.append(result)
    save_to_excel(data)
    # 计算总耗时
    total_time = time.perf_counter() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTotal execution time: {int(hours):0>2}h {int(minutes):0>2}m {seconds:05.2f}s")