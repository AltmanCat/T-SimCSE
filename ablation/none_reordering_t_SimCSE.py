# -*- codeing = utf-8 -*-
# 作者——wwq
# 时间： 2025/3/6 14:42
# -*- codeing = utf-8 -*-
# 作者——wwq
# 时间： 2024/4/1 10:13


import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer
import operator
import metrics
import file_tool_other


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

model_path = "../simcse/"
answer = file_tool_other.parse_file(RESULT_XML)
print(answer)

try:
    # 从本地目录加载 tokenizer 和 model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    print("模型加载成功")
except Exception as e:
    print(f"模型加载失败: {e}")
if model is not None:
    print("模型名称:", model.name_or_path)
else:
    print("模型加载失败")

targets, targetsName = file_tool_other.parse_file_SorT(TA_ADDRESS)
sources, sourcesName = file_tool_other.parse_file_SorT(SA_ADDRESS)

targets_inputs = tokenizer(targets, padding=True, truncation=True, return_tensors="pt")
sources_inputs = tokenizer(sources, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    targets_embeddings = model(**targets_inputs, output_hidden_states=True, return_dict=True).pooler_output
    sources_embeddings = model(**sources_inputs, output_hidden_states=True, return_dict=True).pooler_output
ap, f1, f2 = 0, 0, 0
averagePrecision, averageRecall = 0, 0
averagePrecision_Recall = [0 for i in range(0, 10)]
recall = [i / 10 for i in range(1, 11)]
ans = 0
for i in range(0, len(sources)):
    dict = {}
    label_text = {}
    top = 0
    for j in range(0, len(targets)):
        label_text[targetsName[j]] = j
        cosine_sim_0_1 = 1 - cosine(sources_embeddings[i], targets_embeddings[j])
        dict[targetsName[j]] = cosine_sim_0_1
    list = sorted(dict.items(), key=operator.itemgetter(1), reverse=True)
    if answer.get(sourcesName[i]):
        ans += 1
        ap = metrics.map(sourcesName, answer, i, ap, list)
        p1 = metrics.precision(answer, sourcesName, i, list)
        averagePrecision += p1
        r1 = metrics.recall(answer, sourcesName, i, list)
        averageRecall += r1
        averagePrecision_Recall = metrics.precision_recall(answer, sourcesName, recall, i, averagePrecision_Recall,list)
map = ap / ans
print("map,Precision,Recall,f1,f2")
print(map, averagePrecision / ans, averageRecall / ans,metrics.fn(1, averagePrecision / ans, averageRecall / ans),metrics.fn(2, averagePrecision / ans, averageRecall / ans))

for i in range(0, 10):
    averagePrecision_Recall[i] = averagePrecision_Recall[i] / ans
print("============Precision_Recall")
print(averagePrecision_Recall)

print("ans===")
print(ans)
