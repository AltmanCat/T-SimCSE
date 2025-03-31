# -*- codeing = utf-8 -*-
# 作者——wwq
# 时间： 2025/3/12 9:28

import os, sys
import math
from xml.dom.minidom import parse
import xml.dom.minidom
import re

import os
import xml.etree.ElementTree as ET

#  parse_file 支持第一列为SA第二列为TA EasyClinic、GANNT、dronology
def parse_file(file_path):  # 解析标准集文件XML或TXT，返回answer字典
    # 获取文件扩展名并转为小写
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    print(ext)
    answer = {}

    if ext == '.xml':
        DOMTree = xml.dom.minidom.parse(file_path)
        collection = DOMTree.documentElement
        artifacts = collection.getElementsByTagName("link")
        for artifact in artifacts:
            said = artifact.getElementsByTagName('target_artifact_id')[0]
            saID = said.childNodes[0].data
            content = artifact.getElementsByTagName('source_artifact_id')[0]
            taID = ""
            taID += content.childNodes[0].data
            if answer.__contains__(taID):
                answer.get(taID).append(saID)
            else:
                tmp = []
                tmp.append(saID)
                answer[taID] = tmp
        return answer

    elif ext == '.txt':
        # 读取TXT文件内容（使用utf-8编码）
        with open(file_path, "r") as result:
            for line in result:
                str = line.strip()
                str = re.split(r"[ ,;]+", str)  # 按空格、逗号、分号分割（允许多个分隔符连续出现）
                if len(str) <= 1:
                    continue
                elif len(str) >= 2:
                    taid = str[0] # 第一列为TA第二列为SA
                    said = str[1]
                if answer.__contains__(taid):
                    answer.get(taid).append(said)
                else:
                    tmp = []
                    tmp.append(said)
                    answer[taid] = tmp
        return answer

    else:
        return ValueError(f"不支持的文件类型: {ext}")


def parse_file_SorT(file_path):  # 解析源工件或者目标工件文件XML或TXT或者多个txt文件，返回result_dict字典

    if not os.path.exists(file_path):
        print(FileNotFoundError(f"路径不存在: {file_path}"))
        return

    result_array = []
    result_array_name = []
    word_counts = 0  # 新增存储单词数
    # 处理单个文件的情况
    if os.path.isfile(file_path):
        file_name = os.path.basename(file_path)
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()

        if ext not in ('.xml', '.txt'):
            print(ValueError(f"不支持的文件类型: {ext}"))
            return

        try:
            if ext == '.xml':
                DOMTree = xml.dom.minidom.parse(file_path)
                collection = DOMTree.documentElement
                artifacts = collection.getElementsByTagName("artifact")
                for artifact in artifacts:
                    art_id = artifact.getElementsByTagName("art_id")[0].childNodes[0].data  #   非CM1
                    art_title = artifact.getElementsByTagName("art_title")[0].childNodes[0].data  #   非CM1

                    # 处理可能为空的节点
                    art_parent = artifact.getElementsByTagName("art_parent")[0].childNodes  #   非CM1
                    art_content = artifact.getElementsByTagName("art_content")[0].childNodes  # 非CM1
                    art_type = artifact.getElementsByTagName("art_type")[0].childNodes  # 非CM1

                    art_parent_text = art_parent[0].data if art_parent else " "  # 非CM1
                    art_content_text = art_content[0].data if art_content else " "  # 非CM1
                    art_type_text = art_type[0].data if art_type else " "  # 非CM1

                    word_counts += count_words(art_content_text)
                    word_counts += count_words(art_type_text)
                    word_counts += count_words(art_title)

                    result_array_name.append(art_id)
                    result_array.append(f"{art_title} " +    # 非CM1
                                        # f"{art_parent_text}"+
                                        f" {art_content_text} {art_type_text}");
                print(f"单词数=={word_counts}")
                return result_array, result_array_name
            else:  # txt格式的文件
                with open(file_path, "r") as result:
                    for line in result:
                        str = line.strip()
                        sa_name = str.split(' ', 1)[0]
                        sa_conten = str.split(' ', 1)[1]
                        result_array_name.append(sa_name)
                        result_array.append(sa_conten)
                        word_counts += count_words(sa_conten)
                print(f"单词数=={word_counts}")
                return result_array, result_array_name
        except Exception as e:
            print(f"解析失败: {str(e)}")
            return

    # 处理文件夹的情况
    else:
        dirs = os.listdir(file_path)
        for file in dirs:
            filepath = file_path + file  # 文件所在地址
            result_array_name.append(file[:-4])  # txt格式的文件名
            with open(filepath, 'r',encoding="utf-8", errors="replace") as f:  # 读取文件
                data = f.readlines()
                f.close()  # 关
                # 将文件转换成字符串
                text = ""
                for line in data:
                    text += line
            word_counts += count_words(text)
            result_array.append(text)

        for file_name in os.listdir(file_path):
            file_path = os.path.join(file_path, file_name)
            if os.path.isfile(file_path):
                _, ext = os.path.splitext(file_name)
                ext = ext.lower()

                if ext in ('.txt'):
                    result_array_name.append(file[:-4])  # txt格式的文件名
                elif ext in ('.java'):
                    result_array_name.append(file[:-5])  # txt格式的文件名
                try:
                    with open(filepath, 'r') as f:  # 读取文件
                        data = f.readlines()
                        f.close()  # 关
                        # 将文件转换成字符串
                        text = ""
                        for line in data:
                            text += line
                    result_array.append(text)
                    word_counts += count_words(text)
                except Exception as e:
                    print(f"解析失败: {str(e)}")
                    return
        print(f"单词数=={word_counts}")
        return result_array, result_array_name
def count_words(text):
    """统计文本中的单词数（按非字母数字分割）"""
    return len(re.findall(r'\b\w+\b', text))  # 使用正则匹配单词


# 使用示例
# 使用示例
if __name__ == "__main__":
    TA_ADDRESS1 = "./EasyClinic/3 - test cases/"
    TA_ADDRESS2 = "./GANNT/low/"
    TA_ADDRESS4 = "./dronology/design_definition/"
    SA_ADDRESS1 = "./EasyClinic/1 - use cases/"
    SA_ADDRESS2 = "./GANNT/high/"
    SA_ADDRESS4 = "./dronology/req/"
    try:
        xml_result, xml_name = parse_file_SorT(TA_ADDRESS1)
        xml_result1, xml_name1 = parse_file_SorT(SA_ADDRESS1)
        print("XML解析结果：")
        print(xml_result.tag)


    except Exception as e:
        print(f"错误: {str(e)}")

