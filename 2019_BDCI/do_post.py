# coding=gbk
import pandas as pd
import codecs
import re
def revemove_substring(index):
    for j in index:
        strlabel = result['unknownEntities'][j]
        labels = str(strlabel).strip().split(';')
        if len(labels) < 2:
            continue
        longest_entities = []
        for label1 in labels:
            flag = 0
            for label2 in labels:
                if label1 == label2:
                    continue
                if label2.find(label1) != -1:
                    flag = 1
            if flag == 0:
                longest_entities.append(label1)
        newstr = ''
        for i in range(len(longest_entities)):
            newstr = newstr + longest_entities[i]
            if i != len(longest_entities) - 1:
                newstr = newstr + ';'
        result.loc[j,'unknownEntities'] = newstr


def many_labels():
    list = []
    for i in range(len(result)):
        strlabel = result['unknownEntities'][i]
        label = str(strlabel).strip().split(';')
        if len(label) >= 5:
            list.append(i)
    return list

def divide(index2):
    con1 = 0
    con2 = 0
    train_df = pd.read_csv('D:/Train_Datayuanban.csv', encoding='utf-8')
    train_df2 = pd.read_csv('D:/Round2_trainyuanban.csv', encoding='utf-8')
    train_df = train_df[~train_df['unknownEntities'].isnull()]
    train_df2 = train_df2[~train_df2['unknownEntities'].isnull()]
    train_label = []
    list1 = train_df.index.tolist()
    list2 = train_df2.index.tolist()

    for i in list1:
        l = train_df['unknownEntities'][i]
        l = l.strip().split(';')
        for j in l:
            train_label.append(j)

    for i in list2:
        l = train_df2['unknownEntities'][i]
        l = l.strip().split(';')
        for j in l:
            train_label.append(j)
    train_label = list(set(train_label))
    for j in index2:
        text = test_df['text'][j]
        divide_text = text.strip().split('、')
        label = []
        flag = 0
        last_flag = 0
        for content in divide_text:
            if len(content) >2 and len(content)<7 and content.find(',') == -1:
                flag = 1
                label.append(content)
                con1 = con1 + 1

        for i in train_label:
             label = [content for content in label if content != i]
        con2 = con2 + len(label)
        strlabel = result['unknownEntities'][j]
        labelold = str(strlabel).strip().split(';')
        labelnew = label + labelold
        labelnew = list(set(labelnew))
        print(j,labelnew)
        newstr = ''
        for i in range(len(labelnew)):
            newstr = newstr + labelnew[i]
            if i != len(labelnew) - 1:
                newstr = newstr + ';'
        result.loc[j,'unknownEntities'] = newstr
    print(con1)
    print(con2)

def rule():
    con = 0
    list_index = []
    for i in range(len(test_df)):
       loc = [t.start() for t in re.finditer('、', test_df['text'][i])]
       if len(loc)<15 :
          continue
       for j in range(len(loc)-7):
          now = loc[j+1] -loc[j]
          if loc[j+7]-loc[j]<35 and loc[j+1]-loc[j]>2 and loc[j+2]-loc[j+1]>3\
                  and loc[j+3]-loc[j+2]>3 and loc[j+4]-loc[j+3]>3\
                  and loc[j+5]-loc[j+4]>3 and loc[j+6]-loc[j+5]>3:
              list_index.append(i)
              con =con+1
              break
    print('数量',con)
    return list_index
def clean_zh_text(text):
    # keep English, digital and Chinese
    comp = re.compile('[^a-zA-Z0-9\u4e00-\u9fa5;（）]')
    return re.sub(comp,'', text)

# 一些需要保留的符号

if __name__ == "__main__":

    result = pd.read_csv('D:/final_combine.csv', encoding='utf-8')
    test_df = pd.read_csv('D:/Round2_Test.csv',encoding='utf-8')
    test_df['text'] = test_df['title'].fillna('') + test_df['text'].fillna('')
    #选择出过滤前实体多的数据
    index1 = many_labels()
    #对实体多的数据进行最大字符串过滤
    revemove_substring(index1)
    #选择出过滤后实体多的数据
    index2 = many_labels()
    index3 = rule()
    #根据规则划分数据，并将其与原来数据融合
    divide(index2)
    #对融合后的数据进行最大字符串过滤
    revemove_substring(index2)
    divide(index3)
    revemove_substring(index3)
    for i in range(len(result)):
        if str(result['unknownEntities'][i]) != 'nan':
            result.loc[i,'unknownEntities'] = clean_zh_text(result.loc[i,'unknownEntities'])
    result.to_csv('D:/try.csv', index=False)



