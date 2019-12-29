import pandas as pd
import codecs
import re
from sklearn.model_selection import train_test_split

train_df1 = pd.read_csv('D:/Train_Data.csv',encoding='utf-8')
train_df2 = pd.read_csv('D:/Round2_train.csv',encoding='utf-8')

test_df = pd.read_csv('D:/Round2_Test.csv',encoding='utf-8')
train_df = pd.concat([train_df1,train_df2]).reset_index()

def stop_words(input):
    input = input.replace(",", "，")
    input = input.replace("\xa0", "")
    input = input.replace("\b", "")
    input = input.replace('"', "")
    input = re.sub("\t|\n|\x0b|\x1c|\x1d|\x1e", "", input)
    input = input.strip()
    input = re.sub(r'<.*?>', '',input)
    input = re.sub('\?\?\?\?+', '', input)
    input = re.sub('\{IMG:.?.?.?\}', '', input)
    input = re.sub('\t|\n', '', input)
    return input

train_df['text'] =  train_df['title'].fillna('') + train_df['text'].fillna('')
test_df['text'] =  test_df['title'].fillna('') +test_df['text'].fillna('')
train_df['text'] = train_df['text'].apply(stop_words)
test_df['text'] = test_df['text'].apply(stop_words)
#不要将索引重置
train_df = train_df[~train_df['unknownEntities'].isnull()]
print(len(train_df))
additional_chars = set()
for t in list(test_df.text) + list(train_df.text):
    additional_chars.update(re.findall(u'[^\u4e00-\u9fa5a-zA-Z0-9\*]', t))

# 一些需要保留的符号
extra_chars = set("!#$%&\()*+,-./:;<=>?@[\\]^_`{|}~！#￥%&？《》{}“”，：‘’。（）·、；【】｜")
additional_chars = additional_chars.difference(extra_chars)

def remove_additional_chars(input):
    for x in additional_chars:
        input = input.replace(x, "")
    return input
train_df["text"] = train_df["text"].apply(remove_additional_chars)
test_df["text"] = test_df["text"].apply(remove_additional_chars)

#获取行索引
list1 = train_df.index.tolist()

newtest = []
newlabel = []
def unfind(train_df):
    con2 = 0
    con3 = 0
    con4 = 0
    con5 = 0
    for i in list1:
        label = []
        # len_train = len(train_df)
        label.append(train_df['unknownEntities'][i])
        label = str(label[0]).strip().split(';')
        for data in label:
            if len(train_df['text'][i]) > 1000 and train_df['text'][i].find(data, -512, -1) != -1:
                con2 = con2 + 1
                newtest.append(train_df['text'][i][-510:])
                newlabel.append(train_df['unknownEntities'][i])
                break
            elif train_df['text'][i].find(data, 0, len(train_df['text'][i])) == -1 and len(label) == 1:
                train_df.drop(i, inplace=True)
                con3 = con3 + 1
                break
            elif train_df['text'][i].find(data, 0, 512) == -1 and len(label) ==1:
                   train_df.drop(i, inplace=True)
                   con4 = con4 +1
                   break
            elif train_df['text'][i].find(label[0], 0, 512) == -1 and len(label) != 1:
                if train_df['text'][i].find(label[1], 0, 512) == -1:
                   train_df.drop(i, inplace=True)
                   con5 = con5 +1
                   break
                break

    print(con2)
    print(con3)
    print(con4)
    print(con5)
    d = {'text': newtest, 'unknownEntities': newlabel}
    newpd = pd.DataFrame(data=d)
    train_df = pd.concat([train_df, newpd], axis=0, sort=True, ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=10).reset_index(drop=True)
    print(train_df)
    print(len(train_df))
    return train_df

#unfind是在初赛时使用的方法，它可以过滤一些不符合条件的句子
#train_df = unfind(train_df)
print(train_df.shape)

with codecs.open('D:deep learning/BDCI_ner/train.txt', 'w',encoding='utf-8') as up:
    #itertuples(): 将DataFrame迭代为元祖。
    for row in train_df.iloc[:-5].itertuples():

        text_lbl = row.text
        entitys = str(row.unknownEntities).split(';')
        for entity in entitys:
            text_lbl = text_lbl.replace(entity, 'Ё' + (len(entity) - 1) * 'Ж')

        for c1, c2 in zip(row.text, text_lbl):
            if c2 == 'Ё':
                up.write('{0} {1}\n'.format(c1, 'B-ORG'))
            elif c2 == 'Ж':
                up.write('{0} {1}\n'.format(c1, 'I-ORG'))
            else:
                up.write('{0} {1}\n'.format(c1, 'O'))

        up.write('\n')

with codecs.open('D:deep learning/BDCI_ner/dev.txt', 'w',encoding='utf-8') as up:
    for row in train_df.iloc[-5:].itertuples():
        # print(row.unknownEntities)

        text_lbl = row.text
        entitys = str(row.unknownEntities).split(';')
        for entity in entitys:
            text_lbl = text_lbl.replace(entity, 'Ё' + (len(entity) - 1) * 'Ж')

        for c1, c2 in zip(row.text, text_lbl):
            if c2 == 'Ё':
                up.write('{0} {1}\n'.format(c1, 'B-ORG'))
            elif c2 == 'Ж':
                up.write('{0} {1}\n'.format(c1, 'I-ORG'))
            else:
                up.write('{0} {1}\n'.format(c1, 'O'))

        up.write('\n')



with codecs.open('D:deep learning/BDCI_ner/test_tail.txt', 'w',encoding='utf-8') as up:
    for row in test_df.iloc[:].itertuples():

        text_lbl = row.text
        for c1 in text_lbl:
            up.write('{0} {1}\n'.format(c1, 'O'))

        up.write('\n')

#下面是在复赛中使用的方法
#将句子按照。！，？标点划分（划分后的句子不会超过512长度，这是因为bert识别的最大长度为512）
#划分后的句子若有实体在句中，则保留，否则舍去，生成新的训练集。
'''
text = []
text_id = []
def divide_tet(str,j):
    i = 512
    lasti = -1
    if len(str)>600:
       while(i<len(str)):
           if str[i] == '。' or str[i] == '！'or str[i] == '，' or str[i] == '？':
              text.append(str[lasti+1:i])
              text_id.append(test_df['id'][j])
              lasti = i
              i = i+512
              continue
           i = i -1
           #如果没有上述标点符号
           if i<=lasti:
               text.append(str[lasti + 1:lasti+512])
               text_id.append(test_df['id'][j])
               lasti = lasti + 512
               i = lasti + 512
       if len(str)-lasti>60:
         text.append(str[lasti+1:])
         text_id.append(test_df['id'][j])
    else:
        text.append(str)
        text_id.append(test_df['id'][j])
for j in range(len(test_df)):
   divide_tet(str(test_df['text'][j]).strip(),j)
print(len(text))
print(len(text_id))

train = []
train_label = []
def divide_train(str1,j):
    i = 512
    lasti = -1
    strlabel = train_df['unknownEntities'][j]
    label = str(strlabel).strip().split(';')
    if len(str1)>600:
       while(i<len(str1)):
           if str1[i] == '。' or str1[i] == '！'or str1[i] == '，' or str1[i] == '？':
               for l in label:
                  if str1[lasti+1:i].find(l)!= -1:
                     train.append(str1[lasti + 1:i])
                     train_label.append(strlabel)
                     break
               lasti = i
               i = i + 512
               continue
           i = i -1
           #如果没有上述标点符号
           if i<=lasti:
                   for l in label:
                       if str1[lasti + 1:lasti+512].find(l) != -1:
                           train.append(str1[lasti + 1:lasti+512])
                           train_label.append(strlabel)
                           break
                   lasti = lasti + 512
                   i = lasti + 512
       if len(str1)-lasti>30:
           for l in label:
               if str1[lasti+1:].find(l) != -1:
                   train.append(str1[lasti+1:])
                   train_label.append(strlabel)
                   break
    else:
        if str1.find(train_df['unknownEntities'][j]) != -1:
           train.append(str1)
           train_label.append(strlabel)
for j in list1:
   divide_train(str(train_df['text'][j]).strip(),j)
x_train, x_val, y_train,y_val = train_test_split(train, train_label, test_size=0.001,random_state=18)

with codecs.open('D:deep learning/BDCI_ner/train.txt', 'w',encoding='utf-8') as up:
    i=0
    for text in x_train:
        text_lbl = text
        entitys = str(y_train[i]).split(';')
        i = i+1
        for entity in entitys:
            text_lbl = text_lbl.replace(entity, 'Ё' + (len(entity) - 1) * 'Ж')

        for c1, c2 in zip(text, text_lbl):
            if c2 == 'Ё':
                up.write('{0} {1}\n'.format(c1, 'B-ORG'))
            elif c2 == 'Ж':
                up.write('{0} {1}\n'.format(c1, 'I-ORG'))
            else:
                up.write('{0} {1}\n'.format(c1, 'O'))

        up.write('\n')


with codecs.open('D:deep learning/BDCI_ner/test.txt', 'w',encoding='utf-8') as up:
    for t in text:
        for c1 in t:
            up.write('{0} {1}\n'.format(c1, 'O'))
        up.write('\n')
d = {'id': text_id}
newresult = pd.DataFrame(data=d)
newresult['unknownEntities'] = None
print(len(newresult))
print(newresult)
newresult.to_csv('D:/newresult.csv',index=False)
'''