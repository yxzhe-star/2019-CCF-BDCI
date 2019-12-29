import pandas as pd
import codecs
train_df = pd.read_csv('D:/Train_Datayuanban.csv',encoding='utf-8')
train_df2 = pd.read_csv('D:/Round2_trainyuanban.csv',encoding='utf-8')
oldtest_df = pd.read_csv('D:/Round2_Test.csv',encoding='utf-8')
test_df = pd.read_csv('D:/newresult.csv',encoding='utf-8')
train_df['text'] =  train_df['title'].fillna('') + train_df['text'].fillna('')
train_df2['text'] =  train_df2['title'].fillna('') + train_df2['text'].fillna('')
#不要将索引重置
train_df = train_df[~train_df['unknownEntities'].isnull()]
train_df2 = train_df2[~train_df2['unknownEntities'].isnull()]

#将训练集的实体提取出来
train_label = []

list1 = train_df.index.tolist()
for i in list1:
     l = train_df['unknownEntities'][i]
     l = l.strip().split(';')
     for j in l:
       train_label.append(j)

list2 = train_df2.index.tolist()
for i in list2:
     l = train_df2['unknownEntities'][i]
     l = l.strip().split(';')
     for j in l:
       train_label.append(j)
train_label = list(set(train_label))

#处理结果文件
test_pred = codecs.open('D:deep learning/BDCI_ner/label_test11.txt',encoding='utf-8').readlines()
pred_tag = []
pred_word = []

pred_line_tag = ''
pred_line_word = ''

for line in test_pred:
    line = line.strip()

    if len(line) == 0 or line == '':
        pred_tag.append(pred_line_tag)
        pred_word.append(pred_line_word)
        pred_line_tag = ''
        pred_line_word = ''
        continue

    c, _, tag = line.split(' ')

    if tag != 'O':
        tag = tag[1:]
        pred_line_word += c
    else:
        pred_line_word += ';'

    pred_line_tag += tag

def filter_word(w):
    for wbad in ['？', '《', '🔺', '️?', '!', '#', '%', '%', '，', 'Ⅲ', '》', '丨', '、', '​',',','，','(',')',
                 '👍', '。', '😎', '/', '】', '-', '⚠️', '：', '✅', '㊙️', '“',  '！', '🔥','.','🌹',']','[',
                 '✔','+','……','@','【','❓','$',']','[','📢','🇳','�','┓','｜','◢','🔰','”','“']:
        if wbad in w:
            return ''
    return w

con1 = 0
con2 = 0
with codecs.open('D:/submit.csv', 'w',encoding='utf-8') as up:
    up.write('id,unknownEntities\n')

    for word, id in zip(pred_word, test_df['id'].values):
        word = set([filter_word(x) for x in word.split(';') if x not in ['', ';'] and len(x) > 1])
        #print(word)
        word = [x for x in word if x != '']
        con1 = con1 +len(word)
        for j in train_label:
           word = [x for x in word if x != j]
        con2 = con2 + len(word)
        if len(word) == 0:
            word = ['']

        word = ';'.join(list(word))
        up.write('{0},{1}\n'.format(id, word))

print(con1)
print(con2)


result = pd.read_csv('D:/submit.csv',encoding='utf-8')
list_id = []
for i in range(len(oldtest_df)):
    list_id.append(int(oldtest_df['id'][i]))

for i in list_id:
    now = result.loc[result['id'] == i]
    list1 = now.index.tolist()
    lstr = ''
    for j in list1:
        if str(now['unknownEntities'][j]).strip() != 'nan':
            lstr = lstr + str(now['unknownEntities'][j]) + ';'
    lstr = lstr[:len(lstr)-1]
    oldtest_df.loc[oldtest_df['id'] == i, 'unknownEntities'] = lstr
nb_output = oldtest_df[['id','unknownEntities']]
#nb_output.to_csv('D:/submit.csv',index=False)
cpn = 0

#nb_output = pd.read_csv('D:/submit.csv',encoding='utf-8')
for j in range(len(nb_output)):
    strlabel = nb_output['unknownEntities'][j]
    label = str(strlabel).strip().split(';')
    labelold = str(strlabel).strip().split(';')
    label = list(set(label))
    labelold = list(set(labelold))
    for i in range(len(labelold)):
        if len(labelold[i])%2 == 0 and len(labelold[i])!= 2:
            l = len(labelold[i])//2
            sleft = labelold[i][:l]
            sright = labelold[i][l:]
            if sleft == sright:
              #print(j)
              label.remove(labelold[i])
              flag = 0
              for t in label:
                if sleft ==t:
                    cpn = cpn +1
                    flag =1
                    break
              if flag == 0:
                cpn = cpn + 1
                label.append(sleft)

    if len(label) > 1:
      newstr = ''
      for i in range(len(label)):
           newstr= newstr + label[i]
           if i != len(label)-1:
               newstr = newstr + ';'
      nb_output.loc[j,'unknownEntities']= newstr
    elif len(label) == 1 and label[0]!='nan':
        nb_output.loc[j,'unknownEntities'] = label[0]
print('数量',cpn)
nb_output.to_csv('D:/submit11.csv',index=False)
