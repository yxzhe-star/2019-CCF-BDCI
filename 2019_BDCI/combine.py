import pandas as pd
import codecs


#并集合并，可以首尾预测取并集
def union_combine():
   result1 = pd.read_csv('D:/submit2.csv',encoding='utf-8')
   result2 = pd.read_csv('D:/submit1.csv',encoding='utf-8')
   result3 = pd.DataFrame(columns=['id', 'unknownEntities'])
   result3['id']=result1['id']
   #result3['unknownEntities'] = result3['unknownEntities'].fillna('')

   for j in range(len(result3)):
      strlabel1 = result1['unknownEntities'][j]
      strlabel2 = result2['unknownEntities'][j]
      if str(strlabel1) == 'nan' and str(strlabel2) == 'nan':
        continue
      labels1 = []
      labels2 = []
      if str(strlabel1) != 'nan':
        labels1 = str(strlabel1).strip().split(';')
      if str(strlabel2) != 'nan':
        labels2 = str(strlabel2).strip().split(';')

      label = []
      label = labels1 + labels2
      label = list(set(label))
      newstr = ''
      for i in range(len(label)):
        newstr = newstr + label[i]
        if i != len(label) - 1:
            newstr = newstr + ';'
      result3.loc[j,'unknownEntities'] = newstr
   result3.to_csv('D:/submit_combine.csv',index=False)

def intersection_combine():
    result1 = pd.read_csv('D:/submit1.csv', encoding='utf-8')
    resultfinal = pd.DataFrame(columns=['id', 'unknownEntities'])
    resultfinal['id'] = result1['id']

    result2 = pd.read_csv('D:/submit2.csv', encoding='utf-8')
    result3 = pd.read_csv('D:/submit3.csv', encoding='utf-8')
    result4 = pd.read_csv('D:/submit4.csv', encoding='utf-8')
    result5 = pd.read_csv('D:/submit5.csv', encoding='utf-8')
    result6 = pd.read_csv('D:/submit6.csv', encoding='utf-8')
    result7 = pd.read_csv('D:/submit7.csv', encoding='utf-8')
    result8 = pd.read_csv('D:/submit8.csv', encoding='utf-8')
    finallist = []
    for j in range(len(resultfinal)):
        strlabel = str(result1['unknownEntities'][j])+ ';' + str(result2['unknownEntities'][j])\
                   + ';' + str(result3['unknownEntities'][j])+ ';' + str(result4['unknownEntities'][j])+ ';' + str(result5['unknownEntities'][j])\
                + str(result6['unknownEntities'][j]) + ';' + str(result7['unknownEntities'][j])+ ';' + str(result8['unknownEntities'][j])
        labels = strlabel.strip().split(';')
        finallist.append(labels)
    for i in range(len(finallist)):
        labels = finallist[i]
        dict = {}
        for key in labels:
           dict[key] = dict.get(key, 0) + 1
        if 'nan' in dict.keys():
            del dict['nan']
        labelnew = [k for k,v in dict.items() if v > 2]
        newstr = ''
        for j in range(len(labelnew)):
            newstr = newstr + labelnew[j]
            if j != len(labelnew) - 1:
                newstr = newstr + ';'
        resultfinal.loc[i,'unknownEntities'] = newstr
    resultfinal.to_csv('D:/final_combine.csv', index=False)


if __name__ == "__main__":
    intersection_combine()
    #union_combine()