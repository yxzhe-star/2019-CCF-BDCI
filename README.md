
# 比赛说明
2019-CCF-BDCI 互联网金融新实体发现。比赛地址：https://www.datafountain.cn/competitions/361 
随着互联网的飞速进步和全球金融的高速发展，金融信息呈现爆炸式增长。投资者和决策者在面对浩瀚的互联网金融信息时常常苦于如何高效的获取需要关注的内容。针对这一问题，金融实体识别方案的建立将极大提高金融信息获取效率，从而更好的为金融领域相关机构和个人提供信息支撑。
# 比赛结果 
初赛第7名，复赛第16名。  
# 运行环境
## 环境  
腾讯云ti-one  
python 3.5  
tensorflow 1.12  
## 硬件  
tesla P40, 大约需要12g显存。如果没有大显存，调小maxlen和batch size即可，可能效果会略有下降  
# 模型（bert+bilstm+crf）
本题是自然语言处理的ner（命名实体识别）问题，ner本质其实也是分类问题，只不过是token级别的分类。  
在bert模型出来前，主流的解决方法是将词向量输入bilstm+crf来分类。不过google去年开源了bert之后，nlp的许多任务准确率大幅提升。自然ner也不例外，只不过bert在此处相当于充当词向量，后续一样。不过google开源的源码里好像没有ner任务，此处参考大佬改写的bert ner版：https://github.com/jiangxinyang227/NLP-Project （真的是大佬，将各类任务都改写了一遍），不过需要改几处地方，其中源码中是读取英文数据，我将他改为处理中文数据。之前训练词向量的话一般用word2vec，不过它不能很好的解决同义词问题，而bert的好处是它是google用上亿数据通过tpu来训练，比赛中期开始用roberta（30G原始文本，近3亿个句子，100亿个中文字(token)，产生了2.5亿个训练数据(instance)），因此用bert这种预训练模型来对下游任务fine-tune（和图像领域的vggnet等很像）效果都很棒（包括小数据集）。
## bert
BERT=基于Transformer 的双向编码器表征，顾名思义，BERT模型的根基就是Transformer，来源于论文attention is all you need。其中双向的意思表示它在处理一个词的时候，能综合考虑到该词前面和后面单词的信息，从而获取上下文的语义，而Transformer有很强的特征抽取能力。   
BERT输入表示。输入嵌入是token embeddings, segmentation embeddings 和position embeddings 的总和。  
具体如下：  
（1）使用WordPiece嵌入（Wu et al., 2016）和30,000个token的词汇表。用##表示分词。  
（2）使用学习的positional embeddings，支持的序列长度最多为512个token，该向量包含位置信息，比如区分“我爱你”和”你爱我“。    
（3）每个序列的第一个token始终是特殊分类嵌入（[CLS]）。对应于该token的最终隐藏状态（即，Transformer的输出）被用作分类任务的聚合序列表示。对于非分类任务，将忽略此向量。可以用output_layer = model.get_sequence_output()这个获取每个token的output，输出维度为[batch_size, seq_length, embedding_size]。本比赛是ner任务，因此使用该函数。output_layer = model.get_pooled_output()，这个输出是获取句子的output，对应文本分类任务。  
（4）句子对被打包成一个序列。以两种方式区分句子。首先，用特殊标记（[SEP]）将它们分开。其次，添加一个learned sentence A嵌入到第一个句子的每个token中，一个sentence B嵌入到第二个句子的每个token中。  
（5）对于单个句子输入，只使用 sentence A嵌入。  
## crf
BiLSTM就不介绍了，比较常见。其实完全可以bert后直接接crf层，因为BiLSTM也是提取特征的，实际上bert已经做得很好了，不过我这里没有去掉BiLSTM层。  
加入crf层的目的主要是可以为最终预测标签添加一些约束以确保它们有效。在训练过程中，CRF层可以自动从训练数据集中学习这些约束。  
例如有两种类型的实体：Person和Organization。  
（1）句子中第一个单词的标签应以“B-”或“O”开头，而不是“I-”  
（2）B-label1 I-label2 I-label3 I- …“，在此模式中，label1，label2，label3 …应该是相同的命名实体标签。例如，“B-Person I-Person”有效，但“B-Person I-Organization”无效。  
（3）“O I-label”无效。一个命名实体的第一个标签应以“B-”而非“I-”开头，换句话说，有效模式应为“O B-label”  
（4）利用这些有用的约束，无效预测标签序列的数量将显着减少。    
CRF是它以路径为单位，考虑的是路径的概率。具体来讲，在CRF的序列标注问题中，我们要计算的是条件概率    
P(y1,…,yn|x1,…,xn)=P(y1,…,yn|x),x=(x1,…,xn)
# 关于比赛
## 预处理（BDCI_NER.py）
bert很神奇，有时候你对数据做一些清洗，最后结果反而会下降，但不代表完全不做清洗，需根据具体数据具体对待。比赛中我去除了html，<img>之类的标签。  
用正则表达式：  
```
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
```
之后是将训练集.csv转换成bert可以读入的格式.txt，用BIO标注将实体标注出来。具体处理参见BDCI_NER.py。 
## 训练（bert_blstm_crf.py）
由于github上开源的是处理英文数据，格式不太一样，这里对本任务适配，对bert_blstm_crf.py中的_read_data函数修改
```
    def _read_data(cls, input_file):
        """读取数据."""
        with open(input_file, encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                word = line.strip().split(' ')[0]
                label = line.strip().split(' ')[-1]
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
                # if len(contends) == 0 and words[-1] == '。':
                if len(contends) == 0:
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append([l, w])
                    words = []
                    labels = []
                    continue
                words.append(word)
                labels.append(label)
            #print(lines[0])
            #print(lines[1])
            return lines
```
模型用的是roberta_zh_l12,12层的普通版
可以直接在源码中修改路径和超参数，例如：
```
data_dir = '/cos_person/data/data5/'
bert_config_file = '/cos_person/model/roberta_zh_l12/bert_config.json'
vocab_file = '/cos_person/model/roberta_zh_l12/vocab.txt'
init_checkpoint = '/cos_person/model/roberta_zh_l12/bert_model.ckpt'
#init_checkpoint ='/cos_person/model/fine-tune-ner5/'
output_dir = '/cos_person/model/fine-tune-ner5/'
flags = tf.flags
FLAGS = flags.FLAGS

# 定义必须要传的参数
flags.DEFINE_string(
    "data_dir", data_dir,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", bert_config_file,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", 'ner', "The name of the task to train.")

flags.DEFINE_string("vocab_file",vocab_file,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", output_dir,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "init_checkpoint",init_checkpoint,
    "Initial checkpoint (usually from a pre-trained BERT model).")

```
下面是可选参数
```
flags.DEFINE_bool(
    "do_lower_case",True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train",True, "Whether to run training.")

flags.DEFINE_bool("do_eval",False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict",False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 6, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 6, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 6, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 1e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 15.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_list(
    "hidden_sizes", [256],
    "support multi lstm layers, only add hidden_size into hidden_sizes list."
)

flags.DEFINE_list(
    "layers", [256], "full connection layers"
)

flags.DEFINE_float("dropout_rate", 0.5, "dropout keep rate")
```
## 得到结果文件（getResult.py）
这个脚本完成了两个工作。  
一个工作是轻微后处理，在查看预测结果时发现，举个例子：比如该句的实体是腾讯，但预测的结果是腾讯，腾讯腾讯。显然腾讯腾讯不是准确的结果，看了文本内容发现是文本在介绍某实体的时候没有冒号，应该是腾讯：腾讯是一家······但是冒号没有，导致识别错误。因此我在我在预测的结果上进行纠正。
```
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
```
这是我在初赛时发现的，大概有纠正了100个错误，使用后提升了0.15%，其实效果不是很明显。主要是因为多识别一个正确的实体带来的收益远大于识别错的负面影响。
第二个工作是就是把bert的预测结果转化成csv格式。
```
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
```
## 模型融合（combine.py）
模型融合是很重要的一个步骤，单模得到得结果往往很片面，我使用随机种子（sklearn的train_test_split函数就可以）生成了多个数据分布不同的训练集，分别训练模型并预测。在初赛时我对句子使用了首尾各取512分别预测的策略，对同样一个句子的结果采用了并集合并，见union_combine（）函数。将这些结果再进行交集预测，见intersection_combine()函数。融合5-6个模型后貌似提升效果就很有限了，融合方法是投票表决，对于某个预测样本，在这n个模型中统计某实体出现的次数m，当m>v时，该实体是有效实体，保留，否则舍去。n、v的值就需要自己去尝试效果了。我最后是n=8，v=2。  
核心代码如下：
```
    for i in range(len(finallist)):
        labels = finallist[i]
        dict = {}
        for key in labels:
           dict[key] = dict.get(key, 0) + 1
        if 'nan' in dict.keys():
            del dict['nan']
        labelnew = [k for k,v in dict.items() if v > 2]
```
模型融合的威力是巨大的，我经过上面这样的处理，比单模提升了6-7个百分点。
## 后处理（do_post.py）
在某些任务中，后处理也是十分重要的环节，后处理体现在对结果的再修改。  
### 最大字符串过滤
对于一个字符串a，若b是a的字串，则去除b。这样考虑的目的是有可能出现，八戒网络;猪八戒网;八戒网;猪八戒。模型可能识别很多子串，而且是明显错误的。  
但结果中还可能出现这样的情况：深圳市景华投资发展有限公司;景华。赛题中说明实体的缩写和全称是不同的实体，因此不能简单的直接过滤子串。  
综合考虑上述两种情况，如果某一条数据预测出的实体有n个，当n>k时（这里的k也需要自己尝试，我是k=5），采取最大字符串过滤，否则不处理。
```
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
```
### 基本规则的实体处理
仔细分析文本数据发现，模型在预测时，类似这样的文本段：被骗后保留证据就行了?成?功?处?理?过?的?平?台?点赢策略、策略赢、涨赢策略、天元策略、优选策略、众赢策略众昇策略、中航策略、E策略、期期盈策略、赢远期策略、ETF在线。明显看出、将这些实体分割开了，但模型有时候仍然会识别不出来。  
具体的，当n>k时，将该条文本基于、划分，划分后存入list，当划分后的元素长度大于2且小于7时保留，其余去除。核心代码如下：
```
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
             label = [content for content in label if content != i]    #去除训练集实体
```
由于需要反复调用函数，将上述操作写到函数里，具体见do_post.py：  
```
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
```
经过上面这样的处理，结果又能提升3-4个百分点，效果十分明显。
# 感悟
由于本人时初学者，也是第一次参加该类型比赛，理论知识有限，无法对模型架构动刀，但仍然学到了很多。
