应平台方要求，12月来更代码。 
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
之后是将训练集.csv转换成bert可以读入的格式.txt，用BIO标注将实体标注出来。具体处理参见BDCI_NER.py。  
