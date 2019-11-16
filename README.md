# 2019-CCF-BDCI
本人是初学者。本篇文章主要是开源比赛的代码，并记录一些自己学到的知识，难免有理解不当之处，希望大佬指出。首先感谢一些大佬在比赛前期的开源，使我能快速入门赛题。
# 比赛说明
2019-CCF-BDCI 互联网金融新实体发现。比赛地址：https://www.datafountain.cn/competitions/361 
随着互联网的飞速进步和全球金融的高速发展，金融信息呈现爆炸式增长。投资者和决策者在面对浩瀚的互联网金融信息时常常苦于如何高效的获取需要关注的内容。针对这一问题，金融实体识别方案的建立将极大提高金融信息获取效率，从而更好的为金融领域相关机构和个人提供信息支撑。
# 比赛结果 
初赛第7名  
# 运行环境
## 环境  
腾讯云ti-one  
python 3.5  
tensorflow 1.12  
## 硬件  
tesla P40, 大约需要12g显存。如果没有大显存，调小maxlen和batch size即可，可能效果会略有下降  
## 运行  
修改各种路径，直接run.sh就可以了。
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
