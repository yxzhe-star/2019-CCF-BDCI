
# bert直接用于分类问题
python run_classifier.py --data_dir=data/imdb --task_name=imdb --vocab_file=modelParams/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=modelParams/uncased_L-12_H-768_A-12/bert_config.json --output_dir=output/ --do_train=True --do_eval=True --init_checkpoint=modelParams/uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=16 --learning_rate=5e-5 --num_train_epochs=1.0


# bert + textcnn用于分类问题
python bert_cnn.py --data_dir=data/imdb --task_name=imdb --vocab_file=modelParams/uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=modelParams/uncased_L-12_H-768_A-12/bert_config.json --output_dir=cnn_output/ --do_train=True --do_eval=True --init_checkpoint=modelParams/uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=16 --learning_rate=5e-5 --num_train_epochs=1.0


# bert + bilstm_crf用于命名实体识别
python bert_blstm_crf.py 
--data_dir ='D:deep learning/BDCI_ner/'
--task_name='ner'
--vocab_file = 'E:/roberta_zh_l12/vocab.txt' 
--bert_config_file = 'E:/roberta_zh_l12/bert_config.json' 
--output_dir='D:deep learning/BDCI_ner/fine-tune-ner/'
--do_train=True 
--do_eval=False 
--init_checkpoint = 'E:/roberta_zh_l12/bert_model.ckpt' 
--max_seq_length=256 
--train_batch_size=6


