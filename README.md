文件目录：
hw_boolq
----code
--------main.py
--------loadData.py
----data
--------train.jsonl
--------dev.jsonl
----model
--------model.ckpt（存储的模型）

robert_large_mnli
----config.json
----merges.txt
----pytorch_model.bin
----vocab.json
（这四个文件在分享的roberta中，将文件名前缀roberta-large-mnli-去掉即可）


数据增强可以直接生成新的data_aug.jsonl（格式同train.jsonl和dev.jsonl），然后调用loadData/load_data_aug()读出，拼接在105行的train_data处（train_data = train_data + load_data_aug()）即可

若数据增强使得数据集太大，可以加入sample或用其他方法修改代码，提高速度

运行当前代码需要两张卡，0号卡需要9000+M的显存，其余需要3000+M的显存 
可以通过修改export CUDA_VISIBLE_DEVICES=0,1,2,3来限定用卡，也可以更改device_ids来限定用卡

不同的卡数对应了不同的batch_size，当前大概只能是用n张卡，batch_size就为n。gradient_accu参数为gradient_accumulation参数，用于将多个小batch合成一个大batch再进行梯度更新。可以适当调节gradient_accu和batch_size，使得两者的乘积在16, 32之间（论文推荐batch_size为32）

需要测试eval阶段的代码时，确保model/model.ckpt存在，然后将train_mode参数改为False，max_epochs改为1

当前参数设置，在第4个epoch的acc达到最高，为0.8544

每个epoch的loss: 0.92087 --- 0.52443 --- 0.29871 -- 0.17319