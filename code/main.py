import torch
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel
import transformers
import torch.nn as nn
import torch.nn.functional as F
import loadData
from torch.utils.data import RandomSampler, DataLoader, TensorDataset, SequentialSampler
import numpy as np
from tqdm import tqdm, trange
from torch.utils.checkpoint import checkpoint

class RobertaClassification(nn.Module):
    def __init__(self, config, num_label):
        super(RobertaClassification, self).__init__()
        self.roberta = RobertaModel.from_pretrained('../../roberta_large_mnli/')
        self.fc = nn.Linear(config.hidden_size, num_label)
        self.drop = nn.Dropout(config.hidden_dropout_prob)
        self.loss = nn.CrossEntropyLoss(reduction='sum')

        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.constant_(self.fc.bias, 0.)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        _, pooled_output = self.roberta(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        #_, pooled_output = checkpoint(self.roberta, input_ids, attention_mask, token_type_ids)
        pooled_output = self.drop(pooled_output)
        logits = self.fc(pooled_output)

        if labels is not None:
            loss = self.loss(logits, labels)
            return loss, logits
        else:
            return logits

def get_features(samples, max_len, tknzr, label_list=None):
    '''

    :param samples: sents. sents[i] = [question, title+passage, label]
    :param max_len:
    :param tknzr:
    :param label_list: if label_list is not None, should map elements in label_list to 0, 1, ..., len(label_list)-1
    :return: input_id, segment_id (0 vectors), input_mask_id, label_id
    '''
    label_id = []
    input_ids = []
    input_masks_ids = []
    segment_ids = []
    for idx, ls in enumerate(samples):
        print(idx)
        sent1 = ls[0]
        sent2 = ls[1]
        label = ls[2]
        '''
        tokens1 = tknzr.encode(sent1)[:max_len]
        tokens2 = tknzr.encode(sent2)[:max_len]
        input_id = tknzr.build_inputs_with_special_tokens(tokens1, tokens2)[:max_len]
        '''
        dic = tknzr.encode_plus(sent1, sent2, max_length=max_len, pad_to_max_length=True, return_token_type_ids=True, padding_side='right', return_tensors='pt', truncation_strategy="longest_first")
        '''
        if input_id[-1] != tknzr.eos_token_id:
            # cut the sentence
            input_id[-1] = tknzr.eos_token_id
        segment_id = [0] * len(input_id)
        input_masks_id = [1] * len(input_id)
        while len(input_masks_id) < max_len:
            input_id.append(0)
            input_masks_id.append(0)
            segment_id.append(0)
        '''
        #label_id.append(label)
        label_id.append(label)
        input_ids.append(dic['input_ids'])
        segment_ids.append(dic['token_type_ids'])
        input_masks_ids.append(dic['attention_mask'])
        #input_ids.append(input_id)
        #segment_ids.append(segment_id)
        #input_masks_ids.append(input_masks_id)
    return input_ids, input_masks_ids, segment_ids, label_id

def create_dataloader(input_ids, mask_ids, segments_ids, label_ids, batch_size, train=True):

    data = TensorDataset(input_ids, mask_ids, segments_ids, label_ids)
    if train:
        sampler = RandomSampler(data)
    else:
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size, drop_last=train)

    return dataloader

def main():

    device_ids=[0,1]

    init_lr = 1e-5
    max_epochs = 10
    max_length = 256
    batch_size = 2
    gradient_accu = 16

    num_label = 2

    train_mode = True

    prev_acc = 0.
    max_acc = 0.

    config = RobertaConfig.from_pretrained('../../roberta_large_mnli/')
    tknzr = RobertaTokenizer.from_pretrained('../../roberta_large_mnli/')

    #tknzr = tokenization.FullTokenizer(vocab_file="data/annotated_wikisql_and_PyTorch_bert_param/vocab_uncased_L-12_H-768_A-12_lstm.txt",
    #                                   do_lower_case=True)

    train_data, test_data = loadData.load_data()

    train_input_ids, train_mask_ids, train_segment_ids, train_label_ids = get_features(train_data, max_length, tknzr)
    test_input_ids, test_mask_ids, test_segment_ids, test_label_ids = get_features(test_data, max_length, tknzr)

    all_input_ids = torch.cat(train_input_ids, dim=0).long()
    all_input_mask_ids = torch.cat(train_mask_ids, dim=0).long()
    all_segment_ids = torch.cat(train_segment_ids, dim=0).long()
    all_label_ids = torch.Tensor(train_label_ids).long()
    train_dataloader = create_dataloader(all_input_ids, all_input_mask_ids, all_segment_ids, all_label_ids,
                                         batch_size=batch_size, train=True)

    all_input_ids = torch.cat(test_input_ids, dim=0).long()
    all_input_mask_ids = torch.cat(test_mask_ids, dim=0).long()
    all_segment_ids = torch.cat(test_segment_ids, dim=0).long()
    all_label_ids = torch.Tensor(test_label_ids).long()
    test_dataloader = create_dataloader(all_input_ids, all_input_mask_ids, all_segment_ids, all_label_ids,
                                        batch_size=batch_size, train=False)

    model = RobertaClassification(config, num_label=num_label).cuda(device_ids[0])
    model = torch.nn.DataParallel(model, device_ids=device_ids)


    optimizer = transformers.AdamW(model.parameters(), lr=init_lr, eps=1e-8)
    optimizer.zero_grad()
    #scheduler = transformers.get_constant_schedule_with_warmup(optimizer, len(train_dataloader) // (batch_size * gradient_accu))
    #scheduler = transformers.get_linear_schedule_with_warmup(optimizer, len(train_dataloader) // (batch_size * gradient_accu), (len(train_dataloader) * max_epochs * 2) // (batch_size * gradient_accu), last_epoch=-1)

    global_step = 0
    for epoch in range(max_epochs):
        model.train()
        if train_mode:
            loss_avg = 0.
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                global_step += 1
                batch = [t.cuda() for t in batch]
                input_id, input_mask, segment_id, label_id = batch
                loss, _ = model(input_id, segment_id, input_mask, label_id)
                loss = torch.sum(loss)
                loss_avg += loss.item()
                loss = loss / (batch_size * gradient_accu)
                loss.backward()
                if global_step % gradient_accu == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    #if epoch == 0:
                        #scheduler.step()
            print(loss_avg / len(train_dataloader))

        model.eval()

        final_acc = 0.
        num_test_sample = 0
        tot = [0, 0]
        correct = [0, 0]
        for input_id, input_mask, segment_id, label_id in test_dataloader:
            input_id = input_id.cuda()
            input_mask = input_mask.cuda()
            segment_id = segment_id.cuda()
            label_id = label_id.cuda()

            with torch.no_grad():
                loss, logit = model(input_id, segment_id, input_mask, label_id)
            logit = logit.detach().cpu().numpy()
            label_id = label_id.to('cpu').numpy()
            acc = np.sum(np.argmax(logit, axis=1) == label_id)
            pred = np.argmax(logit, axis=1)
            for i in range(label_id.shape[0]):
                tot[label_id[i]] += 1
                if pred[i] == label_id[i]:
                    correct[label_id[i]] += 1
            final_acc += acc
            num_test_sample += input_id.size(0)

        print("final acc:", final_acc / num_test_sample)
        if final_acc / num_test_sample > max_acc:
            max_acc = final_acc / num_test_sample
            print("save...")
            torch.save(model.state_dict(), "../model/model.ckpt")
            print("finish")
        '''
        if final_acc / num_test_sample <= prev_acc:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.8
                '''
        prev_acc = final_acc / num_test_sample
        tp = correct[1]
        tn = correct[0]
        fp = tot[1] - correct[1]
        fn = tot[0] - correct[0]
        rec = tp / (tp + fn)
        pre = tp / (tp + fp)
        print("recall:{0}, precision:{1}".format(rec, pre))
        print("f:", 2 * pre * rec / (pre + rec))
        print("acc:", (tp + tn) / (tp+tn+fp+fn))


if __name__ == '__main__':
    main()
