import torch
import torch.nn as nn

from transformers import BertTokenizer, BertConfig, BertModel

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained("bert-base-uncased")

'''DNA bert model'''


class BERT(nn.Module):
    def __init__(self, config, Dropout=0.15, pooling='pooler'):
        super(BERT, self).__init__()
        self.config = config

        # 加载预训练模型参数
        self.kmer = config.kmer
        if self.kmer == 3:
            self.pretrainpath = '../pretrain/DNAbert_3mer'
        elif self.kmer == 4:
            self.pretrainpath = '../pretrain/DNAbert_4mer'
        elif self.kmer == 5:
            self.pretrainpath = '../pretrain/DNAbert_5mer'
        elif self.kmer == 6:
            self.pretrainpath = '../pretrain/DNAbert_6mer'
        # 将配置文件搞好\
        self.pooling = pooling
        dropout = Dropout
        self.setting = BertConfig.from_pretrained(
            self.pretrainpath,
            num_labels=2,  # 最后分成两类
            finetuning_task="dnaprom",
            cache_dir=None,
        )

        self.setting.attention_probs_dropout_prob = dropout
        self.setting.hidden_dropout_prob = dropout
        # ----------------------------------------------------
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrainpath)
        self.bert = BertModel.from_pretrained(self.pretrainpath, config=self.setting)

    def forward(self, seqs):
        # print(seqs)
        seqs = list(seqs)
        kmer = [[seqs[i][x:x + self.kmer] for x in range(len(seqs[i]) + 1 - self.kmer)] for i in range(len(seqs))]
        kmers = [" ".join(kmer[i]) for i in range(len(kmer))]
        token_seq = self.tokenizer(kmers, return_tensors='pt')
        # print(token_seq)
        input_ids, token_type_ids, attention_mask = token_seq['input_ids'], token_seq['token_type_ids'], token_seq[
            'attention_mask']
        if self.config.cuda:
            representation = self.bert(input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda(),
                                       output_hidden_states=True)
        else:
            representation = self.bert(input_ids, token_type_ids, attention_mask, output_hidden_states=True)

        if self.pooling == 'cls':
            return representation.last_hidden_state[:, 0]  # [batch, 768]

        if self.pooling == 'pooler':
            return representation.pooler_output  # [batch, 768]

        if self.pooling == 'last-avg':
            last = representation.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]

        if self.pooling == 'first-last-avg':
            first = representation.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = representation.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]

        return representation
