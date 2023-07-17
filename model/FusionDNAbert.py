import torch
import torch.nn as nn
from model import DNAbert, SE_modul, infonce


class FusionBERT(nn.Module):
    def __init__(self, config):
        super(FusionBERT, self).__init__()
        self.config = config

        self.config.kmer = self.config.kmers[0]  # 3
        self.bertone1 = DNAbert.BERT(self.config, Dropout=0.15)
        self.bertone2 = DNAbert.BERT(self.config, Dropout=0.3)
        self.bertone3 = DNAbert.BERT(self.config, Dropout=0.9)

        self.config.kmer = self.config.kmers[1]  # 4
        self.berttwo1 = DNAbert.BERT(self.config, Dropout=0.15)
        self.berttwo2 = DNAbert.BERT(self.config, Dropout=0.3)
        self.berttwo3 = DNAbert.BERT(self.config, Dropout=0.9)

        self.config.kmer = self.config.kmers[2]  # 5
        self.bertthree1 = DNAbert.BERT(self.config, Dropout=0.15)
        self.bertthree2 = DNAbert.BERT(self.config, Dropout=0.3)
        self.bertthree3 = DNAbert.BERT(self.config, Dropout=0.9)

        self.config.kmer = self.config.kmers[3]  # 6
        self.bertfour1 = DNAbert.BERT(self.config, Dropout=0.15)
        self.bertfour2 = DNAbert.BERT(self.config, Dropout=0.3)
        self.bertfour3 = DNAbert.BERT(self.config, Dropout=0.9)

        '''
        new version should use nn.Parameter to initialize the parameters
        '''

        # self.Ws = torch.randn(1, 768).cuda()
        # self.Wh = torch.randn(1, 768).cuda()

        self.classification = nn.Sequential(
            nn.Linear(768, 20),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        self.infonce_loss = infonce.InfoNCE()

    def forward(self, seqs):
        se_model = SE_modul.SE_Block(inchannel=768).cuda()
        # print(seqs)
        representationX1 = self.bertone1(seqs)
        representationX2 = self.bertone2(seqs)
        representationX3 = self.bertone3(seqs)

        representationY1 = self.berttwo1(seqs)
        representationY2 = self.berttwo2(seqs)
        representationY3 = self.berttwo3(seqs)

        representationA1 = self.bertthree1(seqs)
        representationA2 = self.bertthree2(seqs)
        representationA3 = self.bertthree3(seqs)

        representationB1 = self.bertfour1(seqs)
        representationB2 = self.bertfour2(seqs)
        representationB3 = self.bertfour3(seqs)

        self.Ws1 = nn.Parameter(torch.randn(1, 768).cuda(), requires_grad=True)
        self.Wh1 = nn.Parameter(torch.randn(1, 768).cuda(), requires_grad=True)
        self.Wa1 = nn.Parameter(torch.randn(1, 768).cuda(), requires_grad=True)
        self.Wb1 = nn.Parameter(torch.randn(1, 768).cuda(), requires_grad=True)
        self.Ws2 = nn.Parameter(torch.randn(1, 768).cuda(), requires_grad=True)
        self.Wh2 = nn.Parameter(torch.randn(1, 768).cuda(), requires_grad=True)
        self.Wa2 = nn.Parameter(torch.randn(1, 768).cuda(), requires_grad=True)
        self.Wb2 = nn.Parameter(torch.randn(1, 768).cuda(), requires_grad=True)
        self.Ws3 = nn.Parameter(torch.randn(1, 768).cuda(), requires_grad=True)
        self.Wh3 = nn.Parameter(torch.randn(1, 768).cuda(), requires_grad=True)
        self.Wa3 = nn.Parameter(torch.randn(1, 768).cuda(), requires_grad=True)
        self.Wb3 = nn.Parameter(torch.randn(1, 768).cuda(), requires_grad=True)

        F1 = torch.sigmoid(
            self.Ws1 * representationX1 + self.Wh1 * representationY1 + self.Wa1 * representationA1 + self.Wb1 * representationB1)
        F2 = torch.sigmoid(
            self.Ws2 * representationX2 + self.Wh2 * representationY2 + self.Wa2 * representationA2 + self.Wb2 * representationB2)
        F3 = torch.sigmoid(
            self.Ws3 * representationX3 + self.Wh3 * representationY3 + self.Wa3 * representationA3 + self.Wb3 * representationB3)

        # targets=torch.arange(seqs.size(0)).to(seqs.device)

        # cos_sim = nn.functional.cosine_similarity(representationX1, representationY1, dim=0, eps=1e-6)
        # F = torch.sigmoid(cos_sim)

        # print(representationX)
        # print(representationY)

        # print(F)
        representation1 = F1 * representationX1 + F1 * representationY1 + F1 * representationA1 + F1 * representationB1
        representation2 = F2 * representationX2 + F2 * representationY2 + F2 * representationA2 + F2 * representationB2
        representation3 = F3 * representationX3 + F3 * representationY3 + F3 * representationA3 + F3 * representationB3
        X1_se_out = se_model(representation1)
        representation1 = representation1 + X1_se_out
        X2_se_out = se_model(representation2)
        representation2 = representation2 + X2_se_out
        X3_se_out = se_model(representation3)
        representation3 = representation3 + X3_se_out

        loss = self.infonce_loss(representation1, representation2, representation3)


        output = self.classification(representation1)

        return output, representation1, loss
