import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class Graph_reasoning(nn.Module):
    def __init__(self, in_fea):
        super(Graph_reasoning, self).__init__()

        self.hidden_fea = in_fea
        self.hidden_fea_2 = in_fea
        self.final_fea = in_fea
        self.fc_encoder = nn.Linear(in_fea, self.hidden_fea)

        self.fc_o = nn.Linear(self.hidden_fea, self.hidden_fea)
        self.fc_a = nn.Linear(self.hidden_fea, self.hidden_fea)
        self.fc_q = nn.Linear(self.hidden_fea, self.hidden_fea)

        self.fc_o_ = nn.Linear(in_fea + self.hidden_fea, self.hidden_fea)
        self.fc_a_ = nn.Linear(self.hidden_fea, self.hidden_fea)
        self.fc_q_ = nn.Linear(in_fea + self.hidden_fea, self.hidden_fea)

        self.w_s_o = nn.Linear(self.hidden_fea, self.hidden_fea_2)
        self.w_s_a = nn.Linear(self.hidden_fea, self.hidden_fea_2)
        self.w_s_q = nn.Linear(self.hidden_fea, self.hidden_fea_2)
        self.w_s_o_ = nn.Linear(self.hidden_fea, self.hidden_fea_2)
        self.w_s_a_ = nn.Linear(self.hidden_fea, self.hidden_fea_2)
        self.w_s_q_ = nn.Linear(self.hidden_fea, self.hidden_fea_2)

        self.w_g_o = nn.Linear(self.hidden_fea_2, self.final_fea)
        self.w_g_a = nn.Linear(self.hidden_fea_2, self.final_fea)
        self.w_g_q = nn.Linear(self.hidden_fea_2, self.final_fea)

        self.res_w_a = nn.Linear(in_fea*2, in_fea)
        self.res_w_q = nn.Linear(in_fea*2, in_fea)
        self.res_w_o = nn.Linear(in_fea*2, in_fea)

    def forward(self, answer, o_a, q_a):
        bs, num, seq_len, feature = answer.size()
        answer_view = answer.view(bs*num, seq_len, -1)
        o_a_view = o_a.view(bs*num, seq_len, -1)
        q_a_view = q_a.view(bs*num, seq_len, -1)

        encoder_feature = self.fc_encoder(answer_view)
        s_obj = F.softmax(self.fc_o(encoder_feature), -2) * encoder_feature
        s_ans = F.softmax(self.fc_a(encoder_feature), -2) * encoder_feature
        s_que = F.softmax(self.fc_q(encoder_feature), -2) * encoder_feature

        e_obj = self.fc_o_(torch.cat([encoder_feature, o_a_view], -1))
        e_ans = self.fc_a_(encoder_feature)
        e_que = self.fc_q_(torch.cat([encoder_feature, q_a_view], -1))

        A_obj = F.softmax(self.w_g_o(F.relu(self.w_s_o(s_obj) + self.w_s_o_(e_obj))), dim=-2)
        A_ans = F.softmax(self.w_g_a(F.relu(self.w_s_a(s_ans) + self.w_s_a_(e_ans))), dim=-2)
        A_que = F.softmax(self.w_g_q(F.relu(self.w_s_q(s_que) + self.w_s_q_(e_que))), dim=-2)

        a_out = (A_ans * answer_view).view(bs, num, seq_len, feature)
        o_out = (A_obj * o_a_view).view(bs, num, seq_len, feature)
        q_out = (A_que * q_a_view).view(bs, num, seq_len, feature)

        a_out = self.res_w_a(torch.cat([a_out, answer], dim=-1))
        o_out = self.res_w_o(torch.cat([o_out, o_a], dim=-1))
        q_out = self.res_w_q(torch.cat([q_out, q_a], dim=-1))

        return a_out, o_out, q_out