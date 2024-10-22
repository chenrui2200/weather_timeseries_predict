''' Define the Transformer model '''

import torch
import torch.nn as nn
import numpy as np
from InputAndOutput import InputLayer, OutputLayer
from transformer.Layers import EncoderLayer, DecoderLayer
from torch.nn import functional as F

__author__ = "Chen Rui, Oxalate-c"


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    '''
    (1 - ...)：将上三角矩阵中的1变为0，0变为1，得到一个下三角矩阵，其中上三角部分（未来信息）被置为0，底部（当前及之前的信息）被置为1。
    bool()：将张量转换为布尔类型，方便在后续的注意力计算中使用。
    '''
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        '''
        存储非学习参数:通常情况下,我们会将神经网络中的可学习参数(权重、偏移等)注册为 nn.Parameter，
        而一些非学习参数(如均值、方差等)则可以使用 register_buffer 进行注册。这样可以将这些非学习参数与模型本身进行绑定,方便后续的保存和加载。
        减少显存占用:相比于 nn.Parameter，register_buffer 注册的参数不会作为模型的可学习参数进行反向传播更新。
        这样可以减少显存占用,特别是在处理大型模型时非常有用。
        方便模型保存和加载:当我们保存和加载模型时,使用 register_buffer 注册的参数会自动被包含在模型的状态字典中,避免了手动处理这些参数的麻烦。
        增强模型可解释性:通过 register_buffer 注册一些中间计算结果或者辅助性参数,可以帮助理解模型的内部工作机制,增强模型的可解释性。
        '''
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        '''
        作用是生成一个用于位置编码的正弦波表。位置编码是自然语言处理中常用的技术之一,
        它的目的是为输入序列中的每个词添加一个位置信息,从而帮助模型理解词语在序列中的相对位置关系。
        具体来说:
            n_position 表示序列的最大长度,d_hid 表示每个词的隐藏层维度。
            该方法首先定义了一个 get_position_angle_vec 辅助函数,用于计算每个位置对应的正弦波值。
            然后使用 NumPy 生成一个 n_position x d_hid 大小的正弦波编码表 sinusoid_table。其中奇数列存储 sine 值,偶数列存储 cosine 值。
            最后将 sinusoid_table 转换为 PyTorch 张量并增加一个 batch 维度,得到最终的位置编码表。
            这个位置编码表可以作为模型输入的一部分,与词嵌入向量进行拼接,从而为模型提供额外的位置信息。
            相比于学习可训练的位置编码,这种基于正弦波的预定义位置编码在某些情况下可以取得不错的效果,同时计算代价也较低。
        '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, n_position=200, scale_emb=False):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq.long())
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1, scale_emb=False):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.trg_word_emb(trg_seq.long())
        if self.scale_emb:
            dec_output *= self.d_model ** 0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    '''
    n_src_vocab：源语言词汇表大小。表示源语言中不同词的个数。
    n_trg_vocab：目标语言词汇表大小。表示目标语言中不同词的个数。
    d_word_vec：词嵌入维度。表示将每个词映射到固定维度的向量表示的大小。在Transformer模型中，源语言和目标语言的词嵌入维度可以是不同的。
    d_model：模型的隐藏层维度。表示Transformer模型中每个位置的隐藏层表示的大小。它也是词嵌入维度和注意力机制中的查询、键、值向量的维度。
    d_inner：前馈神经网络内部层的维度。表示Transformer模型中前馈神经网络内部隐藏层的大小。
    n_layers：模型的层数。表示Transformer模型中的编码器和解码器堆叠的层数。
    n_head：注意力头的数量。表示Transformer模型中多头自注意力机制的头数，用于捕捉不同的语义信息。
    d_k：注意力机制中的查询（Q）和键（K）向量的维度。
    d_v：注意力机制中的值（V）向量的维度。
    dropout：Dropout概率。表示在模型训练过程中应用的Dropout概率，用于减少过拟合。
    n_position：位置编码的最大长度。表示Transformer模型中位置编码的最大长度，用于表示输入序列中不同位置的相对位置信息。
    '''

    def __init__(
            self,
            n_src_vocab,
            n_trg_vocab,
            d_word_vec=512,
            d_model=512,
            d_inner=2048,
            n_layers=6,
            n_head=8,
            d_k=64,
            d_v=64,
            dropout=0.1,
            n_position=200):

        super().__init__()
        self.d_model = d_model
        self.input_layer = InputLayer()
        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, pad_idx=1, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.output_layer = OutputLayer()

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        '''
        nn.init.xavier_uniform_ 是 PyTorch 中的一个参数初始化方法，用于初始化神经网络模型的参数。
        具体来说，它使用 Xavier（也称为Glorot）初始化方法，对权重矩阵进行均匀分布的初始化。
        Xavier初始化方法旨在使得输入和输出的方差保持一致，从而避免在深度神经网络中出现梯度消失或梯度爆炸的问题。
        这种初始化方法假设输入和输出的方差相等，并且通过适当的缩放因子将权重初始化在一个合适的范围内。
        nn.init.xavier_uniform_ 方法通过从均匀分布中采样权重值，并根据输入和输出维度进行缩放，将初始化的权重赋值给模型的参数。
        这样做可以确保模型的初始权重适合于有效的前向传播和反向传播。
        '''
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_data, device):
        input_data = self.input_layer(input_data).squeeze(-1)
        # 除了最后一行以外的切片
        src_seq = input_data[:, :-1].to(device).to(torch.float32)
        # 除了第一行以外的切片
        trg_seq = input_data[:, 1:].to(device).to(torch.float32)
        trg_mask = get_subsequent_mask(trg_seq)
        enc_output = self.encoder(src_seq)
        dec_output = self.decoder(trg_seq, trg_mask, enc_output)
        trajectory_logit = self.trg_word_prj(dec_output)
        trajectory_logit = self.output_layer(trajectory_logit)

        return trajectory_logit
