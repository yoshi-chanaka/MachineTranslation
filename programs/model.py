from mask import create_mask
from load import load_data

import torch
from torch import Tensor
import torch.nn as nn
import math
from torch.utils.data import DataLoader

"""
参考: LANGUAGE TRANSLATION WITH NN.TRANSFORMER AND TORCHTEXT
https://pytorch.org/tutorials/beginner/translation_transformer.html
"""


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        emb_size: int,
        dropout: float = 0.1,
        maxlen: int = 5000
    ):

        super(PositionalEncoding, self).__init__()
        den = torch.exp(
            - torch.arange(0, emb_size, 2) * math.log(10000) / emb_size
        )
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[:token_embedding.size(0), :]
        )


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class TransformerNMTModel(nn.Module):

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):

        super(TransformerNMTModel, self).__init__()
        """
        torch.nn.Transformer
        https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
        """
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor
    ):

        src = src.transpose(0, 1)
        trg = trg.transpose(0, 1)

        src_mask, tgt_mask = src_mask[0], tgt_mask[0]
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.generator(outs).transpose(0, 1)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )


if __name__ == "__main__":

    dataset, vocab_src, vocab_tgt = load_data()
    batch_size = 8
    print('batch size: {}'.format(batch_size))
    train_dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))

    SRC_VOCAB_SIZE = len(vocab_src)
    TGT_VOCAB_SIZE = len(vocab_tgt)
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    pad_idx = 1

    model = TransformerNMTModel(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        emb_size=EMB_SIZE,
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        nhead=NHEAD,
        dim_feedforward=FFN_HID_DIM,
    )
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    for src, tgt in train_dataloader:

        src = src.transpose(0, 1).to(device)
        tgt = tgt.transpose(0, 1).to(device)

        tgt_input, tgt_output = tgt[:-1, :], tgt[1:, :]
        print('src/tgt shape:\t', src.shape, tgt.shape)

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
        print('mask shape:\t', src_mask.shape, tgt_mask.shape)
        print('padding mask shape:\t', src_padding_mask.shape, tgt_padding_mask.shape)

        src_tr, tgt_input_tr = src.transpose(0, 1), tgt_input.transpose(0, 1)
        with torch.no_grad():
            pred = model.forward(
                    src=src_tr,
                    trg=tgt_input_tr,
                    src_mask=src_mask.repeat(src_tr.shape[0], 1, 1),
                    tgt_mask=tgt_mask.repeat(src_tr.shape[0], 1, 1),
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=tgt_padding_mask,
                    memory_key_padding_mask=src_padding_mask
            )
        pred = pred.transpose(0, 1)
        print('predicted shape\t', pred.shape)
        loss = loss_fn(pred.reshape(-1, pred.shape[-1]), tgt_output.reshape(-1).long())
        print('loss:\t', loss)
        break

"""
Loading Dataset ...
Done
batch size: 8
Using cuda device
src/tgt shape:   torch.Size([152, 8]) torch.Size([152, 8])
mask shape:      torch.Size([152, 152]) torch.Size([151, 151])
padding mask shape:      torch.Size([8, 152]) torch.Size([8, 151])
predicted shape  torch.Size([151, 8, 8000])
loss:    tensor(9.1544, device='cuda:0')
"""
