from load import load_data
import torch

"""
maskの作成
"""


def generate_square_subsequent_mask(sz):
    """
    triu()で対角成分より下を0に置き換える
    0の部分を-inf, 1の部分を0に置き換える
    tensor([[0., -inf, -inf,  ..., -inf, -inf, -inf],
            [0., 0., -inf,  ..., -inf, -inf, -inf],
            [0., 0., 0.,  ..., -inf, -inf, -inf],
            ...,
            [0., 0., 0.,  ..., 0., -inf, -inf],
            [0., 0., 0.,  ..., 0., 0., -inf],
            [0., 0., 0.,  ..., 0., 0., 0.]])
    """
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, pad_idx=1):
    """
    src, tgt: (系列長, バッチサイズ)
    """
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


if __name__ == "__main__":

    dataset, _, _ = load_data()
    src, tgt = dataset['train'][:8]
    src, tgt = src.transpose(0, 1), tgt.transpose(0, 1)
    print('系列長:\t{} {}'.format(src.shape, tgt.shape))

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = \
        create_mask(src, tgt, pad_idx=1)

    print('src_mask:\t{}\n{}\n'.format(src_mask.shape, src_mask))
    print('tgt_mask:\t{}\n{}\n'.format(tgt_mask.shape, tgt_mask))
    print('src_padding_mask: \t{}\n{}\n'.format(
        src_padding_mask.shape, src_padding_mask))
    print('tgt_padding_mask: \t{}\n{}\n'.format(
        tgt_padding_mask.shape, tgt_padding_mask))

"""
Loading Dataset ...
Done
系列長: torch.Size([152, 8]) torch.Size([152, 8])
src_mask:       torch.Size([152, 152])
tensor([[False, False, False,  ..., False, False, False],
        [False, False, False,  ..., False, False, False],
        [False, False, False,  ..., False, False, False],
        ...,
        [False, False, False,  ..., False, False, False],
        [False, False, False,  ..., False, False, False],
        [False, False, False,  ..., False, False, False]])

tgt_mask:       torch.Size([152, 152])
tensor([[0., -inf, -inf,  ..., -inf, -inf, -inf],
        [0., 0., -inf,  ..., -inf, -inf, -inf],
        [0., 0., 0.,  ..., -inf, -inf, -inf],
        ...,
        [0., 0., 0.,  ..., 0., -inf, -inf],
        [0., 0., 0.,  ..., 0., 0., -inf],
        [0., 0., 0.,  ..., 0., 0., 0.]])

src_padding_mask:       torch.Size([8, 152])
tensor([[False, False, False,  ...,  True,  True,  True],
        [False, False, False,  ...,  True,  True,  True],
        [False, False, False,  ...,  True,  True,  True],
        ...,
        [False, False, False,  ...,  True,  True,  True],
        [False, False, False,  ...,  True,  True,  True],
        [False, False, False,  ...,  True,  True,  True]])

tgt_padding_mask:       torch.Size([8, 152])
tensor([[False, False, False,  ...,  True,  True,  True],
        [False, False, False,  ...,  True,  True,  True],
        [False, False, False,  ...,  True,  True,  True],
        ...,
        [False, False, False,  ...,  True,  True,  True],
        [False, False, False,  ...,  True,  True,  True],
        [False, False, False,  ...,  True,  True,  True]])
"""
