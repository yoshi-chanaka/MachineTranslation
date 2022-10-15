from model import TransformerNMTModel
from mask import generate_square_subsequent_mask
from convert_sequence import convert_sent2idx
from beam_search import beam_decode
from process import post_process

import torch
import numpy as np

def load_model(
    path = '../models/model.pth',
    src_vocab_size = 8000,
    tgt_vocab_size = 8000,
    emb_size = 512,
    nhead = 8,
    ffn_hid_dim = 512,
    num_encoder_layers = 3,
    num_decoder_layers = 3,
    model = None,
    device='cuda'
):
    """
    読み込むモデルの構造と同じ構造のモデルを用意して
    model.load_state_dict(torch.load(path, map_location=device)) で呼び出す
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading Model ... : Using {} device'.format(device))

    if model == None:

        model = TransformerNMTModel(
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            emb_size=emb_size,
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            nhead=nhead,
            dim_feedforward=ffn_hid_dim,
        )

    model = model.to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print('Done')
    
    return model

def greedy_decode(model, src, src_mask, max_len, device, bos_idx, eos_idx):

    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask).to(device)
    ys = torch.ones(1, 1).fill_(bos_idx).type(torch.long).to(device)
    for i in range(max_len - 1): # bosがすでに入っているので-1

        # memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == eos_idx:
            break

    return ys


# actual function to translate input sentence into target language
def translate(
    model: torch.nn.Module,
    src_sentence: str,
    vocab_src,
    vocab_tgt,
    device,
    bos_idx=2,
    eos_idx=3,
    post_proc=False,
    beam_width=2,
    method='greedy',
    max_len=None,
    margin=10
):
    src = torch.Tensor(convert_sent2idx(src_sentence.split(' '), vocab_src, set(list(vocab_src.keys())))).long()
    src = src.unsqueeze(0).reshape(-1, 1)
    num_tokens = src.shape[0]
    if max_len == None:
        max_len = num_tokens + margin
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

    model.eval()
    if method == 'greedy':
        tgt_tokens = greedy_decode(
            model, src, src_mask,
            max_len=max_len, device=device,
            bos_idx=bos_idx, eos_idx=eos_idx
        ).flatten()

    elif method == 'beam':
        tgt_tokens_list, nll_list = beam_decode(
            model, src, src_mask,
            max_len=max_len,
            device=device, bos_idx=bos_idx, eos_idx=eos_idx,
            beam_width=beam_width, num_return=beam_width
        )
        tgt_tokens = tgt_tokens_list[np.argmin(nll_list)]

    vocab_decode = list(vocab_tgt.keys())
    output = " ".join([vocab_decode[idx.item()] for idx in tgt_tokens])
    if post_proc:
        return post_process(output.replace("<bos>", "").replace("<eos>", ""))
    else:
        return output.replace("<bos>", "").replace("<eos>", "")



if __name__ == "__main__":

    from load import load_data
    import sentencepiece as spm

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))

    _, vocab_ja, vocab_en = load_data(only_vocab=True)
    model = load_model(device=device)

    sp_tokenizer = spm.SentencePieceProcessor(model_file='../models/kftt_sp_ja.model')

    input_text = "今日は良い天気ですね。"
    proc_text = ' '.join(sp_tokenizer.EncodeAsPieces(input_text))
    print('\n' + proc_text)
    print(translate(model=model, src_sentence=proc_text, vocab_src=vocab_ja, vocab_tgt=vocab_en, device=device, post_proc=True))
    # Today, it is good weather.

    # train
    input_text = "没年は、確実な記録はないが1506年とするものが多い。"
    proc_text = ' '.join(sp_tokenizer.EncodeAsPieces(input_text))
    print('\n' + proc_text)
    print(translate(model=model, src_sentence=proc_text, vocab_src=vocab_ja, vocab_tgt=vocab_en, device=device, post_proc=True))
    # [結果] Although there is no reliable record of his death, many of them say that he died in 1506.
    # [正解] There is no reliable record of the date of his death, but most put it at 1506.

    # dev
    input_text = "一般に禅宗は知識ではなく、悟りを重んじる。"
    proc_text = ' '.join(sp_tokenizer.EncodeAsPieces(input_text))
    print('\n' + proc_text)
    print(translate(model=model, src_sentence=proc_text, vocab_src=vocab_ja, vocab_tgt=vocab_en, device=device, post_proc=True))
    # [結果] Zen sect is generally not knowledge but emphasized enlightenment instead of enlightenment.
    # [正解] The Zen sects generally emphasize enlightenment over knowledge.

    # test
    input_text = "特記事項のないものは1972年に前所属機関区から転入の手続きがとられている。"
    proc_text = ' '.join(sp_tokenizer.EncodeAsPieces(input_text))
    print('\n' + proc_text)
    print(translate(model=model, src_sentence=proc_text, vocab_src=vocab_ja, vocab_tgt=vocab_en, device=device, post_proc=True))
    # [結果] In 1972, the procedure of renting from the former affiliated organization to the prefecture was considered to be a procedure for entering the property.
    # [正解] Unless otherwise noted, the steam locomotives were transferred from the previous engine depots in 1972.

"""
Using cuda device
Loading Dataset ...
Done
Loading Model ... : Using cuda device
Done

▁ 今日 は 良い 天 気 で す ね 。
Today, it is good weather.

▁ 没 年 は 、 確 実 な 記録 はない が 1 50 6 年 とする ものが多い 。
Although there is no reliable record of his death, many of them say that he died in 1506.

▁ 一般に 禅宗 は 知識 ではなく 、 悟 り を 重 んじ る 。
Zen sect is generally not knowledge but emphasized enlightenment instead of enlightenment.

▁ 特 記事 項 の ない ものは 19 7 2 年に 前 所属 機関 区 から 転 入  の手 続き が と られている 。
In 1972, the procedure of renting from the former affiliated organization to the prefecture was considered to be a procedure for entering the property.
"""
