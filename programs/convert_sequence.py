import torch

"""
文章をIDに変換
"""

def convert_sent2idx(sentence, word2id_encoder, vocab, unk_idx=0):
    """
    input
    sentence: ['今日', 'は', '良い', '天気', 'です', 'ね']
    """
    if vocab == None:
        indices = [word2id_encoder[w] for w in sentence]
    else:
        indices = [word2id_encoder[w] if w in vocab else unk_idx for w in sentence]

    return indices


def convert_dataset2idxset(corpora, word2id_encoder, bos_idx=2, eos_idx=3):
    """
    input: corpora
    {
        'train': ['i have a pen', 'this is a pen', ...],
        'valid': [...],
        'test' : [...],
    }
    """
    vocab = set(list(word2id_encoder.keys()))
    id_seqsets_dict = {}
    for key in corpora.keys():
        corpus = corpora[key]
        id_seqsets_dict[key] = [
            torch.Tensor(
                [bos_idx] +\
                convert_sent2idx(sentence.split(), word2id_encoder, vocab) +\
                [eos_idx]
            ).int() for sentence in corpus
        ]

    return id_seqsets_dict
