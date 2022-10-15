from convert_sequence import convert_dataset2idxset

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset

def build_vocab(vocab_filepath):
    """
    .vocabファイルの語彙にidを振り辞書型の変数として返す
    """
    with open(vocab_filepath, 'r') as f:

        vocab_dict = {}
        line = f.readline()
        idx = 0
        while line:
            token, _ = line.split('\t', 1)
            vocab_dict[token] = idx
            line = f.readline()
            idx += 1

    return vocab_dict


def load_corpus(filepath):
    """
    filepathのファイルを行ごとにリストで返す
    """
    sentences = []
    with open(filepath, 'r') as f:
        line = f.readline()
        while line:
            sentences.append(line.strip())
            line = f.readline()

    return sentences


def load_data(max_len=150, only_vocab=False, extension='kftt', pad_idx=1):
    """
    データの読み込み. データセットとvocabがとれる
    """

    print('Loading Dataset ...')
    vocab_src = build_vocab(f'../models/kftt_sp_ja.vocab')
    vocab_tgt = build_vocab(f'../models/kftt_sp_en.vocab')
    if only_vocab:
        print('Done')
        return None, vocab_src, vocab_tgt

    corpus_src, corpus_tgt = {}, {}
    for kind in ['train', 'dev', 'test']:
        corpus_src[kind] = load_corpus(f'../data/{kind}_ja.{extension}')
        corpus_tgt[kind] = load_corpus(f'../data/{kind}_en.{extension}')

        # max_lenより長いものは切り捨て
        # <bos>と<eos>がつくので最終的に最大系列長はmax_len + 2になる
        train_src, train_tgt = [], []
        for text_src, text_tgt in zip(corpus_src[kind], corpus_tgt[kind]):
            if len(text_src.split(' ')) <= max_len and len(text_tgt.split(' ')) <= max_len:
                train_src.append(text_src)
                train_tgt.append(text_tgt)
        corpus_src[kind], corpus_tgt[kind] = train_src, train_tgt

    src_idsequences = convert_dataset2idxset(corpus_src, vocab_src)
    tgt_idsequences = convert_dataset2idxset(corpus_tgt, vocab_tgt)

    dataset = {}
    for kind in ['train', 'dev', 'test']:
        srcset, tgtset = src_idsequences[kind], tgt_idsequences[kind]
        src_size = len(srcset)
        set_all = pad_sequence(srcset + tgtset, batch_first=True, padding_value=pad_idx)
        srcset, tgtset = set_all[:src_size], set_all[src_size:]
        # print(srcset.shape, tgtset.shape)
        dataset[kind] = TensorDataset(srcset, tgtset)
        """ longに変換する """

    print('Done')
    return dataset, vocab_src, vocab_tgt


if __name__ == "__main__":
    dataset, vocab_src, vocab_tgt = load_data()
    print(dataset['train'][:][0].shape, dataset['train'][:][1].shape)
    print(dataset['dev'][:][0].shape, dataset['dev'][:][1].shape)
    print(dataset['test'][:][0].shape, dataset['test'][:][1].shape)

"""
Loading Dataset ...
Done
torch.Size([439136, 152]) torch.Size([439136, 152])
torch.Size([1165, 151]) torch.Size([1165, 151])
torch.Size([1158, 144]) torch.Size([1158, 144])
"""
