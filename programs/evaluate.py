from load import load_data
from translate import load_model, translate
from process import post_process

import sacrebleu
import torch

def compute_bleu(model, srcpath, tgtpath, device,
                vocab_src, vocab_tgt, num_sentences=None):

    with open(srcpath, 'r', encoding='utf-8') as f:
        if num_sentences == None:
            num_sentences = len(f.readlines())
        else:
            num_sentences = min(num_sentences, len(f.readlines()))

    pred_corpus, true_corpus = [], []
    f_src = open(srcpath, 'r', encoding='utf-8')
    f_tgt = open(tgtpath, 'r', encoding='utf-8')

    line_src = f_src.readline()
    line_tgt = f_tgt.readline()

    model.eval()
    for i in range(num_sentences):

        """
        in: '▁ 道 元 ( どう げん ) は 、 鎌倉時代 初期の 禅 僧 。'
        out: '▁Do gen ▁was ▁a ▁Zen ▁monk ▁in ▁the ▁early ▁Kamakura ▁period .'
        """
        pred = translate(
            model=model,
            src_sentence=line_src.strip(),
            vocab_src=vocab_src,
            vocab_tgt=vocab_tgt,
            device=device,
            post_proc=False,
            margin=50
        )

        pred_corpus.append(post_process(pred))
        true_corpus.append(post_process(line_tgt.strip()))

        line_src = f_src.readline()
        line_tgt = f_tgt.readline()

    f_src.close()
    f_tgt.close()

    bleu = sacrebleu.corpus_bleu(pred_corpus, [true_corpus])

    return bleu


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))

    _, vocab_ja, vocab_en = load_data(only_vocab=True)
    model = load_model(path = '../models/model.pth', device=device)

    for k in ['train', 'dev', 'test']:
        srcpath = f'../data/{k}_ja.kftt'
        tgtpath = f'../data/{k}_en.kftt'

        bleu = compute_bleu(model, srcpath, tgtpath, device, 
                            vocab_ja, vocab_en, num_sentences=10000)
        print(f'{k}\t{bleu}')

"""
Using cuda device
Loading Dataset ...
Done
Loading Model ... : Using cuda device
Done
train   BLEU = 26.36 55.9/31.3/20.1/13.7 (BP = 1.000 ratio = 1.017 hyp_len = 274195 ref_len = 269739)
dev     BLEU = 16.96 47.2/21.9/11.8/6.8 (BP = 1.000 ratio = 1.121 hyp_len = 27212 ref_len = 24281)
test    BLEU = 19.75 48.3/24.0/14.3/9.2 (BP = 1.000 ratio = 1.113 hyp_len = 29573 ref_len = 26563)
"""
