import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    plt.figure(figsize=(5, 5))
    for k in ['train', 'dev', 'test']:
        f_src = open(f'../data/{k}_ja.kftt', 'r', encoding='utf-8')
        f_tgt = open(f'../data/{k}_en.kftt', 'r', encoding='utf-8')

        line_src = f_src.readline()
        line_tgt = f_tgt.readline()

        src_lengths, tgt_lengths = [], []

        while line_src:
            len_src, len_tgt = len(line_src.strip().split()), len(line_tgt.strip().split())
            if len_src <= 150 and len_tgt <= 150:
                src_lengths.append(len_src)
                tgt_lengths.append(len_tgt)
            line_src = f_src.readline()
            line_tgt = f_tgt.readline()

        f_src.close()
        f_tgt.close()

        plt.scatter(src_lengths, tgt_lengths, alpha=0.3, label=k, marker='.')

    plt.legend()
    plt.xlabel('src length')
    plt.ylabel('tgt length')
    plt.xlim(-5, 155)
    plt.ylim(-5, 155)
    plt.title(f'corrcoef = {np.corrcoef(src_lengths, tgt_lengths)[0][1]}')
    plt.savefig('../figures/length.png')
    plt.show()

