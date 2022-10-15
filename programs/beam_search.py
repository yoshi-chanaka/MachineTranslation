from mask import generate_square_subsequent_mask

import torch
from heapq import heapify, heappop, heappush

class BeamSearchNode(object):
    def __init__(self, token_ids, logp, length):
        self.token_ids = token_ids  # トークンID (1, self.length)
        self.logp = logp  # 累積対数尤度
        self.length = length  # 生成トークン長

    def negative_log_likelihood(self):
        return - self.logp / float(self.length - 1 + 1e-6)


def beam_decode(
    model,
    src,
    src_mask,
    max_len,
    device,
    bos_idx,
    eos_idx,
    beam_width,
    num_return
):

    best_list, best_nll = [], []

    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask).to(device)
    seq = torch.ones(1, 1).fill_(bos_idx).type(torch.long).to(device)

    # 初期化
    node = BeamSearchNode(token_ids=seq, logp=0, length=1)
    nodes = []
    heappush(nodes, (node.negative_log_likelihood(), id(node), node))

    for iter in range(max_len - 1): # bosがすでに入っているので-1

        if len(best_list) >= num_return:
            break

        # nllが小さいものからビームサイズ分取り出してnodesをリセット
        n_best_nodes = [heappop(nodes)
                        for _ in range(min(len(nodes), beam_width))]
        nodes = []
        heapify(nodes)

        for _, _, node in n_best_nodes:

            # 次のトークンの予測
            seq = node.token_ids
            tgt_mask = (generate_square_subsequent_mask(
                seq.size(0)).type(torch.bool)).to(device)
            out = model.decode(seq, memory, tgt_mask).transpose(0, 1)
            out_prob = torch.softmax(model.generator(out[:, -1]), dim=1)
            topk_prob, topk_idx = torch.topk(out_prob, beam_width)

            # top-Kに対してheappush．eosが出たらpushしない
            # nllが同じの場合heappopの際エラーになるので，idを含める
            for prob, next_tok in zip(topk_prob.squeeze(0), topk_idx.squeeze(0)):

                new_seq = torch.cat(
                    [seq, torch.ones(1, 1).type_as(src.data).fill_(next_tok)],
                    dim=0
                )
                new_node = BeamSearchNode(
                    token_ids=new_seq,
                    logp=node.logp + torch.log(prob).item(),
                    length=node.length + 1
                )

                if next_tok == eos_idx:
                    best_list.append(new_seq.flatten().cpu())
                    best_nll.append(new_node.negative_log_likelihood())
                else:
                    heappush(
                        nodes,
                        (new_node.negative_log_likelihood(), id(new_node), new_node)
                    )

    for i in range(beam_width - len(best_list)):
        _, _, node = heappop(nodes)
        best_list.append(node.token_ids.flatten().cpu())
        best_nll.append(node.negative_log_likelihood())

    return best_list, best_nll
