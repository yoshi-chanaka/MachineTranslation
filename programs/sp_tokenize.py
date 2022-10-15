import sentencepiece as spm

def tokenize(input_filepath, output_filepath, modelpath):
    """
    input_filepathのファイルを1行ずつ読み込んでトークナイズ
    -> output_filepathに書き込み
    """

    sp = spm.SentencePieceProcessor(model_file=modelpath)

    fo = open(output_filepath, 'w')
    with open(input_filepath, 'r', encoding='utf-8') as fi:
        line = fi.readline()
        while line:
            pieces_list = sp.EncodeAsPieces(line.strip())
            line_encoded = ' '.join(pieces_list)
            fo.write(line_encoded + '\n')
            line = fi.readline()
    fo.close()

    return


if __name__ == "__main__":

    data_dirpath = '../data/kftt-data-1.0/data/orig/'
    for lang in ['ja', 'en']:

        # spモデルの学習
        in_filepath = f'{data_dirpath}kyoto-train.{lang}'
        out_filename = f'kftt_sp_{lang}'

        spm.SentencePieceTrainer.train(
            input=in_filepath,
            model_type='unigram',
            model_prefix='../models/' + out_filename,
            vocab_size=8000,
            pad_id=1,
            bos_id=2,
            eos_id=3,
            unk_piece='<unk>',
            pad_piece='<pad>',
            bos_piece='<bos>',
            eos_piece='<eos>',
        )

        # tokenize
        for kind in ['train', 'dev', 'test']:
            tokenize(
                input_filepath = f'{data_dirpath}kyoto-{kind}.{lang}',
                output_filepath = f'../data/{kind}_{lang}.kftt',
                modelpath = f'../models/{out_filename}.model'
            )
