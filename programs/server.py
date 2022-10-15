from load import load_data
from translate import load_model, translate

from flask import Flask, render_template, request
import sentencepiece as spm
import torch
import os

app = Flask(__name__)

"""
https://qiita.com/kujirahand/items/896ea20b28ee2ed96311
"""

@app.route("/", methods=["GET"])
def get():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def post():
    text = request.form["name"]
    if len(text):
        proc_text = ' '.join(sp_tokenizer.EncodeAsPieces(text))
        output = translate(
            model=model, src_sentence=proc_text,
            vocab_src=vocab_ja, vocab_tgt=vocab_en,
            device=device, post_proc=True,
            margin=100,
            method='beam',
            beam_width=10
        )
        return render_template(
            "index.html",
            input=f"{text}",
            output=f"{output}"
        )
    else:
        return render_template(
            "index.html",
            input=f"{text}",
            output=f""
        )

@app.context_processor
def add_staticfile():
    def staticfile_cp(fname):
        path = os.path.join(app.root_path, 'static', fname)
        mtime =  str(int(os.stat(path).st_mtime))
        return '/static/' + fname + '?v=' + str(mtime)
    return dict(staticfile=staticfile_cp)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {} device'.format(device))

    sp_tokenizer = spm.SentencePieceProcessor(
        model_file='../models/kftt_sp_ja.model')

    _, vocab_ja, vocab_en = load_data(only_vocab=True)
    model = load_model(device=device)

    app.run()
