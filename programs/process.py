def post_process(input_text):
    """
    input: ▁There ▁is ▁no ▁reliable ▁record ▁of ▁the ▁date ▁of ▁his ▁death , ▁but ▁most ▁put ▁it ▁at ▁15 06 .
    output: There is no reliable record of the date of his death, but most put it at 1506.
    """
    tokens = input_text.strip().split()
    out_tokens = []
    for i, tok in enumerate(tokens):
        if tok[0] == '▁':
            out_tokens.append(tok[1:])
        elif i == 0:
            out_tokens.append(tok)
        else:
            out_tokens[-1] += tok
    return ' '.join(out_tokens)
