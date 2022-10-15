# 機械翻訳
## 概要
* ディープラーニングを用いた機械翻訳モデル
* 翻訳モデルはOpenNMTやfairseqなどニューラル翻訳のツールキット等は使用せず，PyTorchのみを用いて実装
    * [PyTorchのチュートリアル](https://pytorch.org/tutorials/beginner/translation_transformer.html)を参考
* 学習は[京都フリー翻訳タスク (KFTT)コーパス](http://www.phontron.com/kftt/index-ja.html)を利用
* `programs/server.py`を実行することでブラウザ上での翻訳が可能
* ブラウザでの翻訳例は[screenshots/translate_example*.png](https://github.com/yoshi-chanaka/MachineTranslation/tree/master/screenshots)にて確認可能

## 翻訳モデルの構築
* sp_tokenize.py
    * sentencepieceのモデルの学習と，コーパスのトークナイズ
* train.py
    * 機械翻訳モデルの学習
* evaluate.py
    * BLEUスコアの計測
* translate.py
    * translate関数により翻訳
    * 貪欲法による翻訳と，ビーム探索による翻訳が選択可能
    * ビーム探索は`programs/beam_search.py`にて実装

## 参考
[LANGUAGE TRANSLATION WITH NN.TRANSFORMER AND TORCHTEXT](https://pytorch.org/tutorials/beginner/translation_transformer.html)

