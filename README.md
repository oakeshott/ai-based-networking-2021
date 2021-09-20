# AI-based Networking ジュニアチャレンジ

## 前提 (動作確認済みの環境)

- Python 3.8.6を想定
- Ubuntu 18.04.5 LTS

## プログラムの実行方法

### Pythonパッケージのインストール

```
pip install -r requirements.txt
```
- (Pytorch cpu版をインストールするための定義ファイルであるため注意)

### 利用データのDL

```
# 動画ファイルのDLと解凍
# USER，PASSWORDはそれぞれ指定の値を入れる
$ make download USER=${USER} PASSWORD=${PASSWORD}
```

### 訓練・テストデータの作成

```
# 画像データの前処理 (リサイズ，グレースケール化)
$ make preprocessing
# PNSRとSSIMの計算
$ make similarity
```

- 詳細は`scripts/{preprocessing,similarity}.sh`を参照
- `similarity_measures.tar.gz`には，前処理後のデータ(PSNRとSSIM)が含まれている．
  - `tar zxvf similarity_measures.tar.gz`で解凍後，学習・テストが実行できる．

### 学習とテスト

```
# 学習
$ make train
# テスト
$ make test
```

- 詳細は`scripts/{train,test}.sh`を参照

