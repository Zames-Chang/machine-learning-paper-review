# machine-learning-paper-review


## enviroment
```
tensorflow 1.1.0 
```
## dataset
```
Mnist
```
## Usage
假設你想要跑 bidirectionRNN
```
$ cd bidirectionRNN
```
```
$ python model.py
```
## accurency base
### bidirectionRNN
收斂相比下面的model效果挺不好的，加上振幅很大，代表說他學習的時候並沒有很快地抓到特徵，也沒辦法做到很好的泛化效果<br>
<img src="bidirectionRNN/images/acc.png"/><br>
### crnn
試過的 nn solution 中收斂最快，且振幅很小，代表他抓到特徵的速度很快，也有較高的泛化能力。<br>
<img src="crnn/images/acc.png"/><br>
### google net
介於 crnn 跟 bidirection 表現之間，有著不錯的收斂速度與泛化能力，但是都沒有 crnn 表現得來的優秀。
<img src="google_net/images/acc.png"/><br>
### Random Forest
random forest 是很一群決策樹叢集，所以收斂的非常快，也不太會有 nn 收斂時震盪的問題，當然這是因為他本身的參數小，可能樹的數量多一點就會有震盪的問題<br>
<img src="Random Forest/img/acc.png"/><br>
### Residual neural network
效果意外沒有比傳統的 其他 cnn base 的 model 來的好，不過也許可以把 residual 的概念用在譬如說 crnn 中，也許會有不錯的效果。<br>
<img src="Residual neural network/images/acc.png"/><br>
## loss base (unsurpirvise learning)
### DeepAutoencoder
可以看出來，test 的 loss 跟 train 的 loss 有一大段差距，這是因為資料是圖片的關係，所以導致學習的時候沒有抓到特徵，若圖片的資料應該要使用 cnn base 的 deep auto encoder<br>
<img src="DeepAutoIncoder/img/loss.png"/><br>
### SpectralNet
下圖是經過 SpectralNet 壓縮後，在經過 K-mean 分群的結果<br>
<img src="SpectralNet/images/SpectralNet.png"/><br>
下圖是經過 PCA 降維後，在經過 K-mean 分群的結果<br>
<img src="SpectralNet/images/without_SpectralNet.png"/><br>
效果來說，比單純的 PCA 降維好多了，至少大部分同一種類都有正確的分到鄰近的點，而不是像pca降維的的結果基本上就是亂數分類。
### seqToSeq
擬合的挺不錯，不過這仍是非監督式學習的seq2seq，監督式學習下的成果仍未知。<br>
<img src="seqToSeq\seq2seq-signal-prediction\images\E1.png"/><br>
