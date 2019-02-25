# Residual neural network
### 1. 論文模型架構圖
其結構如下:<br>
<img src="images/model.png"/><br>

### 2. 架構特性與概念<br>
深度學習可以有效的fit非常複雜的系統，但是有一個問題是當模型深度過深的時候常常會梯度消失，為此Residual neural network要解決的就是這個問題。<br>
他解決的方式是利用把中間filter產生的結果加到後面的輸出中，使其就算發生梯度消失的問題也可以保有原本的輸入值，不會使後面的連接層全部梯度消失。
### 3. 模型運算邏輯
左邊是普通的cnn 右邊是 Residual neural network 版本的cnn，簡單來說就是把Convlution前的結果+回到output
<img src="images/arg.png"/><br>
<img scr="images/formula.pmg"/><br>
主要要注意的點是假設你的input跟output的filter數量不一樣的時候<br>
ex input 17x17x3 的圖片經過convlution 可能變成 17x17x32，這時候要怎麼做相加，她處理的方式也不困難，就是把原本17x17x3中間缺少的14個 0 filter補齊，然後與原本的input concat 然後再把input跟output相加。
### 4. 閱讀後認為可以發展的方向或心得
Residual neural network雖然有程度的解決梯度消失的問題，但是因為網路通常很深，運算起來的速度實在很慢。

## reference
paper: https://arxiv.org/abs/1512.03385<br>
tensorflow sample code
