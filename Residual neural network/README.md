# Residual neural network
### 1. 論文模型架構圖
其結構如下:<br>
<img src="images/model.png"/><br>

### 2. 架構特性與概念<br>
深度學習可以有效的fit非常複雜的系統，但是有一個問題是當模型深度過深的時候常常會梯度消失，為此Residual neural network要解決的就是這個問題。<br>
他解決的方式是利用把中間filter產生的結果加到後面的輸出中，使其就算發生梯度消失的問題也可以保有原本的輸入值，不會使後面的連接層全部梯度消失。
### 3. 數學
對每個堆疊層採用殘差學習。<br>
構建塊如圖所示。在本文中，形式正式構建塊定義為：<br>
<img src="http://latex.codecogs.com/gif.latex?y = F(x,{W{i}}) + x" /><br>
這裡x和y是所考慮的層的輸入和輸出向量。 函數F（x,{Wi}）表示<br>
要學習的殘差映射。 對於圖中的示例<br>
具有兩層，F =W2σ（W1x），其中σ表示<br>
為了簡化符號，省略了ReLU 和bias的操作F + x由方式執行<br>
連接和元素添加。 我們採用加法後的第二個非線性（即σ（y））。<br>
方程<br>
<img src="http://latex.codecogs.com/gif.latex?y = F(x,{W{i}}) + x" /><br>
中的快捷方式連接既不引入額外參數也不引入計算複雜性。 這不僅僅是<br>
在實踐中有吸引力，但在比較中也很重要<br>
普通和剩餘網絡之間。 我們可以公平地比較同時擁有的普通/殘餘網絡<br>
相同數量的參數，深度，寬度和計算成本（除了可忽略的元素加法）。<br>
公式中x和F的尺寸必須相等。<br>
如果不是這種情況（例如，在改變輸入/輸出時）<br>
通道），我們可以執行Ws的線性投影<br>
用於匹配維度的快捷方式連接：<br>
<img src="http://latex.codecogs.com/gif.latex?y = F(x, {W{i}}) + W{sx.}" /><br>
我們也可以在方程<br>
<img src="http://latex.codecogs.com/gif.latex?y = F(x,{W{i}}) + x" /><br>
中使用Ws(一個轉換維度的操作)。<br>
但論文中的實驗表明，理想映射<br>
<img src="http://latex.codecogs.com/gif.latex?y = F(x,{W{i}}) + x" />就足夠了<br>
解決梯度消失<br>
因此Ws僅在匹配尺寸時使用。<br>
殘餘函數F的形式是靈活的。 本文中的實驗涉及具有兩個或兩個的函數F.<br>
三層，而更多的層是可能的。 但如果<br>
F只有一層，方程<br>
<img src="http://latex.codecogs.com/gif.latex?y = F(x,{W{i}}) + x" /><br>
類似於線性層：<br>
<img src="http://latex.codecogs.com/gif.latex?y = F(x, {W{i}}) + W{s}x" /><br>
，是沒意義的。<br>
我們還注意到，雖然上述符號是關於<br>
完全連接的層為簡單起見，它們適用於<br>
卷積層。 函數F（x，{Wi}）可以表示多個卷積層。 每個元素的加法是在兩個特徵映射上每個通道執行的。
### 4. 模型運算邏輯
左邊是普通的cnn 右邊是 Residual neural network 版本的cnn，簡單來說就是把Convlution前的結果+回到output
<img src="images/arg.png"/><br>
<img scr="images/formula.pmg"/><br>
主要要注意的點是假設你的input跟output的filter數量不一樣的時候<br>
ex input 17x17x3 的圖片經過convlution 可能變成 17x17x32，這時候要怎麼做相加，她處理的方式也不困難，就是把原本17x17x3中間缺少的14個 0 filter補齊，然後與原本的input concat 然後再把input跟output相加。
### 5. 閱讀後認為可以發展的方向或心得
Residual neural network雖然有程度的解決梯度消失的問題，但是因為網路通常很深，運算起來的速度實在很慢。

## reference
paper: https://arxiv.org/abs/1512.03385<br>
tensorflow sample code
