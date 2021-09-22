I have conducted some experiments for this RFP using Compact Transformers:

_Escaping the Big Data Paradigm with Compact Transformers_: https://arxiv.org/abs/2104.05704

PyTorch implementation: https://github.com/SHI-Labs/Compact-Transformers

I used a mid-late August version captured in my fork: https://github.com/anhinga/Compact-Transformers

The experiments are rather inconclusive so far, but I am going to document them here.

---

I am referencing a version captured in my fork:

The fastest version was just run by default: 

```shell
python main.py directory_to_download_cifar10_to_if_it's_not_yet_there
```

I reproduced it on the latest PyTorch (a later version of PyTorch than their README suggested; I used Windows 10, but it should work elsewhere too).

The result was a bit lower than in their table:

```
[Epoch 200] Top-1 88.34 Time: 64.41
Script finished in 64.41 minutes, best top-1: 88.34, final top-1: 88.34
```

They had more luck and got 89.17 (some of the hyperparameters might have been different in the paper; the training was a bit jittery, but generally OK).

The file to change for my RFP was

https://github.com/anhinga/Compact-Transformers/blob/main/src/utils/transformers.py

and the lines which are potentially involved were lines 207 and 313 which say

```python
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
```

It turned out that one needed to modify line 207.

I started with a buggy modification

```python
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), F.softmax(x, dim=0)).squeeze(-2)
```

which did not make much sense at all, because the code uses 3D tensors and "batched matrix multiplication", 
and I did not take it into account, and with this bug the extent of degradation was as follows:

```
[Epoch 200] Top-1 84.56 Time: 58.91
Script finished in 58.91 minutes, best top-1: 84.61, final top-1: 84.56
```

The correct version was

```python
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), F.softmax(x, dim=1)).squeeze(-2)
```

and that worked better, but still trailed the baseline:

```
[Epoch 200] Top-1 86.99 Time: 60.12
Script finished in 60.12 minutes, best top-1: 87.10, final top-1: 86.99
```

Then I includes a number of printouts to see the magnitude of the values involved.

The code preceeding "line 207" in my local copy of `transformers.py` now looks like

```python
            #print("x.size() = ", x.size())
            #print("torch.max(x) = ", torch.max(x))
            #print("torch.min(x) = ", torch.min(x))
            #mishka_0 = 10*F.softmax(x, dim=1)+x
            #print("mishka_0.size() = ", mishka_0.size())
            #print("torch.max(mishka_0) = ", torch.max(mishka_0))
            #print("torch.min(mishka_0) = ", torch.min(mishka_0))
            #mishka_1 = self.attention_pool(x)
            #print("mishka_1.size() = ", mishka_1.size())
            #mishka_2 = F.softmax(mishka_1, dim=1)
            #print("mishka_2.size() = ", mishka_2.size())
            #mishka_3 = mishka_2.transpose(-1, -2)
            #print("mishka_3.size() = ", mishka_3.size())
            #mishka_4 = torch.matmul(mishka_3, x)
            #print("mishka_4.size() = ", mishka_4.size())
            #mishka_5 = mishka_4.squeeze(-2)
            #print("mishka_5.size() = ", mishka_5.size())
            #time.sleep(1)
            #x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
            #x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), F.softmax(x, dim=1)+x).squeeze(-2)
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x-F.softmax(x, dim=1)).squeeze(-2)
```

allowing me to monitor max and mix values of `x` and similar things. Because `x` was having an order of magnitude larger range
than `softmax(x)`, I decided to try

```python
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), 10*F.softmax(x, dim=1)).squeeze(-2)
```

but that performed worse:

```
[Epoch 200] Top-1 85.64 Time: 60.00
Script finished in 60.00 minutes, best top-1: 85.77, final top-1: 85.64
```

(The remark from my notes was: 'Perhaps, one does need to play with learning rate schedule; 
at the beginning it looked like this is better, but then it started to look like this "10*" is counter-productive'.)
