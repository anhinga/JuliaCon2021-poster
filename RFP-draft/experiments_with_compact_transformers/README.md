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

---

Then I recalled the penultimate slide of my RPF talk, slide 9 of https://github.com/anhinga/JuliaCon2021-poster/blob/main/RFP-draft/slides-for-rfp.pdf
on mixing `x` and `softmax(x)`, and I've also recalled signed normalization results: https://github.com/anhinga/JuliaCon2021-poster/tree/main/signed-normalization

I think the fine-grained details were somewhat better with `(softmax,softmax^T)` than with signed normalization, but
signed normalization also looked pretty good, and `x` was almost balanced in this sense (`min(x)` tended to be below -3, and `max(x)` tended to be above 5, if I remember correctly).

So, this prompted me to run a mixture, namely `x+softmax(x)` and I started to get what looked like a very tentative and very mild improvement over baseline:

```python
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), F.softmax(x, dim=1)+x).squeeze(-2)
```

with result

```
[Epoch 200] Top-1 88.47 Time: 70.49
Script finished in 70.49 minutes, best top-1: 88.49, final top-1: 88.47
```

Then I decided that since absolute value of `max(x)` tended to be larger than absolute value of `min(x)`, it would make sense to rather consider

```python
            x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x-F.softmax(x, dim=1)).squeeze(-2)
```

Suddently, the optimization became much less stable, getting ahead the previous one, then falling behind as epochs progressed,
and it ended somewhat behind:

```
[Epoch 200] Top-1 88.12 Time: 58.26
Script finished in 58.26 minutes, best top-1: 88.20, final top-1: 88.12
```

I decided to rerun (and to check in the process, how much non-determinism is from run to run).

And this time I got a much better result:

```
[Epoch 200] Top-1 88.67 Time: 58.07
Script finished in 58.07 minutes, best top-1: 88.67, final top-1: 88.67
```

But comparing in the presence of this much jitter is a nightmare, unless one configuration is overwhelmingly better.

One might need to do tons of reruns (in parallel, perhaps) to get statistics...

I made a pause here to ponder the situation a bit.

---
---

I reported this here

https://mlcollective.org/research-jam-4/

https://www.youtube.com/watch?v=SQXIFgcJay4 (11 miniutes starting from 35:25 mark)

and Jason noted that a grid of `(V +/- alpha * sofmax^T(V))` should be computed for a variety of _alpha_.

Then the training jitters would not matter all that much, because we would get a trend curve depending on _alpha_, and this would
provide a reasonably clear picture.

I was offered to use a bit of ML Collective Google Cloud Platform resources to do these series of traininng experiments.
