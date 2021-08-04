# Cross-norm

"Request for Plot": is softmax cross-normalization of values fruitful in Transformers?

---

There are various ways to remix the classical Transformers and the proposed scheme.

E.g. instead of __softmax^T (V)__ one can consider __alpha * softmax^T (V)+(1-alpha) * V__.

And if one does that, then instead of training this new version of Transformer from
scratch, one can start with a classical pretrained Transformer and __alpha=0__ and
fine-tune it while gradually increasing __alpha__.

(There is a newly added slide in the slide deck to reflect this.)
