# Cross-norm

"Request for Plot": is softmax cross-normalization of values fruitful in Transformers?

---

There are various ways to remix the classical Transformers and the proposed scheme.

E.g. instead of __softmax^T (V)__ one can consider __alpha * softmax^T (V)+(1-alpha) * V__.

And if one does that, then instead of training this new version of Transformer from
scratch, one can start with a classical pretrained Transformer and __alpha=0__ and
fine-tune it while gradually increasing __alpha__.

(There is a newly added slide in the slide deck to reflect this.)

---

This was presented on August 4 at 

https://mlcollective.org/research-jam-3/

Video recording (about 5 min) is at 28:46 of the video recording of the presentation part of
the ML Collective Open Collab Research Jam:

https://www.youtube.com/watch?v=EktncBW69lQ&t=1726s
