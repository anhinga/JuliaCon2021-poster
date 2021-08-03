# Cross-norm

"Request for Plot": is softmax cross-normalization of values fruitful in Transformers?

---

There are various ways to remix the classical Transformers and the proposed scheme.

E.g. instead of __softmax^T(V)__ one can consider __alpha * softmax^T(V)+(1-alpha) * V__.
