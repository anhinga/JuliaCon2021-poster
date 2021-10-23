Materials for Stuttgart Julia Programming Language Meetup on Oct 23, 2021:

https://www.meetup.com/stuttgart-julia-programming-language-meetup-gruppe/events/281500979/

---

Julia 1.6.3. All Julia packages are updated today. (Using Windows 10 machine to run this at home. Anaconda is a bit old, from early Summer 2021.)

If reproducing the `oct-23-2021.ipynb` Julia Jupyter notebook, run

```
steps!(changing_m, target_value, loss, opt, m_list, 30)
```

and subsequent steps as many times as needed (the last parameter is how many gradient steps you want to perform at once;
I've done the first call with 10, and then 4 calls with 30, there are still differences between `Gray.(value3(changing_m))`
and `Gray.(target_value)`, although they are barely visible and would disappear after a few dozen more steps).
