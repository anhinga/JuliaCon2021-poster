Materials for Stuttgart Julia Programming Language Meetup on Oct 23, 2021:

https://www.meetup.com/stuttgart-julia-programming-language-meetup-gruppe/events/281500979/

---

Julia 1.6.3. All Julia packages are updated today. (I am using Windows 10 machine to run this at home. My Anaconda is a bit old, from early Summer 2021.)

GitHub is not performing well when rendering large Julia Jupyter notebooks, but it should be visible on the `nbviewer`:

https://nbviewer.org/github/anhinga/JuliaCon2021-poster/blob/main/Stuttgart-Julia-meetup/oct-23-2021.ipynb

If reproducing the `oct-23-2021.ipynb` Julia Jupyter notebook, run

```julia
steps!(changing_m, target_value, loss, opt, m_list, 30)
```

and subsequent cells as many times as needed (the last parameter is how many gradient steps you want to perform at once;
I've done the first call with 10, and then 4 calls with 30, there are still some minor differences between `Gray.(value3(changing_m))`
and `Gray.(target_value)`, although they are barely visible and would disappear after a few dozen more steps).
