```bash
conda create --name gpu_stack python=3.8 -y
conda activate gpu_stack
conda install ipykernel jupyter nb_conda_kernels pandas numba cudatoolkit
conda install -c conda-forge cupy cudnn cutensor nccl
conda install -c conda-forge jupyter_contrib_nbextensions
conda install -c conda-forge jupyter_nbextensions_configurator
conda install tbb
conda install -c numba icc_rt
pip install RISE
```


1. JIT takes more time to compile function 1st time. - Done
2. If things are running in object mode, code written in numba will take more time than pure python mode. - Done
3. `@njit` or `@jit(nopython=True)` is same. Always `njit` to ensure performant code. 
4. Use numpy arrays, not python lists with numba.
5. Whenever possible use `@vectorize`. Write a function as scaler. Use `@vectorize`. It will work for scaler & vector both.
6. When writing numba code you need to have a trade off between pythonic code vs c-ish code. Example, `for index in indexes` will change to `for index in range(len(indexes))`.



7. To use all threads using `@njit(nogil=True)`. If you don't want use it, you will not be benefited from using `ThreadPoolExecutor`.
8. Using thread pool executor is same as using `parallel=True` flag in `@njit` decorator. Make sure the problem is `embarrassingly parallel`. In case if you are using this use `prange` in place of `range` and install TBB.


9. `float32` is great. use it wherever possible.


10. If you don't care much about floating point precision `fastmath=True` is your friend.
11. Remember numba supports limited functions. A good starting point is to check numba support for numpy & python.


12. If you really with efficient python code always use `@njit`. It will fail to execute if you code is falling back to object mode. This is not the case with `@jit`. It will execute your code in object mode and will throw some warning. Due to this it can be very slow and defeat the purpose if the objective is to achieve speed. 
13. To call back python inside `@njit` mode use `nb.objmode` context manager. Numba is `JIT` compiler. Static typing is a must have for cython code. This makes cython less flexible compared to numba.



14. LLVM takes care of the backend and different architecture. Hence if you want to leverage GPU or CPU, it is less stressful.
15. To check all the dependencies use `numba -s`.
16. MKL, BLAS, SVML, TBB is great explore them. If you have intel CPU MKL + Intel Python is great to generate synthetic data.
17. Talk about `ufuncs` & `gufuncs`. Also, about `vectorize` & `guvectorize`. 
18. If have GPU used, `target='cuda'`.
19. `numba.stencil` is great for convolution or sliding window or any other neighborhood computation. For C callbacks use `numba.cfunc`. 
20. Show how to use numba with pandas.
21. There is a test suite present in numpy with all sort of numerical comparison. Wherever possible use that. 
22. Measure, measure & measure. Snakeviz is a good profiler if you prefer web UIs. Prioritize what needs to be optimized.
23. deepcopy does not work with numba.
24. Talk about default profiler which comes with numba. `from time import pref_counter`. `foo.py_func()` we can use to call python function.
25. Talk about single and multiple instructions in numba.
26. Example that how we are using this to calculate DTW distance using numba in multiplier.






Reference:

* http://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/
