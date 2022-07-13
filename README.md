# Task 7 - Performance

This weeks exercise requires you to optimize matrix multiplication as well as mandelbrot set calculation. You're provided with implementations in `matrix.cpp` and `mandelbrot.cpp`. 

## Task 7.1 - Matrix

Optimize the code in `matrix.cpp` in terms of performance. Your task is to get as close as possible to our optimized code that we've committed as binaries (or even surpass it!).

## Task 7.2 - Mandelbrot

Optimize the code in `mandelbrot.cpp` in terms of performance. Your task is to get as close as possible to our optimized code that we've committed as binaries (or even surpass it!).

## Deliveries

Deliver `libmatrix.so` and `libmandelbrot.so` in the project root directory exporting the symbols from `matrix.h` and `mandelbrot.h`.

## Notes

- If you want to multi-thread your code, use **a maximum of 4 (four)** threads. You can expect all inputs of your code to be a multiple of 4 in size.
- If you want to use SIMD in your code, use **only SSE and SSE2** intrinsics. You can filter the instructions sets on the "Intel Intrinsics Guide" (see references).
- We know that there are faster algorithms for matrix multiplication such as [Strassen Algorithm](https://en.wikipedia.org/wiki/Strassen_algorithm). We **do not** want you to implement that. We want you to **optimize the naive version** given in `matrix.cpp`. The same goes for mandelbrot if you find a faster algorithm.
- For mandelbrot, you're required to use the parameters given in `tests/mandelbrot_params.h`. Do not change them.
- Regarding Rust, use the instrinsics from [core::arch::x86_64](https://doc.rust-lang.org/core/arch/x86_64/index.html) and **not** [std::simd](https://doc.rust-lang.org/nightly/std/simd/index.html).


## References

- [Matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication)
- [Mandelbrot Set](https://en.wikipedia.org/wiki/Mandelbrot_set)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [Rust Intrinsics](https://doc.rust-lang.org/core/arch/x86_64/index.html)

