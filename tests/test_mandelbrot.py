#!/usr/bin/env python3

from testsupport import (
    subtest,
    test_root,
    run,
    ensure_library,
    warn,
    info,
    project_root,
)
import tempfile
import sys


def main() -> None:
    with subtest("Testing mandelbrot performance"):
        mandelbrot_driver = test_root().joinpath("mandelbrot_driver")
        mandelbrot_driver_output = test_root().joinpath("mandelbrot_driver_output")

        if not mandelbrot_driver.exists():
            run(["make", "-C", str(test_root()), "mandelbrot_driver"])

        if not mandelbrot_driver_output.exists():
            libmandelbrot_dir = str(project_root())
            libmandelbrot_optimized_dir = str(test_root())
            library_path = ":".join([libmandelbrot_dir, libmandelbrot_optimized_dir])
            extra_env = {"LD_LIBRARY_PATH": library_path}

            with open(str(mandelbrot_driver_output), mode="w+") as stdout:
                run([str(mandelbrot_driver)], extra_env=extra_env, stdout=stdout)

        with open(str(mandelbrot_driver_output), mode="r") as stdin:
            lines = stdin.read().splitlines()

        times = [float(x) for x in lines]
        optimized_time = times[0]
        solution_time = times[1]
        fraction = optimized_time / solution_time

        info(
            f"Optimized: {optimized_time}, Solution: {solution_time}, Fraction: {fraction}"
        )

        needed_fraction = float(sys.argv[1])
        if fraction < needed_fraction:
            warn(f"Not fast enough: {fraction} < {needed_fraction}")
            sys.exit(1)


if __name__ == "__main__":
    main()
