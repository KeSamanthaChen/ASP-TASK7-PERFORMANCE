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
    with subtest("Testing matrix multiplication performance"):
        matrix_driver = test_root().joinpath("matrix_driver")
        matrix_driver_output = test_root().joinpath("matrix_driver_output")

        if not matrix_driver.exists():
            run(["make", "-C", str(test_root()), "matrix_driver"])

        if not matrix_driver_output.exists():
            libmatrix_dir = str(project_root())
            libmatrix_optimized_dir = str(test_root())
            library_path = ":".join([libmatrix_dir, libmatrix_optimized_dir])
            extra_env = {"LD_LIBRARY_PATH": library_path}

            with open(str(matrix_driver_output), mode="w+") as stdout:
                run([str(matrix_driver)], extra_env=extra_env, stdout=stdout)

        with open(str(matrix_driver_output), mode="r") as stdin:
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
