from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

test_deps = [
    "pytest",
    "autograd",
    "numdifftools",
    "click",
    "pandas",
]
extras = {
    'test': test_deps,
}

setup(
    name="delsmm",
    version="0.0.1",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "torch>=1.7",
        "matplotlib",
        "numpy",
        "scipy",
        "ceem>=0.0.2",
        "pykalman>=0.9.5"
    ],
    tests_require=test_deps,
    extras_require=extras,
)