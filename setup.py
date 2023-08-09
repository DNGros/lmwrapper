from setuptools import setup


def get_requirments(requirements_file):
    # https://stackoverflow.com/a/53069528
    import os

    lib_folder = os.path.dirname(os.path.realpath(__file__))
    requirement_path = lib_folder + "/" + requirements_file
    install_requires = []
    if os.path.isfile(requirement_path):
        with open(requirement_path) as f:
            install_requires = f.read().splitlines()
    return install_requires


setup(
    name="lmwrapper",
    version="0.03.01",
    author="David Gros",
    description="Wrapper around language model APIs",
    license="MIT",
    packages=["lmwrapper"],
    install_requires=get_requirments("requirements.txt"),
    extras_require={
        "huggingface": get_requirments("requirements-hf.txt"),
        "ort": get_requirments("requirements-hf.txt") + get_requirments("requirements-ort.txt")
        + ["optimum[onnxruntime]>=1.11.0"],
        "ort-gpu": get_requirments("requirements-hf.txt") + get_requirments("requirements-ort.txt")
        + ["optimum[onnxruntime-gpu]>=1.11.0"],
    },
    python_requires=">=3.10",
)
