from setuptools import setup


def get_requirments():
    # https://stackoverflow.com/a/53069528
    import os
    lib_folder = os.path.dirname(os.path.realpath(__file__))
    requirement_path = lib_folder + '/requirements.txt'
    install_requires = []
    if os.path.isfile(requirement_path):
        with open(requirement_path) as f:
            install_requires = f.read().splitlines()
    return install_requires


setup(
    name='lmwrapper',
    version='0.03.01',
    author='David Gros',
    description='Wrapper around language model APIs',
    license='MIT',
    packages=['lmwrapper'],
    install_requires=get_requirments(),
    python_requires='>=3.10',
)

"""
mamba install -c pytorch onnxruntime onnx datasets evaluate protobuf accelerate
mamba install -c huggingface safetensors
mamba install -c xformers xformers
openai peft salesforce-codetf
"""