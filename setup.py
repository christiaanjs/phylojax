import setuptools

setuptools.setup(
    name="phylojax",
    version="0.0.1",
    packages=setuptools.find_packages(),
    install_requires=["jax[cpu]"],
)
