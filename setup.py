import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
def read(fname):
    return open(os.path.join(here, fname)).read()

setup(
    name = "lc_pulse_avalanche",
    version = "1.0.0",

    author = "Anastasia Tsvetkova",
    author_email = "tsvetkova.lea@gmail.com",

    description = ("Creates GRB light curve generating a stochastic pulse avalanche according to Stern & Svensson (1996)."),
    long_description=read('README.md'),

    license = "MIT",
    url = "https://github.com/anastasia-tsvetkova/lc_pulse_avalanche/",

    packages = ["lc_pulse_avalanche"],
)
