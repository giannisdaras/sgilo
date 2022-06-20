import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="score-ilo",
    version="0.0.1",
    author="giannisdaras",
    author_email="daras.giannhs@gmail.com",
    description="Score-Guided Intermediate Layer Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/giannisdaras/score-ilo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.9',
)
