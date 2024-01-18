# Contributing

## Clone the Software Repository

Use `git` to clone the software repository:

```default
git clone http://gitlab.asa.dcta.mil.br/asa/asapy.git
```

## Install `asapy` in Development Mode

Go to the source code folder:

```default
cd asapy
```

Install in development mode:

```default
pip install -r requirements.txt
pip install -e .
```

**NOTE**: If you want to create a new *Python Virtual Environment*, please, follow this instruction:

**1.** Enter the following command at the command line to create an environment named asapy using  Anaconda:

```default
conda create --name asapy python=3.8
```

**2.** Activate the new environment:

```default
conda activate asapy
```

**3.** Update pip and setuptools:

```default
pip install --upgrade pip
pip install --upgrade setuptools
```

## Build the Documentation

Generate the documentation:

```default
sphinx-build -b html docs/ docs/build/html
```

The above command will generate the documentation in HTML and it will place it under:

```default
docs/sphinx/_build/html/
```

You can open the above documentation in your favorite browser, as:

```default
firefox docs/sphinx/_build/html/index.html
```
