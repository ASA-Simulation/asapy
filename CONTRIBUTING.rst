Contributing
------------

Clone the Software Repository
+++++++++++++++++++++++++++++


Use ``git`` to clone the software repository::

    git clone http://gitlab.asa.dcta.mil.br/asa/asapy.git


Install ``asapy`` in Development Mode
++++++++++++++++++++++++++++++++++++++++++


Go to the source code folder::

    cd asapy


Install in development mode::

    pip install -r requirements.txt
    pip install -e .


.. note::

    If you want to create a new *Python Virtual Environment*, please, follow this instruction:

    **1.** Enter the following command at the command line to create an environment named asapy using  Anaconda::

        conda create --name asapy python=3.8


    **2.** Activate the new environment::

        conda activate asapy


    **3.** Update pip and setuptools::

        pip install --upgrade pip
        pip install --upgrade setuptools



Build the Documentation
+++++++++++++++++++++++


Generate the documentation::

    sphinx-build -b html docs/ docs/build/html


The above command will generate the documentation in HTML and it will place it under::

    docs/sphinx/_build/html/


You can open the above documentation in your favorite browser, as::

    firefox docs/sphinx/_build/html/index.html
