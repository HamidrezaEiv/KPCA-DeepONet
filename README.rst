===============
KPCA-DeepONet
===============
``kpca_deeponet`` is a library that utilizes nonlinear model reduction for operator learning.

Operator learning provides methods to approximate mappings between infinite-dimensional function spaces. Deep operator networks (DeepONets) are a notable architecture in this field. Recently, an extension of DeepONet based on the combination of model reduction and neural networks, POD-DeepONet, has been able to outperform other architectures in terms of accuracy for several benchmark tests. In this contribution, we extend this idea towards nonlinear model order reduction by proposing an efficient framework that combines neural networks with kernel principal component analysis (KPCA) for operator learning. Our results demonstrate the superior performance of KPCA-DeepONet over POD-DeepONet.

.. image:: examples/err_vs_d_1d.png
.. image:: examples/err_vs_d_cavity.png
.. image:: examples/err_vs_d_NS.png

Installation
------------

Clone the repository and locally install it in editable mode:

.. code::

  git clone https://github.com/HamidrezaEiv/KPCA-DeepONet.git
  cd KPCA-DeepONet
  pip install -e .
  pip install -r requirements.txt

You can also just pip install the library:


.. code::
  
  pip install kpca_deeponet