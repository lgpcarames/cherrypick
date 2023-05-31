==========
cherrypick
==========


.. image:: https://img.shields.io/pypi/v/cherrypick.svg
        :target: https://pypi.python.org/pypi/cherrypick

.. image:: https://img.shields.io/travis/lgpcarames/cherrypick.svg
        :target: https://travis-ci.com/lgpcarames/cherrypick

.. image:: https://readthedocs.org/projects/cherrypick/badge/?version=latest
        :target: https://cherrypick.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Some tools to help the process of feature selection


* Free software: MIT license
* Documentation: https://cherrypick.readthedocs.io. (work in progress!)


Features
--------

* CherryPick: utilizes the competitive scoring technique, offering a comprehensive pipeline
that incorporates multiple techniques to measure feature importance. It provides a ranked
list of the most important variables, sorted based on their calculated scores.

* cherry_score: unique score developed exclusively for this library.  It assesses the
importance of a variable by evaluating its ability to classify a row based on the
performance of other variables.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
