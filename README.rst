.. image:: docs/figs/cherrypick.png

==========
cherrypick
==========

..
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

How to Use?
----------
You can install the library by using the command

**- pip install cherrypick**


Features
--------

* CherryPick: This feature utilizes competitive scoring techniques, offering a comprehensive pipeline that incorporates multiple methods to measure feature importance. It provides a ranked list of the most crucial variables, sorted based on their calculated scores.


* cherry_score: A unique scoring metric developed exclusively for this library. It evaluates the importance of a variable by assessing its ability to classify data rows based on the performance of other variables.


How it Works?
-------------
In this section, we provide a detailed description of each tool's functionality within this library.

Competitive Score
=================

This technique involves evaluating and ranking the performance of each explanatory variable concerning the dependent variable. After conducting multiple independent evaluation processes, the results are combined to provide an overall understanding. It's called "competitive" because it resembles a competition among explanatory variables, with the top performer winning.

The model supports a standard pipeline with various metrics and classifiers that can be applied to the scoring process. Alternatively, you can create a custom pipeline by fitting the dataframe, where one column represents explanatory variables, and the remaining columns correspond to each evaluation process.

Using the example of the Breast Cancer Wisconsin dataset, we obtain the following dataframe:

.. image:: docs/figs/competitive_score.png
   :width: 1800px
   :alt: competitive_score_winsconsin_dataset


In the table above, we present the entire process of constructing the competitive scoring. The first column displays the explanatory variables, sorted by their final scores in the last column. The intermediate columns represent the evaluation stages and the performance of each variable in those stages. Variables closer to the top have a higher degree of explainability with the target variable.

For example, in this sample, "worst_area" has the highest degree of explainability with target variable, while "worst_radius" performs less well.

cherry_score
============
This score, developed exclusively for this library, is based on the accuracy rate of each explanatory variable's rows. It helps assess how well each row in the dataset is classified by different variables, revealing classification difficulty. Rows are split into two groups: easy and difficult to classify. By examining scores and accuracy rates in each group, we can evaluate variable performance. Important variables should have high accuracy rates for both easy and difficult rows.

.. image:: docs/figs/cherry_score.png
   :width: 1800px
   :alt: competitive_score_winsconsin_dataset

However, when dealing with random variables, relying solely on accuracy rates is insufficient. The ability to classify a row correctly or incorrectly depends largely on the class distribution within the target variable. For example, if the target variable has an equal distribution of easy and challenging cases (50/50), random variables tend to have an equal chance of correctly classifying both types of cases. This observation highlights an additional aspect: the inconsistency inherent in the nature of random variables.

Conversely, in scenarios where a variable exhibits a high correlation with the target, it may achieve a high rate of correct classification for both easy and challenging cases. This situation occurs when the variable reliably predicts outcomes across the entire spectrum of complexity, demonstrating its significant association with the target variable.

However, when a variable shows a higher accuracy rate for challenging cases compared to easy ones, it suggests that the variable struggles to differentiate simpler cases that are readily classified. Instead, it performs better with more complex cases. This behavior indicates a potential random relationship between the variable and the target variable, as the likelihood of correctly classifying both easy and difficult cases tends to be the same, suggesting no significant correlation

To gain a better understanding of how this technique operates, let's delve into a real-world example that served as its inspiration. In Brazil, a nationwide examination known as the Exame Nacional do Ensino Médio (ENEM) plays a pivotal role in determining students' eligibility for admission to higher education institutions. Given the vast scope of this examination, meticulous measures were taken in its design, particularly concerning the scoring system.

Consider this scenario: If the exam were to assign scores solely based on the accuracy rate (i.e., the number of correctly answered questions), a significant issue would emerge. This challenge arises because the majority of the exam comprises multiple-choice questions, with the exception of an essay section. This format allows some individuals to potentially achieve respectable scores by simply making educated guesses. Consequently, there is a risk that university placements could be allocated to individuals who happened to guess a higher number of questions correctly, rather than to those who earnestly prepared for the exam. Such an outcome would deny deserving candidates the opportunity to secure a place at a university.

To mitigate this risk, a mechanism was introduced to discourage random guessing. Therefore, in addition to recognizing and rewarding high-performing students, this mechanism also imposes penalties on those who answer questions randomly.


While the precise mechanism remains undisclosed, we strive to develop an approximation that qualitatively reproduces the underlying scoring concept. Examining the analogy, we can envision the exam questions as the variables under study, and the rows within the columns as the questions participants must address. In the context of the ENEM, participants select from five alternatives, with only one being correct. Within our library, where we primarily address binary classification scenarios, it aligns with participants having the choice between just two alternatives.

Within this framework, the concept of easy and difficult questions plays a crucial role. If a participant correctly answers more difficult questions than easy ones, it implies a likelihood of guessing on the difficult ones, impacting the final score. In essence, individuals with the same number of correct answers may receive different scores. This assumption arises from the premise that correctly answering the most challenging questions necessitates a strong foundational knowledge, which should manifest as a relatively high success rate on easier questions. Failure to exhibit this knowledge suggests a probable reliance on guesswork.

This same logic applies to the cherry_score metric. If a variable excels at correctly classifying the most challenging cases while struggling with the seemingly "obvious" ones, it likely reflects random fluctuations (guessing) rather than a genuine correlation between the variable and the target (indicating knowledge).

In this manner, the library's approach maintains consistency with these foundational principles. We can use the Wisconsin breast cancer dataset, which was previously used for competitive scoring, to test the cherry_score, and the results are quite decent.

.. image:: docs/figs/validation_cherry_score.png
   :width: 1800px
   :alt: competitive_score_winsconsin_dataset

The image presented above captures a snapshot in which the first three and last three variables from the total of 30 variables in the Wisconsin breast cancer dataset were selected based on the cherry_score criteria. Accompanying the original dataset variables, a random variable was introduced. Within this set of 30 variables, it becomes evident that cherry_score effectively distinguishes the random variable from the rest. At the top of the list, we find the same variables as identified in the competitive scoring process, albeit with alternating positions.

I really hope that the features presented can be useful to you. Enjoy!


Credits
-------
Developed by `Lucas Caramês`_.

.. _`Lucas Caramês`: https://github.com/lgpcarames

With code contributions by `Vinicius Ormenesse`_.

.. _`Vinicius Ormenesse`: https://github.com/ormenesse


This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

