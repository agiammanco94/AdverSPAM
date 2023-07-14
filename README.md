AdverSPAM: Adversarial SPam Account Manipulation in Online Social Networks
========

AdverSPAM is an Adversarial Machine Learning attacks aiming at preserving statistical correlations and semantic dependences among features in a Spam Online Social Network Account detection scenario.

#### Background 

![AttackScenario](/docs/images/attack_scenario.pdf)
 

#### Reference

If you use adverspam in your research, we would appreciate a citation to the following paper ([bibtex](/docs/references/concone2023adverspam.bib)!)


## Installation

Run the following snippet in a Unix terminal to install adverspam.  

```
git clone https://github.com/agiammanco94/AdverSPAM
cd AdverSPAM
pip install -e . 		# install in editable mode  
```

In order to test the correct installation of the adverspam module, the following example script can be run by opening a Unix terminal in the project root folder and running:

```
python adverspam/example/toy_example.py
```

### Tested Python and Modules Versions

The project has been tested with Python v. 3.10.9 on macos v. 10.15.7; it has also been tested on Python v. 3.10.6 on Linux Ubuntu 22.04.2 LTS.

The following versions of the additional modules has been tested:

- numpy v. 1.25.1
- pandas v. 2.0.3
- scipy v. 1.11.1
- scikit-learn v. 1.3.0


## *Documentation*

Code is documented with both [type hints](https://docs.python.org/3/library/typing.html) and [Google docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).