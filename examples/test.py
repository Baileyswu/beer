import copy
import sys
sys.path.append('/home/danliwoo/gplab/beer')
from beer import __init__
import beer
import numpy as np
import torch

from tests.test_normal import TestNormalFullCovarianceSet

case = TestNormalFullCovarianceSet
case.setUp()
case.test_create()