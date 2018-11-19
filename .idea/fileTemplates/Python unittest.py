#parse("CeCILL-B license.txt")
#parse("snakeToCamel")

"""Test of the #snakeToCamel(${NAME}) class."""

import unittest

import tests.qtestcase as qtest

class #snakeToCamel(${NAME})TestCase(qtest.QTestCase):
    """Unit-tests for the #snakeToCamel(${NAME}) class."""
    pass

if __name__ == '__main__':
    unittest.main()
