#parse("CeCILL-B license.txt")
#parse("snakeToCamel")

import unittest

import tests.qtestcase as qtest

class #snakeToCamel(${NAME})TestCase(qtest.QTestCase):
    pass

if __name__ == '__main__':
    unittest.main()
