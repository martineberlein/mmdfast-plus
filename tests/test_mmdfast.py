import unittest

from mmdfast.core import MMDFast

class TestMMDFast(unittest.TestCase):
    def test_initialize(self):
        mmdfast = MMDFast()

        mmdfast.run()

if __name__ == '__main__':
    unittest.main()
