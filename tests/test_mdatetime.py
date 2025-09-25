#!/usr/bin/env python3
"""
  Mylibpy Datetime

  Copyright (c) 2023, Augusto Damasceno.
  All rights reserved.

  SPDX-License-Identifier: BSD-2-Clause
"""

__author__ = "Augusto Damasceno"
__version__ = "0.1"
__copyright__ = "Copyright (c) 2023, Augusto Damasceno."
__license__ = "BSD-2-Clause"

import unittest
import datetime

from mylibpy import mdatetime


class MyTestCase(unittest.TestCase):
    def test_third_friday(self):
        for year in range(1000, 9999):
            for month in range(1, 13):
                tf = mdatetime.third_friday(year=year, month=month)
                self.assertEqual(4, datetime.datetime.strptime(tf, '%Y-%m-%d').weekday())


if __name__ == '__main__':
    unittest.main()
