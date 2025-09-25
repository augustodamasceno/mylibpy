# !/usr/bin/python3
# -*- coding: utf-8 -*-
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

import datetime


def timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def timestamp_file() -> str:
    stamp = timestamp()
    file_stamp = stamp.replace(':', '-').replace('+', 'Z-')
    return file_stamp


def third_friday(year: int, month: int):
    """
    third_friday(year: int, month: int) -> str

    This function calculates the day of the third Friday in a given year and month.

    Args:
        year (int): The year.
        month (int): The month (1-12).

    Returns:
        str: The date of the third Friday in the specified year and month, formatted as "YYYY-MM-DD".

    Raises:
         AssertionError: If the provided month is invalid (1-12).

    Examples:
        >>> third_friday(2023, 12)
        '2023-12-15'

        >>> third_friday(2024, 1)
        '2024-01-19'
    """
    assert isinstance(year, int), f"year must be an integer, passed values is {type(year)}"
    assert isinstance(month, int), f"month must be an integer, passed values is {type(month)}"
    assert 1 <= year <= 9999, f"month must be between 1 and 9999, passed values is {year}"
    assert 1 <= month <= 12, f"month must be between 1 and 12,passed values is {month}"

    first_day = datetime.date(year, month, 1)
    first_day_weekday = first_day.weekday()

    if first_day_weekday == 4:
        the_day = first_day + datetime.timedelta(days=14)
    elif first_day_weekday < 4:
        the_day = first_day + datetime.timedelta(days=((4 - first_day_weekday) + 14))
    else:
        the_day = first_day + datetime.timedelta(days=(25 - first_day_weekday))

    assert 4 == the_day.weekday(), f'third_friday must be Friday (weekday 4) and is {the_day.weekday()}'
    assert the_day.day > 14, f'third_friday must be in the third week: the day is {the_day.day}'

    return the_day.strftime('%Y-%m-%d')


if __name__ == "__main__":
    pass
