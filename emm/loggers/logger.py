# Copyright (c) 2023 ING Analytics Wholesale Banking
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""We define the logger that is going to be used in the entire package.
We should not configure the logger, that is the responsibility of the user.
By default, in Python the log level is set to WARNING.
"""

import logging

logger = logging.getLogger("emm")


def set_logger(level=logging.INFO, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s"):
    # setup logging for ALL loggers
    # this will print all messages >= INFO (default logging level is WARNING)
    logging.basicConfig(level=level, format=format)


def logSchema(df):
    # Equivalent of printSchema() but for logging
    logger.debug(df._jdf.schema().treeString())


def logShow(df, n: int = 20, truncate: bool = True, vertical: bool = False):
    """Equivalent of show() but for logging
    Copy pasted from
    https://spark.apache.org/docs/latest/api/python/_modules/pyspark/sql/dataframe.html#DataFrame.show
    """
    if not isinstance(n, int) or isinstance(n, bool):
        msg = "Parameter 'n' (number of rows) must be an int"
        raise TypeError(msg)

    if not isinstance(vertical, bool):
        msg = "Parameter 'vertical' must be a bool"
        raise TypeError(msg)

    if isinstance(truncate, bool) and truncate:
        logger.debug(df._jdf.showString(n, 20, vertical))
    else:
        try:
            int_truncate = int(truncate)
        except ValueError as e:
            msg = f"Parameter 'truncate={truncate}' should be either bool or int."
            raise TypeError(msg) from e
        logger.debug(df._jdf.showString(n, int_truncate, vertical))
