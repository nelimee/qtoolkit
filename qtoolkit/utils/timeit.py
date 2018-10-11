# ======================================================================
# Copyright CERFACS (October 2018)
# Contributor: Adrien Suau (suau@cerfacs.fr)
#
# This software is governed by the CeCILL-B license under French law and
# abiding  by the  rules of  distribution of free software. You can use,
# modify  and/or  redistribute  the  software  under  the  terms  of the
# CeCILL-B license as circulated by CEA, CNRS and INRIA at the following
# URL "http://www.cecill.info".
#
# As a counterpart to the access to  the source code and rights to copy,
# modify and  redistribute granted  by the  license, users  are provided
# only with a limited warranty and  the software's author, the holder of
# the economic rights,  and the  successive licensors  have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using, modifying and/or  developing or reproducing  the
# software by the user in light of its specific status of free software,
# that  may mean  that it  is complicated  to manipulate,  and that also
# therefore  means that  it is reserved for  developers and  experienced
# professionals having in-depth  computer knowledge. Users are therefore
# encouraged  to load and  test  the software's  suitability as  regards
# their  requirements  in  conditions  enabling  the  security  of their
# systems  and/or  data to be  ensured and,  more generally,  to use and
# operate it in the same conditions as regards security.
#
# The fact that you  are presently reading this  means that you have had
# knowledge of the CeCILL-B license and that you accept its terms.
# ======================================================================

"""Set of function/decorators related to timing."""

import time
import typing


def time_single_execution(function_name: str):
    """Parametrised decorator to time each execution of the decorated function.

    :param function_name: name of the decorated function. This name is printed
    alongside the execution time of the function.
    :return: A decorator that will print the execution time of the decorated
    callable after each call of this callable.
    """

    def real_timeit(method: typing.Callable[..., typing.Any]):
        """Decorator to time each execution of the decorated function.

        :param method: the decorated function.
        :return: a new function that wraps the decorated function and prints
        the execution time of the decorated function.
        """

        def timed(*args, **kw):
            start = time.time()
            result = method(*args, **kw)
            end = time.time()
            elapsed_time_ms = int(1000 * (end - start))
            print("Execution time for {}: {}ms".format(function_name,
                                                       elapsed_time_ms),
                  flush=True)
            return result

        return timed

    return real_timeit


class Timer:
    """Compute and display the elapsed time between tic() and toc()."""

    def __init__(self) -> None:
        """Initialise the Timer instance."""
        self._time: int = 0

    def tic(self) -> None:
        """Set the internal reference time of the timer as "now"."""
        self._time = time.time()

    def toc(self, text: str) -> None:
        """Print the elapsed time since the last call to self.tic().

        :param text: Description of the task executed between the last
        call to self.tic() and the call to self.toc(text).
        """
        end_time: int = time.time()
        print("Task '{}' ended in {}s".format(text, end_time - self._time))
