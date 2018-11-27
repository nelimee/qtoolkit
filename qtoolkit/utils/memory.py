# ======================================================================
# Copyright CERFACS (November 2018)
# Contributor: Adrien Suau (adrien.suau@cerfacs.fr)
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

import inspect
import sys
import typing
from collections import Set, Mapping, deque
from numbers import Number

zero_depth_bases = (str, bytes, Number, range, bytearray)
iteritems = "items"


def getsize(obj_0: typing.Any) -> int:
    """Recursively iterate to sum size of object & members.

    Copied from https://stackoverflow.com/a/30316760/4810787.

    :param obj_0: the object whose size will be computed.
    :return: the size (in bytes) of the given object.
    """
    _seen_ids = set()

    def inner(obj):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if inspect.isclass(obj):
            pass  # bypass remaining control flow and return
        elif isinstance(obj, zero_depth_bases):
            pass  # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, "__dict__"):
            size += inner(vars(obj))
        if hasattr(obj, "__slots__"):  # can have __slots__ with __dict__
            size += sum(
                inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s)
            )
        return size

    return inner(obj_0)


def first_level_sizes_report(obj_0: typing.Any, _seen_ids: set = None) -> dict:
    """Compute the sizes of the first-level members.

    First-level members are the entries of obj_0.__dict__. The printed report
    will display the pairs "[__dict__ key]: [memory consumption in bytes]"
    in ascending order of memory consumption.

    :param obj_0: the object whose size will be computed.
    :param _seen_ids: a set of the already counted IDs.
    """
    if _seen_ids is None:
        _seen_ids = set()

    def inner(obj):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if inspect.isclass(obj):
            pass  # bypass remaining control flow and return
        elif isinstance(obj, zero_depth_bases):
            pass  # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, iteritems):
            size += sum(inner(k) + inner(v) for k, v in getattr(obj, iteritems)())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, "__dict__"):
            size += inner(vars(obj))
        if hasattr(obj, "__slots__"):  # can have __slots__ with __dict__
            size += sum(
                inner(getattr(obj, s)) for s in obj.__slots__ if hasattr(obj, s)
            )
        return size

    if hasattr(obj_0, "__dict__"):
        dictionary = obj_0.__dict__
    elif isinstance(obj_0, Mapping):
        dictionary = obj_0
    else:
        raise NotImplementedError("No dictionary to analyse!")

    sizes = {"__object__": sys.getsizeof(obj_0), "__dict__": sys.getsizeof(dictionary)}
    for k, v in dictionary.items():
        sizes[k] = inner(v) + inner(k)
    return sizes


def list_first_level_size_report(obj_list, _seen_ids: set = None):
    """Build a memory report for the whole list.

    :param obj_list: the list containing objects whose size will be computed.
    :param _seen_ids: a set of the already counted IDs.
    """
    if _seen_ids is None:
        _seen_ids = set()

    final_report = merge_reports(
        (first_level_sizes_report(obj, _seen_ids) for obj in obj_list)
    )
    return final_report


def byte_number_to_human_format(byte_number: int) -> str:
    """Transform a byte number to a human-readable format (1024 -> 1Kio).

    :param byte_number: The number of bytes.
    :return: The human-readable version.
    """
    bin_prefix = ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi", "Yi"]
    current_bin_prefix = 0
    while byte_number > 1024:
        current_bin_prefix += 1
        byte_number /= 1024
    return f"{byte_number:.4g}" + " " + bin_prefix[current_bin_prefix] + "B"


def size_report(raw_size_report: dict) -> str:
    """Construct a report

    :param raw_size_report: The size report returned by the methods in this
        module.
    :return: a human-readable report as a string.
    """
    sorted_sizes = sorted([(v, k) for k, v in raw_size_report.items()])
    max_key_len = max(map(lambda x: len(str(x[1])), sorted_sizes))

    report = ""
    format_string = "{0:<" + str(max_key_len) + "}: {1}"
    total_size = 0
    for size, key in sorted_sizes:
        report += format_string.format(key, byte_number_to_human_format(size))
        total_size += size
    report += format_string.format("TOTAL", byte_number_to_human_format(total_size))
    return report


def merge_reports(reports: typing.Iterable[dict]) -> dict:
    """Merge the size reports in reports.

    :param reports: The reports to merge.
    :return: the merged report.
    """
    final_report = dict()
    for report in reports:
        for k, v in report.items():
            final_report[k] = final_report.get(k, 0) + v
    return final_report
