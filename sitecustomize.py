
# -*- coding: utf-8 -*-
"""
Windows + Python 3.14 workaround.

Avoid platform.uname() / platform.machine() hanging or failing inside WMI calls
when pandas / streamlit import platform metadata.
"""

import platform
from collections import namedtuple

_UnameResult = namedtuple(
    "uname_result",
    ["system", "node", "release", "version", "machine", "processor"],
)

def _safe_uname():
    return _UnameResult(
        system="Windows",
        node="localhost",
        release="10",
        version="10.0.0",
        machine="AMD64",
        processor="AMD64",
    )

platform.uname = _safe_uname
platform.system = lambda: "Windows"
platform.machine = lambda: "AMD64"
platform.processor = lambda: "AMD64"
platform.release = lambda: "10"
platform.version = lambda: "10.0.0"
platform.win32_ver = lambda *args, **kwargs: ("10", "10.0.0", "SP0", "Multiprocessor Free")

if hasattr(platform, "_wmi_query"):
    platform._wmi_query = lambda *args, **kwargs: ["9"]
