#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scilpy.io.deprecator import deprecate_script
from scripts.scil_bundle_mean_std import main as new_main


DEPRECATION_MSG = """
This script has been merged with new script scil_bundle_mean_std.py. It is 
now available through option '--per_point'.
Please change your existing pipelines accordingly.
"""


@deprecate_script("scil_compute_bundle_mean_std_per_point.py",
                  DEPRECATION_MSG, '1.7.0')
def main():
    new_main()


if __name__ == "__main__":
    main()
