#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to convert the format of a surface mesh.

Supported formats are ".vtk", ".vtp", ".ply", ".stl", ".xml", ".obj"
and Freesurfer surface (e.g. lh.white).
"""

import argparse

from trimeshpy.io import load_mesh_from_file

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)

EPILOG = """
References:
[1] St-Onge, E., Daducci, A., Girard, G. and Descoteaux, M. 2018.
    Surface-enhanced tractography (SET). NeuroImage.
"""


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_surface',
                   help='Input surface')

    p.add_argument('out_surface',
                   help='Output surface')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_surface)
    assert_outputs_exist(parser, args, args.out_surface)

    mesh = load_mesh_from_file(args.in_surface)
    mesh.save(args.out_surface)


if __name__ == "__main__":
    main()
