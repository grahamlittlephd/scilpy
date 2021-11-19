#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute powder average (mean diffusion weighted image) from set of
diffusion images.

By default will output an average image calculated from all images with
non-zero bvalue.

specify --bvalue to output an image for a single shell

Script currently does not take into account the diffusion gradient directions
being averaged.
"""

from os.path import splitext
import re

import argparse
import logging

import nibabel as nib
import numpy as np

from dipy.io.gradients import read_bvals_bvecs

# Aliased to avoid clashes with images called mode.
from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.utils.filenames import add_filename_suffix, split_name_with_nii
from nibabel.tmpdirs import InTemporaryDirectory

logger = logging.getLogger("Compute_Powder_Average")
logger.setLevel(logging.INFO)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_dwi',
                   help='Path of the input diffusion volume.')
    p.add_argument('in_bval',
                   help='Path of the bvals file, in FSL format.')
    p.add_argument('out_avg',
                   help='Path of the output file')

    add_overwrite_arg(p)

    p.add_argument('--mask', dest='mask', metavar='file',
                   help='Path to a binary mask.\nOnly data inside the '
                   'mask will be used for powder avg. (Default: %(default)s)')

    p.add_argument('--shell', dest='shell', type=int, default=None,
                   help='bvalue (shell) to include in powder average.\nIf '
                   'not specified will include all volumes with a non-zero bvalue')

    p.add_argument('--shell_thr', dest='shell_thresh', type=int, default='50',
                   help='Include volumes with bvalue +- the specified '
                   'threshold.\ndefault: 50')

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    inputs = [args.in_dwi, args.in_bval]
    if args.mask:
        inputs.append(args.mask)

    assert_inputs_exist(parser, inputs)

    assert_outputs_exist(parser, args, args.out_avg)

    img = nib.load(args.in_dwi)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine
    if args.mask is None:
        mask = None
    else:
        mask = get_data_as_mask(nib.load(args.mask), dtype='uint8')

    # Read bvals (bvecs not needed at this point)
    logging.info('Performing powder average')
    bvals, bvecs = read_bvals_bvecs(args.in_bval, None)

    # Select diffusion volumes to average
    if not(args.shell):
        # If no shell given, average all diffusion weigthed images
        bval_idx = bvals > 0 + args.shell_thresh
    else:
        min_bval = args.shell - args.shell_thresh
        max_bval = args.shell + args.shell_thresh
        bval_idx = np.logical_and(bvals > min_bval, bvals < max_bval)

    powder_avg = np.squeeze(np.mean(data[:, :, :, bval_idx], axis=3))

    if args.mask:
        powder_avg = powder_avg * mask

    powder_avg_img = nib.Nifti1Image(powder_avg.astype(np.float32), affine)
    nib.save(powder_avg_img, args.out_avg)

    del powder_avg_img


if __name__ == "__main__":
    main()