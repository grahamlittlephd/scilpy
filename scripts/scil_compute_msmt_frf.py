#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute response functions for multi-shell multi-tissue (MSMT)
constrained spherical deconvolution from DWI data.

The script computes a response function for white-matter (wm),
gray-matter (gm), csf and the mean b=0.

In the wm, we compute the response function in each voxels where
the FA is superior at threshold_fa_wm.

In the gm (or csf), we compute the response function in each voxels where
the FA is below at threshold_fa_gm (or threshold_fa_csf) and where
the MD is below threshold_md_gm (or threshold_md_csf).

Based on B. Jeurissen et al., Multi-tissue constrained spherical
deconvolution for improved analysis of multi-shell diffusion
MRI data. Neuroimage (2014)
"""

import argparse
import logging

from dipy.core.gradients import unique_bvals_tolerance
from dipy.io.gradients import read_bvals_bvecs
import nibabel as nib
import numpy as np

from scilpy.io.image import get_data_as_mask
from scilpy.io.utils import (add_force_b0_arg,
                             add_overwrite_arg, add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist)
from scilpy.reconst.frf import compute_msmt_frf
from scilpy.utils.bvec_bval_tools import extract_dwi_shell


def buildArgsParser():

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('input',
                   help='Path to the input diffusion volume.')
    p.add_argument('bvals',
                   help='Path to the bvals file, in FSL format.')
    p.add_argument('bvecs',
                   help='Path to the bvecs file, in FSL format.')
    p.add_argument('wm_frf_file',
                   help='Path to the output WM frf file, in .txt format.')
    p.add_argument('gm_frf_file',
                   help='Path to the output GM frf file, in .txt format.')
    p.add_argument('csf_frf_file',
                   help='Path to the output CSF frf file, in .txt format.')

    p.add_argument(
        '--mask',
        help='Path to a binary mask. Only the data inside the mask will be '
             'used for computations and reconstruction. Useful if no tissue '
             'masks are available.')
    p.add_argument(
        '--mask_wm',
        help='Path to the WM mask file.')
    p.add_argument(
        '--mask_gm',
        help='Path to the GM mask file.')
    p.add_argument(
        '--mask_csf',
        help='Path to the CSF mask file.')

    p.add_argument(
        '--fa_thr_wm', default=0.7, type=float,
        help='If supplied, use this threshold to select single WM fiber '
             'voxels from the FA inside the WM mask defined by mask_wm. Each '
             'voxel above this threshold will be selected. [%(default)s]')
    p.add_argument(
        '--fa_thr_gm', default=0.2, type=float,
        help='If supplied, use this threshold to select GM voxels from the FA '
             'inside the GM mask defined by mask_gm. Each voxel below this '
             'threshold will be selected. [%(default)s]')
    p.add_argument(
        '--fa_thr_csf', default=0.1, type=float,
        help='If supplied, use this threshold to select CSF voxels from the '
             'FA inside the CSF mask defined by mask_csf. Each voxel below '
             'this threshold will be selected. [%(default)s]')
    p.add_argument(
        '--md_thr_gm', default=0.0007, type=float,
        help='If supplied, use this threshold to select GM voxels from the MD '
             'inside the GM mask defined by mask_gm. Each voxel below this '
             'threshold will be selected. [%(default)s]')
    p.add_argument(
        '--md_thr_csf', default=0.003, type=float,
        help='If supplied, use this threshold to select CSF voxels from the '
             'MD inside the CSF mask defined by mask_csf. Each voxel below '
             'this threshold will be selected. [%(default)s]')

    p.add_argument(
        '--min_nvox', default=100, type=int,
        help='Minimal number of voxels needed for each tissue masks '
             'in order to proceed to frf estimation. [%(default)s]')
    p.add_argument(
        '--tolerance', type=int, default=20,
        help='The tolerated gap between the b-values to '
             'extract\nand the current b-value. [%(default)s]')
    p.add_argument(
        '--roi_radii', default=[10], nargs='+', type=int,
        help='If supplied, use those radii to select a cuboid roi '
             'to estimate the response functions. The roi will be '
             'a cuboid spanning from the middle of the volume in '
             'each direction with the different radii. The type is '
             'either an int or an array-like (3,). [%(default)s]')
    p.add_argument(
        '--roi_center', metavar='tuple(3)', nargs=3, type=int,
        help='If supplied, use this center to span the cuboid roi '
             'using roi_radii. [center of the 3D volume]')

    p.add_argument(
        '--wm_frf_mask', metavar='file', default='',
        help='Path to the output WM frf mask file, the voxels used '
             'to compute the WM frf.')
    p.add_argument(
        '--gm_frf_mask', metavar='file', default='',
        help='Path to the output GM frf mask file, the voxels used '
             'to compute the GM frf.')
    p.add_argument(
        '--csf_frf_mask', metavar='file', default='',
        help='Path to the output CSF frf mask file, the voxels used '
             'to compute the CSF frf.')

    p.add_argument(
        '--frf_table', metavar='file', default='',
        help='Path to the output frf table file. Saves the frf for '
             'each b-value, in .txt format.')

    add_force_b0_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():

    parser = buildArgsParser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    assert_inputs_exist(parser, [args.input, args.bvals, args.bvecs])
    assert_outputs_exist(parser, args, [args.wm_frf_file, args.gm_frf_file,
                                        args.csf_frf_file])

    if len(args.roi_radii) == 1:
        roi_radii = args.roi_radii[0]
    elif len(args.roi_radii) == 2:
        parser.error('--roi_radii cannot be of size (2,).')
    else:
        roi_radii = args.roi_radii
    roi_center = args.roi_center

    vol = nib.load(args.input)
    data = vol.get_fdata(dtype=np.float32)
    bvals, bvecs = read_bvals_bvecs(args.bvals, args.bvecs)

    tol = args.tolerance

    list_bvals = unique_bvals_tolerance(bvals, tol=tol)
    if not np.all(list_bvals <= 1200):
        outputs = extract_dwi_shell(vol, bvals, bvecs,
                                    list_bvals[list_bvals <= 1200],
                                    tol=tol)
        _, data_dti, bvals_dti, bvecs_dti = outputs
        bvals_dti = np.squeeze(bvals_dti)
    else:
        data_dti = None
        bvals_dti = None
        bvecs_dti = None

    mask = None
    if args.mask is not None:
        mask = get_data_as_mask(nib.load(args.mask), dtype=bool)
        if mask.shape != data.shape[:-1]:
            raise ValueError("Mask is not the same shape as data.")
    mask_wm = None
    mask_gm = None
    mask_csf = None
    if args.mask_wm:
        mask_wm = get_data_as_mask(nib.load(args.mask_wm), dtype=bool)
    if args.mask_gm:
        mask_gm = get_data_as_mask(nib.load(args.mask_gm), dtype=bool)
    if args.mask_csf:
        mask_csf = get_data_as_mask(nib.load(args.mask_csf), dtype=bool)

    force_b0_thr = args.force_b0_threshold
    responses, frf_masks = compute_msmt_frf(data, bvals, bvecs,
                                            data_dti=data_dti,
                                            bvals_dti=bvals_dti,
                                            bvecs_dti=bvecs_dti,
                                            mask=mask, mask_wm=mask_wm,
                                            mask_gm=mask_gm, mask_csf=mask_csf,
                                            fa_thr_wm=args.fa_thr_wm,
                                            fa_thr_gm=args.fa_thr_gm,
                                            fa_thr_csf=args.fa_thr_csf,
                                            md_thr_gm=args.md_thr_gm,
                                            md_thr_csf=args.md_thr_csf,
                                            min_nvox=args.min_nvox,
                                            roi_radii=roi_radii,
                                            roi_center=roi_center,
                                            tol=tol,
                                            force_b0_threshold=force_b0_thr)

    masks_files = [args.wm_frf_mask, args.gm_frf_mask, args.csf_frf_mask]
    for mask, mask_file in zip(frf_masks, masks_files):
        if mask_file:
            nib.save(nib.Nifti1Image(mask.astype(np.uint16), vol.get_affine()),
                     mask_file)

    frf_out = [args.wm_frf_file, args.gm_frf_file, args.csf_frf_file]

    for frf, response in zip(frf_out, responses):
        np.savetxt(frf, response)

    if args.frf_table:
        if list_bvals[0] < tol:
            bvals = list_bvals[1:]
        else:
            bvals = list_bvals
        response_csf = responses[2]
        response_gm = responses[1]
        response_wm = responses[0]
        iso_responses = np.concatenate((response_csf[:, :3],
                                        response_gm[:, :3]), axis=1)
        responses = np.concatenate((iso_responses, response_wm[:, :3]), axis=1)
        frf_table = np.vstack((bvals, responses.T)).T
        np.savetxt(args.frf_table, frf_table)


if __name__ == "__main__":
    main()
