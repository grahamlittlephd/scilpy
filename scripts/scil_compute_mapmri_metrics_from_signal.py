#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute the SH coefficient directly on the raw DWI signal.
"""

import argparse

from dipy.core.gradients import gradient_table
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst import mapmri

import nibabel as nib
import numpy as np

from scilpy.io.image import (get_data_as_mask, assert_same_resolution)
from scilpy.io.utils import (add_force_b0_arg, add_overwrite_arg,
                             add_sh_basis_args, assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.reconst.raw_signal import compute_sh_coefficients
from scilpy.reconst.multi_processes import fit_from_model

def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_dwi',
                   help='Path of the dwi volume.')
    p.add_argument('in_bval',
                   help='Path of the b-value file, in FSL format.')
    p.add_argument('in_bvec',
                   help='Path of the b-vector file, in FSL format.')
    p.add_argument('big_delta', type=float,
                   help='Big delta from the diffusion acquisition protocol.')
    p.add_argument('small_delta', type=float,
                   help='Small delta from the diffusion acquisition protocol.')
    p.add_argument('out_mapmri',
                   help='Name of the output MAPMRI file to save.')

    p.add_argument('--radial_order', type=int, default=2,
                   help='Radial order to fit MAPMRI model (int). [%(default)s]')
    add_sh_basis_args(p)
    p.add_argument('--laplacian_regularization', type=bool, default=True,
                   help='Use laplacian regularization in the MAPMRI fit '
                        '(True or False). [%(default)s]')
    p.add_argument('--laplacian_weighting', type=float, default=None,
                   help='regularization weighting for laplacian in the MAPMRI fit '
                        '(float). [Default: will estimate weighting using generalized cross-validation (GCV)]')
    p.add_argument('--positivity_constraint', type=bool, default=True,
                   help='Constrain the solution of the diffusion propogator to positive values '
                        '(True or False). [%(default)s]')
    
    add_force_b0_arg(p)
    p.add_argument('--mask',
                   help='Path to a binary mask.\nOnly data inside the mask '
                        'will be used for computations and reconstruction ')
    add_overwrite_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.in_dwi, args.in_bval, args.in_bvec])
    assert_outputs_exist(parser, args, args.out_mapmri)

    vol = nib.load(args.in_dwi)
    dwi = vol.get_fdata(dtype=np.float32)

    bvals, bvecs = read_bvals_bvecs(args.in_bval, args.in_bvec)
    gtab = gradient_table(args.in_bval, args.in_bvec, b0_threshold=bvals.min(), 
                        big_delta=args.big_delta,
                        small_delta=args.small_delta)

    mask = None
    if args.mask is None:
        mask = None
    else:
        mask_img = nib.load(args.mask)
        assert_same_resolution((vol, mask_img))
        mask = mask_img.get_fdata().astype(np.uint8)
        mask = np.repeat(mask[:,:,:,np.newaxis], dwi.shape[3], axis=3)
        print(mask.shape)
        dwi *= mask
        mask = None
        
        #mask = get_data_as_mask(mask_img, dtype='bool')

    # If laplacian regularization but no weighting given use GCV
    if args.laplacian_regularization and not args.laplacian_weighting:
        laplacian_weighting = 'GCV'
    else:
        laplacian_weighting = args.laplacian_weighting

    mapmri_model = mapmri.MapmriModel(gtab, radial_order=args.radial_order,
                                      laplacian_regularization=args.laplacian_regularization,
                                      laplacian_weighting=laplacian_weighting,
                                      positivity_constraint=args.positivity_constraint)
    # Computing CSD fit
    mapfit = fit_from_model(mapmri_model, dwi,
                             mask=mask, nbr_processes=6)
    
    nib.save(nib.Nifti1Image(mapfit.rtop().astype(np.float32), 
            vol.affine), args.out_mapmri + '_rtop.nii.gz')
    nib.save(nib.Nifti1Image(mapfit.msd().astype(np.float32),
             vol.affine), args.out_mapmri + '_msd.nii.gz')
    nib.save(nib.Nifti1Image(mapfit.qiv().astype(np.float32),
             vol.affine), args.out_mapmri + '_qiv.nii.gz')
    nib.save(nib.Nifti1Image(mapfit.rtap().astype(np.float32),
             vol.affine), args.out_mapmri + '_rtap.nii.gz')
    nib.save(nib.Nifti1Image(mapfit.rtpp().astype(np.float32),
             vol.affine), args.out_mapmri + '_rtpp.nii.gz')

if __name__ == "__main__":
    main()
