#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local streamline HARDI tractography using scilpy-only methods -- no dipy (i.e
no cython). The goal of this is to have a python-only version that can be
modified more easily by our team when testing new algorithms and parameters,
and that can be used as parent classes in sub-projects of our lab such as in
dwi_ml.

As in scil_compute_local_tracking:

    The tracking direction is chosen in the aperture cone defined by the
    previous tracking direction and the angular constraint.
    - Algo 'det': the maxima of the spherical function (SF) the most closely
    aligned to the previous direction.
    - Algo 'prob': a direction drawn from the empirical distribution function
    defined from the SF.
    - Algo 'eudx' is not yet available!

Contrary to scil_compute_local_tracking:
    - Input nifti files do not necessarily need to be in isotropic resolution.
    - The script works with asymmetric input ODF.
    - The interpolation for the tracking mask and spherical function can be
      one of 'nearest' or 'trilinear'.
    - Runge-Kutta integration is supported for the step function.

A few notes on Runge-Kutta integration.
    1. Runge-Kutta integration is used to approximate the next tracking
       direction by estimating directions from future tracking steps. This
       works well for deterministic tracking. However, in the context of
       probabilistic tracking, the next tracking directions cannot be estimated
       in advance, because they are picked randomly from a distribution. It is
       therefore recommanded to keep the rk_order to 1 for probabilistic
       tracking.
    2. As a rule of thumb, doubling the rk_order will double the computation
       time in the worst case.

References: [1] Girard, G., Whittingstall K., Deriche, R., and
            Descoteaux, M. (2014). Towards quantitative connectivity analysis:
            reducing tractography biases. Neuroimage, 98, 266-278.
"""
import argparse
from doctest import debug_script
import logging
import math
import time

import dipy.core.geometry as gm
import nibabel as nib

from numpy import loadtxt
from dipy.io.stateful_tractogram import StatefulTractogram, Space, \
                                        set_sft_logger_level
from dipy.io.stateful_tractogram import Origin
from dipy.io.streamline import save_tractogram, load_trk

from scilpy.io.utils import (add_processes_arg, add_sphere_arg,
                             add_verbose_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             verify_compression_th)
from scilpy.image.volume_space_management import DataVolume
from scilpy.tracking.propagator import ODFPropagatorMesh
from scilpy.tracking.seed import SeedGeneratorExplicit
from scilpy.tracking.tools import get_theta
from scilpy.tracking.tracker import Tracker
from scilpy.tracking.utils import (add_mandatory_options_tracking,
                                   add_out_options, add_seeding_options,
                                   add_tracking_options,
                                   verify_streamline_length_options,
                                   verify_seed_options)

import numpy as np
from numpy import asarray
from nibabel.affines import apply_affine

import open3d as o3d

def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    # Some options aren't mandatory for my mesh based tracking
    #add_mandatory_options_tracking(p)
    p.add_argument('in_odf',
                   help='File containing the orientation diffusion function \n'
                        'as spherical harmonics file (.nii.gz). Ex: ODF or '
                        'fODF.')
    p.add_argument('in_seed_list',
                   help='Text file containing explicit list of seed points ')
    p.add_argument('in_mask',
                   help='Tracking mask (.nii.gz).\n'
                        'Tracking will stop outside this mask.')
    p.add_argument('out_tractogram',
                   help='Tractogram output file (must be .trk or .tck).')

    track_g = add_tracking_options(p)
    track_g.add_argument('--algo', default='prob',
                         choices=['det', 'prob'],
                         help='Algorithm to use [%(default)s]')
    add_sphere_arg(track_g, symmetric_only=False)
    track_g.add_argument('--sfthres_init', metavar='sf_th', type=float,
                         default=0.5, dest='sf_threshold_init',
                         help="Spherical function relative threshold value "
                              "for the \ninitial direction. [%(default)s]")
    track_g.add_argument('--rk_order', metavar="K", type=int, default=1,
                         choices=[1, 2, 4],
                         help="The order of the Runge-Kutta integration used "
                              "for the step function.\n"
                              "For more information, refer to the note in the"
                              " script description. [%(default)s]")
    track_g.add_argument('--max_invalid_length', metavar='MAX', type=float,
                         default=1,
                         help="Maximum length without valid direction, in mm. "
                              "[%(default)s]")

    track_g.add_argument('--sh_interp', default='trilinear',
                         choices=['nearest', 'trilinear'],
                         help="Spherical harmonic interpolation: "
                              "nearest-neighbor \nor trilinear. [%(default)s]")
    track_g.add_argument('--mask_interp', default='trilinear',
                         choices=['nearest', 'trilinear'],
                         help="Mask interpolation: nearest-neighbor or "
                              "trilinear. [%(default)s]")

    r_g = p.add_argument_group('Random seeding options')
    r_g.add_argument('--rng_seed', type=int, default=0,
                     help='Initial value for the random number generator. '
                          '[%(default)s]')
    r_g.add_argument('--skip', type=int, default=0,
                     help="Skip the first N random number. \n"
                          "Useful if you want to create new streamlines to "
                          "add to \na previously created tractogram with a "
                          "fixed --rng_seed.\nEx: If tractogram_1 was created "
                          "with -nt 1,000,000, \nyou can create tractogram_2 "
                          "with \n--skip 1,000,000.")

    mesh_g = p.add_argument_group('Mesh based tracking options')
    mesh_g.add_argument('--in_norm_list',default=None,
                        help='List of normals per vertex coordinate. If given'
                        'initiate tracking in the direction of normal.')
    mesh_g.add_argument('--nbr_init_norm_steps', type=int,
                         default=0, dest='nbr_init_norm_steps',
                         help="Number of steps to take in the initial "
                              "direction. [%(default)s]")
    mesh_g.add_argument('--nbr_sps', type=int,
                         default=1, dest='nbr_sps',
                         help="Number of streamlines per seed [%(default)s]")

    mesh_repulsion = p.add_argument_group('Mesh based attachtion/repulsion options')
    mesh_repulsion.add_argument('--repulsion_force_map', default=None,
                        help='Force map to use for repulsion.')
    mesh_repulsion.add_argument('--repulsion_force_weight', type=float, default=0.1,
                        help='Weight of the repulsion force.')

    mesh_set = p.add_argument_group('Surface Enhanced Tracking options')
    mesh_set.add_argument('--set_mesh', default=None,
                        help='Mesh in LPS coordinates for surface enhanced tracking.')
    mesh_set.add_argument('--set_laplacian_smooth_iter', type=float, default=2,
                        help='Number of iterations for the initial laplacian smoothing.')
    mesh_set.add_argument('--set_laplacian_smooth_weight', type=float, default=10.0,
                        help='Weight of the initial laplacian smoothing.')
    mesh_set.add_argument('--set_stiffness_flow_iter', type=float, default=5,
                        help='Number of iterations for the stiffness flow.')
    mesh_set.add_argument('--set_stiffness_flow_diffusion_step', type=float, default=10.0,
                        help='Diffusion step for the stiffness flow.')


    m_g = p.add_argument_group('Memory options')
    add_processes_arg(m_g)

    add_out_options(p)
    add_verbose_arg(p)

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error('Invalid output streamline file format (must be trk or ' +
                     'tck): {0}'.format(args.out_tractogram))

    inputs = [args.in_odf, args.in_seed_list, args.in_mask]
    assert_inputs_exist(parser, inputs)
    assert_outputs_exist(parser, args, args.out_tractogram)

    verify_streamline_length_options(parser, args)
    verify_compression_th(args.compress)
    #verify_seed_options(parser, args)

    theta = gm.math.radians(get_theta(args.theta, args.algo))

    max_nbr_pts = int(args.max_length / args.step_size)
    min_nbr_pts = int(args.min_length / args.step_size) + 1
    max_invalid_dirs = int(math.ceil(args.max_invalid_length / args.step_size))

    # Only track in the direction specified by the input norms
    forward_only = True

    logging.debug("Loading explicit seed points and normals")
    # If seeds are in trk then
    if args.in_seed_list.endswith('.trk'):
        trk_streamlines = load_trk(args.in_seed_list, 'same')
        trk_streamlines.to_voxmm()
        trk_streamlines.to_corner()
        print(trk_streamlines._space)
        print(trk_streamlines._origin)

        seeds = []
        normals = []

        ## TODO check that streamlines always have 2 points
        for coord1, coord2 in trk_streamlines.streamlines:
            seeds.append(tuple(coord1))
            normals.append(tuple(coord2 - coord1))
        seeds = tuple(seeds)
        normals = tuple(normals)
        #print(seeds)
    elif args.in_seed_list.endswith('.txt'):
        seeds = tuple(map(tuple, loadtxt(args.in_seed_list)))
    else:
        parser.error('Seeds must be in .trk or .txt format.')

    if args.in_norm_list is not None:
        normals = tuple(map(tuple, loadtxt(args.in_norm_list)))
    else:
        normals = None

    seed_generator = SeedGeneratorExplicit(seeds, normals)

    nbr_seeds = len(seed_generator.seeds)
    if nbr_seeds == 0:
        parser.error('No seeds points provided.')

    logging.debug("Loading tracking mask.")
    mask_img = nib.load(args.in_mask)
    mask_data = mask_img.get_fdata(caching='unchanged', dtype=float)
    mask_res = mask_img.header.get_zooms()[:3]
    mask = DataVolume(mask_data, mask_res, args.mask_interp)

    logging.debug("Loading ODF SH data.")
    odf_sh_img = nib.load(args.in_odf)
    odf_sh_data = odf_sh_img.get_fdata(caching='unchanged', dtype=float)
    odf_sh_res = odf_sh_img.header.get_zooms()[:3]
    dataset = DataVolume(odf_sh_data, odf_sh_res, args.sh_interp)

    if args.repulsion_force_map is not None:
        logging.debug("Loading repulsion force map.")
        repulsion_force_map_img = nib.load(args.repulsion_force_map)
        repulsion_force_map_data = repulsion_force_map_img.get_fdata(caching='unchanged', dtype=float)
        repulsion_force_map_res = repulsion_force_map_img.header.get_zooms()[:3]
        repulsion_force_map = DataVolume(repulsion_force_map_data, repulsion_force_map_res, 'trilinear')
    else:
        repulsion_force_map = None

    # Calculate stiffness flow
    if args.set_mesh is not None:
        print('Placeholder SET not implemented yet')



    logging.debug("Instantiating tracker.")
    rng_seed = args.rng_seed
    streamlines =[]
    new_seeds = []
    for i in range(0,args.nbr_sps):
        logging.debug("Instantiating propagator.")
        from scilpy.tracking.propagator import ODFPropagatorMesh
        propagator = ODFPropagatorMesh(dataset, args.step_size, args.rk_order, args.algo, args.sh_basis,
                        args.sf_threshold, args.sf_threshold_init, theta, args.sphere,
                        nbr_init_norm_steps=args.nbr_init_norm_steps,
                        repulsion_force=repulsion_force_map,
                        repulsion_weight=args.repulsion_force_weight)

        tracker = Tracker(propagator, mask, seed_generator, nbr_seeds, min_nbr_pts,
                      max_nbr_pts, max_invalid_dirs,
                      compression_th=args.compress,
                      nbr_processes=args.nbr_processes,
                      save_seeds=args.save_seeds,
                      mmap_mode='r+', rng_seed=args.rng_seed,
                      track_forward_only=forward_only,
                      skip=args.skip)

        start = time.time()

        logging.debug("Tracking seed round {}...".format(i+1))
        this_streamlines, this_seeds = tracker.track()
        streamlines = streamlines + this_streamlines
        new_seeds = new_seeds + this_seeds
        rng_seed = rng_seed + 1

    seeds = new_seeds

    str_time = "%.2f" % (time.time() - start)
    logging.debug("Tracked {} streamlines (out of {} seeds), in {} seconds.\n"
                  "Now saving..."
                  .format(len(streamlines), nbr_seeds, str_time))

    # save seeds if args.save_seeds is given
    data_per_streamline = {'seeds': seeds} if args.save_seeds else {}

    # Silencing SFT's logger if our logging is in DEBUG mode, because it
    # typically produces a lot of outputs!
    set_sft_logger_level('WARNING')

    # Compared with scil_compute_local_tracking, using sft rather than
    # LazyTractogram to deal with space.
    # Contrary to scilpy or dipy, where space after tracking is vox, here
    # space after tracking is voxmm.
    # Smallest possible streamline coordinate is (0,0,0), equivalent of
    # corner origin (TrackVis)
    sft = StatefulTractogram(streamlines, mask_img, Space.VOXMM,
                             Origin.TRACKVIS,
                             data_per_streamline=data_per_streamline)
    save_tractogram(sft, args.out_tractogram)


if __name__ == "__main__":
    main()
