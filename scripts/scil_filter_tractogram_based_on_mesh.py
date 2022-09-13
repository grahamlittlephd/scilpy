#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""
Now supports sequential filtering condition and mixed filtering object.
For example, --atlas_roi ROI_NAME ID MODE CRITERIA
- ROI_NAME is the filename of a Nifti
- ID is the integer value in the atlas
- MODE must be one of these values: ['any', 'all', 'either_end', 'both_ends']
- CRITERIA must be one of these values: ['include', 'exclude']

If any meant any part of the streamline is contained in the mesh, all means that
all parts of the mesh.

When used with exclude, it means that a streamline entirely in the mesh will
be excluded.
"""

import argparse
import json
import logging
import os
from copy import deepcopy

from dipy.io.stateful_tractogram import set_sft_logger_level
from dipy.io.streamline import save_tractogram
import nibabel as nib
import numpy as np
from scipy import ndimage

from scilpy.io.image import get_data_as_label, get_data_as_mask
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist,
                             read_info_from_mb_bdo)
from scilpy.segment.streamlines import (filter_cuboid, filter_ellipsoid,
                                        filter_grid_roi)


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_tractogram',
                   help='Path of the input tractogram file.')
    p.add_argument('out_tractogram',
                   help='Path of the output tractogram file.')

    p.add_argument('--mesh_roi', nargs=3, action='append',
                   metavar=('ROI_NAME', 'MODE', 'CRITERIA'),
                   help='Filename of a hand drawn ROI (.nii or .nii.gz).')

    p.add_argument('--filtering_list',
                   help='Text file containing one rule per line\n'
                   '(i.e. mesh_roi left_striatum.ply both_ends include).')


    p.add_argument('--no_empty', action='store_true',
                   help='Do not write file if there are no streamlines.')
    p.add_argument('--display_counts', action='store_true',
                   help='Print streamline count before and after filtering')
    p.add_argument('--save_rejected', metavar='FILENAME',
                   help='Save rejected streamlines to output tractogram.')

    add_reference_arg(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    add_json_args(p)

    return p


def prepare_filtering_list(parser, args):
    roi_opt_list = []
    only_filtering_list = True

    if args.mesh_roi:
        only_filtering_list = False
        for roi_opt in args.drawn_roi:
            roi_opt_list.append(['drawn_roi'] + roi_opt)

    if args.filtering_list:
        with open(args.filtering_list) as txt:
            content = txt.readlines()
        for roi_opt in content:
            roi_opt_list.append(roi_opt.strip().split())

    for roi_opt in roi_opt_list:

        _, _, filter_mode, filter_criteria = roi_opt
        
        if filter_mode not in ['any', 'all', 'either_end', 'both_ends']:
            parser.error('{} is not a valid option for filter_mode'.format(
                filter_mode))

        if filter_criteria not in ['include', 'exclude']:
            parser.error('{} is not a valid option for filter_criteria'.format(
                filter_criteria))

    return roi_opt_list, only_filtering_list

def filter_mesh_roi(sft, mesh, filter_type, is_exclude):
    """
    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines to segment.
    mask : numpy.ndarray
        Binary mask in which the streamlines should pass.
    filter_type: str
        One of the 3 following choices, 'any', 'all', 'either_end', 'both_ends'.
    is_exclude: bool
        Value to indicate if the ROI is an AND (false) or a NOT (true).
    Returns
    -------
    new_sft: StatefulTractogram
        Filtered sft.
    ids: list
        Ids of the streamlines passing through the mask.
    """
    line_based_indices = []
    if filter_type in ['any', 'all']:
        line_based_indices = streamlines_in_mask(sft, mask,
                                                 all_in=filter_type == 'all')
    else:
        sft.to_vox()
        sft.to_corner()
        streamline_vox = sft.streamlines
        # For endpoint filtering, we need to keep 2 separately
        # Could be faster for either end, but the code look cleaner like this
        line_based_indices_1 = []
        line_based_indices_2 = []
        for i, line_vox in enumerate(streamline_vox):
            voxel_1 = line_vox[0].astype(np.int16)[:, None]
            voxel_2 = line_vox[-1].astype(np.int16)[:, None]
            if map_coordinates(mask, voxel_1, order=0, mode='nearest'):
                line_based_indices_1.append(i)
            if map_coordinates(mask, voxel_2, order=0, mode='nearest'):
                line_based_indices_2.append(i)

        # Both endpoints need to be in the mask (AND)
        if filter_type == 'both_ends':
            line_based_indices = np.intersect1d(line_based_indices_1,
                                                line_based_indices_2)
        # Only one endpoint need to be in the mask (OR)
        elif filter_type == 'either_end':
            line_based_indices = np.union1d(line_based_indices_1,
                                            line_based_indices_2)

    # If the 'exclude' option is used, the selection is inverted
    if is_exclude:
        line_based_indices = np.setdiff1d(range(len(sft)),
                                          np.unique(line_based_indices))
    line_based_indices = np.asarray(line_based_indices, dtype=np.int32)

    # From indices to sft
    streamlines = sft.streamlines[line_based_indices]
    data_per_streamline = sft.data_per_streamline[line_based_indices]
    data_per_point = sft.data_per_point[line_based_indices]

    new_sft = StatefulTractogram.from_sft(
        streamlines, sft,
        data_per_streamline=data_per_streamline,
        data_per_point=data_per_point)

    return new_sft, line_based_indices

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_tractogram)
    assert_outputs_exist(parser, args, args.out_tractogram, args.save_rejected)
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        set_sft_logger_level('WARNING')

    roi_opt_list, only_filtering_list = prepare_filtering_list(parser, args)
    o_dict = {}

    logging.debug("Loading the tractogram...")
    sft = load_tractogram_with_reference(parser, args, args.in_tractogram)
    if args.save_rejected:
        initial_sft = deepcopy(sft)
    bin_struct = ndimage.generate_binary_structure(3, 2)

    # Streamline count before filtering
    o_dict['streamline_count_before_filtering'] = len(sft.streamlines)

    total_kept_ids = np.arange(len(sft.streamlines))
    for i, roi_opt in enumerate(roi_opt_list):
        logging.debug("Preparing filtering from option: {}".format(roi_opt))
        curr_dict = {}
        
        filter_type, filter_arg, filter_mode, filter_criteria = roi_opt

        curr_dict['filename'] = os.path.abspath(filter_arg)
        curr_dict['type'] = filter_type
        curr_dict['mode'] = filter_mode
        curr_dict['criteria'] = filter_criteria

        is_exclude = False if filter_criteria == 'include' else True

        # Read in mesh and filter
        if filter_type == 'mesh_roi':
            
            img = nib.load(filter_arg)
            mask = get_data_as_mask(img)

            filtered_sft, kept_ids = filter_mesh_roi(sft, mask,
                                                     filter_mode, is_exclude)

        
        logging.debug('The filtering options {0} resulted in '
                      '{1} streamlines'.format(roi_opt, len(filtered_sft)))

        sft = filtered_sft

        if only_filtering_list:
            filtering_Name = 'Filter_' + str(i)
            curr_dict['streamline_count_after_filtering'] = len(
                sft.streamlines)
            o_dict[filtering_Name] = curr_dict

        total_kept_ids = total_kept_ids[kept_ids]

    # Streamline count after filtering
    o_dict['streamline_count_final_filtering'] = len(sft.streamlines)
    if args.display_counts:
        print(json.dumps(o_dict, indent=args.indent))

    if not filtered_sft:
        if args.no_empty:
            logging.debug("The file {} won't be written (0 streamline)".format(
                args.out_tractogram))

            return

        logging.debug('The file {} contains 0 streamlines'.format(
            args.out_tractogram))

    save_tractogram(sft, args.out_tractogram)

    if args.save_rejected:
        rejected_ids = np.setdiff1d(np.arange(len(initial_sft.streamlines)),
                                    total_kept_ids)

        if len(rejected_ids) == 0 and args.no_empty:
            logging.debug("Rejected streamlines file won't be written (0 "
                          "streamline).")
            return

        sft = initial_sft[rejected_ids]
        save_tractogram(sft, args.save_rejected)


if __name__ == "__main__":
    main()
