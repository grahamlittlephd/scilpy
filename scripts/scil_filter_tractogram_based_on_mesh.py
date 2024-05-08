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
import numpy as np

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_json_args,
                             add_overwrite_arg,
                             add_reference_arg,
                             add_verbose_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)

from dipy.io.stateful_tractogram import StatefulTractogram

import open3d as o3d


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_tractogram',
                   help='Path of the input tractogram file.')
    p.add_argument('out_tractogram',
                   help='Path of the output tractogram file.')

    p.add_argument('--mesh_roi', nargs=3, action='append',
                   metavar=('ROI_NAME', 'MODE', 'CRITERIA'),
                   help='Filename of 3D mesh in RAS mm.')

    p.add_argument('--mesh_intersect', nargs=3, action='append',
                   metavar=('ROI_NAME', 'MODE', 'CRITERIA'),
                   help='Filename of 3D mesh in RAS mm.')

    p.add_argument('--filtering_list',
                   help='Text file containing one rule per line\n'
                   '(i.e. mesh_roi left_striatum.ply both_ends include).')

    p.add_argument('--dist_thr', type=float, default=0.1,
                   help='distance in mm, if mesh_roi used points within this threshold are considered in the mesh. [%(default)s]')

    p.add_argument('--intersect_min_angle', type=float, default=0.5,
                   help='minimum angle in radians, if intersect_roi used remove streamlines that\n'
                   'intersect the mesh at an anlge smaller than this value. [%(default)s]')
    p.add_argument('--intersect_max_angle', type=float, default=1.0,
                   help='maximum angle in radians, if intersect_roi used remove streamlines that\n'
                   'intersect the mesh at an angle larger than this value. [%(default)s]')
    p.add_argument('--intersect_dist', type=float, default=3.0,
                   help='distance in mm, only look this far from ends of streamlines for mesh intersections. [%(default)s]')

    p.add_argument('--no_empty', action='store_true',
                   help='Do not write file if there is no streamline.')

    p.add_argument('--display_counts', action='store_true',
                   help='Print streamline count before and after filtering')

    p.add_argument('--save_rejected', metavar='FILENAME',
                   help='Save rejected streamlines to output tractogram.')

    add_reference_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)
    add_json_args(p)

    return p


def prepare_filtering_list(parser, args):
    roi_opt_list = []
    only_filtering_list = True

    if args.mesh_roi:
        only_filtering_list = False
        for roi_opt in args.mesh_roi:
            roi_opt_list.append(['mesh_roi'] + roi_opt)

    if args.mesh_intersect:
        only_filtering_list = False
        for roi_opt in args.mesh_intersect:
            roi_opt_list.append(['mesh_intersect'] + roi_opt)

    if args.filtering_list:
        with open(args.filtering_list) as txt:
            content = txt.readlines()
        for roi_opt in content:
            roi_opt_list.append(roi_opt.strip().split())

    for roi_opt in roi_opt_list:

        _, _, filter_mode, filter_criteria = roi_opt

        if filter_mode not in ['any', 'all', 'either_end', 'both_ends', 'only_start', 'only_end']:
            parser.error('{} is not a valid option for filter_mode'.format(
                filter_mode))

        if filter_criteria not in ['include', 'exclude']:
            parser.error('{} is not a valid option for filter_criteria'.format(
                filter_criteria))

    return roi_opt_list, only_filtering_list


def streamline_endpoints_in_mesh(sft, target_mesh, both_ends=False, only_start=False, only_end=False, dist_thr=0.1):
    """
    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines to segment.
    target_mesh : o3d.geometry.TriangleMesh
        Mesh oriented RAS mm in which.
    both_ends : bool
        If True, both end points must be within mesh.
    only_start : bool
        If True, only start point must be within mesh.
    only_end : bool
        If True, only end point must be within mesh.
    dist_thr : float
        Distance threshold in mm to consider a point in the mesh.
    Returns
    -------
    ids : list
        Ids of the streamlines that pass filter (any or all).
    """
    # tractogram should be in RASmm same with mesh
    sft.to_rasmm()
    sft.to_corner()

    # Legacy mesh needed for ray casting
    legacy_mesh = o3d.t.geometry.TriangleMesh.from_legacy(target_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(legacy_mesh)

    # TODO definitely a faster way to do this, the casting is fast but the for loop sucks
    # probably do a reshape get all signed distances and then reshape back

    # use ray casting to find distance from each point in tractogram to mesh
    filter_list = []
    for i in range(0, len(sft.streamlines)):
        # Get first and last endpoint of each streamline
        query_points = [sft.streamlines[i][0], sft.streamlines[i][-1]]

        # Cast rays and get distance (negative values if inside)
        signed_distance = np.array(scene.compute_signed_distance(query_points))

        if both_ends:
            # Check if both endpoints are within mesh
            if signed_distance[0] < dist_thr and signed_distance[1] < dist_thr:
                filter_list.append(i)
        elif only_end:
            # Check to see if end point of streamline is within mesh
            if signed_distance[-1] < dist_thr:
                filter_list.append(i)
        elif only_start:
            # Check to see if start point of streamline is within mesh
            if signed_distance[0] < dist_thr:
                filter_list.append(i)
        else:
            # Check if either endpoint is within mesh
            if signed_distance[0] < dist_thr or signed_distance[1] < dist_thr:
                filter_list.append(i)

    return filter_list


def streamlines_in_mesh(sft, target_mesh, all_in=False, dist_thr=0.1):
    """
    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines to segment.
    target_mesh : o3d.geometry.TriangleMesh
        Mesh oriented RAS mm in which the streamlines should pass.
    all_in : bool
        If True, all points of the streamline must be in the mesh.
    dist_thr : float
        Distance threshold in mm to consider a point in the mesh.
    Returns
    -------
    ids : list
        Ids of the streamlines that pass filter (any or all).
    """
    # tractogram should be in RASmm same with mesh
    sft.to_rasmm()
    sft.to_corner()

    # Legacy mesh needed for ray casting
    legacy_mesh = o3d.t.geometry.TriangleMesh.from_legacy(target_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(legacy_mesh)

    # TODO definitely a faster way to do this, the casting is fast but the for loop sucks
    # probably do a reshape get all signed distances and then reshape back

    # use ray casting to find distance from each point in tractogram to mesh
    filter_list = []
    for i in range(0, len(sft.streamlines)):
        query_points = sft.streamlines[i]

        # Cast rays and get distance (negative values if inside)
        signed_distance = np.array(scene.compute_signed_distance(query_points))

        # decide if this streamline should be filtered
        if all_in:
            if np.all(signed_distance < dist_thr):
                filter_list.append(i)
        else:
            if np.any(signed_distance < dist_thr):
                filter_list.append(i)

    return filter_list

def streamlines_intersect_with_mesh(sft, target_mesh, both_ends=False, only_start=False, only_end=False, intersect_min_angle=0.0, intersect_max_angle=0.5, intersect_dist=3.0):
    """
    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines to segment.
    target_mesh : o3d.geometry.TriangleMesh
        Mesh oriented RAS mm in which.
    both_ends : bool
        If True, both end points must be within mesh.
    only_start : bool
        If True, only the start point must be within mesh.
    only_end : bool
        If True, only the end point must be within mesh.
    intersect_min_angle : float
        Minimum angle between streamline and mesh surface in radians.
    intersect_max_angle : float
        Maximum angle between streamline and mesh surface in radians.
    intersect_dist : float
        Distance in mm from end of streamline to search for intersection.
    Returns
    -------
    ids : list
        Ids of the streamlines that pass filter (any or all).
    """
    # tractogram should be in RASmm same with mesh
    sft.to_rasmm()
    sft.to_corner()

    # Legacy mesh needed for ray casting
    legacy_mesh = o3d.t.geometry.TriangleMesh.from_legacy(target_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(legacy_mesh)

    # TODO definitely a faster way to do this, the casting is fast but the for loop sucks
    # maybe do a reshape get all signed distances and then reshape back

    # use ray casting to find distance from each point in tractogram to mesh
    filter_list = []
    for i in range(0, len(sft.streamlines)):
        # Determine if there is an interection at either end of the streamline

        # Step size
        step_size = np.linalg.norm(sft.streamlines[i][0] - sft.streamlines[i][1])
        nbr_of_steps = np.int(np.ceil(intersect_dist / step_size))

        if len(sft.streamlines[i]) > nbr_of_steps*2:
            first_pnts = sft.streamlines[i][0:nbr_of_steps]
            last_pnts = sft.streamlines[i][-nbr_of_steps:]
            query_points = np.concatenate((first_pnts, last_pnts))
        else:
            query_points = sft.streamlines[i]

        # Cast rays and get distance (negative values if inside)
        signed_distance = np.sign(np.array(scene.compute_signed_distance(query_points)))

        #Determine if there is an intersection (negative to positive flip)
        intersection = np.diff(signed_distance) != 0

        # Set middle index False incase intersection detected because firstpnt and lastpnt concatenation (above)
        if len(sft.streamlines[i]) > nbr_of_steps*2:
            intersection[nbr_of_steps-1] = False

        # check intersection points to see if angle is within range
        intersect_ind = np.where(intersection)[0]
        intersect_start = query_points[intersect_ind]
        intersect_end = query_points[intersect_ind+1]

        # If both_ends then check to make sure at least two intersections exists (one is at the end one at the beginning)
        if both_ends:
            if len(intersect_ind) < 2:
                continue
            if not (np.any(intersect_ind < nbr_of_steps) and np.any(intersect_ind >= nbr_of_steps)):
                continue

        # If only_start then check to make sure at least one intersection exists at the beginning
        if only_start:
            if len(intersect_ind) < 1 or len(intersect_ind) >= 2:
                continue
            if not np.any(intersect_ind < nbr_of_steps):
                continue

        # If only_end then check to make sure at least one intersection exists at the end
        if only_end:
            if len(intersect_ind) < 1 or len(intersect_ind) >= 2:
                continue
            if not np.any(intersect_ind >= nbr_of_steps):
                continue

        # For each intersection throw a ray from the start_pnt to the mesh and get angle
        angles = []
        for start_pnt, end_pnt in zip(intersect_start, intersect_end):
            # Get vector from start_pnt to end_pnt
            vector = end_pnt - start_pnt
            vector = vector / np.linalg.norm(vector) # normalize to unit vector

            # Get normal at intersection point by casting a ray in both directions
            thisRay_pos = o3d.core.concatenate((start_pnt,vector),0).reshape((1,6))
            thisRay_neg = o3d.core.concatenate((start_pnt,-vector),0).reshape((1,6))
            thisRay = o3d.core.concatenate((thisRay_pos,thisRay_neg),0)
            ans = scene.cast_rays(thisRay)
            mesh_normal = np.squeeze(ans['primitive_normals'].numpy()[np.where(ans['t_hit'].numpy() == np.min(ans['t_hit'].numpy()))[0][0]])

            # Get angle between vector and normal (dot product) vectors should already be normalized
            angles.append(np.abs(np.dot(vector, mesh_normal)))

        angles = np.array(angles)
        if both_ends:
            start_angles = angles[intersect_ind < nbr_of_steps]
            end_angles = angles[intersect_ind >= nbr_of_steps]
            if (np.any(np.logical_and(start_angles > intersect_min_angle, start_angles < intersect_max_angle)) and
                      np.any(np.logical_and(end_angles > intersect_min_angle, end_angles < intersect_max_angle))):
                filter_list.append(i)
        elif only_start:
            start_angles = angles[intersect_ind < nbr_of_steps]
            if (np.any(np.logical_and(start_angles > intersect_min_angle, start_angles < intersect_max_angle))):
                filter_list.append(i)
        elif only_end:
            end_angles = angles[intersect_ind >= nbr_of_steps]
            if (np.any(np.logical_and(end_angles > intersect_min_angle, end_angles < intersect_max_angle))):
                filter_list.append(i)
        else:
            if np.any(np.logical_and(angles > intersect_min_angle, angles < intersect_max_angle)):
                filter_list.append(i)

    return filter_list

def filter_mesh_roi(sft, mesh, filter_type, is_exclude, dist_thr=0.1):
    """
    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines to segment.
    mesh : o3d.geometry.TriangleMesh
        Mesh oriented RAS mm.
    filter_type: str
        One of the 5 following choices, 'any', 'all', 'either_end', 'both_ends', 'only_start', 'only_end'.
    is_exclude: bool
        Value to indicate if the ROI is an AND (false) or a NOT (true).
    dist_thr : float
        Distance threshold in mm to consider a point in the mesh.
    Returns
    -------
    new_sft: StatefulTractogram
        Filtered sft.
    ids: list
        Ids of the streamlines passing through the mask.
    """

    line_based_indices = []
    if filter_type in ['any', 'all']:
        # Point based filtering
        line_based_indices = streamlines_in_mesh(sft, mesh,
                                                 all_in=filter_type == 'all',
                                                 dist_thr=dist_thr)
    else:
        # End point filtering
        line_based_indices = streamline_endpoints_in_mesh(sft, mesh,
                                                          both_ends=filter_type == 'both_ends',
                                                          only_start=filter_type == 'only_start',
                                                          only_end=filter_type == 'only_end',
                                                          dist_thr=dist_thr)

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


def filter_mesh_intersect(sft, mesh, filter_type, is_exclude, intersect_min_angle=0.0, intersect_max_angle=0.5,intersect_dist=3.0):
    """
    Parameters
    ----------
    sft : StatefulTractogram
        StatefulTractogram containing the streamlines to segment.
    mesh : o3d.geometry.TriangleMesh
        Mesh oriented RAS mm.
    filter_type: str
        One of the 4 following choices, 'either_end', 'both_ends', 'only_start', 'only_end'.
    is_exclude: bool
        Value to indicate if the ROI is an AND (false) or a NOT (true).
    intersect_min_angle : float
        Minimum angle in radians between the streamline and the mesh.
    intersect_max_angle : float
        Maximum angle in radians between the streamline and the mesh.
    Returns
    -------
    new_sft: StatefulTractogram
        Filtered sft.
    ids: list
        Ids of the streamlines passing through the mask.
    """

    line_based_indices = []
    if filter_type in ['either_end', 'both_ends', 'only_start', 'only_end']:
        line_based_indices = streamlines_intersect_with_mesh(sft, mesh,
                                                             both_ends=filter_type == 'both_ends',
                                                             only_start=filter_type == 'only_start',
                                                             only_end=filter_type == 'only_end',
                                                             intersect_min_angle=intersect_min_angle,
                                                             intersect_max_angle=intersect_max_angle,
                                                             intersect_dist=intersect_dist)
    else:
        raise ValueError("Filter type not recognized for mesh intersection")

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
            mesh = o3d.io.read_triangle_mesh(filter_arg)

            filtered_sft, kept_ids = filter_mesh_roi(sft, mesh,
                                                     filter_mode, is_exclude, args.dist_thr)

        if filter_type == 'mesh_intersect':
            if filter_mode == 'any' or filter_mode == 'all':
                raise ValueError(
                    "Only 'either_end' or 'both_ends' are valid for mesh_intersect")

            mesh = o3d.io.read_triangle_mesh(filter_arg)
            filtered_sft, kept_ids = filter_mesh_intersect(sft, mesh,
                                                           filter_mode, is_exclude,
                                                           args.intersect_min_angle,
                                                           args.intersect_max_angle,
                                                           args.intersect_dist)

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
