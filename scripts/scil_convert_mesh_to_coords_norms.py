#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to convert a surface to coordinates and normals.
    mesh types supported (same as in open3d):
    ".ply"

> scil_convert_mesh_to_coords_norms.py surf.ply coords.txt norms.txt
"""

import argparse
from copy import deepcopy

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist,
                             assert_outputs_exist)
from scilpy.image.volume_space_management import DataVolume

import open3d as o3d
import nibabel as nib
import numpy as np

from numpy import (asarray, hstack, ones, matmul, squeeze, savetxt)
from numpy.linalg import (inv, norm)
from nibabel.affines import apply_affine

from dipy.io.stateful_tractogram import StatefulTractogram, Space, \
                                        set_sft_logger_level
from dipy.io.stateful_tractogram import Origin
from dipy.io.streamline import save_tractogram

EPILOG = """
"""

# From trimeshpy mesh transformations
def vtk_to_vox(vts, nibabel_img):
    inv_affine = np.linalg.inv(nibabel_img.affine)
    flip = np.diag([-1, -1, 1, 1])
    vts = apply_affine(np.dot(inv_affine, flip), vts)
    return vts

def vtk_to_voxmm(vts, nibabel_img):
    scale = np.array(nibabel_img.header.get_zooms())
    return vtk_to_vox(vts, nibabel_img) * scale

def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('in_surface',
                   help='Input a surface (FreeSurfer or supported by VTK).')

    p.add_argument('out_coords',
                   help='Text file with the coordinates of the surface.')

    p.add_argument('out_norms',
                     help='Text file with the normals of the surface.')

    p.add_argument('--apply_transform', default=None,
                    help='If given apply a transformation to the mesh, from a nifti file')

    p.add_argument('--output_mesh', default=None,
                    help='If given output the mesh after flips and reoirentations are applied')

    p.add_argument('--ras', action='store_true',
                       help='Set to true if input mesh is in RAS')

    p.add_argument('--las', action='store_true',
                       help='Set to true if input mesh is in LAS')

    p.add_argument('--flip_normals', action='store_true',
                       help='If given normals will be flipped in the inward direction')

    p.add_argument('--flip_normals_xy', action='store_true',
                       help='If given normals will be flipped in the inward direction')

    p.add_argument('--within_mask', default=None,
                    help='If given, only output the vertices/normals within the mask')

    p.add_argument('--output_indices', default=None,
                    help='If given, output the indices of the vertices within the mask')

    p.add_argument('--output_trk', default=None,
                   help='If given, output a trk file with the seed points for visualization, --apply_transform must be provided to save stateful tractogram')
    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, args.in_surface)
    assert_outputs_exist(parser, args, (args.out_coords, args.out_norms))

    if args.apply_transform is not None:
        assert_inputs_exist(parser, args.apply_transform)

    mesh = o3d.io.read_triangle_mesh(args.in_surface)
    mesh.compute_vertex_normals()

    coords = asarray(mesh.vertices)
    norms = asarray(mesh.vertex_normals)

    # Flip RAS to LPS
    if args.ras:
        coords[:,0] = -1 * coords[:,0]
        coords[:,1] = -1 * coords[:,1]

    if args.las:
        coords[:,1] = -1 * coords[:,1]

    # output LPS mesh for MI-brain visualization
    lps_mesh = deepcopy(mesh)
    lps_coords=deepcopy(coords)
    lps_mesh.vertices = o3d.utility.Vector3dVector(coords)

    if args.apply_transform is not None:
        coords = vtk_to_voxmm(coords, nib.load(args.apply_transform))

        # Recalculate normals after transformation
        mesh.vertices = o3d.utility.Vector3dVector(coords)
        affine = nib.load(args.apply_transform).affine
        diagnol_ones = [affine[0,0]/abs(affine[0,0]), affine[1,1]/abs(affine[1,1]), affine[2,2]/abs(affine[2,2])]
        norms = norms * diagnol_ones

    if args.flip_normals:
        norms = -1 * norms

    if args.flip_normals_xy:
        norms[:,0] = -1 * norms[:,0]
        norms[:,1] = -1 * norms[:,1]

    if args.within_mask is not None:
        new_coords = []
        new_norms = []
        indices = []

        #Select coords and norms within mask
        mask_img = nib.load(args.within_mask)
        mask_data = mask_img.get_fdata()
        mask_res = mask_img.header.get_zooms()[:3]
        mask = DataVolume(mask_data, mask_res,'nearest')

        for i, coord in enumerate(asarray(coords)):
            if mask._voxmm_to_value(coord[0], coord[1], coord[2], Origin('corner')) > 0:
                new_coords.append(coord)
                new_norms.append(norms[i])
                indices.append(i)

        coords = np.array(new_coords)
        norms = np.array(new_norms)

        if args.output_indices is not None:
            savetxt(args.output_indices, indices, fmt='%d')

    savetxt(args.out_coords, coords)
    savetxt(args.out_norms, norms)

    # Write a temp mesh of vtk with transformed coords
    if args.output_mesh:
        mesh.vertices = o3d.utility.Vector3dVector(coords)
        o3d.io.write_triangle_mesh(args.output_mesh, lps_mesh)

    # Write seeds into a trk file for visualization
    if args.output_trk:
        assert(args.apply_transform is not None)
        seeds=[]
        for i, coord in enumerate(coords):
            seeds.append([coord, coord + norms[i]*2])
        seeds = np.array(seeds)

        sft = StatefulTractogram(seeds, nib.load(args.apply_transform),
                                 Space.VOXMM,
                                 Origin.NIFTI)

        save_tractogram(sft, args.output_trk)


if __name__ == "__main__":
    main()
