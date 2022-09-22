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
import open3d as o3d
import nibabel as nib
import numpy as np

from numpy import (asarray, hstack, ones, matmul, squeeze, savetxt)
from numpy.linalg import (inv, norm)
from nibabel.affines import apply_affine
from scilpy.image.datasets import DataVolume

EPILOG = """
"""

# From trimeshpy mesh transformations
def vtk_to_vox(vts, nibabel_img):
    inv_affine = np.linalg.inv(nibabel_img.get_affine())
    flip = np.diag([-1, -1, 1, 1])
    vts = apply_affine(np.dot(inv_affine, flip), vts)
    return vts

def vtk_to_voxmm(vts, nibabel_img):
    scale = np.array(nibabel_img.get_header().get_zooms())
    return vtk_to_vox(vts, nibabel_img) * scale

def calculate_force(pos, mesh, repulsion_radius, mag_direction):
    """
    Calculate the force at a given position

    Parameters
    ----------
    pos: np.array
        Position to calculate the force at (position must be within radius at this point)
    mesh: o3d.geometry.TriangleMesh
        Open3d mesh object (with vertex normals already computed)
    repulsion_radius: float
        radius of the repulsion force (points oustide this radius are given a zero)
    mag_direction: np.array 
        Same size as mesh.vertices determines whether repulsion or attractive force should be calculated
         1 indicates repulstion, -1 for attraction and 0 for no force

    Return
    ------
    force: np.array
        Repulsion force at the given position
    """
    # Find Points within radius of pos.
    distance_3d = mesh.vertices - pos
    distance = np.linalg.norm(distance_3d, axis=1)
    
    distance_3d_within_range = distance_3d[np.where(distance < repulsion_radius)]
    distance_within_range = distance[np.where(distance < repulsion_radius)]
    mag_direction_within_range = mag_direction[np.where(distance < repulsion_radius)]

    # Sum up repulsion forces calculated for each vertex within range
    mag = 0
    for thisDist3d, thisDist, thisMagDir in zip(distance_3d_within_range, distance_within_range, mag_direction_within_range):
        # Calculate repulsion force magnitude (similar to Schuh 2017, IEEE)
        mag -= thisMagDir * (thisDist/repulsion_radius - 1)**2 * thisDist3d/thisDist # signed force magnitude

    if len(distance_within_range) > 0:
        force = mag / len(distance_within_range)
    else:
        force = np.zeros((3,))

    return force

def generate_force_map(mesh, wm_mask_img, repulsion_radius, invert_force_map, mag_direction=None):
    """
    Given a set of vertex coordinates and normals,
    generate a repulsion force map

    Parameters
    ----------
    mesh: o3d.geometry.TriangleMesh
        Open3d mesh object
    wm_mask: nibabel.nifti1.Nifti1Image
        White matter mask where the force map is calculated
    
    repulsion_radius: float
        radius of the repulsion force (points oustide this radius are given a zero)
    
    Optional Parameters
    -------------------
    mag_direction: np.array
        Same size as mesh.vertices determines whether repulsion or attractive force should be calculated

    Return
    ------
    force_map: nibabel.nifti1.Nifti1Image
        Repulsion force map calculated at each wm voxel
    force_normals: nibabel.nifti1.Nifti1Image
        Normal vector from mesh extracted from the closest vertex to each wm voxel
    """
    if mag_direction is None:
        mag_direction = np.ones((asarray(mesh.vertices).shape[0],))
    
    wm_mask_data = wm_mask_img.get_fdata(caching='unchanged', dtype=float)
    wm_mask_res = wm_mask_img.header.get_zooms()[:3]
    wm_mask = DataVolume(wm_mask_data, wm_mask_res, 'nearest')
    
    force_map_data = np.zeros(wm_mask.data.shape[:3] + (3,))

    # indices of wm_voxels
    wm_indices = np.argwhere(wm_mask_data > 0)
    wm_pnts = np.float32(wm_indices * wm_mask_res)
   
    # Remove points outside force radius
    legacy_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(legacy_mesh)

    mesh_distance = np.array(scene.compute_distance(wm_pnts))
    
    if np.where(mesh_distance>repulsion_radius) == None:
        raise ValueError("White matter not within range of mesh, try increasing repulsion radius")
    
    wm_indices = wm_indices[mesh_distance<repulsion_radius,:]
    wm_pnts = wm_pnts[mesh_distance<repulsion_radius,:]
    
    #nib.save(nib.Nifti1Image(test_wm_data, wm_mask_img.affine), "/home/graham/DATA/dump/test_wm.nii.gz")

    for thisPnt in range(0,wm_indices.shape[0]):
        if thisPnt % 1000 == 0:
            print("Processing point {}/{}".format(thisPnt, wm_indices.shape[0]))
            
        pos = np.asarray([wm_pnts[thisPnt,0], wm_pnts[thisPnt,1], wm_pnts[thisPnt,2]])
        
        thisForce = calculate_force(pos, mesh, repulsion_radius, mag_direction)
        
        force_map_data[wm_indices[thisPnt,0], wm_indices[thisPnt,1], wm_indices[thisPnt,2],:] = thisForce

    if invert_force_map:
        force_map_data = -1 * force_map_data

    force_map_data = force_map_data*10

    return nib.Nifti1Image(force_map_data, wm_mask_img.affine)
    



def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, epilog=EPILOG,
                                formatter_class=argparse.RawTextHelpFormatter)

    p.add_argument('mesh',
                   help='Input Mesh file')

    p.add_argument('wm_mask',
                   help='Text file with the coordinates of the surface.')

    p.add_argument('force_map',
                    help='Force map calculated at each wm voxel from mesh')
    
    p.add_argument('--force_resolution', type=float, default=None,
                    help='Resolution of the force map. Default is the resolution of the wm mask')
    
    p.add_argument('--repulsion_radius', type=float, default=2.0,
                     help='Radius of the repulsion force (points oustide this radius are given a zero)')

    p.add_argument('--flip_orientation', action='store_true',
                       help='Set to true if you want to flip the mesh from LPS to RAS or vice versa prior to calculating the force map')

    p.add_argument('--invert_force_map', action='store_true',
                       help='Negates the force map such that negative values become positive')
    p.add_argument('--force_attaction', default=None,
                    help='Text file with indices for each vertices that should use attraction')
    p.add_argument('--force_null', default=None,
                    help='Text file with indices for each vertices that should use no force')

    add_overwrite_arg(p)
    return p

    p.add_argument('')

    add_overwrite_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    assert_inputs_exist(parser, [args.mesh, args.wm_mask])
    assert_outputs_exist(parser, args, args.force_map)

    # Load wm mask
    mask_img = nib.load(args.wm_mask)
    
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    mesh.compute_vertex_normals()
    coords = asarray(mesh.vertices)

    # Flip RAS to LPS
    if args.flip_orientation:
        coords[:,0] = -1 * coords[:,0]
        coords[:,1] = -1 * coords[:,1]
        mesh.vertices = o3d.utility.Vector3dVector(coords)

    # Convert to voxel_mm using wm mask
    coords_voxmm = vtk_to_voxmm(coords, nib.load(args.wm_mask))
    mesh.vertices = o3d.utility.Vector3dVector(coords_voxmm)
    mesh.compute_vertex_normals()

    mag_direction = np.ones((asarray(mesh.vertices).shape[0],))
    if args.force_attaction is not None:
        mag_direction[np.loadtxt(args.force_attaction, dtype=int)] = -1
    if args.force_null is not None:
        mag_direction[np.loadtxt(args.force_null, dtype=int)] = 0

    # Calculate force map at each wm voxel
    force_map_img = generate_force_map(mesh, mask_img, args.repulsion_radius, args.invert_force_map, mag_direction=mag_direction)

    # Save mesh
    if args.force_map is not None:
        nib.save(force_map_img, args.force_map)

if __name__ == "__main__":
    main()
