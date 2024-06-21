import numpy as np
import os
os.environ['NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS'] = '0'
# os.environ['NUMBA_ENABLE_CUDASIM'] = '1'
from numba import cuda
import torch

@cuda.jit
def update(gaussians_xyz, full_projection_transform, map, width, height, transients):
    i = cuda.grid(1)
    if i >= gaussians_xyz.shape[0]:
        return
    xyz = gaussians_xyz[i, :]
    x = xyz[0]*full_projection_transform[0,0]+\
        xyz[1]*full_projection_transform[1,0]+\
        xyz[2]*full_projection_transform[2,0]+full_projection_transform[3,0]
    y = xyz[0]*full_projection_transform[0,1]+\
        xyz[1]*full_projection_transform[1,1]+\
        xyz[2]*full_projection_transform[2,1]+full_projection_transform[3,1]
    w = xyz[0]*full_projection_transform[0,3]+\
        xyz[1]*full_projection_transform[1,3]+\
        xyz[2]*full_projection_transform[2,3]+full_projection_transform[3,3]
    w = 1/(w+0.000001)

    x,y = int((x*w+1)*width*0.5), int((y*w+1)*height*0.5)

    if x < 0 or x>=width or y<0 or y>=height:
        return

    transients[i] = map[y, x]

def update_transient(gaussians, viewpoint_camera, transient_map, update_filter, radii):
    imgid = viewpoint_camera.uid
    # full_proj_matrix = cuda.to_device(viewpoint_camera.full_proj_transform.cpu().numpy())
    # gaussians_xyz = cuda.to_device(gaussians.get_xyz[update_filter].detach().cpu().numpy()) 
    # map = cuda.to_device(transient_map.detach().cpu().numpy())
    # transients = cuda.to_device(gaussians.get_transient[:, imgid].cpu().numpy())
    full_proj_matrix = cuda.as_cuda_array(viewpoint_camera.full_proj_transform)
    gaussians_xyz = cuda.as_cuda_array(gaussians.get_xyz[update_filter].detach()) 
    map = cuda.as_cuda_array(transient_map.detach())
    transients = cuda.device_array(shape=(gaussians_xyz.shape[0]), dtype=np.float32)
    # transients = cuda.as_cuda_array(gaussians.get_transient[update_filter, imgid])

    threads_per_block = 256
    blocks_per_grid = (gaussians_xyz.shape[0]+threads_per_block-1)//threads_per_block
    update[blocks_per_grid, threads_per_block](gaussians_xyz, full_proj_matrix, map, map.shape[1], map.shape[0], transients)
    gaussians.get_transient[update_filter, imgid] = torch.tensor(transients.copy_to_host(), device='cuda')
    