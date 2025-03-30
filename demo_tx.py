import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

import numpy as np
import cv2
import os
from PIL import Image


def sample_frames_from_video(video_path, image_root, max_num_frames):
    # --- Handle video ---
    video_name = video_path.split('/')[-1].split('.')[0]
    image_dir = os.path.join(image_root, video_name)
    os.makedirs(image_dir, exist_ok=True)

    vs = cv2.VideoCapture(video_path)
    fps = vs.get(cv2.CAP_PROP_FPS)
    # frame_interval = int(fps * 1)  # 1 frame/sec\
    frame_interval = 1

    count = 0
    video_frame_num = 0
    image_paths = []
    while True:
        gotit, frame = vs.read()
        if not gotit:
            break
        
        count += 1
        if count % frame_interval == 0:
            image_path = os.path.join(image_dir, f"{video_frame_num:06}.png")
            cv2.imwrite(image_path, frame)
            image_paths.append(image_path)
            video_frame_num += 1
        
        if count == max_num_frames:
            break

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    return image_paths


######################

video_name = 'vecteezy_a-man-plays-basketball-alone-in-a-public-area-having-fun_46467372'
video_path = f'/projects_vol/gp_slab/tianxing001/project/vggt/examples/videos/{video_name}.mp4'
image_root = '/projects_vol/gp_slab/tianxing001/project/vggt/examples/video_frames'
# max_num_frames = 200
max_num_frames = 100


save_root = f'/projects_vol/gp_slab/tianxing001/project/vggt/outputs/{video_name}'
os.makedirs(save_root)


# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"

# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# # Load and preprocess example images (replace with your own image paths)
# image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]  
# images = load_and_preprocess_images(image_names).to(device)

# load frames from video
image_names = sample_frames_from_video(video_path, image_root, max_num_frames)
images = load_and_preprocess_images(image_names).to(device)

# with torch.no_grad():
#     with torch.cuda.amp.autocast(dtype=dtype):
#         # Predict attributes including cameras, depth maps, and point maps.
#         predictions = model(images)


with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images)
                
    # Predict Cameras
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    # Predict Depth Maps
    depth_maps, depth_confs = model.depth_head(aggregated_tokens_list, images, ps_idx)

    # Predict Point Maps
    point_maps, point_confs = model.point_head(aggregated_tokens_list, images, ps_idx)
        
    # Construct 3D Points from Depth Maps and Cameras
    # which usually leads to more accurate 3D points than point map branch
    point_maps_by_unprojection = unproject_depth_map_to_point_map(depth_maps.squeeze(0), 
                                                                extrinsic.squeeze(0), 
                                                                intrinsic.squeeze(0))

    # # Predict Tracks
    # # choose your own points to track, with shape (N, 2) for one scene
    # query_points = torch.FloatTensor([[100.0, 200.0], 
    #                                     [60.72, 259.94]]).to(device)
    # track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])

    
    
    point_maps = point_maps[0].cpu().numpy()
    point_maps_normalized = (point_maps - point_maps.min()) / (point_maps.max() - point_maps.min())  # normalize
    point_imgs = []
    for i, point_map in enumerate(point_maps_normalized):
        # Normalize each coordinate (assuming already normalized between 0 and 1)
        point_map_rgb = (point_map * 255).astype(np.uint8)  # Scale to 0-255 for RGB mapping

        # Convert (H, W, 3) into an RGB image
        # # save pointmaps per frame
        # img_path = f'{outdir}/pointmap_frame_{i:04d}.png'
        # cv2.imwrite(img_path, point_map_rgb)

        # Convert BGR (OpenCV default) to RGB for visualization
        img = cv2.cvtColor(point_map_rgb, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        point_imgs.append(img)
    # images[0].save(f'{outdir}/_pointmaps.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
    point_imgs[0].save(f'{save_root}/pointmaps.gif', save_all=True, append_images=point_imgs[1:], duration=100, loop=0)

    point_map_by_unprojections_normalized = (point_maps_by_unprojection - point_maps_by_unprojection.min()) / (point_maps_by_unprojection.max() - point_maps_by_unprojection.min())  # normalize
    point_imgs = []
    for i, point_map in enumerate(point_map_by_unprojections_normalized):
        # Normalize each coordinate (assuming already normalized between 0 and 1)
        point_map_rgb = (point_map * 255).astype(np.uint8)  # Scale to 0-255 for RGB mapping

        # Convert (H, W, 3) into an RGB image
        # # save pointmaps per frame
        # img_path = f'{outdir}/pointmap_frame_{i:04d}.png'
        # cv2.imwrite(img_path, point_map_rgb)

        # Convert BGR (OpenCV default) to RGB for visualization
        img = cv2.cvtColor(point_map_rgb, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        point_imgs.append(img)
    # images[0].save(f'{outdir}/_pointmaps.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
    point_imgs[0].save(f'{save_root}/pointmaps_by_unprojection.gif', save_all=True, append_images=point_imgs[1:], duration=100, loop=0)  
