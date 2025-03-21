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

    vs = cv2.VideoCapture(video_path)
    fps = vs.get(cv2.CAP_PROP_FPS)
    # frame_interval = int(fps * 1)  # 1 frame/sec\
    frame_interval = 1

    count = 0
    video_frame_num = 0
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


######################


video_path = '/mnt/petrelfs/sichenyang.p/txwu/dataset/human_videos/vecteezy/dance/12002501-young-hip-couple-with-tattoos-embrace-on-a-city-street.mp4'
image_root = '/mnt/petrelfs/sichenyang.p/txwu/project/vggt/examples/sampled_frames'
max_num_frames = 200



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


    for i, point_map in enumerate(point_maps):
        # Normalize each coordinate (assuming already normalized between 0 and 1)
        point_map_rgb = (point_map * 255).astype(np.uint8)  # Scale to 0-255 for RGB mapping

        # Convert (H, W, 3) into an RGB image
        # # save pointmaps per frame
        # img_path = f'{outdir}/pointmap_frame_{i:04d}.png'
        # cv2.imwrite(img_path, point_map_rgb)

        # Convert BGR (OpenCV default) to RGB for visualization
        img = cv2.cvtColor(point_map_rgb, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        images.append(img)
    # images[0].save(f'{outdir}/_pointmaps.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
    images[0].save(f'_pointmaps.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
