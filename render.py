import os, sys, torch, cv2, numpy
import matplotlib.pyplot as plt
from random import uniform
from pathlib import Path
from tqdm.auto import tqdm

from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.ops import cubify
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings, 
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex
)


class RenderBot:
    def __init__(self, view = [30, 30, 60], at = [1.0, 0, -0.25]):
        # Select PyTorch device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")
        # Instantiate internal objects
        self.use_tqdm = True
        self.frames = [] # Frame buffer
        self.sequence_offset = 0 # Offset when calling self.export_frames()
        self.exported_frames = [] # List of exported frame filenames
        # Objects to construct during render_from_file()
        self.cube_faces = None
        self.cube_textures = None
        self.ground = None
        # Instantiate renderer objects
        R, T = look_at_view_transform(*view, up = [[0, 0, 1]], at = [at]) 
        cameras = FoVPerspectiveCameras(device = self.device, R = R, T = T)
        raster_settings = RasterizationSettings(
            image_size = 512, blur_radius = 0.0, faces_per_pixel = 1)
        lights = PointLights(device = self.device, location = [[4.0, 3.0, 5.0]])
        # self.renderer is called on a mesh to produce a rendered image
        self.renderer = MeshRenderer(
            rasterizer = MeshRasterizer(
                cameras = cameras, raster_settings = raster_settings),
            shader = SoftPhongShader(
                device = self.device, cameras = cameras, lights = lights))
        
    # Returns a mesh of a (1 x 1 x 1) cube
    def cube(self):
        voxels = torch.zeros(size = (1, 3, 3, 3),
                             dtype = torch.float64, device = self.device)
        voxels[0, 1, 1, 1] = 1.0 # A 1.0 surrounded by 8 + (2 * 9) 0.0's
        mesh = cubify(voxels, 1.0, device = self.device, align = "topleft")
        return mesh

    # Renders a Meshes object
    def render_meshes(self, meshes):
        image = self.renderer(meshes)
        assert len(image) == 1 # Multiple meshes have to be wrapped in a scene
        return (image[0, ..., :3] * 255).cpu().numpy().astype(numpy.uint8)
        
    # Export the frame buffer to a sequence of output images
    # Sequence ordering is preserved between calls through self.sequence_offset
    def export_frames(self, output_fn):
        output_fn = Path(output_fn).resolve()
        self.extension = output_fn.suffix
        if self.use_tqdm:
            progress = tqdm(total = len(self.frames))
        for i, frame in enumerate(self.frames):
            name = f"{output_fn.stem}_{self.sequence_offset + i}{output_fn.suffix}"
            fn = str(output_fn.parents[0] / name)
            plt.figure(figsize = (10, 10))
            plt.axis("off")
            plt.imshow(frame)
            plt.savefig(fn)
            plt.close()
            self.exported_frames += fn
            if self.use_tqdm:
                progress.update(1)
        if self.use_tqdm:
            progress.close()
        self.sequence_offset += i + 1
        self.frames = []
        
    # Export a frame buffer to a video of the image sequence
    def export_video(self, result_path = None, fps = 1000, use_sequenced = False):
        """
        Exports the result video
        Will save a compressed result video to result_path if provided
        If use_sequenced, use what was generated from self.export_frames(),
            otherwise use the frame buffer self.frames
        """
        assert ((self.frames and result_path)
                or (self.exported_frames and use_sequenced))
        # Create runtime objects
        if result_path: # Create video writer if save path provided
            vid_width, vid_height, vid_fps = (512, 512, fps)
            codec = cv2.VideoWriter_fourcc(*'XVID')
            recorder = cv2.VideoWriter(
                result_path, codec, vid_fps, (vid_width, vid_height))
        if self.use_tqdm:
            total = self.sequence_offset if use_sequenced else len(self.frames)
            progress = tqdm(total = total)
        iterator = self.exported_frames if use_sequenced else self.frames
        # Event loop
        for frame in iterator:
            if result_path:
                recorder.write(frame)
            if self.use_tqdm:
                progress.update(1)
        # Destroy export runtime objects
        if not use_sequenced:
            self.frames = []
        else:
            self.exported_frames = []
        if recorder:
            recorder.release()
        if self.use_tqdm:
            progress.close()
            
    # Builds ground mesh, a green 40 x 40 plane at z = 0
    def build_ground(self):
        plane_verts = torch.Tensor(
            [[-20.0, 20.0, 0.0], [-20.0, -20.0, 0.0],
             [20.0, -20.0, 0.0], [20.0, 20.0, 0.0]]).to(self.device)
        plane_faces = torch.Tensor([[0, 1, 2], [2, 3, 0]]).to(self.device)
        self.ground = Meshes(verts = plane_verts[None], faces = plane_faces[None])
        plane_verts_rgb = torch.ones_like(self.ground.verts_list()[0])[None]
        plane_verts_rgb[0, :, :] /= 8
        plane_verts_rgb[0, :, 1] *= 5 # Green
        self.ground.textures = TexturesVertex(
            verts_features = plane_verts_rgb.to(self.device))

    # Renders a video from a .pt file containing frames/animation of a cubeoid mesh
    def render_from_file(self, input_fn, result_fn, fps = 1000, every = 1):
        assert not self.frames # Assumes it is empty at call time
        assert every > 0
        input_fn = Path(input_fn)
        assert input_fn.exists() and input_fn.suffix == ".pt"
        coords = torch.tensor(torch.load(str(input_fn))).to(self.device).float()
        skip_shape = [*coords.shape]
        skip_shape[0] = int(skip_shape[0] / every)
        print(f"Input shape after skip: {skip_shape}")
        if skip_shape[1] == 8:
            # Crossed single cube
            springs = [[0, 1], [0, 2], [1, 3], [2, 3], # Square 1
                       [4, 5], [4, 6], [5, 7], [6, 7], # Square 2
                       [0, 4], [1, 5], [2, 6], [3, 7], # Depth between squares
                       [0, 7], [1, 6], [2, 5], [3, 4]] # Diagonals
        elif skip_shape[1] == 16:
            # Crossed triple-cube, as in RandomSearchRobot
            springs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13),
                       (14, 15), (0, 8), (2, 10), (4, 12), (6, 14), (1, 9), (3, 11),
                       (5, 13), (7, 15), (0, 11), (1, 10), (8, 3), (9, 2), (0, 2),
                       (1, 3), (8, 10), (9, 11), (2, 13), (3, 12), (10, 5), (11, 4),
                       (2, 4), (3, 5), (10, 12), (11, 13), (4, 15), (5, 14),
                       (12, 7), (13, 6), (4, 6), (5, 7), (12, 14), (13, 15)]
        else:
            print("Bad input shape! Quitting...")
            quit()
                   
        # Build needed object attributes
        cube = self.cube() # Mesh of a 1 x 1 x 1 cube
        spring_textures = []
        for _ in springs: # Generate a unique texture for each spring in a frame
            mesh_verts_rgb = torch.ones_like(cube._verts_list[0]) # (1, V, 3)
            for face in mesh_verts_rgb: # Randomize the colors per-vertex
                for channel in face:
                    channel *= uniform(0.55, 0.85)
            spring_textures.append(TexturesVertex(
                verts_features = mesh_verts_rgb[None].to(self.device)))
        if not self.cube_faces: # shape: 12 x 3
            # Get faces vector from a generic cube
            # None-index to wrap in a batch of 1 (instead of in a list here)
            self.cube_faces = cube._faces_list[0].to(self.device)[None]
        if not self.ground:
            self.build_ground()
        if self.use_tqdm:
            progress = tqdm(total = skip_shape[0])
            
        # Render each frame and store in memory
        for i, frame in enumerate(coords):
            if i % every != 0: # Allows for subsampling of large datasets
                continue
            meshes = [self.ground]
            # Build each spring and render the mesh to an image
            for (point0, point1), texture in zip(springs, spring_textures):
                mesh = cube.clone() # Begin at 1 x 1 x 1
                # Scale to get a mesh of size 0.2 x 0.2 x 0.2
                verts = mesh._verts_list[0] / 5
                # Offset the spring according to its bounding points
                verts[:4] += frame[point0]
                verts[4:] += frame[point1]
                # Permute to increase x and y thickness
                verts[:2, 1] += 0.2
                verts[6:, 1] -= 0.2
                verts[2:4, 1] += 0.025
                verts[4:6, 1] -= 0.025
                # Update the mesh with its appropriate verticies and texture
                mesh._verts_list = [verts]
                mesh.textures = texture
                meshes.append(mesh.clone())
            # Render composite mesh (with ground) and append to frame buffer
            self.frames.append(self.render_meshes(join_meshes_as_scene(meshes)))
            if self.use_tqdm:
                progress.update(1)
                
        if self.use_tqdm:
            progress.update(progress.total - progress.n)
            progress.close()
        self.export_video(result_path = result_fn, fps = int(fps / every))
        self.frames = []
