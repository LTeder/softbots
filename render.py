import os, sys, torch, cv2, numpy
import matplotlib.pyplot as plt
from random import uniform
from pathlib import Path
from tqdm.auto import tqdm

from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.ops import cubify
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
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
    def __init__(self):
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
        R, T = look_at_view_transform(14, 30, 60,
                                      up = [[0, 0, 1]], at = [[-1.2, 0, -0.5]]) 
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

    # Renders a video from a .pt file containing frames/animation of a cubeoid mesh
    def render_from_file(self, input_fn, result_fn, fps = 1000):
        assert not self.frames # Assumes it is empty at call time
        input_fn = Path(input_fn)
        assert input_fn.exists() and input_fn.suffix == ".pt"
        coords = torch.tensor(torch.load(str(input_fn))).to(self.device).float()
        print(f"Input shape: {coords.shape}")
        # Build needed object attributes
        if not self.cube_faces or self.cube_textures:
            cube = self.cube() # Mesh of a 1 x 1 x 1 cube
        if not self.cube_faces: # shape: 12 x 3
            # Get faces vector from a generic cube
            # None-index to wrap in a batch of 1 (instead of in a list here)
            self.cube_faces = cube._faces_list[0].to(self.device)[None]
        if not self.cube_textures:
            cube_verts_rgb = torch.ones_like(cube.verts_list()[0])[None]
            for face in cube_verts_rgb[0]:
                for channel in face:
                    channel *= uniform(0.6, 0.98)
            self.cube_textures = TexturesVertex(
                verts_features = cube_verts_rgb.to(self.device))
        if not self.ground: # Green ground mesh, a 20 x 20 plane at z = 0
            plane_verts = torch.Tensor(
                [[-10.0, 10.0, 0.0], [-10.0, -10.0, 0.0],
                 [10.0, -10.0, 0.0], [10.0, 10.0, 0.0]]).to(self.device)
            plane_faces = torch.Tensor([[0, 1, 2], [2, 3, 0]]).to(self.device)
            self.ground = Meshes(verts = plane_verts[None], faces = plane_faces[None])
            plane_verts_rgb = torch.ones_like(self.ground.verts_list()[0])[None]
            plane_verts_rgb[0, :, :] /= 8
            plane_verts_rgb[0, :, 1] *= 5 # Green
            self.ground.textures = TexturesVertex(
                verts_features = plane_verts_rgb.to(self.device))
        if self.use_tqdm:
            progress = tqdm(total = coords.size(0))
        # Render each frame and store in memory
        for i, verts in enumerate(coords):
            # verts.shape for a cube: 8 x 3
            cube = Meshes(verts = verts[None], faces = self.cube_faces)
            cube.textures = self.cube_textures # Add texture info
            self.frames.append( # Append render with ground to frame buffer
                self.render_meshes(join_meshes_as_scene([cube, self.ground])))
            if self.use_tqdm:
                progress.update(1)
        if self.use_tqdm:
            progress.close()
        self.export_video(result_path = result_fn, fps = fps)
        self.frames = []
