### script to generate frontal shots of 3D target models in front of green background for overview images

import blenderproc as bproc
import os
import imageio as iio
import glob
import argparse

parser = argparse.ArgumentParser()
# accept a single string for output dir (not a list)
parser.add_argument('--output-dir', dest='output_dir', default='../../output/frontal_shot/',
                    help='Directory to write output images to')
args = parser.parse_args()

def load_objects(path: str = '../../assets/objects/', file_extension: str = 'ply'):
    object_files = []
    if os.path.isdir(path):
        print(f"Loading objects from folder {path} with extension {file_extension}...")
        object_files = sorted(
            glob.glob(os.path.join(path, f'**/*.{file_extension}'), recursive=True),
            key=lambda x: os.path.basename(x).lower()
        )
        print(f"Found {len(object_files)} files.")
    else:
        print(f"Loading single object file: {path}")
        object_files = [path]

    if not object_files:
        raise ValueError(f"No object files found in the specified path: {path}")

    all_objs = []
    for file_path in object_files:
        loaded_objs = bproc.loader.load_obj(file_path)
        
        if not loaded_objs:
            print(f"WARNING: No objects loaded from {file_path}. Skipping this file.")
            continue

        obj = loaded_objs[0]
        if len(loaded_objs) > 1:
            obj.join_with_other_objects(loaded_objs[1:])

        obj.hide(True)
        obj.set_shading_mode('auto')
        all_objs.append(obj)

    return all_objs

def load_tools():
    hammer_objs = load_objects('../../assets/objects/tools/hammer/', file_extension='glb')

    screwdriver_objs = load_objects('../../assets/objects/tools/screwdriver/', file_extension='glb')

    wrench_objs = load_objects('../../assets/objects/tools/wrench/', file_extension='glb')

    plier_objs = load_objects('../../assets/objects/tools/pliers/', file_extension='glb')

    target_objs = []
    target_objs.extend(hammer_objs)
    target_objs.extend(screwdriver_objs)
    target_objs.extend(wrench_objs)
    target_objs.extend(plier_objs)

    return target_objs

def load_hammers():
    hammer_objs = load_objects('../../assets/objects/hammers/', file_extension='ply')

    for obj in hammer_objs:
        s = 0.002 # from visual inspection of the models in blender
        obj.set_scale([s, s, s])
        obj.persist_transformation_into_mesh()

    return hammer_objs

def load_targets(scenario: int = 0):
    if scenario == 0:
        print("Using tools as target objects.")
        target_objs = load_tools()
    elif scenario == 1:
        print("Using hammers as target objects.")
        target_objs = load_hammers()
    else:
        raise ValueError("Invalid scenario number. Please choose 0 or 1.")

    return target_objs

bproc.init()
bproc.renderer.set_max_amount_of_samples(128)

out_dir = os.path.abspath(os.path.expanduser(args.output_dir))
os.makedirs(out_dir, exist_ok=True)

print(f"Writing frontal shots to: {out_dir}")

bproc.camera.set_resolution(512, 1024)

cam_pose = bproc.math.build_transformation_mat(
    [1, -1, 0.5],
    bproc.camera.rotation_from_forward_vec([-1, 1, -0.5])
)
bproc.camera.add_camera_pose(cam_pose)

light = bproc.types.Light()
light.set_type("SUN")
light.set_location([5, -5, 5])
light.set_energy(2.0)

bproc.renderer.set_world_background([1, 1, 1], strength=1.0)

# background plane
green = bproc.material.create("ChromaGreen")
green.set_principled_shader_value("Base Color", [0.0, 0.5, 0.0, 1.0])  # #00FF00

back = bproc.object.create_primitive('PLANE', size=10)
back.set_location([0, 1.2, 0])
back.set_rotation_euler([1.5708, 0.0, 3.1416])
back.replace_materials(green)

objs0 = load_targets(scenario=0)
objs1 = load_targets(scenario=1)
for o in objs0 + objs1:
    o.set_location([0, 0, 0])
    o.set_rotation_euler([0, 1.5708, 0])
    o.hide(False)
    data = bproc.renderer.render()

    img = data["colors"][0]
    name = o.get_name()
    iio.imwrite(os.path.join(out_dir, f"{name}.png"), img)

    o.hide(True)