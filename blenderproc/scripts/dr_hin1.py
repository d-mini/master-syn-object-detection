import blenderproc as bproc
import numpy as np
from collections import defaultdict
import time
import sys
import os
import argparse
import shutil
import yaml
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from helpers import enable_physics, clean_up, load_targets, set_basic_light


# ----- Argument and config parsing ------

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", type=int, required=True, choices=[1, 2])
parser.add_argument("--inst-per-class", type=int, default=1, help="Number of instances per class to generate")
parser.add_argument("--scene", type=str, default="apartment", help="Scene to use as background", choices=["apartment", "basement", "warehouse", "construction", "garden", "city"])
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
args = parser.parse_args()

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# ----- Helper Functions -----

def sample_pose(obj: bproc.types.MeshObject):
    # Sample the spheres location above the surface
    obj.set_location(bproc.sampler.upper_region(
        objects_to_sample_on=[ground],
        face_sample_range=(0.1, 0.9),
        min_height=0.1,
        max_height=1
    ))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

# -----------------------------

OUTPUT_DIR = f"../output/scenario{args.scenario}/dr_hin1/{args.scene}/inst{args.inst_per_class}_seed{args.seed}"
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

random.seed(args.seed); np.random.seed(args.seed)

start_time = time.time()

# ----- Initialization -----

bproc.init()
bproc.camera.set_resolution(cfg['camera']['resolution']['width'], cfg['camera']['resolution']['height'])
bproc.renderer.set_max_amount_of_samples(cfg['rendering']['num_samples']//2)

# ----- Load Objects -----
target_objs, label_mapping = load_targets(args.scenario)

# ----- Load Materials -----
materials = bproc.material.collect_all()

# ----- Setup Scene -----

background = []
ground = None
scene_type = "indoor" if args.scene in ["apartment", "basement", "warehouse"] else "outdoor"

scene = bproc.loader.load_blend(f"../assets/scenes/{args.scene}.blend", obj_types=['mesh', 'light'])

boundaries = []

for obj in scene:
    b = obj.blender_obj

    if hasattr(b, "data") and hasattr(b.data, "users") and b.data.users > 1:
        b.data = b.data.copy() 

    if isinstance(obj, bproc.types.Light):
        obj.set_color([1, 1, 1])
        continue

    obj.hide(False)
    if obj.get_name() == "sampling_plane":
        obj.hide(True)
        obj.enable_rigidbody(active=False, collision_shape='MESH', collision_margin=0.0)
        ground = obj

    background.append(obj)

if scene_type == "outdoor":
    bproc.world.set_world_background_hdr_img("hdr/DaySkyHDRI022A_1K-HDR.exr")

if ground is None:
    ground = bproc.object.create_primitive('PLANE', scale=[15, 15, 0.1])
    ground.set_location([0, 0, 0])
    ground.enable_rigidbody(active=False, collision_shape='BOX', collision_margin=0.0)
background.append(ground)

original_objs = set(target_objs + background)

# -----------------------------

class_counts = defaultdict(int)
instances_per_class = args.inst_per_class

object_counts = defaultdict(int)
instances_per_object = max(1, instances_per_class//5)

neg_ratio = cfg['sampling']['neg_ratio']
neg_count = 0

target_class_ids = {o.get_cp("category_id") for o in target_objs}
object_ids = {o.get_cp("object_id") for o in target_objs}

scene_id = 0
while min(class_counts.get(cid, 0) for cid in target_class_ids) < instances_per_class or min(object_counts.get(oid, 0) for oid in object_ids) < instances_per_object:
    scene_id += 1
    print(f"\nGenerating Scene {scene_id}")
    print("Class counts:", dict(class_counts))
    print("Object counts:", dict(object_counts))

    bproc.utility.reset_keyframes()

    dup_prob = cfg['sampling']['dup_prob']
    max_target_objs = cfg['sampling']['max_target_objs']

    eligible = [o for o in target_objs if object_counts[o.get_cp("object_id")] < instances_per_object] # only consider objects of classes that still need more instances

    if not eligible:
        eligible = [
            o for o in target_objs
            if class_counts[o.get_cp("category_id")] < instances_per_class
        ]

    if not eligible:
        print("No eligible objects left, stopping generation.")
        break

    num_objs = np.random.randint(1, min(max_target_objs, len(eligible)) + 1)

    base_objs = []  
    base_objs = np.random.choice(eligible, size=num_objs, replace=False)

    sampled_target_objs = [] # sampled target objects including duplicates
    for obj in base_objs:
        sampled_target_objs.append(obj.duplicate())
        if np.random.rand() < dup_prob:
            dup = obj.duplicate()
            sampled_target_objs.append(dup)

    for obj in target_objs:
        obj.hide(True)
    for obj in sampled_target_objs:
        enable_physics(obj)
        obj.hide(False)

    bproc.object.sample_poses_on_surface(sampled_target_objs, ground, sample_pose, min_distance=cfg['sampling']['min_distance'], max_distance=cfg['sampling']['max_distance'])
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3, max_simulation_time=10, check_object_interval=1)

    bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_objs)

    if scene_type == "outdoor":
        radius_sets = cfg["camera"]["radius_sets"]
    else: 
        radius_sets = [[1, 1.5], [2, 3]]
    poses_per_set = cfg["camera"]["poses_per_set"]

    for radius_set in radius_sets:
        radius_min, radius_max = radius_set
        poses = 0
        while poses < poses_per_set:
            poi = bproc.object.compute_poi(sampled_target_objs) if sampled_target_objs else np.array([0, 0, 0])
            location = bproc.sampler.shell(center=poi,
                                            radius_min=radius_min,
                                            radius_max=radius_max,
                                            elevation_min=10,
                                            elevation_max=80,
                                            uniform_volume=False)
            # Compute rotation based on vector going from location towards poi
            rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
            # Add homog cam pose based on location an rotation
            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

            if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bvh_tree): # check if camera pose is valid / desired distance to objects
                bproc.camera.add_camera_pose(cam2world_matrix)
                poses += 1

    # Render the scene

    bproc.renderer.enable_segmentation_output(map_by=["category_id", "object_id", "instance", "name"], default_values={"category_id": 0, "object_id": 0, "instance": 0, "name": "background"})
    data = bproc.renderer.render(file_prefix=f"{args.scene}")

    bproc.writer.write_coco_annotations(OUTPUT_DIR,
                                        file_prefix=f"{args.scene}",
                                        instance_segmaps=data["instance_segmaps"],
                                        instance_attribute_maps=data["instance_attribute_maps"],
                                        colors=data["colors"],
                                        color_file_format="JPEG",
                                        label_mapping=label_mapping,
                                        append_to_existing_output=True)

    # Count instances of each class
    for attr_map in data["instance_attribute_maps"]:
        has_target = False
        for entry in attr_map:
            cat_id = entry.get("category_id", 0)
            if cat_id in target_class_ids and cat_id != 0:
                has_target = True
                class_counts[cat_id] += 1

            obj_id = entry.get("object_id", None)
            if obj_id is not None and obj_id in object_ids:
                object_counts[obj_id] += 1

        if not has_target:
            neg_count += 1

    current_objs = set(bproc.object.get_all_mesh_objects())
    to_remove = current_objs - original_objs
    clean_up(to_remove)

# ---- Generate additional negative images if needed ----
bproc.utility.reset_keyframes()

neg_images_needed = int(scene_id * 4 * neg_ratio)

if neg_count < neg_images_needed:
    print(f"\nGenerating additional negative image from {neg_count + 1} of {neg_images_needed}")
    for count in range(neg_images_needed-neg_count):

        for obj in target_objs:
            obj.hide(True)

        poi = bproc.sampler.upper_region(
            objects_to_sample_on=[ground],
            face_sample_range=(0.1, 0.9),
            min_height=0.1,
            max_height=0.5
        )
        location = bproc.sampler.shell(center=poi,
                                        radius_min=radius_min,
                                        radius_max=radius_max,
                                        elevation_min=10,
                                        elevation_max=80,
                                        uniform_volume=False)
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

        bproc.camera.add_camera_pose(cam2world_matrix)
        poses += 1

    data = bproc.renderer.render(file_prefix=f"{args.scene}")

    bproc.writer.write_coco_annotations(OUTPUT_DIR,
                                        file_prefix=f"{args.scene}",
                                        instance_segmaps=data["instance_segmaps"],
                                        instance_attribute_maps=data["instance_attribute_maps"],
                                        colors=data["colors"],
                                        color_file_format="JPEG",
                                        label_mapping=label_mapping,
                                        append_to_existing_output=True)


# -----------------------------

end_time = time.time()
elapsed = end_time - start_time
print(f"\nFinished in {elapsed:.2f} Seconds ({elapsed/60:.2f} Minutes).")
