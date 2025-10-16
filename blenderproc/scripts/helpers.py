import os
import glob
import blenderproc as bproc
from blenderproc.python.utility.LabelIdMapping import LabelIdMapping
from blenderproc.python.loader.CCMaterialLoader import _CCMaterialLoader as CCMaterialLoader
from blenderproc.python.types.MaterialUtility import Material
import numpy as np
import math

def ensure_single_uv_layer(objs=None, keep="ACTIVE"):
    candidates = []
    for o in objs:
        try:
            bo = o.blender_obj if hasattr(o, "blender_obj") else o.get_blender_obj()
        except Exception:
            bo = None
        if bo and bo.type == 'MESH':
            candidates.append(bo)

    fixed, added, skipped = 0, 0, 0
    for obj in candidates:
        me = obj.data
        uvs = me.uv_layers

        if len(uvs) == 0:
            uvs.new(name="UVMap")
            added += 1
            continue

        if len(uvs) == 1:
            continue

        # Mehr als eine UV: wähle, welche du behältst
        keep_idx = uvs.active_index if keep.upper() == "ACTIVE" else 0
        # Entferne alle bis auf die behaltene
        # Wichtig: erst die zu löschenden Indizes sammeln, dann entfernen
        to_remove = [i for i in range(len(uvs)) if i != keep_idx]
        # Beim Entfernen von hinten nach vorn iterieren, damit Indizes stabil bleiben
        for i in sorted(to_remove, reverse=True):
            uvs.remove(uvs[i])

        # Sicherstellen, dass genau 1 übrig ist
        if len(uvs) != 1:
            skipped += 1  # sollte nicht passieren, nur zur Diagnose
        else:
            fixed += 1

    print(f"[UV-Fix] added:{added} fixed:{fixed} skipped:{skipped} total:{len(candidates)}")


def load_objects(path: str = '../assets/objects/filtered/', file_extension: str = 'ply', is_target: bool = True, category_offset: int = 1, same_category: bool = False):
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
    for i, file_path in enumerate(object_files):
        loaded_objs = bproc.loader.load_obj(file_path)
        
        if not loaded_objs:
            print(f"WARNING: No objects loaded from {file_path}. Skipping this file.")
            continue

        obj = loaded_objs[0]
        if len(loaded_objs) > 1:
            obj.join_with_other_objects(loaded_objs[1:])

        if same_category:
            category_id = category_offset * is_target
            obj.set_cp("category_id", category_id)
        else:
            category_id = (i+category_offset) * is_target
            obj.set_cp("category_id", category_id)

        object_id = 1 if not same_category else (i+1)
        obj.set_cp("object_id", f"{category_id}_{object_id}")

        obj.hide(True)
        obj.set_shading_mode('auto')
        all_objs.append(obj)

    return all_objs

def load_tools():
    raw_label_mapping = {
        "background": 0,
        "hammer": 1,
        "screwdriver": 2,
        "wrench": 3,
        "pliers": 4,
    }
    label_mapping = LabelIdMapping.from_dict(raw_label_mapping)

    hammer_objs = load_objects('../assets/objects/tools/hammer/', file_extension='glb', category_offset=1, same_category=True)

    screwdriver_objs = load_objects('../assets/objects/tools/screwdriver/', file_extension='glb', category_offset=2, same_category=True)

    wrench_objs = load_objects('../assets/objects/tools/wrench/', file_extension='glb', category_offset=3, same_category=True)

    plier_objs = load_objects('../assets/objects/tools/pliers/', file_extension='glb', category_offset=4, same_category=True)

    target_objs = []
    target_objs.extend(hammer_objs)
    target_objs.extend(screwdriver_objs)
    target_objs.extend(wrench_objs)
    target_objs.extend(plier_objs)

    return target_objs, label_mapping

def load_hammers():
    raw_label_mapping = {
        "background": 0,
        "hammer_1": 1,
        "hammer_2": 2,
        "hammer_3": 3,
        "hammer_4": 4,
        "hammer_5": 5,
        "hammer_6": 6,
        "hammer_7": 7,
        "hammer_8": 8,
        "hammer_9": 9,
    }
    label_mapping = LabelIdMapping.from_dict(raw_label_mapping)

    hammer_objs = load_objects('../assets/objects/hammers/', file_extension='ply', category_offset=1, same_category=False)

    for obj in hammer_objs:
        s = 0.002 # from visual inspection of the models in blender
        obj.set_scale([s, s, s])
        obj.persist_transformation_into_mesh()

    return hammer_objs, label_mapping

def load_targets(scenario: int = 0):
    if scenario == 1:
        print("Using tools as target objects.")
        target_objs, label_mapping = load_tools()
    elif scenario == 2:
        print("Using hammers as target objects.")
        target_objs, label_mapping = load_hammers()
    else:
        raise ValueError("Invalid scenario number. Please choose 1 or 2.")

    ensure_single_uv_layer(target_objs, keep="ACTIVE")
    
    return target_objs, label_mapping

def load_distractors():
    distractor_objs = load_objects('../assets/objects/distractors/', file_extension='ply', is_target=False)
    for obj in distractor_objs:
        s = 0.002
        obj.set_scale([s, s, s])
        obj.persist_transformation_into_mesh()
    return distractor_objs

def load_materials(path: str = '../assets/materials/'):
    materials = []

    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            base_image_path = os.path.join(folder_path, f"{folder}_1K-PNG_Color.png")
            ambient_occlusion_image_path = base_image_path.replace("Color", "AmbientOcclusion")
            metallic_image_path = base_image_path.replace("Color", "Metalness")
            roughness_image_path = base_image_path.replace("Color", "Roughness")
            alpha_image_path = base_image_path.replace("Color", "Opacity")
            normal_image_path = base_image_path.replace("Color", "NormalDX")
            if not os.path.exists(normal_image_path):
                normal_image_path = base_image_path.replace("Color", "NormalGL")
            displacement_image_path = base_image_path.replace("Color", "Displacement")

        new_mat = bproc.material.create_new_cc_material(f"CCMaterial_{folder}", dict())
        CCMaterialLoader.create_material(new_mat, base_image_path, ambient_occlusion_image_path, metallic_image_path, roughness_image_path, alpha_image_path, normal_image_path, displacement_image_path)

        materials.append(Material(new_mat))

    if not materials:
        print(f"WARNING: No materials found in the specified path: {path}")
    
    return materials

def random_color_material(o):
    color = np.random.uniform(0.0, 1.0, size=3)
    color = np.append(color, 1.0)
    material = bproc.material.create("RandomColor")
    material.set_principled_shader_value("Base Color", color)
    o.replace_materials(material)

def random_material(o, materials):
    if not materials:
        print("No materials available for random assignment.")
        return
    material = np.random.choice(materials)
    if o.has_uv_mapping():
        o.add_uv_mapping("smart", True)
        o.scale_uv_coordinates(np.random.uniform(1.0, 5.0))
    o.replace_materials(material)

def jitter_scale(o, value: float = 0.1, per_axis: bool = False, reset: bool = False):
    if reset:
        o.set_scale([1.0, 1.0, 1.0])

    scale_base = o.get_scale()
    if per_axis:
        s = scale_base * np.random.uniform(1 - value, 1 + value, size=3)
    else:
        s = scale_base * np.random.uniform(1 - value, 1 + value)
    o.set_scale(s)

def enable_physics(o):
    o.enable_rigidbody(
        active=True,
        mass=1.0,
        collision_shape='CONVEX_HULL',
        collision_margin=0.0
    )

def clean_up(objs):
    for o in objs:
        o.disable_rigidbody()
        o.delete()

def create_primitive_objects():
    objects = []
    prim_types = ['CUBE', 'SPHERE', 'CYLINDER', 'CONE']
    for prim_type in prim_types:
        for _ in range(5):
            obj = bproc.object.create_primitive(prim_type)
            obj.set_scale([0.25, 0.25, 0.25])
            obj.persist_transformation_into_mesh()
            obj.set_cp("category_id", 0)
            obj.set_shading_mode('auto')
            obj.hide(True)
            objects.append(obj)
    return objects

def set_basic_light():
    light = bproc.types.Light()
    light.set_type("SUN")
    light.set_location([0, 0, 10])
    light.set_rotation_euler([math.radians(-60), 0, math.radians(45)])
    light.set_energy(1)
    light.set_color([1, 1, 1])
    return light

import mathutils

def set_random_light_aimed(poi, types=("SUN","POINT","AREA","SPOT"),
                           xy_radius=10.0, z_min=3.0, z_max=15.0):
    light = bproc.types.Light()
    ltype = np.random.choice(types)
    light.set_type(ltype)

    # Position oberhalb der Szene, zufällig um den POI herum
    angle = np.random.uniform(0, 2*math.pi)
    r = np.random.uniform(0.0, xy_radius)
    x = poi[0] + r * math.cos(angle)
    y = poi[1] + r * math.sin(angle)
    z = np.random.uniform(z_min, z_max)  # immer > 0 => über der Szene
    light.set_location([x, y, z])

    # Richtung auf POI (für SUN/AREA/SPOT). Hinweis: Blender-Lichter "schauen" entlang -Z.
    if ltype in ("SUN", "AREA", "SPOT"):
        forward = np.array(poi, dtype=float) - np.array([x, y, z], dtype=float)
        R = bproc.camera.rotation_from_forward_vec(forward, inplane_rot=np.random.uniform(0, 2*math.pi))
        eul = mathutils.Matrix(R).to_euler()   # Matrix -> Euler
        light.set_rotation_euler([eul.x, eul.y, eul.z])

    # Energie breit streuen (hell/dunkel)
    if ltype == "SUN":
        energy = np.random.uniform(0.1, 5.0)
        strength = energy / 5.0
    elif ltype == "POINT":
        energy = np.random.uniform(10, 1000)
        strength = energy / 1000
    elif ltype == "AREA":
        energy = np.random.uniform(10, 2000)
        strength = energy / 2000
    else:  # SPOT
        energy = np.random.uniform(10, 500)
        strength = energy / 500
    light.set_energy(energy)

    # Bunte Farbe erlaubt
    rgb = np.random.uniform(0.0, 1.0, size=3)
    light.set_color(rgb.tolist())

    bproc.renderer.set_world_background(rgb.tolist(), strength=strength * 0.2)

    return light

def sample_pose(obj: bproc.types.MeshObject, surface):
    # Sample location above surface
    obj.set_location(bproc.sampler.upper_region(
        objects_to_sample_on=[surface],
        face_sample_range=(0.1, 0.9),
        min_height=0.1,
        max_height=1
    ))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())