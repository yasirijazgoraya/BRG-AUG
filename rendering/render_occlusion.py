# --- THIS MUST BE FIRST ---
import blenderproc as bproc
import signal, sys

# ========== STANDARD IMPORTS ==========
import os, glob, csv, random
import numpy as np
import cv2
from PIL import Image
import concurrent.futures
import atexit
from datetime import datetime

# Blender API
import bpy
import mathutils
import pandas as pd

# ===================== USER SETTINGS =====================
BLEND      = "/home/yasir/yasir_mnt/external3/Rendered_data/HRI/Occlusions/occlusions_HRI.blend"
HDRI_DIR   = "/home/yasir/yasir_mnt/external3/Rendered_data/HRI/Occlusions/assets/hdrs/industrial/"
OUT_DIR    = "/home/yasir/yasir_mnt/external3/Rendered_data/HRI/Occlusions/output_images_Occlusions/"

# Annotation / ROI (DO NOT CHANGE)
OBJ_NAME_ANNOTATION     = "HRI_loadstation"   # YOLO target
OBJ_NAME_OCCLUSION_BASE = "HRI_ROI"           # ROI used for IoU / Occ%
OBJ_NAME_OCCLUSION_PLACE = "ROI_occ"   # used for placement

# === EDIT THESE NAMES TO MATCH YOUR .blend ===
OCCLUDER_GROUPS = {
    "Screw_Driver": ["screwdriver1", "screwdriver2", "screwdriver3"],
    "Pliers"      : ["plier1",       "plier2",       "plier3"],
    "Gloves"      : ["glove1",       "glove2",       "glove3"],
    "Bottles"     : ["bottle1",      "bottle2",      "bottle3"],
    "Cubes"       : ["cube1",        "cube2",        "cube3"],
    "PaperStack"  : ["paper1",       "paper2",       "paper3"],
}

# ===================== INIT =====================
bproc.init()
try:
    bproc.renderer.set_device("GPU")
except Exception:
    print("[WARN] Could not force GPU device; continuing with default device.")
bproc.renderer.set_denoiser("optix")

# Output folders
IMG_DIR = os.path.join(OUT_DIR, "images")
LBL_DIR = os.path.join(OUT_DIR, "labels")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

# Per-group shot plan (154 total × 6 groups = 924 per HDRI)
SHOTS_PER_GROUP = {"single": 100, "dual": 50, "triple": 4}

# Output image resolution (DO NOT CHANGE vs control)
RES = (1280, 1280)

# ---- Camera sweep (DO NOT CHANGE vs control) ----
STEP_X = 10.0
STEP_Y = 5.0
STEP_Z = 10.0
WORLD_X_RANGE = (100.0, 200.0)
WORLD_Y_RANGE = (-15.0, 15.0)
WORLD_Z_RANGE = (50.0, 160.0)

# Keep simple/robust to avoid white frames; re-enable later if desired
HDRI_ROTATE_PER_SHOT = False
# If True, add "_FAILED" to filenames when occluder placement fails
TAG_FAILED_SHOTS = True


# Minimum XY gap between occluders (meters); set >0 to avoid touching
SEP_MARGIN = 0.1

# ---- batching ----
BATCH_SIZE     = 200
FLUSH_INTERVAL = 200

# Where to “park” inactive occluders
OFF_POS = (1e6, 1e6, 1e6)

# CSVs
MASTER_CSV  = os.path.join(OUT_DIR, "shot_log.csv")
RUN_SUMMARY = os.path.join(OUT_DIR, "run_summary.csv")

def _append_csv_row(row_dict):
    write_header = not os.path.exists(MASTER_CSV)
    with open(MASTER_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row_dict)

def _write_run_summary(summary_dict):
    write_header = not os.path.exists(RUN_SUMMARY)
    with open(RUN_SUMMARY, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_dict.keys()))
        if write_header:
            w.writeheader()
        w.writerow(summary_dict)

# ===================== LOAD SCENE =====================
# if not os.path.isfile(BLEND):
#     raise FileNotFoundError(BLEND)
# bproc.loader.load_blend(BLEND)

# objs = bproc.object.get_all_mesh_objects()

# ===================== LOAD SCENE =====================
if not os.path.isfile(BLEND):
    raise FileNotFoundError(BLEND)
bproc.loader.load_blend(BLEND)

objs = bproc.object.get_all_mesh_objects()

# --- Fix for glass/transparent materials rendering black ---
import bpy

scene = bpy.context.scene
scene.render.engine = 'CYCLES'

# Increase light bounces for glass
scene.cycles.max_bounces = 12
scene.cycles.transparent_max_bounces = 12
scene.cycles.transmission_bounces = 12
scene.cycles.glossy_bounces = 12
scene.cycles.samples = 128

# Ensure transparent materials don't render black
scene.render.film_transparent = True

# Boost HDRI brightness (glass needs more light)
world = bpy.data.worlds['World']
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
if bg:
    bg.inputs[1].default_value = 2.0  # bump up if still dark








def _get(name):
    return next((o for o in objs if o.get_name() == name), None)

anno_obj = _get(OBJ_NAME_ANNOTATION)
base_obj = _get(OBJ_NAME_OCCLUSION_BASE)
occ_place_obj = _get(OBJ_NAME_OCCLUSION_PLACE)  # ⬅️ NEW

if anno_obj is None or base_obj is None or occ_place_obj is None:
    missing = []
    if anno_obj is None: missing.append(OBJ_NAME_ANNOTATION)
    if base_obj is None: missing.append(OBJ_NAME_OCCLUSION_BASE)
    if occ_place_obj is None: missing.append(OBJ_NAME_OCCLUSION_PLACE)
    raise RuntimeError(f"❌ Missing required object(s) in scene: {', '.join(missing)}")



# --- Forbidden zones from scene (only those explicitly created in .blend) ---
forb_objects = [o for o in objs if o.get_name().lower().startswith("forbidden_zone")]
# --- make helper meshes invisible in renders (but still usable for math) ---
helpers = [obj for obj in [base_obj, occ_place_obj] if obj is not None] + forb_objects
for h in helpers:
    try:
        h.blender_obj.hide_render = True                   # don't render
        h.blender_obj.hide_viewport = False                # keep visible in viewport
        vis = h.blender_obj.cycles_visibility
        vis.camera = False
        vis.diffuse = False
        vis.glossy = False
        vis.transmission = False
        vis.scatter = False
        vis.shadow = False
        # 🚀 extra safety: unlink from collections so it *never* gets drawn
        for coll in list(h.blender_obj.users_collection):
            coll.objects.unlink(h.blender_obj)
    except Exception:
        pass
# --- Hide default Camera object (prevents cylinder showing up) ---
cam_obj = bpy.data.objects.get("Camera")
if cam_obj:
    cam_obj.hide_render = True
    cam_obj.hide_viewport = True






# --- Helpers: world-space AABBs ---
def _set_rot_key(obj, rot_euler, frame_idx):
    """Keyframe occluder rotation with CONSTANT interpolation (no tweening)."""
    obj.blender_obj.rotation_euler = rot_euler
    obj.blender_obj.keyframe_insert(data_path="rotation_euler", frame=frame_idx)
    ad = obj.blender_obj.animation_data
    if ad and ad.action:
        for fcu in ad.action.fcurves:
            for kp in fcu.keyframe_points:
                kp.interpolation = 'CONSTANT'



def _world_aabb_xy(bpy_obj):
    deps = bpy.context.evaluated_depsgraph_get()
    ob_eval = bpy_obj.evaluated_get(deps)
    corners = [ob_eval.matrix_world @ mathutils.Vector(c) for c in ob_eval.bound_box]
    xmin, xmax = min(v.x for v in corners), max(v.x for v in corners)
    ymin, ymax = min(v.y for v in corners), max(v.y for v in corners)
    return xmin, xmax, ymin, ymax

def _world_aabb(bpy_obj):
    """Full 3D AABB in world space (x_min, x_max, y_min, y_max, z_min, z_max)."""
    deps = bpy.context.evaluated_depsgraph_get()
    ob_eval = bpy_obj.evaluated_get(deps)
    corners = [ob_eval.matrix_world @ mathutils.Vector(c) for c in ob_eval.bound_box]
    xmin, xmax = min(v.x for v in corners), max(v.x for v in corners)
    ymin, ymax = min(v.y for v in corners), max(v.y for v in corners)
    zmin, zmax = min(v.z for v in corners), max(v.z for v in corners)
    return xmin, xmax, ymin, ymax, zmin, zmax


def _aabb_overlap_3d(a, b, margin=0.0):
    """Check if two 3D AABBs overlap (with optional margin)."""
    axmin, axmax, aymin, aymax, azmin, azmax = a
    bxmin, bxmax, bymin, bymax, bzmin, bzmax = b
    return (axmin - margin < bxmax and axmax + margin > bxmin and
            aymin - margin < bymax and aymax + margin > bymin and
            azmin - margin < bzmax and azmax + margin > bzmin)


# --- ROI sampling window (shrunken for safety) ---
# --- ROI sampling window (shrunken for safety) ---
bxmin, bxmax, bymin, bymax = _world_aabb_xy(occ_place_obj.blender_obj)
SAFE_MARGIN_X = 1.0
SAFE_MARGIN_Y = 5.0
OCCL_X_RANGE = (bxmin + SAFE_MARGIN_X, bxmax - SAFE_MARGIN_X)
OCCL_Y_RANGE = (bymin + SAFE_MARGIN_Y, bymax - SAFE_MARGIN_Y)
print(f"[INFO] Occluder placement window from ROI (safe) → X={OCCL_X_RANGE}, Y={OCCL_Y_RANGE}")

# --- Forbidden zones from scene (only those explicitly created in .blend) ---
forb_objects = [o for o in objs if o.get_name().lower().startswith("forbidden_zone")]

def _static_forbidden_aabbs():
    aabbs = []
    for o in forb_objects:
        try:
            axmin, axmax, aymin, aymax = _world_aabb_xy(o.blender_obj)
            aabbs.append((axmin, axmax, aymin, aymax))
        except Exception:
            pass
    return aabbs

FORB_AABBS = _static_forbidden_aabbs()
print(f"[INFO] Found {len(FORB_AABBS)} forbidden zones from scene.")

# --- Loadstation (MLO_loadstation) is NOT added as forbidden zone ---
loadstation_obj = _get(OBJ_NAME_ANNOTATION)  # "MLO_loadstation"
if loadstation_obj:
    LOADSTATION_AABB = _world_aabb(loadstation_obj.blender_obj)
    _, _, _, _, lzmin, lzmax = LOADSTATION_AABB
    print(f"[INFO] Loadstation Z range = ({lzmin:.2f}, {lzmax:.2f})")
else:
    LOADSTATION_AABB = None



def _world_aabb(bpy_obj):
    """Full 3D AABB of any object in world space"""
    deps = bpy.context.evaluated_depsgraph_get()
    ob_eval = bpy_obj.evaluated_get(deps)
    corners = [ob_eval.matrix_world @ mathutils.Vector(c) for c in ob_eval.bound_box]
    xs = [v.x for v in corners]
    ys = [v.y for v in corners]
    zs = [v.z for v in corners]
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)






# # ===================== OBJECTS =====================
group_objs = {}
all_occluders = []
for gname, names in OCCLUDER_GROUPS.items():
    lst = []
    for n in names:
        o = _get(n)
        if o is None:
            raise RuntimeError(f"❌ Missing occluder object '{n}' in group '{gname}'.")
        lst.append(o)
        all_occluders.append(o)
    group_objs[gname] = lst

# ✅ Cache each occluder's original Z from the .blend
# occluder_rest_z = {oc.get_name(): oc.get_location()[2] for oc in all_occluders}

# ✅ Cache each occluder's original Z and rotation from the .blend
occluder_rest_z = {oc.get_name(): oc.get_location()[2] for oc in all_occluders}
occluder_rest_rot = {oc.get_name(): tuple(oc.blender_obj.rotation_euler[:]) for oc in all_occluders}







# Assign segmentation IDs (class segmaps; anno fixed at 2)
for idx, o in enumerate(objs):
    o.set_cp("category_id", idx + 1)
anno_obj.set_cp("category_id", 2)
anno_seg_id = 2

# Ensure occluders visible; we “disable” by moving them to OFF_POS per frame
for oc in all_occluders:
    try:
        oc.blender_obj.hide_render = False
        oc.blender_obj.hide_viewport = False
    except Exception:
        pass

# ===================== CAMERA/RENDER CONFIG =====================
def K_from_fov(fov_deg, width, height):
    fov = np.deg2rad(fov_deg)
    fx = fy = (width/2.0)/np.tan(fov/2.0)
    cx, cy = width/2.0, height/2.0
    return np.array([[fx,0,cx], [0,fy,cy], [0,0,1]], dtype=np.float32)

K = K_from_fov(50, RES[0], RES[1])
bproc.camera.set_intrinsics_from_K_matrix(K, RES[0], RES[1])
bproc.renderer.set_output_format(file_format="JPEG", color_depth=8,
                                 enable_transparency=False, jpg_quality=90)
# IMPORTANT: class segmaps since we set category_id
bproc.renderer.enable_segmentation_output(map_by="class")

# ===================== HELPERS =====================
def save_rgb(rgb_np, path_jpg):
    arr = rgb_np
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    Image.fromarray(arr, mode="RGB").save(path_jpg, quality=90)

def coco_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    return [(x + w/2)/img_w, (y + h/2)/img_h, w/img_w, h/img_h]

def set_look_at_pose(cam_pos, look_at):
    forward = (look_at - cam_pos)
    rot = bproc.camera.rotation_from_forward_vec(forward)
    T = bproc.math.build_transformation_mat(cam_pos, rot)
    bproc.camera.add_camera_pose(T)
    return T

def _aabb_overlap_xy(a, b, margin=0.0):
    axmin, axmax, aymin, aymax = a
    bxmin, bxmax, bymin, bymax = b
    axmin -= margin; aymin -= margin
    axmax += margin; aymax += margin
    bxmin -= margin; bymin -= margin
    bxmax += margin; bymax += margin
    return (axmin < bxmax) and (axmax > bxmin) and (aymin < bymax) and (aymax > bymin)

def _aabb_overlap_3d(a, b, margin=0.0):
    axmin, axmax, aymin, aymax, azmin, azmax = a
    bxmin, bxmax, bymin, bymax, bzmin, bzmax = b
    axmin -= margin; aymin -= margin; azmin -= margin
    axmax += margin; aymax += margin; azmax += margin
    bxmin -= margin; bymin -= margin; bzmin -= margin
    bxmax += margin; bymax += margin; bzmax += margin
    return (axmin < bxmax and axmax > bxmin and
            aymin < bymax and aymax > bymin and
            azmin < bzmax and azmax > bzmin)

COLLISION_MARGIN_3D = 0.01  # small tolerance

def _set_loc_key(obj, loc, frame_idx):
    """Keyframe occluder location with CONSTANT interpolation (no tweening)."""
    obj.blender_obj.location = loc
    obj.blender_obj.keyframe_insert(data_path="location", frame=frame_idx)
    ad = obj.blender_obj.animation_data
    if ad and ad.action:
        for fcu in ad.action.fcurves:
            for kp in fcu.keyframe_points:
                kp.interpolation = 'CONSTANT'

def _sample_valid_position(obj, occupied_aabbs):
    """Sample XY inside ROI, keep Z from .blend, reject on:
       - forbidden XY zones
       - overlap with already placed occluders (XY)"""

    z = occluder_rest_z.get(obj.get_name(), 0.0)
    retries = 500
    last = ((0.0, 0.0, z), (0, 0, 0, 0))

    for attempt in range(retries):
        x = float(np.random.uniform(*OCCL_X_RANGE))
        y = float(np.random.uniform(*OCCL_Y_RANGE))

        # Temporarily move to compute AABB
        obj.set_location([x, y, z])
        bpy.context.view_layer.update()  # force Blender to recompute

        axmin, axmax, aymin, aymax = _world_aabb_xy(obj.blender_obj)
        aabb2d = (axmin, axmax, aymin, aymax)
        last = ((x, y, z), aabb2d)

        # Reject if overlaps forbidden XY zones (with negative margin = stricter)
        if any(_aabb_overlap_xy(aabb2d, forb, margin=-0.05) for forb in FORB_AABBS):
            continue

        # Reject if overlaps previously placed occluders
        if any(_aabb_overlap_xy(aabb2d, o, margin=SEP_MARGIN) for o in occupied_aabbs):
            continue

        # ✅ Valid position found — log it for analysis
        print(f"[INFO] Accepted placement for {obj.get_name()} → "
              f"pos=({x:.2f},{y:.2f},{z:.2f}), "
              f"AABB_X=({axmin:.2f},{axmax:.2f}), "
              f"AABB_Y=({aymin:.2f},{aymax:.2f})")

        return (x, y, z), aabb2d

    # ⚠ Fallback if no valid found
    print(f"[WARN] ⚠ Fallback placement for {obj.get_name()} → {last[0]}")
    return last







def compute_topdown_occlusion_multi(base_obj, occl_list):
    """IoU/Occ% (XY) between base ROI and union of occluders (sum of pairwise intersections)."""
    deps = bpy.context.evaluated_depsgraph_get()
    base_eval = base_obj.blender_obj.evaluated_get(deps)
    bc = [base_eval.matrix_world @ mathutils.Vector(c) for c in base_eval.bound_box]
    bxmin, bxmax = min(v.x for v in bc), max(v.x for v in bc)
    bymin, bymax = min(v.y for v in bc), max(v.y for v in bc)
    base_area = max(0.0, (bxmax - bxmin)) * max(0.0, (bymax - bymin))

    inter_area_sum = 0.0
    occ_area_sum   = 0.0
    for occ in occl_list:
        oe = occ.blender_obj.evaluated_get(deps)
        oc = [oe.matrix_world @ mathutils.Vector(c) for c in oe.bound_box]
        oxmin, oxmax = min(v.x for v in oc), max(v.x for v in oc)
        oymin, oymax = min(v.y for v in oc), max(v.y for v in oc)
        occ_area_sum += max(0.0, (oxmax-oxmin)) * max(0.0, (oymax-oymin))

        ixmin, ixmax = max(bxmin, oxmin), min(bxmax, oxmax)
        iymin, iymax = max(bymin, oymin), min(bymax, oymax)
        if ixmin < ixmax and iymin < iymax:
            inter_area_sum += (ixmax - ixmin) * (iymax - iymin)

    union_area = base_area + occ_area_sum - inter_area_sum
    iou = (inter_area_sum / union_area * 100.0) if union_area > 0 else 0.0
    occ_pct = (inter_area_sum / base_area * 100.0) if base_area > 0 else 0.0
    return base_area, occ_area_sum, inter_area_sum, union_area, iou, occ_pct

# ===================== RESUMPTION =====================
progress_per_hdri = {}
existing_rows = 0
global_start_index = 0
if os.path.exists(MASTER_CSV):
    try:
        df = pd.read_csv(MASTER_CSV)
        progress_per_hdri = df.groupby("hdri").size().to_dict()
        existing_rows = len(df)
        if "shot_idx" in df.columns and pd.api.types.is_numeric_dtype(df["shot_idx"]):
            global_start_index = int(df["shot_idx"].max()) + 1
        else:
            global_start_index = 0
    except Exception as e:
        print(f"[WARN] Could not parse CSV for resumption: {e}")
        progress_per_hdri, existing_rows, global_start_index = {}, 0, 0
shot_index = global_start_index

# ===================== PRINT SUMMARY =====================
grid_x = np.arange(WORLD_X_RANGE[0], WORLD_X_RANGE[1] + 1e-6, STEP_X)
grid_y = np.arange(WORLD_Y_RANGE[0], WORLD_Y_RANGE[1] + 1e-6, STEP_Y)
grid_z = np.arange(WORLD_Z_RANGE[0], WORLD_Z_RANGE[1] + 1e-6, STEP_Z)
hdris = sorted(glob.glob(os.path.join(HDRI_DIR, "*.hdr")) +
               glob.glob(os.path.join(HDRI_DIR, "*.exr")))
if not hdris:
    raise RuntimeError(f"No HDRIs found in: {HDRI_DIR}")

obj_center = np.array(anno_obj.get_location(), dtype=float)

per_group_total = sum(SHOTS_PER_GROUP.values())  # 154
total_per_hdri = per_group_total * len(OCCLUDER_GROUPS)  # 924

print("\n========== RENDER JOB SUMMARY (OCCLUSION + BATCHED) ==========")
print(f"Blend file : {BLEND}")
print(f"HDRI dir   : {HDRI_DIR}  |  #HDRIs: {len(hdris)}")
print(f"Output dir : {OUT_DIR}")
print(f"Resolution : {RES[0]}x{RES[1]}  |  JPEG quality: 90")
print(f"Camera grid: X={len(grid_x)} × Y={len(grid_y)} × Z={len(grid_z)} = {len(grid_x)*len(grid_y)*len(grid_z)} shots per HDRI (should be 924)")
print(f"Plan/group : {SHOTS_PER_GROUP} → {per_group_total} each; groups={list(OCCLUDER_GROUPS.keys())}")
print(f"Occl XY    : X={OCCL_X_RANGE}  Y={OCCL_Y_RANGE}  |  Forbidden planes: {len(FORB_AABBS)}")
print(f"Batch      : {BATCH_SIZE}  |  Flush: {FLUSH_INTERVAL}")
print("==============================================================\n")

_write_run_summary({
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "blend": BLEND, "hdri_dir": HDRI_DIR, "out_dir": OUT_DIR,
    "res_w": RES[0], "res_h": RES[1],
    "groups": list(OCCLUDER_GROUPS.keys()),
    "shots_per_group": SHOTS_PER_GROUP, "per_group_total": per_group_total,
    "per_hdri": total_per_hdri, "num_hdris": len(hdris),
    "batch_size": BATCH_SIZE, "flush_interval": FLUSH_INTERVAL,
    "hdr_rotate_per_shot": HDRI_ROTATE_PER_SHOT,
    "occl_x_range": OCCL_X_RANGE, "occl_y_range": OCCL_Y_RANGE,
    "forbidden_planes": len(FORB_AABBS),
})

# ===================== THREAD POOL / SAFE EXIT =====================
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
pending = 0

def flush_executor():
    global executor, pending
    if executor:
        executor.shutdown(wait=True)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    pending = 0

def safe_exit():
    print("[INFO] Final flush before exit...")
    try:
        if executor:
            executor.shutdown(wait=True)
    except Exception:
        pass

atexit.register(safe_exit)
signal.signal(signal.SIGINT,  lambda s, f: (safe_exit(), sys.exit(0)))
signal.signal(signal.SIGTERM, lambda s, f: (safe_exit(), sys.exit(0)))

# ===================== SCHEDULER (924 shots order per HDRI) =====================
def yield_group_mode_sequence():
    """Yields (group_name, mode) in required order per HDRI."""
    for gname in OCCLUDER_GROUPS.keys():
        for _ in range(SHOTS_PER_GROUP["single"]): yield (gname, "single")
        for _ in range(SHOTS_PER_GROUP["dual"]):   yield (gname, "dual")
        for _ in range(SHOTS_PER_GROUP["triple"]): yield (gname, "triple")

print(f"[INFO] Occluder XY range: X={OCCL_X_RANGE}, Y={OCCL_Y_RANGE}")
print("[INFO] Forbidden zones (AABBs):")
for i, f in enumerate(FORB_AABBS, 1):
    print(f"   Zone {i}: xmin={f[0]:.2f}, xmax={f[1]:.2f}, ymin={f[2]:.2f}, ymax={f[3]:.2f}")
















for h_idx, hdri in enumerate(hdris, 1):
    # Load HDRI
    try:
        bproc.world.set_world_background_hdr_img(hdri, strength=1.0)
    except Exception as e:
        print(f"[WARN] Failed to load HDRI: {hdri} -> {e}")
        continue

    # For resumption: how many shots of this HDRI already in CSV
    done_for_hdri = progress_per_hdri.get(os.path.basename(hdri), 0)

    # ⚠️ Only skip if we *really* finished all shots
    if done_for_hdri >= total_per_hdri:
        print(f"[RESUME] HDRI {os.path.basename(hdri)} already complete → skipping.")
        continue
    else:
        print(f"[HDRI] {os.path.basename(hdri)} — resuming at {done_for_hdri}/{total_per_hdri}")

    # Prepare mapping node
    world = bpy.data.worlds['World']
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    if "Env Mapping" not in nodes:
        mapping = nodes.new(type='ShaderNodeMapping')
        mapping.name = "Env Mapping"
        mapping.inputs['Scale'].default_value = (1.0, 1.0, 1.0)
        tex_coord = nodes.get("Texture Coordinate") or nodes.new(type='ShaderNodeTexCoord')
        env = next((n for n in nodes if n.type == 'TEX_ENVIRONMENT'), None)
        if env:
            links.new(tex_coord.outputs['Generated'], mapping.inputs['Vector'])
            links.new(mapping.outputs['Vector'], env.inputs['Vector'])
    else:
        mapping = nodes["Env Mapping"]

    # --- HDRI rotation per-frame keyframer (so batched renders use per-shot values) ---
    def _set_env_rotation_key(mapping, world, theta, frame_idx):
        # Set Z-rotation only (X,Y = 0)
        mapping.inputs['Rotation'].default_value = (0.0, 0.0, float(theta))
        # Keyframe this frame so Blender uses different rotation on each frame in the batch
        mapping.inputs['Rotation'].keyframe_insert("default_value", frame=frame_idx)
        # Make interpolation CONSTANT (no tweening between frames)
        ad = world.node_tree.animation_data
        if ad and ad.action:
            for fcu in ad.action.fcurves:
                if 'nodes["Env Mapping"].inputs[2].default_value' in fcu.data_path:
                    for kp in fcu.keyframe_points:
                        kp.interpolation = 'CONSTANT'
        # Debug print (optional)
        print(f"[HDRI ROTATE] frame={frame_idx}, theta={theta:.2f} rad")

    # Buffers accumulated for one render() call
    cam_positions = []
    active_lists  = []
    occl_pos_list = []
    group_used    = []
    mode_used     = []
    failed_flags  = []   # marks placement failure for this frame

    # Reset any previous animation before starting the HDRI run
    bproc.utility.reset_keyframes()
    for oc in all_occluders:
        if oc.blender_obj.animation_data:
            oc.blender_obj.animation_data_clear()
    if world.node_tree.animation_data:
        world.node_tree.animation_data_clear()

    # Camera grid iterator — must match schedule length (924 per HDRI)
    cam_positions_all = [(x, y, z)
                         for z in np.arange(WORLD_Z_RANGE[0], WORLD_Z_RANGE[1] + 1e-6, STEP_Z)
                         for y in np.arange(WORLD_Y_RANGE[0], WORLD_Y_RANGE[1] + 1e-6, STEP_Y)
                         for x in np.arange(WORLD_X_RANGE[0], WORLD_X_RANGE[1] + 1e-6, STEP_X)]

    # Skip already-completed shots for this HDRI (resumption)
    shot_iter = iter(list(yield_group_mode_sequence())[done_for_hdri:])
    cam_iter  = iter(cam_positions_all[done_for_hdri:])

    def render_and_dispatch_batch():
        """Render all queued frames, process and log, then clear buffers/animations."""
        global shot_index, pending
        if not cam_positions:
            return

        # fallback: if TAG_FAILED_SHOTS wasn't defined, default to True
        tag_failed = TAG_FAILED_SHOTS if 'TAG_FAILED_SHOTS' in globals() else True

        data = bproc.renderer.render()
        num_frames = min(len(cam_positions), len(data["colors"]))

        for i in range(num_frames):
            bpy.context.scene.frame_set(i)
            bpy.context.evaluated_depsgraph_get().update()

            cam_pos = cam_positions[i]
            actives = active_lists[i]
            occ_pos = occl_pos_list[i]
            gname   = group_used[i]
            mode    = mode_used[i]

            # Did this frame fail placement?
            failed = failed_flags[i] if i < len(failed_flags) else False

            rgb_np  = data["colors"][i].copy()
            seg     = data["class_segmaps"][i].copy()

            # Annotation bbox for SegID=2
            mask = (seg == anno_seg_id).astype(np.uint8)
            bbox_x = bbox_y = bbox_w = bbox_h = None
            bboxes = []
            if np.any(mask):
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                if num_labels > 1:
                    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    x_min, y_min, w_box, h_box, area = stats[largest_idx]
                    if w_box > 0 and h_box > 0:
                        x_min = int(x_min); y_min = int(y_min); w_box = int(w_box); h_box = int(h_box)
                        bboxes.append({"class_id": 0, "bbox": (x_min, y_min, w_box, h_box)})
                        bbox_x, bbox_y, bbox_w, bbox_h = x_min, y_min, w_box, h_box

            base_area, occ_area, inter_area, union_area, iou2d, occ_pct = \
                compute_topdown_occlusion_multi(base_obj, actives)

            # ---- filenames (add _FAILED suffix when placement failed) ----
            suffix    = "_FAILED" if (tag_failed and failed) else ""
            base_name = f"HDRI_{h_idx:03d}_{gname}_{mode}_{shot_index:05d}{suffix}"
            jpg_path  = os.path.join(IMG_DIR, base_name + ".jpg")
            txt_path  = os.path.join(LBL_DIR, base_name + ".txt")

            # Save image + label
            def save_outputs(rgb=rgb_np.copy(), bb=bboxes, jpg=jpg_path, txt=txt_path):
                save_rgb(rgb, jpg)
                with open(txt, "w") as f:
                    for b in bb:
                        yolo_box = coco_to_yolo(b["bbox"], RES[0], RES[1])
                        f.write("1 " + " ".join(f"{v:.6f}" for v in yolo_box) + "\n")

            executor.submit(save_outputs)

            # CSV row (blank occluder names when failed; include marker column)
            occl_names = "" if failed else ";".join(o.get_name() for o in actives)
            row = {
                "shot_idx": shot_index,
                "base": base_name,
                "image": jpg_path,
                "label": txt_path,
                "hdri": os.path.basename(hdri),
                "group": gname,
                "mode": mode,
                "cam_x": float(cam_pos[0]),
                "cam_y": float(cam_pos[1]),
                "cam_z": float(cam_pos[2]),
                "occluders": occl_names,
                "occl_positions": ";".join(f"({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})" for p in occ_pos),
                "bbox_x": bbox_x, "bbox_y": bbox_y, "bbox_w": bbox_w, "bbox_h": bbox_h,
                "base_area": float(base_area),
                "obs_area": float(occ_area),
                "inter_area": float(inter_area),
                "union_area": float(union_area),
                "iou2d_pct": float(iou2d),
                "occ_pct_xy": float(occ_pct),
                "placement_failed": int(failed),   # 1 if failed placement, else 0
            }
            _append_csv_row(row)

            shot_index += 1
            pending += 1
            if pending >= FLUSH_INTERVAL:
                flush_executor()

        # Reset animations
        bproc.utility.reset_keyframes()
        for oc in all_occluders:
            if oc.blender_obj.animation_data:
                oc.blender_obj.animation_data_clear()
        if world.node_tree.animation_data:
            world.node_tree.animation_data_clear()

        # Clear buffers
        cam_positions.clear()
        active_lists.clear()
        occl_pos_list.clear()
        group_used.clear()
        mode_used.clear()
        failed_flags.clear()  # important

    # ===================== FRAME BUILD LOOP (per HDRI) =====================
    frames_built = 0
    total_needed = total_per_hdri - done_for_hdri
    skipped_shots = 0

    while frames_built < total_needed:
        try:
            gname, mode = next(shot_iter)
            cam = next(cam_iter)
        except StopIteration:
            break

        frame_idx = len(cam_positions)
        failed_this_frame = False

        # Randomize HDRI orientation per shot (only if enabled)
        if HDRI_ROTATE_PER_SHOT:
            theta = float(np.random.uniform(0.0, 2.0*np.pi))
            _set_env_rotation_key(mapping, world, theta, frame_idx)

        objs_in_group = group_objs[gname]

        # Choose occluders
        if mode == "single":
            active = [random.choice(objs_in_group)]
        elif mode == "dual":
            active = random.sample(objs_in_group, 2)
        else:  # triple
            active = list(objs_in_group)

        occupied_aabbs = []
        per_frame_occl_positions = []
        shot_valid = True

        fallback_used = False  # tracks whether we already tried the single-occluder fallback


        for oc in all_occluders:
            if oc in active:
                placed = False
                for attempt in range(100):
                    (x, y, z), aabb = _sample_valid_position(oc, occupied_aabbs)
                    _set_loc_key(oc, (x, y, z), frame_idx)

                    # keep original XY rotation, randomize Z
                    orig_rot = occluder_rest_rot[oc.get_name()]
                    rand_rot_z = float(np.random.uniform(0, 2*np.pi))
                    _set_rot_key(oc, (orig_rot[0], orig_rot[1], rand_rot_z), frame_idx)

                    bpy.context.view_layer.update()

                    if any(_aabb_overlap_xy(aabb, forb, margin=0.0) for forb in FORB_AABBS):
                        continue
                    if any(_aabb_overlap_xy(aabb, other, margin=SEP_MARGIN) for other in occupied_aabbs):
                        continue

                    occupied_aabbs.append(aabb)
                    per_frame_occl_positions.append((x, y, z))
                    placed = True
                    break

                # if not placed:
                #     print(f"[ERROR] Could not place {oc.get_name()} → skipping shot")
                #     shot_valid = False
                #     break



                if not placed:
                    # Fallback: try with ONLY the first occluder from this active set
                    if not fallback_used and len(active) >= 1:
                        fallback_used = True

                    # pick the first occluder in the original active list
                    first_only = active[0]
                    # reset per-frame state for a clean retry
                    occupied_aabbs.clear()
                    per_frame_occl_positions.clear()

                    # make this the only active occluder for this frame;
                    # the loop below will automatically park all others at OFF_POS
                    active = [first_only]

                    # attempt to place only this one
                    placed_single = False
                    for attempt2 in range(100):
                        (x2, y2, z2), aabb2 = _sample_valid_position(first_only, occupied_aabbs)
                        _set_loc_key(first_only, (x2, y2, z2), frame_idx)

                        orig_rot2 = occluder_rest_rot[first_only.get_name()]
                        rand_rot_z2 = float(np.random.uniform(0, 2*np.pi))
                        _set_rot_key(first_only, (orig_rot2[0], orig_rot2[1], rand_rot_z2), frame_idx)

                        bpy.context.view_layer.update()

                        if any(_aabb_overlap_xy(aabb2, forb, margin=0.0) for forb in FORB_AABBS):
                            continue
                        if any(_aabb_overlap_xy(aabb2, other, margin=SEP_MARGIN) for other in occupied_aabbs):
                            continue

                        occupied_aabbs.append(aabb2)
                        per_frame_occl_positions.append((x2, y2, z2))
                        placed_single = True
                        break

                    if placed_single:
                        # success with single fallback; continue the outer loop so
                        # non-active occluders get parked by the existing 'else' branch
                        continue
                    else:
                        print(f"[ERROR] Fallback (single only) also failed → mark shot as failed")
                        shot_valid = False
                        break
                # else:
                #     print(f"[ERROR] Could not place {oc.get_name()} → skipping shot")
                #     shot_valid = False
                #     break




            else:
                _set_loc_key(oc, OFF_POS, frame_idx)
                _set_rot_key(oc, (0.0, 0.0, 0.0), frame_idx)

        if not shot_valid:
            for oc in all_occluders:
                _set_loc_key(oc, OFF_POS, frame_idx)
                orig_rot = occluder_rest_rot[oc.get_name()]
                _set_rot_key(oc, orig_rot, frame_idx)
            shot_valid = True            # keep index alignment
            failed_this_frame = True     # mark failure
            skipped_shots += 1           # ⬇️ count this failure in the HDRI summary
            print("[FAIL] Failed_this_frame")  # ⬇️ explicit log line

        cam_pos = np.array(cam, dtype=float)
        set_look_at_pose(cam_pos, obj_center)  # <-- this registers the camera pose

        cam_positions.append(cam_pos)
        active_lists.append(active)
        occl_pos_list.append(per_frame_occl_positions)
        group_used.append(gname)
        mode_used.append(mode)
        failed_flags.append(failed_this_frame)
        frames_built += 1

        if len(cam_positions) >= BATCH_SIZE:
            render_and_dispatch_batch()

    # Render leftovers for this HDRI
    render_and_dispatch_batch()
    flush_executor()

    # HDRI summary
    print("==============================================================")
    print(f"[SUMMARY HDRI {h_idx}] {os.path.basename(hdri)} finished.")
    print(f"[SUMMARY] Total attempted frames : {total_needed}")
    print(f"[SUMMARY] Successfully built     : {frames_built}")
    print(f"[SUMMARY] Skipped (placement fail): {skipped_shots}")
    print("==============================================================")
