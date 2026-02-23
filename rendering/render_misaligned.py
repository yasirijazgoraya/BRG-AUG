# --- THIS MUST BE FIRST ---
import blenderproc as bproc
import signal, sys
import math


# ========== STANDARD IMPORTS ==========
import os, glob, csv
import numpy as np
import cv2                     # used for mask cleanup and bounding box extraction
from PIL import Image
import concurrent.futures       # for async saving
import atexit                   # flush threads on exit

# Blender Python API (only for visibility toggles)
import bpy

# ===================== USER SETTINGS =====================
BLEND      = "/home/yasir/yasir_mnt/external3/Rendered_data/HRI/Misaligned/misaligned_HRI.blend"
HDRI_DIR   = "/home/yasir/yasir_mnt/external3/Rendered_data/HRI/Misaligned/assets/hdrs/industrial/"
OUT_DIR    = "/home/yasir/yasir_mnt/external3/Rendered_data/HRI/Misaligned/output_images_Misaligned/"

OBJ_NAME_ANNOTATION     = "HRI_loadstation"
OBJ_NAME_OCCLUSION_BASE = "HRI_ROI"
OBJ_NAME_OCCLUDER       = "obs_Pliers"

RES = (1280, 1280)

# STEP_X = 10.0
# STEP_Y = 5.0
# STEP_Z = 10.0
# WORLD_X_RANGE = (100.0, 200.0)
# WORLD_Y_RANGE = (-15.0, 15.0)
# WORLD_Z_RANGE = (50.0, 160.0)
# # STEP_X = 50.0
# STEP_Y = 20.0
# STEP_Z = 30.0
# WORLD_X_RANGE = (550.0, 950.0)
# WORLD_Y_RANGE = (-120.0, 120.0)
# WORLD_Z_RANGE = (50.0, 250.0)

STEP_X = 10.0
STEP_Y = 5.0
STEP_Z = 10.0
WORLD_X_RANGE = (100.0, 200.0)
WORLD_Y_RANGE = (-15.0, 15.0)
WORLD_Z_RANGE = (50.0, 160.0)


HDRI_ROTATE_PER_SHOT = False

# ---- runtime controls ----
BATCH_SIZE     = 200   # render this many poses per render() call (saves frequently)
FLUSH_INTERVAL = 200   # force-finish background writes after this many shots (for testing)

# ===================== INIT =====================
bproc.init()

try:
    bproc.renderer.set_device("GPU")
except Exception:
    print("[WARN] Could not force GPU device; continuing with default device.")


IMG_DIR = os.path.join(OUT_DIR, "images")
LBL_DIR = os.path.join(OUT_DIR, "labels")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

MASTER_CSV = os.path.join(OUT_DIR, "shot_log.csv")

def _append_csv_row(row_dict):
    write_header = not os.path.exists(MASTER_CSV)
    with open(MASTER_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "shot_idx","base","image","label","hdri",
            "cam_x","cam_y","cam_z",
            "bbox_x","bbox_y","bbox_w","bbox_h",
            "obj_y_deg",
        ])
        if write_header:
            w.writeheader()
        w.writerow(row_dict)

# ===================== LOAD SCENE =====================
if not os.path.isfile(BLEND):
    raise FileNotFoundError(f"Blend file not found: {BLEND}")
bproc.loader.load_blend(BLEND)

objs = bproc.object.get_all_mesh_objects()
anno_obj = next((o for o in objs if o.get_name() == OBJ_NAME_ANNOTATION), None)
base_obj = next((o for o in objs if o.get_name() == OBJ_NAME_OCCLUSION_BASE), None)
occl_obj = next((o for o in objs if o.get_name() == OBJ_NAME_OCCLUDER), None)

if anno_obj is None:
    raise RuntimeError(f"❌ Annotation target '{OBJ_NAME_ANNOTATION}' not found. Found: {[o.get_name() for o in objs]}")

# Assign segmentation IDs and force annotation object's ID
for idx, o in enumerate(objs):
    o.set_cp("category_id", idx + 1)


# check id for the target object for annnotations

anno_obj.set_cp("category_id", 4)
anno_seg_id = anno_obj.get_cp("category_id")

# Disable occluder if present
if occl_obj is not None:
    try:
        occl_obj.blender_obj.hide_render = True
        occl_obj.blender_obj.hide_viewport = True
    except Exception:
        pass
    try:
        occl_obj.set_location([1e6, 1e6, 1e6])
    except Exception:
        pass

# ===================== CAMERA/RENDER CONFIG =====================
def K_from_fov(fov_deg, width, height):
    fov = np.deg2rad(fov_deg)
    fx = fy = (width/2.0)/np.tan(fov/2.0)
    cx, cy = width/2.0, height/2.0
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)

K = K_from_fov(50, RES[0], RES[1])
bproc.camera.set_intrinsics_from_K_matrix(K, RES[0], RES[1])
bproc.renderer.set_output_format(file_format="JPEG", color_depth=8,
                                 enable_transparency=False, jpg_quality=90)
bproc.renderer.enable_segmentation_output(map_by="instance")

# Force GPU denoiser (OptiX if available)
bproc.renderer.set_denoiser("optix")

# ===================== HELPERS =====================
def save_rgb(rgb_np, path_jpg):
    arr = rgb_np
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    Image.fromarray(arr, mode="RGB").save(path_jpg, quality=90)

def coco_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w/2) / img_w
    y_center = (y + h/2) / img_h
    return [x_center, y_center, w/img_w, h/img_h]

def set_look_at_pose(cam_pos, look_at):
    forward = (look_at - cam_pos)
    rot = bproc.camera.rotation_from_forward_vec(forward)
    T = bproc.math.build_transformation_mat(cam_pos, rot)
    bproc.camera.add_camera_pose(T)
    return T



# ===================== ROTATION SEQUENCER (HRI_loadstation, Y-axis) =====================
# Ping-pong cycle: -10,-15,...,-90,-85,...,-10  → repeats forever
def _build_hrils_y_cycle():
    down = list(range(-10, -91, -5))   # -10 → -90 (inclusive)
    up   = list(range(-85, -9,  5))    # -85 → -10 (inclusive)
    return down + up

_HRILS_Y_CYCLE = _build_hrils_y_cycle()
_HRILS_Y_LEN   = len(_HRILS_Y_CYCLE)   # 32

def hrils_angle_for_k(k: int) -> float:
    """Return Y-rotation (degrees) for shot index k within an HDRI."""
    return float(_HRILS_Y_CYCLE[k % _HRILS_Y_LEN])







# ===================== GRID PREPARATION =====================
obj_center = np.array(anno_obj.get_location(), dtype=float)

grid_x = np.arange(WORLD_X_RANGE[0], WORLD_X_RANGE[1] + 1e-6, STEP_X)
grid_y = np.arange(WORLD_Y_RANGE[0], WORLD_Y_RANGE[1] + 1e-6, STEP_Y)
grid_z = np.arange(WORLD_Z_RANGE[0], WORLD_Z_RANGE[1] + 1e-6, STEP_Z)

hdris = sorted(glob.glob(os.path.join(HDRI_DIR, "*.hdr")) +
               glob.glob(os.path.join(HDRI_DIR, "*.exr")))
if not hdris:
    raise RuntimeError(f"No HDRIs found in: {HDRI_DIR}")

# ===================== RESUMPTION LOGIC (PER HDRI) =====================
import pandas as pd

progress_per_hdri = {}
existing_rows = 0
global_start_index = 0
if os.path.exists(MASTER_CSV):
    try:
        df = pd.read_csv(MASTER_CSV)
        # Count how many shots per HDRI are already finished
        progress_per_hdri = df.groupby("hdri").size().to_dict()

        existing_rows = len(df)
        # continue from max shot_idx + 1 (not row count!)
        if "shot_idx" in df.columns and pd.api.types.is_numeric_dtype(df["shot_idx"]):
            global_start_index = int(df["shot_idx"].max()) + 1
        else:
            global_start_index = 0



        # existing_rows = len(df)
        # # continue global numbering from max shot_idx + 1
        # if "shot_idx" in df.columns and pd.api.types.is_numeric_dtype(df["shot_idx"]):
        #     global_start_index = int(df["shot_idx"].max()) + 1
        # else:
        #     global_start_index = existing_rows
    except Exception as e:
        print(f"[WARN] Could not parse CSV for resumption: {e}")
        progress_per_hdri = {}
        existing_rows = 0
        global_start_index = 0

# Global shot index continues growing across HDRIs
shot_index = global_start_index

# ===================== PRINT SUMMARY =====================
total_per_hdri = len(grid_x) * len(grid_y) * len(grid_z)
total_shots = total_per_hdri * len(hdris)
to_render = max(0, total_shots - existing_rows)

print("\n========== RENDER JOB SUMMARY (NO OCCLUSION) ==========")
print(f"Blend file : {BLEND}")
print(f"HDRI dir   : {HDRI_DIR}  |  #HDRIs: {len(hdris)}")
print(f"Output dir : {OUT_DIR}")
print(f"Resolution : {RES[0]}x{RES[1]}  |  JPEG quality: 90")
print(f"Camera grid: X={len(grid_x)} × Y={len(grid_y)} × Z={len(grid_z)} = {total_per_hdri} shots per HDRI")
print(f"Total shots: {total_shots}")
print(f"Resume     : found {existing_rows} completed shots in CSV → will render {to_render} shots")
print("\n[DEBUG] Objects & SegIDs:")
for idx, o in enumerate(objs):
    sid = o.get_cp('category_id') if 'category_id' in o.blender_obj.keys() else None
    print(f"  idx={idx:02d}, SegID={sid}, Name='{o.get_name()}'")
print(f"\n[INFO] Annotation target '{OBJ_NAME_ANNOTATION}' → SegID={anno_seg_id}")
print("[INFO] Occluder (if present) has been disabled / moved far away.")
print("=======================================================\n")

# ===================== THREAD POOL / SAFE EXIT =====================
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
pending = 0  # shots since last flush

def flush_executor():
    """Wait for all queued writes to finish, then reopen a fresh pool."""
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

# ===================== MAIN LOOP (MINI-BATCHED) =====================
for h_idx, hdri in enumerate(hdris, 1):
    try:
        # Load HDRI once per outer loop
        bproc.world.set_world_background_hdr_img(hdri, strength=1.0)
    except Exception as e:
        print(f"[WARN] Failed to load HDRI: {hdri} -> {e}")
        continue

    # === Resume handling for this HDRI ===
    done_for_hdri = progress_per_hdri.get(os.path.basename(hdri), 0)
    print(f"[RESUME] HDRI {hdri} already has {done_for_hdri} shots, skipping those...")

    shots_done = 0  # counter only for this HDRI

    if done_for_hdri >= total_per_hdri:
        print(f"[RESUME] HDRI {os.path.basename(hdri)} already complete ({done_for_hdri}/{total_per_hdri}). Skipping.")
        continue

    # --- Setup mapping node for rotation (only once per HDRI) ---
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

    # start fresh set of camera poses for this HDRI
    bproc.utility.reset_keyframes()

    poses = []
    cam_positions = []



    shot_angles = []          # parallel list to cam_positions, one angle per shot in the current batch
    shots_emitted = 0         # counts how many NEW shots we’re emitting for THIS HDRI (respects resume)





    # def render_and_dispatch_batch():
    #     """Render the current mini-batch and queue all post-processing asynchronously."""
    #     global shot_index, pending
    #     if not poses:
    #         return



    #     # Render current batch
    #     data = bproc.renderer.render()


    def render_and_dispatch_batch():
        """Render the current mini-batch and queue all post-processing asynchronously."""
        global shot_index, pending
        if not poses:
            return
        # === NEW: keyframe Y-rotation for HRI_loadstation for each frame in this batch ===
        # (one keyframe per index in shot_angles)
        obj = bpy.data.objects.get("HRI_loadstation")
        if obj is not None and shot_angles:
            scn = bpy.context.scene
            for fidx, ang_deg in enumerate(shot_angles):
                scn.frame_set(fidx)                        # frame index for this batch
                obj.rotation_euler[1] = math.radians(ang_deg)  # Y-axis
                obj.keyframe_insert(data_path="rotation_euler", index=1)

        # Render current batch
        data = bproc.renderer.render()


        # For the next batch, clear camera keyframes so we don't re-render old poses
        bproc.utility.reset_keyframes()

        # Queue saves (image + label + CSV) for each shot in this batch
        num_frames = min(len(cam_positions), len(data["colors"]))
        for i in range(num_frames):
            cam_pos = cam_positions[i]
            rgb_np  = data["colors"][i].copy()
            seg     = data["instance_segmaps"][i].copy()

            # Freeze name/paths/indices per job
            base_name = f"HDRI_{h_idx:03d}_VIEW_{shot_index:05d}"
            jpg_path  = os.path.join(IMG_DIR, base_name + ".jpg")
            txt_path  = os.path.join(LBL_DIR, base_name + ".txt")
            shot_i    = shot_index
            cam_i     = cam_pos.copy()
            hdri_name = os.path.basename(hdri)

            angle_deg = float(shot_angles[i])   # ← NEW: rotation for this frame


            def save_outputs(rgb_np=rgb_np, seg=seg, cam_pos=cam_i,
                             hdri_name=hdri_name, shot_idx=shot_i,
                             base_name=base_name, jpg_path=jpg_path, txt_path=txt_path,
                              angle_deg=angle_deg):
                # Extract bbox
                mask = (seg == anno_seg_id).astype(np.uint8)
                bboxes = []
                bbox_x = bbox_y = bbox_w = bbox_h = None
                if np.any(mask):
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                    if num_labels > 1:
                        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                        x_min, y_min, w_box, h_box, area = stats[largest_idx]
                        if w_box > 0 and h_box > 0:
                            bboxes.append({"class_id": 0, "bbox": (x_min, y_min, w_box, h_box)})
                            bbox_x, bbox_y, bbox_w, bbox_h = x_min, y_min, w_box, h_box

                # Save files
                save_rgb(rgb_np, jpg_path)
                with open(txt_path, "w") as f:
                    for bbox in bboxes:
                        yolo_box = coco_to_yolo(bbox["bbox"], RES[0], RES[1])
                        f.write("0 " + " ".join(f"{v:.6f}" for v in yolo_box) + "\n")

                # Log CSV
                row = {
                    "shot_idx": shot_idx,
                    "base": base_name,
                    "image": jpg_path,
                    "label": txt_path,
                    "hdri": hdri_name,
                    "cam_x": float(cam_pos[0]),
                    "cam_y": float(cam_pos[1]),
                    "cam_z": float(cam_pos[2]),
                    "bbox_x": bbox_x, "bbox_y": bbox_y, "bbox_w": bbox_w, "bbox_h": bbox_h,
                    "obj_y_deg": angle_deg,    # ← NEW
                }
                _append_csv_row(row)

                print(f"[SHOT] {base_name}: Cam=({cam_pos[0]:.2f},{cam_pos[1]:.2f},{cam_pos[2]:.2f})"
                      f"  |  BBox={'None' if bbox_w is None else (bbox_x, bbox_y, bbox_w, bbox_h)}")

            executor.submit(save_outputs)

            # Advance to next shot index after scheduling this one
            shot_index += 1
            pending    += 1

            # Periodic flush for safety
            if pending >= FLUSH_INTERVAL:
                print(f"[INFO] Flushing after {pending} shots...")
                flush_executor()

        # ✅ Clear batch buffers so next batch starts fresh
        poses.clear()
        cam_positions.clear()
        shot_angles.clear()


    # Sweep grid; build and render mini-batches
    for z in grid_z:
        for y in grid_y:
            for x in grid_x:
                # Skip past already completed shots for THIS HDRI
                if shots_done < done_for_hdri:
                    shots_done += 1
                    # shot_index += 1   # still advance global index
                    continue

                # Optional: random HDRI rotation per shot (mapping node only)
                if HDRI_ROTATE_PER_SHOT:
                    theta = float(np.random.uniform(0.0, 2.0*np.pi))
                    mapping.inputs['Rotation'].default_value = (0.0, 0.0, theta)

                cam_pos = np.array([x, y, z], dtype=float)
                set_look_at_pose(cam_pos, obj_center)
                poses.append(True)              # marker; BlenderProc tracks pose internally
                cam_positions.append(cam_pos.copy())



                # === NEW: record per-shot Y angle for HRI_loadstation (ping-pong sequence) ===
                # k = (already-finished shots for this HDRI) + (new shots we have emitted so far)
                k = done_for_hdri + shots_emitted
                shot_angles.append(hrils_angle_for_k(k))
                shots_emitted += 1








                # Render when we hit batch size
                if len(cam_positions) >= BATCH_SIZE:
                    render_and_dispatch_batch()

    # Render any leftover poses at the end of this HDRI
    render_and_dispatch_batch()

# Final flush to ensure everything is written
flush_executor()

# ===================== DONE =====================
print("\n✅ Done. (NO OCCLUSION)")
print(f"Images  → {IMG_DIR}")
print(f"Labels  → {LBL_DIR}")
print(f"CSV log → {MASTER_CSV}")
