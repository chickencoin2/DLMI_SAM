import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os
import threading
import numpy as np

import torch
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import logging
import contextlib
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
from util.gui_view import AppView
from util.customutil import (
    rgb_to_tkinter_hex, get_bbox_from_mask,
    calculate_iou, get_stabilized_bbox, process_sam_mask, draw_star_marker,
    get_hashable_obj_id, is_bbox_on_edge, compute_dlmi_logits
)
from util import sam_interaction
from util import autolabel_workflow
from util import propagation_controller
from util import input_handlers
from util import batch_controller

try:
    from transformers import (
        Sam3VideoModel, Sam3VideoProcessor,
        Sam3TrackerVideoModel, Sam3TrackerVideoProcessor,
        Sam3Model, Sam3Processor
    )
    SAM3_AVAILABLE = True
except ImportError as e:
    print(f"ImportError: {e}")
    print("SAM3 library (transformers) not found.")
    print("Run: pip install transformers accelerate")
    SAM3_AVAILABLE = False

try:
    from transformers import Sam2Model, Sam2Processor
    SAM2_AVAILABLE = True
except ImportError as e:
    print(f"SAM2 ImportError: {e}")
    print("SAM2 library unavailable - option disabled")
    SAM2_AVAILABLE = False

SAM3_MODEL_ID = "facebook/sam3"
DEFAULT_MIN_BBOX_AREA_FOR_REPROMPT = 50
DEFAULT_EROSION_KERNEL_SIZE = 3
DEFAULT_EROSION_ITERATIONS = 1
AUTOLABEL_FOLDER = "autolabel_sam"
LABELME_VERSION = "5.10.1"
ALPHA_NORMAL = 153; ALPHA_SELECTED = 220
ALPHA_PROBLEM_HIGHLIGHT = 100; ALPHA_CORRECTION_MODE = 70
DEFAULT_SAM_CLOSING_KERNEL_SIZE = 5
DEFAULT_EDGE_MARGIN = 10

logger = logging.getLogger("DLMI_SAM_LABELER")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class SAM3AutolabelApp:

    def __init__(self, root_window):
        self.root = root_window
        self.root.title("DLMI-SAM labeler v1.1_test")

        self.video_source_path = None
        self.cap = None
        self.video_frames_cache = []

        self.pcs_model = None
        self.pcs_processor = None
        self.pcs_inference_session = None
        self.pcs_streaming_session = None
        self.tracker_model = None
        self.tracker_processor = None
        self.inference_session = None
        self.model_dtype = torch.float32
        self.image_model = None
        self.image_processor = None

        self.app_state = "IDLE"
        self.prompt_mode_var = tk.StringVar(value="PVS")
        self._previous_prompt_mode = "PVS"  

        self.propagated_results = {}
        self.propagation_segments = []
        self.propagation_progress = 0
        self.propagation_stop_requested = False
        self.propagation_paused = False
        self.propagation_pause_event = threading.Event()
        self.propagation_pause_event.set()  # Not paused initially
        self.propagation_current_frame_idx = 0
        self.dlmi_pending_injection = False
        self.dlmi_pending_masks = {}
        self.dlmi_hook_active = False
        self.cut_point_frame = None
        self.cut_start_frame = 0

        self.review_current_frame = 0
        self.is_reviewing = False
        self.discarded_frames = set()

        self.chunk_error_threshold_var = tk.DoubleVar(value=0.15)
        self.chunk_temp_save_dir = None
        self.chunk_processing = False

        self.tracked_objects = {}
        self.next_obj_id_to_propose = 1
        self.playback_paused = True
        self.autolabel_active = False
        self.current_cv_frame = None
        self.current_frame_pil_rgb_original = None
        self.current_frame_idx_conceptual = 0
        self.processing_thread = None
        self.object_colors = {}
        self.is_tracking_ever_started = False
        self.last_active_tracked_sam_ids = set()
        self.selected_object_sam_id = None
        self.selected_objects_sam_ids = set()  
        self.object_groups = {} 
        self.sam_id_to_group = {}  
        self.next_group_id = 1  
        self.is_ctrl_pressed = False
        self.is_shift_pressed = False
        self.is_alt_pressed = False  
        self.bbox_start_canvas_coords = None
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.just_reset_sam = False
        self.sam_operation_in_progress = False
        self._tracking_fatal_error = None
        self._resize_job_id = None

        self.AUTOLABEL_FOLDER_val = AUTOLABEL_FOLDER
        self.LABELME_VERSION_val = LABELME_VERSION

        self.suppressed_sam_ids = set()

        self.pcs_text_prompt_var = tk.StringVar(value="")
        self.pcs_detection_threshold_var = tk.DoubleVar(value=0.5)
        self.pcs_mask_threshold_var = tk.DoubleVar(value=0.5)
        self.pcs_exemplar_boxes = []
        self.pcs_exemplar_labels = []

        self.erosion_kernel_size = tk.IntVar(value=DEFAULT_EROSION_KERNEL_SIZE)
        self.erosion_iterations = tk.IntVar(value=DEFAULT_EROSION_ITERATIONS)
        self.min_bbox_area_for_reprompt = tk.IntVar(value=DEFAULT_MIN_BBOX_AREA_FOR_REPROMPT)
        self.default_object_label_var = tk.StringVar(value="object")
        self.labeling_mode_var = tk.StringVar(value="Instance")
        self.overwrite_policy = "rename"
        self.sam_apply_closing_var = tk.BooleanVar(value=False)
        self.sam_closing_kernel_size_var = tk.IntVar(value=DEFAULT_SAM_CLOSING_KERNEL_SIZE)

        self.problematic_objects_flagged = {}
        self.interaction_correction_pending = None
        self.problematic_highlight_active_sam_id = None
        self.reassign_bbox_mode_active_sam_id = None

        self.polygon_mode_active = False
        self.polygon_points = []
        self.polygon_objects = []

        self.ignore_edge_labels_var = tk.BooleanVar(value=False)
        self.edge_margin_var = tk.IntVar(value=DEFAULT_EDGE_MARGIN)

        self.new_object_method_var = tk.StringVar(value="reset")

        self.mid_new_object_method_var = tk.StringVar(value="off")
        self._suppress_mid_session_guard = False

        self.use_custom_save_path_var = tk.BooleanVar(value=False)
        self.custom_save_dir_var = tk.StringVar(value=os.path.join(os.getcwd(), AUTOLABEL_FOLDER, "custom_output"))
        self.custom_folder_name_var = tk.StringVar(value="{video_name}_dataset")
        self.custom_file_name_var = tk.StringVar(value="{video_name}_frame")

        self.batch_processing_mode_var = tk.BooleanVar(value=False)
        self.batch_source_dir_var = tk.StringVar()
        self.batch_video_files = []
        self.batch_current_index = -1
        self.batch_save_option_var = tk.StringVar(value="subfolder")
        self.batch_filename_option_var = tk.StringVar(value="video_name")
        self.is_batch_running = False
        self.is_batch_video_finished = False

        self.video_display_name = ""
        self.video_total_frames = 0
        self.video_fps = 0
        self.video_resolution = ""
        self.info_video_name_var = tk.StringVar(value="N/A")
        self.info_video_resolution_var = tk.StringVar(value="N/A")
        self.info_video_total_frames_var = tk.StringVar(value="N/A")
        self.info_video_fps_var = tk.StringVar(value="N/A")
        self.info_batch_progress_var = tk.StringVar(value="N/A")

        self.save_format_var = tk.StringVar(value="labelme")
        self.yolo_class_names_for_save = []
        self.yolo_nc = 0
        self.yolo_dataset_initialized = False

        self.batch_move_completed_var = tk.BooleanVar(value=False)
        self.batch_completed_dir_var = tk.StringVar(value=os.path.join(os.getcwd(), "completed_videos"))

        self.allow_image_source_var = tk.BooleanVar(value=False)
        self.is_image_source = False

        self.mask_alpha_var = tk.IntVar(value=153)

        self.filter_small_objects_var = tk.BooleanVar(value=False)
        self.small_object_threshold_var = tk.DoubleVar(value=0.001)

        self.filter_small_contours_var = tk.BooleanVar(value=False)
        self.small_contour_threshold_var = tk.DoubleVar(value=0.0001)
        self.small_contour_base_var = tk.StringVar(value="image")


        self.low_level_api_enabled_var = tk.BooleanVar(value=False)
        self.dlmi_alpha_var = tk.DoubleVar(value=10.0)
        self.dlmi_boundary_mode_var = tk.StringVar(value="Fixed")
        self.dlmi_gradient_falloff_var = tk.IntVar(value=20)
        self.dlmi_preserve_memory_var = tk.BooleanVar(value=False)
        self.dlmi_boost_cond_var = tk.BooleanVar(value=False)

        self.sam2_enabled_var = tk.BooleanVar(value=False)
        self.sam2_tracking_enabled_var = tk.BooleanVar(value=False)
        self.sam2_model = None
        self.sam2_processor = None
        self.sam2_model_id = "facebook/sam2.1-hiera-large"
        self.sam2_masks = {}
        self.sam2_prompt_points = {}
        self.sam2_image_embeddings = {}
        self.sam2_loading_in_progress = False

        self.label_font_size_percent_var = tk.DoubleVar(value=0.7)
        self.polygon_point_size_percent_var = tk.DoubleVar(value=0.4)
        self.show_object_border_var = tk.BooleanVar(value=False)
        self.tabs_visible_var = tk.BooleanVar(value=True)
        self.show_prompt_visualization_var = tk.BooleanVar(value=False)
        self.show_prompt_per_object_var = tk.BooleanVar(value=False)
        self.object_prompt_history = {}

        try:
            from util import pose_ui as _pose_ui
            self._pose_ui = _pose_ui
        except Exception as _pose_ui_err:
            logger.warning(f"pose_ui module load failed: {_pose_ui_err}")
            self._pose_ui = None

        app_dir = os.path.dirname(os.path.abspath(__file__))
        self.pose_config_path = os.path.join(app_dir, "pose_config.json")
        if self._pose_ui is not None:
            self.pose_config = self._pose_ui.load_pose_config(self.pose_config_path)
        else:
            self.pose_config = {}
        self.pose_tapnext_enabled_var = tk.BooleanVar(value=False)
        self.pose_add_mode_var = tk.BooleanVar(value=False)
        self.pose_chain_mode_var = tk.BooleanVar(value=False)
        self.pose_automatch_var = tk.BooleanVar(value=False)
        self.selected_pose_points = set()
        self.pose_tracker = None
        self.yolo_pose_detector = None

        try: self.label_font = ImageFont.truetype("arial.ttf", 15)
        except IOError: self.label_font = ImageFont.load_default()

        if torch.cuda.is_available(): self.device = torch.device("cuda")
        elif torch.backends.mps.is_available(): self.device = torch.device("mps"); logger.info("Using MPS.")
        else: self.device = torch.device("cpu")
        logger.info(f"Device: {self.device}")

        self.autocast_context = torch.autocast("cuda", dtype=torch.float32) if self.device.type == "cuda" else contextlib.nullcontext()
        
        self.sam_interaction_module = sam_interaction
        self.autolabel_workflow_module = autolabel_workflow
        
        self.view = AppView(self.root, self)

        self._suppress_pose_class_trace = False
        self.root.after(200, self._refresh_pose_class_menu)

        self.root.after(100, self._init_models)
        logger.info("SAM3AutolabelApp initialized.")

    def _canvas_to_image_coords(self, canvas_x, canvas_y):
        if not self.current_frame_pil_rgb_original:
            logger.warning("_canvas_to_image_coords: current_frame_pil_rgb_original is None")
            return canvas_x, canvas_y

        if self.scale_x == 0 or self.scale_y == 0:
            logger.warning("_canvas_to_image_coords: scale_x/y not set or zero")
            return canvas_x, canvas_y

        # Recompute canvas size drift between last display_image call and now.
        # Window resize, scroll-bar layout changes, etc. can make the cached
        # scale_x/offset_x stale relative to the current canvas geometry; if
        # the canvas is now visibly different from what display_image saw, we
        # refresh by asking the view to re-render which will update these
        # values before we use them.
        try:
            canvas_w = self.view.canvas.winfo_width()
            canvas_h = self.view.canvas.winfo_height()
            orig_w, orig_h = self.current_frame_pil_rgb_original.size
            expected_display_w = orig_w / self.scale_x
            expected_display_h = orig_h / self.scale_y
            expected_offset_x = (canvas_w - expected_display_w) // 2
            expected_offset_y = (canvas_h - expected_display_h) // 2
            drift = (abs(expected_offset_x - self.offset_x) > 2 or
                     abs(expected_offset_y - self.offset_y) > 2)
            if drift and self.current_cv_frame is not None:
                logger.info(
                    f"_canvas_to_image_coords drift detected: "
                    f"canvas={canvas_w}x{canvas_h} cached offset=({self.offset_x},{self.offset_y}) "
                    f"expected=({expected_offset_x},{expected_offset_y}); refreshing display."
                )
                self._display_cv_frame_on_view(
                    self.current_cv_frame, self._get_current_masks_for_display()
                )
        except (tk.TclError, AttributeError, ZeroDivisionError):
            pass

        orig_w, orig_h = self.current_frame_pil_rgb_original.size
        img_x = (canvas_x - self.offset_x) * self.scale_x
        img_y = (canvas_y - self.offset_y) * self.scale_y
        result = (int(max(0, min(img_x, orig_w - 1))),
                  int(max(0, min(img_y, orig_h - 1))))
        logger.debug(
            f"click canvas=({canvas_x},{canvas_y}) "
            f"scale=({self.scale_x:.3f},{self.scale_y:.3f}) "
            f"offset=({self.offset_x},{self.offset_y}) -> img={result}"
        )
        return result

    def _handle_sam_prompt_wrapper(self, prompt_type, coords, label=None,
                                  proposed_obj_id_for_new=None, target_existing_obj_id=None,
                                  custom_label=None):
        if self.app_state == "PAUSED":
            self.update_status("Paused: BBox/Point prompts blocked (session protection). Use polygon mode + DLMI injection instead.")
            return

        is_new_object = (proposed_obj_id_for_new is not None) and (target_existing_obj_id is None)
        has_existing_state = bool(self.tracked_objects) or bool(self.propagated_results)
        is_mid_session = (self.app_state == "REVIEWING") or bool(self.is_tracking_ever_started)
        if is_new_object and has_existing_state and is_mid_session and not self._suppress_mid_session_guard:
            method = self.mid_new_object_method_var.get() if hasattr(self, 'mid_new_object_method_var') else "off"
            if method == "off":
                messagebox.showwarning(
                    "New Object Blocked",
                    "Existing tracked/propagated labels remain.\n\n"
                    "Adding a new object now would reset the session and lose anchor on the current frame.\n\n"
                    "In the Propagate tab, set 'Mid-session new object' to 'DLMI inject' or 'Load labels' to preserve existing labels and continue from this frame.",
                    parent=self.root
                )
                self.update_status("New object blocked: mid-session option is Off.")
                return
            elif method in ("dlmi", "load"):
                self._mid_session_macro_add(
                    prompt_type, coords, label,
                    proposed_obj_id_for_new, custom_label,
                    use_load=(method == "load"),
                )
                return

        if self.sam2_enabled_var.get() and self.sam2_model is not None:
            self._handle_sam2_prompt(prompt_type, coords, label,
                                    proposed_obj_id_for_new, target_existing_obj_id,
                                    custom_label)
        else:
            self.sam_interaction_module.handle_sam_prompt(
                self, prompt_type, coords, label,
                proposed_obj_id_for_new, target_existing_obj_id,
                custom_label
            )

    def _mid_session_snapshot_precise_masks(self):
        """Collect exact masks for existing objects at the current frame, so
        we can restore them after running SAM3's standard new-object flow
        (which would otherwise degrade existing masks via center-point
        reprompting)."""
        snap = {}
        for oid, data in self.tracked_objects.items():
            mask = data.get('last_mask')
            if mask is None or not mask.any():
                continue
            snap[oid] = {
                'last_mask': mask.copy(),
                'custom_label': data.get('custom_label'),
            }
        return snap

    def _mid_session_polygon_roundtrip(self, snapshot):
        """Emulate 'save labels then reload' by converting each preserved mask
        into a polygon and back into a filled mask. This loses sub-polygon
        detail (which mirrors what the actual Load Labels flow would produce)
        but keeps obj_ids intact."""
        converted = {}
        for oid, data in snapshot.items():
            mask = data['last_mask']
            mask_u8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            epsilon = 0.002 * cv2.arcLength(largest, True)
            poly = cv2.approxPolyDP(largest, epsilon, True).reshape(-1, 2)
            if len(poly) < 3:
                continue
            h, w = mask.shape[:2]
            rebuilt = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(rebuilt, [poly.astype(np.int32).reshape(-1, 1, 2)], 255)
            rebuilt_bool = rebuilt > 0
            if not rebuilt_bool.any():
                continue
            converted[oid] = {
                'last_mask': rebuilt_bool,
                'custom_label': data.get('custom_label'),
            }
        return converted

    def _mid_session_macro_add(self, prompt_type, coords, label,
                               proposed_obj_id_for_new, custom_label,
                               use_load=False):
        """One-click macro for mid-session new-object addition.

        Steps:
          1. Snapshot exact masks of all existing objects at the current frame.
          2. Clear tracked_objects and the inference session.
          3. Apply the new bbox/point as a FRESH first-prompt on the current
             frame. SAM3 produces a clean mask for the new object because the
             session is brand new (no DLMI memory interference).
          4. Merge the snapshotted existing masks back into tracked_objects.
          5. Re-seed the session via DLMI (input_masks) using all masks,
             exactly preserving their shape. For `use_load=True`, each mask is
             first polygon-roundtripped to mimic the labelme file path.

        The result: user sees their existing labels + the new object, session
        is primed, and Start Propagate continues from here."""
        if self.current_cv_frame is None:
            messagebox.showerror("Error", "No current frame.", parent=self.root)
            return
        snapshot = self._mid_session_snapshot_precise_masks()
        if not snapshot:
            messagebox.showwarning("Notice", "No valid existing masks to preserve.", parent=self.root)
            return

        if use_load:
            snapshot = self._mid_session_polygon_roundtrip(snapshot)
            if not snapshot:
                messagebox.showwarning("Notice", "Polygon roundtrip lost all masks.", parent=self.root)
                return

        new_id = proposed_obj_id_for_new
        if new_id is None or new_id in snapshot:
            new_id = max(snapshot.keys()) + 1
        self.next_obj_id_to_propose = max(self.next_obj_id_to_propose, new_id + 1)

        self.tracked_objects.clear()
        self.object_groups.clear()
        self.sam_id_to_group.clear()
        self.next_group_id = 1
        self.suppressed_sam_ids.clear()
        if hasattr(self, 'object_prompt_history'):
            self.object_prompt_history.clear()
        self.is_tracking_ever_started = False
        self.inference_session = None

        self._suppress_mid_session_guard = True
        try:
            if self.sam2_enabled_var.get() and self.sam2_model is not None:
                self._handle_sam2_prompt(
                    prompt_type, coords, label,
                    new_id, None, custom_label
                )
            else:
                self.sam_interaction_module.handle_sam_prompt(
                    self, prompt_type, coords, label,
                    new_id, None, custom_label
                )
        finally:
            self._suppress_mid_session_guard = False

        new_obj_mask = None
        if new_id in self.tracked_objects:
            new_obj_mask = self.tracked_objects[new_id].get('last_mask')

        for oid, data in snapshot.items():
            if oid == new_id:
                continue
            self.tracked_objects[oid] = {
                'custom_label': data.get('custom_label'),
                'last_mask': data['last_mask'],
                'mid_session_added': False,
            }

        if new_id in self.tracked_objects:
            self.tracked_objects[new_id]['mid_session_added'] = True

        if new_obj_mask is None or not new_obj_mask.any():
            logger.warning(
                f"Mid-session add: new object {new_id} produced an empty mask from the SAM3 prompt. "
                f"Existing labels preserved."
            )

        try:
            self.inject_low_level_mask_prompt(force=True)
            if new_id in self.tracked_objects:
                self.tracked_objects[new_id]['mid_session_added'] = True
        except Exception as e:
            logger.exception(f"Mid-session DLMI re-seed failed: {e}")

        if self.current_cv_frame is not None:
            self._display_cv_frame_on_view(self.current_cv_frame, self._get_current_masks_for_display())

        valid_count = sum(
            1 for data in self.tracked_objects.values()
            if data.get('last_mask') is not None and data['last_mask'].any()
        )
        mode_name = "Load" if use_load else "DLMI"
        logger.info(
            f"Mid-session {mode_name} macro complete. "
            f"tracked_objects={sorted(self.tracked_objects.keys())} "
            f"({valid_count} with valid mask). new_id={new_id}."
        )
        self.update_status(
            f"Mid-session {mode_name}: added object {new_id}, {len(snapshot)} existing preserved."
        )
    
    def update_status(self, message): 
        if self.view: self.view.update_status(message)

    def _update_obj_id_info_label(self): 
        if self.view: self.view.update_obj_id_info_label()

    def _init_models(self):
        if not SAM3_AVAILABLE:
            self.update_status("SAM3 library is not installed.")
            return

        self._init_sam3_models()
        if self.tracker_model is not None:
            self.update_status("SAM3 model loaded. Source selection available.")
            self.view.set_ui_element_state("btn_select_source", tk.NORMAL)
        else:
            self.update_status("SAM3 model load failed.")

    def _init_sam3_models(self):
        self.update_status("Loading SAM3 models... (May be downloading from HuggingFace)")
        try:
            model_dtype = torch.float32
            self.model_dtype = model_dtype

            logger.info(f"Loading SAM3 PCS Video model: {SAM3_MODEL_ID}")
            self.pcs_model = Sam3VideoModel.from_pretrained(
                SAM3_MODEL_ID, torch_dtype=model_dtype
            ).to(self.device).eval()
            self.pcs_processor = Sam3VideoProcessor.from_pretrained(SAM3_MODEL_ID)
            logger.info("SAM3 PCS Video model loaded")

            logger.info(f"Loading SAM3 Tracker model: {SAM3_MODEL_ID}")
            self.tracker_model = Sam3TrackerVideoModel.from_pretrained(
                SAM3_MODEL_ID, torch_dtype=model_dtype
            ).to(self.device).eval()
            self.tracker_processor = Sam3TrackerVideoProcessor.from_pretrained(SAM3_MODEL_ID)
            logger.info("SAM3 Tracker model loaded")

            logger.info(f"Loading SAM3 Image model: {SAM3_MODEL_ID}")
            self.image_model = Sam3Model.from_pretrained(
                SAM3_MODEL_ID, torch_dtype=model_dtype
            ).to(self.device).eval()
            self.image_processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
            logger.info("SAM3 Image model loaded")

            self.update_status("SAM3 models loaded.")
            self.view.set_ui_element_state("btn_clear_tracked", tk.NORMAL)
            self.view.set_ui_element_state("btn_set_custom_label", tk.NORMAL)

        except Exception as e:
            logger.exception("SAM3 model load failed:")
            self.update_status(f"SAM3 load error: {e}")
            self.pcs_model = None
            self.pcs_processor = None
            self.tracker_model = None
            self.tracker_processor = None
            self.view.set_ui_element_state("btn_set_custom_label", tk.DISABLED)

    def _init_inference_session(self, for_pcs_mode=False):
        self._reset_group_and_polygon_state()

        model_dtype = torch.float32

        if for_pcs_mode:
            if self.pcs_processor is None:
                logger.error("PCS processor is not initialized.")
                return False
            try:
                if not self.video_frames_cache:
                    logger.error("Video frame cache required for PCS mode.")
                    return False

                self.pcs_inference_session = self.pcs_processor.init_video_session(
                    video=self.video_frames_cache,
                    inference_device=self.device,
                    processing_device="cpu",
                    video_storage_device="cpu",
                    dtype=model_dtype,
                )
                logger.info("SAM3 PCS inference session initialized")
                return True
            except Exception as e:
                logger.exception("SAM3 PCS inference session init failed:")
                self.pcs_inference_session = None
                self.pcs_streaming_session = None
                return False
        else:
            if self.tracker_processor is None:
                logger.error("Tracker processor is not initialized.")
                return False
            try:
                self.inference_session = self.tracker_processor.init_video_session(
                    inference_device=self.device,
                    processing_device="cpu",
                    video_storage_device="cpu",
                    dtype=model_dtype,
                )
                logger.info("SAM3 Tracker inference session initialized (streaming mode)")
                return True
            except Exception as e:
                logger.exception("SAM3 Tracker inference session init failed:")
                self.inference_session = None
                return False

    def _reset_inference_session(self):
        self._remove_dlmi_persistent_hooks()
        if self.inference_session is not None:
            try:
                self.inference_session.reset_inference_session()
                logger.info("SAM3 inference session reset complete")
            except Exception as e:
                logger.warning(f"SAM3 inference session reset error: {e}")
        return self._init_inference_session()

    @property
    def predictor(self):
        return self.tracker_model

    @predictor.setter
    def predictor(self, value):
        self.tracker_model = value

    @property
    def is_predictor_loaded_first_frame(self):
        return self.inference_session is not None

    @is_predictor_loaded_first_frame.setter
    def is_predictor_loaded_first_frame(self, value):
        pass

    def _on_labeling_mode_change(self):
        logger.info(f"Labeling mode changed: {self.labeling_mode_var.get()}")

    def _set_custom_label_for_selected(self):
        if self._is_any_special_mode_active():
             messagebox.showwarning("Notice", "Auto-correction/reassignment interaction in progress. Label assignment only available when paused.", parent=self.root)
             return
        if not self.playback_paused: messagebox.showwarning("Notice", "Label assignment only available when paused.", parent=self.root); return
        if self.selected_object_sam_id is None: messagebox.showinfo("Notice", "Select an object first (Ctrl+Left click).", parent=self.root); return

        current_data = self.tracked_objects.get(self.selected_object_sam_id, {}); current_custom_label = current_data.get("custom_label", "")

        if self.save_format_var.get() in ["yolo", "both"] and self.yolo_class_names_for_save:
            from tkinter import ttk
            dialog = tk.Toplevel(self.root)
            dialog.title("Select Object Class")
            dialog.geometry("300x150")
            dialog.transient(self.root)
            dialog.grab_set()

            tk.Label(dialog, text=f"Select class for object ID {self.selected_object_sam_id}:").pack(pady=10)

            class_var = tk.StringVar(value=current_custom_label if current_custom_label in self.yolo_class_names_for_save else self.yolo_class_names_for_save[0])
            combo = ttk.Combobox(dialog, textvariable=class_var, values=self.yolo_class_names_for_save, state="readonly")
            combo.pack(pady=5)

            result = {"label": None}

            def on_ok():
                result["label"] = class_var.get()
                dialog.destroy()

            def on_cancel():
                dialog.destroy()

            tk.Button(dialog, text="OK", command=on_ok).pack(side=tk.LEFT, padx=20, pady=10)
            tk.Button(dialog, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=20, pady=10)

            dialog.wait_window()
            new_label = result["label"]
        else:
            new_label = simpledialog.askstring("Set Object Label", f"Enter name for object ID {self.selected_object_sam_id}:", initialvalue=current_custom_label, parent=self.root)

        if new_label is not None:
            if self.save_format_var.get() in ["yolo", "both"] and self.yolo_class_names_for_save:
                if new_label not in self.yolo_class_names_for_save:
                    response = messagebox.askyesnocancel(
                        "Class Mismatch",
                        f"Label '{new_label}' is not in the registered class list.\n\n"
                        f"Yes: Add as new class\n"
                        f"No: Re-enter label\n"
                        f"Cancel: Abort operation",
                        parent=self.root
                    )
                    if response is None:
                        return
                    elif response:
                        self.yolo_class_names_for_save.append(new_label)
                        self.yolo_nc = len(self.yolo_class_names_for_save)
                        self._update_yolo_yaml()
                        logger.info(f"New class '{new_label}' added. Total {self.yolo_nc} classes")
                    else:
                        return self._set_custom_label_for_selected()

            if self.selected_object_sam_id in self.tracked_objects:
                self.tracked_objects[self.selected_object_sam_id]["custom_label"] = new_label
                logger.info(f"Object {self.selected_object_sam_id} label set to '{new_label}'."); self.update_status(f"Object {self.selected_object_sam_id} label: '{new_label}'")
                if self.current_cv_frame is not None: self._display_cv_frame_on_view(self.current_cv_frame, self._get_current_masks_for_display())
            else: logger.warning(f"Label set attempt: Selected ID {self.selected_object_sam_id} not in tracked_objects.")

    def merge_selected_objects(self):
        if len(self.selected_objects_sam_ids) < 2:
            messagebox.showwarning("Notice", "Select 2 or more objects.\nUse Ctrl+Left click to multi-select.", parent=self.root)
            return

        existing_groups = set()
        for sam_id in self.selected_objects_sam_ids:
            if sam_id in self.sam_id_to_group:
                existing_groups.add(self.sam_id_to_group[sam_id])

        if existing_groups:
            target_group_id = min(existing_groups)
            for other_group_id in existing_groups:
                if other_group_id != target_group_id:
                    for member_id in self.object_groups.get(other_group_id, set()):
                        self.object_groups[target_group_id].add(member_id)
                        self.sam_id_to_group[member_id] = target_group_id
                    if other_group_id in self.object_groups:
                        del self.object_groups[other_group_id]
            group_id = target_group_id
        else:
            group_id = self.next_group_id
            self.next_group_id += 1
            self.object_groups[group_id] = set()

        for sam_id in self.selected_objects_sam_ids:
            self.object_groups[group_id].add(sam_id)
            self.sam_id_to_group[sam_id] = group_id

        first_obj_id = min(self.selected_objects_sam_ids)
        first_obj_data = self.tracked_objects.get(first_obj_id, {})
        group_label = first_obj_data.get("custom_label", self.default_object_label_var.get())

        logger.info(f"Object group created/merged: Group ID={group_id}, members={self.object_groups[group_id]}, label={group_label}")
        self.update_status(f"Object group {group_id} created: {len(self.object_groups[group_id])} objects merged")

        self.selected_objects_sam_ids.clear()
        self.selected_object_sam_id = None

        if self.view and hasattr(self.view, 'btn_merge_objects'):
            self.view.btn_merge_objects.config(state='disabled')

        if self.current_cv_frame is not None:
            self._display_cv_frame_on_view(self.current_cv_frame, self._get_current_masks_for_display())

        self._update_interaction_status_and_label()

    def unmerge_object_group(self, group_id):
        if group_id not in self.object_groups:
            logger.warning(f"Group {group_id} to unmerge does not exist.")
            return

        for sam_id in self.object_groups[group_id]:
            if sam_id in self.sam_id_to_group:
                del self.sam_id_to_group[sam_id]

        del self.object_groups[group_id]
        logger.info(f"Object group {group_id} unmerged")

        if self.current_cv_frame is not None:
            self._display_cv_frame_on_view(self.current_cv_frame, self._get_current_masks_for_display())

    def get_group_merged_mask(self, group_id, frame_masks=None):
        if group_id not in self.object_groups:
            return None

        merged_mask = None
        for sam_id in self.object_groups[group_id]:
            if frame_masks and sam_id in frame_masks:
                mask_data = frame_masks[sam_id]
                if isinstance(mask_data, dict):
                    mask = mask_data.get('last_mask')
                else:
                    mask = mask_data
            elif sam_id in self.tracked_objects and "last_mask" in self.tracked_objects[sam_id]:
                mask = self.tracked_objects[sam_id]["last_mask"]
            else:
                continue

            if mask is not None:
                if merged_mask is None:
                    merged_mask = mask.copy().astype(bool)
                else:
                    merged_mask = merged_mask | mask.astype(bool)

        return merged_mask

    def is_sam_id_in_group(self, sam_id):
        return sam_id in self.sam_id_to_group

    def get_group_id_for_sam_id(self, sam_id):
        return self.sam_id_to_group.get(sam_id, None)

    def _reset_group_and_polygon_state(self):
        self.object_groups.clear()
        self.sam_id_to_group.clear()
        self.next_group_id = 1

        self.polygon_mode_active = False
        self.polygon_points.clear()
        self.polygon_objects.clear()

        if hasattr(self, 'view') and self.view is not None:
            try:
                self.view.update_polygon_mode_ui()
            except Exception:
                pass

        logger.debug("Group/polygon state initialized")

    def prepare_dlmi_mid_propagation(self):
        """Prepare DLMI injection to apply on the next frame when propagation resumes."""
        if not self.propagation_paused:
            messagebox.showwarning("Notice", "Propagation is not paused.\nPause propagation first, then inject.", parent=self.root)
            return

        if not self.low_level_api_enabled_var.get():
            messagebox.showwarning("Notice", "Low-level API (DLMI) is not enabled.\nEnable it in Advanced settings.", parent=self.root)
            return

        if not self.tracked_objects:
            messagebox.showwarning("Notice", "No objects to inject.\nModify or add objects first.", parent=self.root)
            return

        masks_to_inject = {}
        processed_sam_ids = set()

        for group_id, member_sam_ids in self.object_groups.items():
            merged_mask = self.get_group_merged_mask(group_id)
            if merged_mask is None:
                continue
            representative_id = min(member_sam_ids)
            first_obj_data = self.tracked_objects.get(representative_id, {})
            label = first_obj_data.get('custom_label', self.default_object_label_var.get())
            masks_to_inject[representative_id] = {
                'mask': merged_mask.astype(np.uint8),
                'label': label,
            }
            processed_sam_ids.update(member_sam_ids)

        for obj_id, obj_data in self.tracked_objects.items():
            if obj_id in processed_sam_ids:
                continue
            mask = obj_data.get('last_mask')
            if mask is None or not mask.any():
                continue
            masks_to_inject[obj_id] = {
                'mask': mask.astype(np.uint8),
                'label': obj_data.get('custom_label', self.default_object_label_var.get()),
            }

        if not masks_to_inject:
            messagebox.showwarning("Notice", "No valid masks to inject.", parent=self.root)
            return

        self.dlmi_pending_masks = masks_to_inject
        self.dlmi_pending_injection = True
        self.update_status(f"DLMI injection prepared ({len(masks_to_inject)} objects). Click 'Resume' to apply.")
        logger.info(f"DLMI mid-propagation injection prepared: {len(masks_to_inject)} objects at paused frame {self.propagation_current_frame_idx}")

    def inject_low_level_mask_prompt(self, force=False):
        import torch.nn.functional as F

        # If paused during propagation, use mid-propagation injection
        if self.propagation_paused:
            self.prepare_dlmi_mid_propagation()
            return

        if not force and not self.low_level_api_enabled_var.get():
            messagebox.showwarning("Notice", "Low-level API is not enabled.\nEnable it in advanced settings.", parent=self.root)
            return

        if not self.tracked_objects:
            messagebox.showwarning("Notice", "No masks to inject.\nDetect objects first.", parent=self.root)
            return

        if self.tracker_model is None:
            messagebox.showerror("Error", "SAM3 Tracker model not loaded.", parent=self.root)
            return

        if not hasattr(self.tracker_model, '_encode_new_memory'):
            logger.error("tracker_model does not have _encode_new_memory method.")
            messagebox.showerror("Error", "SAM3 model does not support Low-level API.\n_encode_new_memory method not found.", parent=self.root)
            return

        original_encode = None
        try:
            logger.info("Low-level API: Starting mask injection...")

            masks_to_inject = {}
            processed_sam_ids = set()

            for group_id, member_sam_ids in self.object_groups.items():
                merged_mask = self.get_group_merged_mask(group_id)
                if merged_mask is None:
                    continue

                representative_id = min(member_sam_ids)
                first_obj_data = self.tracked_objects.get(representative_id, {})
                label = first_obj_data.get('custom_label', self.default_object_label_var.get())

                masks_to_inject[representative_id] = {
                    'mask': merged_mask.astype(np.uint8),
                    'label': label,
                    'is_group': True,
                    'group_id': group_id,
                    'member_count': len(member_sam_ids)
                }
                processed_sam_ids.update(member_sam_ids)
                logger.info(f"Group {group_id}: {len(member_sam_ids)} objects -> merged to object ID {representative_id}")

            for obj_id, obj_data in self.tracked_objects.items():
                if obj_id in processed_sam_ids:
                    continue
                mask = obj_data.get('last_mask')
                if mask is None or not mask.any():
                    continue
                masks_to_inject[obj_id] = {
                    'mask': mask.astype(np.uint8),
                    'label': obj_data.get('custom_label', self.default_object_label_var.get()),
                    'is_group': False
                }

            if not masks_to_inject:
                messagebox.showwarning("Notice", "No valid masks to inject.", parent=self.root)
                return

            logger.info("Initializing SAM3 session...")
            self.inference_session = None
            if not self._init_inference_session():
                messagebox.showerror("Error", "SAM3 session initialization failed.", parent=self.root)
                return

            old_tracked_objects = self.tracked_objects.copy()
            self.tracked_objects.clear()
            self.object_groups.clear()
            self.sam_id_to_group.clear()
            self.next_group_id = 1
            self.suppressed_sam_ids.clear()
            logger.info("Low data injection: suppressed_sam_ids cleared")
            if hasattr(self, 'object_prompt_history'):
                self.object_prompt_history.clear()
                logger.info("Low data injection: object_prompt_history cleared")

            frame_idx = 0
            dtype = self.model_dtype
            injected_count = 0

            if self.current_cv_frame is None:
                messagebox.showerror("Error", "No current frame.", parent=self.root)
                return

            frame_rgb = cv2.cvtColor(self.current_cv_frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            inputs = self.tracker_processor(images=frame_pil, device=self.device, return_tensors="pt")

            frame_tensor = inputs.pixel_values[0]
            if hasattr(self, 'model_dtype') and self.model_dtype == torch.float32:
                frame_tensor = frame_tensor.to(dtype=torch.float32)

            obj_ids_list = list(masks_to_inject.keys())
            input_masks_list = [masks_to_inject[oid]['mask'] for oid in obj_ids_list]

            logger.info(f"Passing {len(obj_ids_list)} object prompts via input_masks...")
            self.tracker_processor.add_inputs_to_inference_session(
                inference_session=self.inference_session,
                frame_idx=frame_idx,
                obj_ids=list(obj_ids_list),
                input_masks=input_masks_list,
                original_size=inputs.original_sizes[0],
            )

            dlmi_mode = self.dlmi_boundary_mode_var.get()
            dlmi_intensity = self.dlmi_alpha_var.get()
            dlmi_falloff = self.dlmi_gradient_falloff_var.get()

            logger.info(f"Preparing DLMI logits: {len(obj_ids_list)} objects, "
                        f"mode={dlmi_mode}, intensity={dlmi_intensity}, falloff={dlmi_falloff}")

            from util import dlmi_hooks
            injection_queue = dlmi_hooks.build_injection_queue(
                obj_ids=obj_ids_list,
                masks_by_oid={oid: masks_to_inject[oid]['mask'] for oid in obj_ids_list},
                intensity=dlmi_intensity, mode=dlmi_mode, falloff=dlmi_falloff,
                device=self.device,
            )
            logger.info(f"Injection queue ready: {len(injection_queue)} logit maps")

            original_encode = self.tracker_model._encode_new_memory
            injection_state = {"idx": 0}
            self.tracker_model._encode_new_memory = dlmi_hooks.create_injection_hook(
                injection_queue, original_encode, log_prefix="low-level", state=injection_state
            )

            logger.info("Forward pass starting (Hook replaces masks)")
            with torch.no_grad():
                try:
                    outputs = self.tracker_model(
                        inference_session=self.inference_session,
                        frame=frame_tensor,
                    )
                    logger.info(f"Forward pass complete. Injected masks: {injection_state['idx']}")
                except Exception as e:
                    logger.exception(f"Forward pass error: {e}")
                    self.tracker_model._encode_new_memory = original_encode
                    messagebox.showerror("Error", f"Forward pass failed: {e}", parent=self.root)
                    return

            self.tracker_model._encode_new_memory = original_encode

            # Install persistent DLMI hooks (Preserve + Boost) after forward pass
            self._install_dlmi_persistent_hooks()

            if injection_state["idx"] == 0:
                messagebox.showwarning("Notice", "No masks injected. Hook was not called.", parent=self.root)
                return

            injected_count = injection_state["idx"]

            for obj_id, obj_info in masks_to_inject.items():
                mask = obj_info['mask']
                label = obj_info['label']
                self.tracked_objects[obj_id] = {
                    'custom_label': label,
                    'last_mask': mask.astype(bool),
                    'is_injected_group': obj_info.get('is_group', False)
                }

            if injected_count > 0:
                self.is_tracking_ever_started = True
                self.update_status(f"Low-level API: {injected_count} objects injected to new SAM3 session. Start propagation.")
                logger.info(f"Low-level API: {injected_count} objects injected (groups merged into single objects)")

                if self.current_cv_frame is not None:
                    self._display_cv_frame_on_view(self.current_cv_frame, self._get_current_masks_for_display())
            else:
                messagebox.showwarning("Notice", "No masks injected.", parent=self.root)

            if self.view and hasattr(self.view, 'update_low_data_inject_button_state'):
                self.view.update_low_data_inject_button_state()

        except Exception as e:
            logger.exception(f"Low-level API mask injection failed: {e}")
            messagebox.showerror("Error", f"Error during mask injection:\n{e}", parent=self.root)
            if 'original_encode' in dir() and original_encode is not None:
                try:
                    self.tracker_model._encode_new_memory = original_encode
                except:
                    pass

    def _install_dlmi_persistent_hooks(self):
        """Install persistent DLMI hooks for Preserve Memory and Boost Conditioning.
        These hooks persist across propagation frames (not cleaned up per-frame)."""
        # --- Preserve: set max_cond_frame_num = -1 to keep ALL conditioning frames ---
        if self.dlmi_preserve_memory_var.get():
            if not hasattr(self, '_dlmi_original_max_cond_frame_num'):
                self._dlmi_original_max_cond_frame_num = self.tracker_model.config.max_cond_frame_num
            self.tracker_model.config.max_cond_frame_num = -1
            logger.info(f"DLMI Preserve: max_cond_frame_num set to -1 "
                        f"(was {self._dlmi_original_max_cond_frame_num})")

        # --- Boost: hook _gather_memory_frame_outputs to triple conditioning entries ---
        if self.dlmi_boost_cond_var.get():
            if not hasattr(self, '_dlmi_original_gather'):
                original_gather = self.tracker_model._gather_memory_frame_outputs
                self._dlmi_original_gather = original_gather
                app_ref = self

                def boosted_gather(inference_session, obj_idx, frame_idx,
                                   track_in_reverse_time=False):
                    result = original_gather(
                        inference_session, obj_idx, frame_idx,
                        track_in_reverse_time=track_in_reverse_time
                    )
                    # Check at runtime so user can toggle on/off
                    if not app_ref.dlmi_boost_cond_var.get():
                        return result

                    # Conditioning entries have offset=0; duplicate them 3x
                    cond_entries = [(off, data) for off, data in result if off == 0 and data is not None]
                    non_cond_entries = [(off, data) for off, data in result if off != 0]
                    # Triple the conditioning entries
                    boosted = cond_entries * 3 + non_cond_entries
                    logger.debug(f"DLMI Boost: {len(cond_entries)} cond entries -> "
                                 f"{len(cond_entries)*3} (total {len(boosted)} memories)")
                    return boosted

                self.tracker_model._gather_memory_frame_outputs = boosted_gather
                logger.info("DLMI Boost: _gather_memory_frame_outputs hooked (3x conditioning)")

    def _remove_dlmi_persistent_hooks(self):
        """Remove persistent DLMI hooks (called on session reset/cleanup)."""
        if not hasattr(self, 'tracker_model') or self.tracker_model is None:
            return

        # Restore max_cond_frame_num
        if hasattr(self, '_dlmi_original_max_cond_frame_num'):
            self.tracker_model.config.max_cond_frame_num = self._dlmi_original_max_cond_frame_num
            logger.info(f"DLMI Preserve: max_cond_frame_num restored to "
                        f"{self._dlmi_original_max_cond_frame_num}")
            del self._dlmi_original_max_cond_frame_num

        # Restore _gather_memory_frame_outputs
        if hasattr(self, '_dlmi_original_gather'):
            self.tracker_model._gather_memory_frame_outputs = self._dlmi_original_gather
            logger.info("DLMI Boost: _gather_memory_frame_outputs restored to original")
            del self._dlmi_original_gather

    def toggle_polygon_mode(self):
        self.polygon_mode_active = not self.polygon_mode_active
        if self.polygon_mode_active:
            self.polygon_points = []
            self.update_status("Polygon mode: Left click to add points. Click 'Complete Object' when done.")
            logger.info("Polygon add mode activated")
        else:
            self.polygon_points = []
            self.update_status("Polygon mode deactivated")
            logger.info("Polygon add mode deactivated")

        if self.view and hasattr(self.view, 'update_polygon_mode_ui'):
            self.view.update_polygon_mode_ui(self.polygon_mode_active)

        if self.current_cv_frame is not None:
            self._display_cv_frame_on_view(self.current_cv_frame, self._get_current_masks_for_display())

    def add_polygon_point(self, x, y):
        if not self.polygon_mode_active:
            return False

        self.polygon_points.append((int(x), int(y)))
        logger.debug(f"Polygon point added: ({x}, {y}), total {len(self.polygon_points)}")
        self.update_status(f"{len(self.polygon_points)} polygon points entered. Add more or click 'Complete Object'.")

        if self.current_cv_frame is not None:
            self._display_cv_frame_on_view(self.current_cv_frame, self._get_current_masks_for_display())

        return True

    def undo_last_polygon_point(self):
        if not self.polygon_mode_active or not self.polygon_points:
            return

        removed = self.polygon_points.pop()
        logger.debug(f"Polygon point removed: {removed}, remaining {len(self.polygon_points)}")
        self.update_status(f"Point removed. Remaining: {len(self.polygon_points)}")

        if self.current_cv_frame is not None:
            self._display_cv_frame_on_view(self.current_cv_frame, self._get_current_masks_for_display())

    def complete_polygon_object(self):
        if not self.polygon_mode_active:
            messagebox.showwarning("Notice", "Polygon mode is not active.", parent=self.root)
            return

        if len(self.polygon_points) < 3:
            messagebox.showwarning("Notice", "Polygon requires at least 3 points.", parent=self.root)
            return

        if self.current_cv_frame is None:
            messagebox.showwarning("Notice", "No current frame.", parent=self.root)
            return

        try:
            h, w = self.current_cv_frame.shape[:2]

            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(self.polygon_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
            mask_bool = mask > 0

            if not mask_bool.any():
                messagebox.showwarning("Notice", "Generated mask is empty.", parent=self.root)
                return

            new_obj_id = self.next_obj_id_to_propose
            self.next_obj_id_to_propose += 1

            label = self.default_object_label_var.get()
            self.tracked_objects[new_obj_id] = {
                'custom_label': label,
                'last_mask': mask_bool,
                'is_polygon_object': True,
                'polygon_points': self.polygon_points.copy()
            }

            self.polygon_objects.append({
                'obj_id': new_obj_id,
                'points': self.polygon_points.copy(),
                'mask': mask_bool,
                'label': label
            })

            logger.info(f"Polygon object created: ID={new_obj_id}, points={len(self.polygon_points)}, label={label}")
            self.update_status(f"Polygon object {new_obj_id} created. Input to SAM3 or continue adding polygons.")

            self.polygon_points = []

            if self.current_cv_frame is not None:
                self._display_cv_frame_on_view(self.current_cv_frame, self._get_current_masks_for_display())

            self._update_obj_id_info_label()

        except Exception as e:
            logger.exception(f"Polygon object creation failed: {e}")
            messagebox.showerror("Error", f"Error creating polygon object:\n{e}", parent=self.root)

    def input_polygon_to_sam3(self):
        if self.app_state == "PAUSED":
            self.update_status("Paused: Cannot input to SAM3 session directly. Use DLMI injection instead.")
            return

        polygon_objs = [obj_id for obj_id, data in self.tracked_objects.items()
                       if data.get('is_polygon_object', False)]

        if not polygon_objs:
            messagebox.showwarning("Notice", "No polygon objects to input to SAM3.\nDraw polygons first and click 'Complete Object'.", parent=self.root)
            return

        if self.tracker_model is None or self.tracker_processor is None:
            messagebox.showerror("Error", "SAM3 model not loaded.", parent=self.root)
            return

        if self.inference_session is None:
            if not self._init_inference_session():
                messagebox.showerror("Error", "SAM3 session initialization failed.", parent=self.root)
                return

        self.suppressed_sam_ids.clear()
        logger.info("Polygon to SAM3 input: suppressed_sam_ids cleared")

        try:
            frame_idx = 0

            obj_ids_list = []
            input_masks_list = []

            for obj_id in polygon_objs:
                obj_data = self.tracked_objects.get(obj_id, {})
                mask = obj_data.get('last_mask')
                if mask is not None and mask.any():
                    obj_ids_list.append(obj_id)
                    input_masks_list.append(mask.astype(np.uint8))

            if not obj_ids_list:
                messagebox.showwarning("Notice", "No valid masks.", parent=self.root)
                return

            logger.info(f"Inputting {len(obj_ids_list)} polygon objects to SAM3...")
            self.tracker_processor.add_inputs_to_inference_session(
                inference_session=self.inference_session,
                frame_idx=frame_idx,
                obj_ids=obj_ids_list,
                input_masks=input_masks_list,
            )

            for obj_id in polygon_objs:
                if obj_id in self.tracked_objects:
                    self.tracked_objects[obj_id]['is_polygon_object'] = False
                    self.tracked_objects[obj_id]['is_sam3_object'] = True

            self.is_tracking_ever_started = True
            self.update_status(f"{len(obj_ids_list)} polygon objects input to SAM3. Start propagation.")
            logger.info(f"Polygon objects input to SAM3 complete: {obj_ids_list}")

            self.polygon_mode_active = False
            self.polygon_points = []
            if self.view and hasattr(self.view, 'update_polygon_mode_ui'):
                self.view.update_polygon_mode_ui(False)

        except Exception as e:
            logger.exception(f"SAM3 polygon input failed: {e}")
            messagebox.showerror("Error", f"Error during SAM3 input:\n{e}", parent=self.root)

    def cancel_polygon_mode(self):
        self.polygon_mode_active = False
        self.polygon_points = []
        self.update_status("Polygon mode cancelled")
        logger.info("Polygon mode cancelled")

        if self.view and hasattr(self.view, 'update_polygon_mode_ui'):
            self.view.update_polygon_mode_ui(False)

        if self.current_cv_frame is not None:
            self._display_cv_frame_on_view(self.current_cv_frame, self._get_current_masks_for_display())

    def _on_ctrl_press(self, event=None):
        input_handlers.on_ctrl_press(self, event)

    def _on_spacebar_press(self, event=None):
        input_handlers.on_spacebar_press(self, event)

    def _on_ctrl_release(self, event=None):
        input_handlers.on_ctrl_release(self, event)

    def _on_shift_press(self, event=None):
        input_handlers.on_shift_press(self, event)

    def _on_shift_release(self, event=None):
        input_handlers.on_shift_release(self, event)

    def _on_alt_press(self, event=None):
        input_handlers.on_alt_press(self, event)

    def _on_alt_release(self, event=None):
        input_handlers.on_alt_release(self, event)

    def _update_interaction_status_and_label(self):
        input_handlers.update_interaction_status_and_label(self)

    def _is_any_special_mode_active(self):
        return input_handlers.is_any_special_mode_active(self)

    def _get_object_id_at_coords(self, img_x, img_y):
        return input_handlers.get_object_id_at_coords(self, img_x, img_y)

    def _on_left_mouse_press(self, event):
        input_handlers.on_left_mouse_press(self, event)

    def _on_left_mouse_drag(self, event):
        input_handlers.on_left_mouse_drag(self, event)

    def _on_left_mouse_release(self, event):
        input_handlers.on_left_mouse_release(self, event)

    def _on_ctrl_shift_left_click(self, event):
        input_handlers.on_ctrl_shift_left_click(self, event)

    def _on_ctrl_middle_click_for_point(self, event):
        input_handlers.on_ctrl_middle_click_for_point(self, event)

    def _on_ctrl_right_click_for_point(self, event):
        input_handlers.on_ctrl_right_click_for_point(self, event)

    def _on_right_mouse_press(self, event):
        input_handlers.on_right_mouse_press(self, event)

    def _on_right_mouse_drag(self, event):
        input_handlers.on_right_mouse_drag(self, event)

    def _on_right_mouse_release(self, event):
        input_handlers.on_right_mouse_release(self, event)

    def _emergency_stop(self):
        input_handlers.emergency_stop(self)

    def _handle_ctrl_point_click_event(self, event, label):
        input_handlers.handle_ctrl_point_click_event(self, event, label)

    def _on_canvas_resize(self, event):
        input_handlers.on_canvas_resize(self, event)

    def _perform_resize(self):
        self._resize_job_id = None
        if self.current_cv_frame is not None:
            logger.debug("Canvas resized. Redrawing current frame.")
            self._display_cv_frame_on_view(
                self.current_cv_frame,
                self._get_current_masks_for_display(),
                None
            )

    def toggle_reassign_bbox_mode(self):
        if self.app_state == "PAUSED":
            self.update_status("Paused: Cannot reassign BBox. Resume propagation first.")
            return
        if not self.playback_paused:
            messagebox.showwarning("Notice", "BBox reassignment only available when paused.", parent=self.root)
            return
        if self.selected_object_sam_id is None:
            messagebox.showinfo("Notice", "Select an object to reassign BBox first (Ctrl+Left click).", parent=self.root)
            return
        if self._is_any_special_mode_active() and self.reassign_bbox_mode_active_sam_id != self.selected_object_sam_id :
             messagebox.showwarning("Notice", "Another interaction (auto-correction etc.) is active.", parent=self.root)
             return

        if self.reassign_bbox_mode_active_sam_id == self.selected_object_sam_id:
            self.reassign_bbox_mode_active_sam_id = None
            logger.info(f"Object {self.selected_object_sam_id} BBox reassignment mode deactivated.")
        else:
            self.reassign_bbox_mode_active_sam_id = self.selected_object_sam_id
            logger.info(f"Object {self.reassign_bbox_mode_active_sam_id} BBox reassignment mode activated. Draw new BBox.")

        self._update_interaction_status_and_label()
        if self.current_cv_frame is not None:
            self._display_cv_frame_on_view(self.current_cv_frame, self._get_current_masks_for_display())

    def delete_selected_object(self):
        if not self.playback_paused:
            messagebox.showwarning("Notice", "Object deletion only available when paused.", parent=self.root)
            return
        if self.selected_object_sam_id is None:
            messagebox.showinfo("Notice", "Select an object to delete first.", parent=self.root)
            return

        obj_to_delete = self.selected_object_sam_id
        is_correction_deletion = (self.interaction_correction_pending is not None and
                                  self.interaction_correction_pending == obj_to_delete)

        if not is_correction_deletion and (self.reassign_bbox_mode_active_sam_id is not None or \
                                           self.problematic_highlight_active_sam_id is not None):
             messagebox.showwarning("Notice", "Cannot delete selected object while another interaction is active.", parent=self.root)
             return

        confirm_message = f"Are you sure you want to delete object ID {obj_to_delete}?"
        if is_correction_deletion:
            confirm_message = f"Delete problem object ID {obj_to_delete} during auto-correction?"

        if messagebox.askyesno("Delete Object Confirmation", confirm_message, parent=self.root):
            reason = "Problem object deleted during auto-correction" if is_correction_deletion else "Manual deletion (button click)"
            self._delete_object_by_id(obj_to_delete, reason)

            if is_correction_deletion:
                self.interaction_correction_pending = None
                if obj_to_delete in self.problematic_objects_flagged:
                    del self.problematic_objects_flagged[obj_to_delete]

                self.autolabel_active = False
                self.playback_paused = True
                self.update_status(f"Problem object {obj_to_delete} deleted. Click 'Resume Auto-labeling'.")
                self._update_ui_for_autolabel_state(False)
            
    def _delete_object_by_id(self, obj_id_to_delete, reason=""):
        if obj_id_to_delete in self.tracked_objects:
            if obj_id_to_delete not in self.suppressed_sam_ids:
                self.suppressed_sam_ids.add(obj_id_to_delete)
                logger.info(f"SAM ID {obj_id_to_delete} deleted and added to suppression list. This object will no longer be labeled or displayed.")

            del self.tracked_objects[obj_id_to_delete]
            logger.info(f"Object ID {obj_id_to_delete} removed from tracked_objects. Reason: {reason}")

            if self.selected_object_sam_id == obj_id_to_delete: self.selected_object_sam_id = None
            if self.reassign_bbox_mode_active_sam_id == obj_id_to_delete: self.reassign_bbox_mode_active_sam_id = None
            if self.interaction_correction_pending == obj_id_to_delete: self.interaction_correction_pending = None
            if self.problematic_highlight_active_sam_id == obj_id_to_delete: self.problematic_highlight_active_sam_id = None
            if obj_id_to_delete in self.problematic_objects_flagged: del self.problematic_objects_flagged[obj_id_to_delete]

            self.update_status(f"Object {obj_id_to_delete} deleted.")
            self._update_interaction_status_and_label()
            if self.current_cv_frame is not None:
                self._display_cv_frame_on_view(self.current_cv_frame, self._get_current_masks_for_display())
        else:
            logger.warning(f"Delete attempt: Object ID {obj_id_to_delete} not in tracked_objects.")

    def _release_video_capture(self):
        if self.cap and self.cap.isOpened(): logger.info("Video capture released."); self.cap.release(); self.cap = None
        if hasattr(self, '_temp_video_path') and self._temp_video_path:
            try:
                if os.path.exists(self._temp_video_path):
                    os.remove(self._temp_video_path)
                    logger.info(f"Temp video file deleted: {self._temp_video_path}")
            except Exception as e:
                logger.warning(f"Temp video file delete failed: {e}")
            self._temp_video_path = None

    def _reset_internal_states_for_new_source(self):
        logger.info("Resetting internal states for new video source.")
        self.is_predictor_loaded_first_frame = False; self.tracked_objects.clear()
        self.next_obj_id_to_propose = 1; self._update_obj_id_info_label()
        self.current_cv_frame = None; self.current_frame_pil_rgb_original = None
        self.current_frame_idx_conceptual = 0; self.is_tracking_ever_started = False
        self.last_active_tracked_sam_ids.clear(); self.selected_object_sam_id = None
        self.autolabel_active = False; self.just_reset_sam = False; self.sam_operation_in_progress = False
        self.problematic_objects_flagged.clear()
        self.interaction_correction_pending = None; self.problematic_highlight_active_sam_id = None
        self.reassign_bbox_mode_active_sam_id = None
        self.suppressed_sam_ids.clear()
        self.is_batch_video_finished = False
        self.yolo_dataset_initialized = False

        self._reset_group_and_polygon_state()

        self.inference_session = None

        self.pcs_inference_session = None
        self.pcs_streaming_session = None
        self.video_frames_cache = []
        self.pcs_exemplar_boxes = []
        self.pcs_exemplar_labels = []

        self.cut_start_frame = 0
        self.cut_point_frame = None
        self.propagated_results = {}

        self.discarded_frames.clear()
        if hasattr(self, 'view') and hasattr(self.view, 'update_discarded_frames_display'):
            self.view.update_discarded_frames_display(set())

        self.video_display_name = ""
        self.video_total_frames = 0
        self.video_fps = 0
        self.video_resolution = ""
        self.info_video_name_var.set("N/A")
        self.info_video_resolution_var.set("N/A")
        self.info_video_total_frames_var.set("N/A")
        self.info_video_fps_var.set("N/A")

        if not self.is_batch_running:
            self.info_batch_progress_var.set("N/A")

        if self.view and hasattr(self.view, 'clear_original_canvas'):
            self.view.clear_original_canvas()

        if self.view: self.view.clear_canvas_image()
        if self.predictor and hasattr(self.predictor, 'reset_state'):
            try: self.predictor.reset_state(); logger.info("SAM3 predictor.reset_state() called.")
            except Exception as e:
                if 'point_inputs_per_obj' in str(e): logger.warning(f"SAM3 predictor.reset_state() known issue: {e} (ignored)")
                else: logger.warning(f"SAM3 predictor.reset_state() error: {e} (ignored)")

    def _prompt_yolo_class_info(self):
        return batch_controller.prompt_yolo_class_info(self)

    def _init_yolo_dataset_structure(self, save_dir):
        return batch_controller.init_yolo_dataset_structure(self, save_dir)

    def _update_yolo_yaml(self):
        batch_controller.update_yolo_yaml(self)

    def _check_existing_yolo_dataset(self, save_dir):
        return batch_controller.check_existing_yolo_dataset(self, save_dir)

    def _get_save_directory(self):
        return batch_controller.get_save_directory(self)

    def start_batch_processing(self):
        batch_controller.start_batch_processing(self)

    def skip_current_batch_video(self):
        batch_controller.skip_current_batch_video(self)

    def _move_completed_video(self, video_path, skipped=False):
        batch_controller.move_completed_video(self, video_path, skipped)

    def select_batch_completed_dir(self):
        batch_controller.select_batch_completed_dir(self)

    def select_video_source(self):
        if self.autolabel_active or self._is_any_special_mode_active():
            messagebox.showwarning("Busy", "Stop auto-labeling or interaction before changing source.", parent=self.root)
            return

        if self.batch_processing_mode_var.get():
            messagebox.showinfo("Notice", "Batch processing mode is active.\nClick 'Start Batch Processing'.", parent=self.root)
            return

        self._release_video_capture(); self._reset_internal_states_for_new_source()
        self.is_image_source = False

        if self.allow_image_source_var.get():
            source_type = messagebox.askquestion("Select Source", "Load from video/image file?\n(Camera 0 is No)", parent=self.root)
            if source_type == 'yes':
                self.video_source_path = filedialog.askopenfilename(
                    title="Select Video/Image File",
                    filetypes=(
                        ("Video", "*.mp4 *.avi *.mov *.mkv"),
                        ("Image", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                        ("All Files", "*.*")
                    ),
                    parent=self.root
                )
                if not self.video_source_path: return

                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
                file_ext = os.path.splitext(self.video_source_path)[1].lower()
                if file_ext in image_extensions:
                    self._load_image_as_video_source(self.video_source_path)
                    return
            else:
                self.video_source_path = 0
        else:
            source_type = messagebox.askquestion("Select Source", "Load from video file?\n(Camera 0 is No)", parent=self.root)
            if source_type == 'yes':
                self.video_source_path = filedialog.askopenfilename(title="Select Video File", filetypes=(("MP4", "*.mp4"),("AVI", "*.avi"),("All Files", "*.*")), parent=self.root)
                if not self.video_source_path: return
            else: self.video_source_path = 0

        self.cap = cv2.VideoCapture(self.video_source_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open video source: {self.video_source_path}", parent=self.root); self.cap = None; return

        self.video_total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_resolution = f"{width}x{height}"
        self.video_display_name = "Camera 0" if self.video_source_path == 0 else os.path.basename(self.video_source_path)

        self.info_video_name_var.set(self.video_display_name)
        self.info_video_resolution_var.set(self.video_resolution)
        self.info_video_total_frames_var.set(f"{self.video_total_frames} frames" if self.video_total_frames > 0 else "N/A (live)")
        self.info_video_fps_var.set(f"{self.video_fps:.2f} FPS" if self.video_fps > 0 else "N/A (live)")
        self.update_status(f"Source: {self.video_source_path}. Loading first frame..."); self.playback_paused = True

        max_slider_frame = self.video_total_frames - 1 if self.video_total_frames > 0 else 0
        self.review_current_frame = 0
        self.view.update_review_slider_range(max_slider_frame)
        self.view.review_frame_slider.set(0)
        self.view.update_review_frame_info(0, max_slider_frame)

        ret, frame_bgr = self.cap.read()
        if ret:
            self.current_cv_frame = frame_bgr.copy(); self.current_frame_idx_conceptual = 0
            self._display_cv_frame_on_view(frame_bgr)
            self.update_status(f"Source loaded. Detect objects then click 'Start Propagation'.")
            self._update_interaction_status_and_label()
        else: messagebox.showerror("Error", "Failed to read first frame from video.", parent=self.root); self._release_video_capture()

    def _load_image_as_video_source(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                messagebox.showerror("Error", f"Cannot open image: {image_path}", parent=self.root)
                return

            self.is_image_source = True
            self.video_source_path = image_path

            import tempfile
            self._temp_video_path = tempfile.NamedTemporaryFile(suffix='.avi', delete=False).name

            height, width = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(self._temp_video_path, fourcc, 1.0, (width, height))
            out.write(img)
            out.write(img)
            out.release()

            self.cap = cv2.VideoCapture(self._temp_video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Failed to convert image to video.", parent=self.root)
                self.cap = None
                return

            self.video_total_frames = 2
            self.video_fps = 1.0
            self.video_resolution = f"{width}x{height}"
            self.video_display_name = f"[Image] {os.path.basename(image_path)}"

            self.info_video_name_var.set(self.video_display_name)
            self.info_video_resolution_var.set(self.video_resolution)
            self.info_video_total_frames_var.set("1 frame (image)")
            self.info_video_fps_var.set("N/A (image)")
            self.update_status(f"Image loaded: {image_path}")
            self.playback_paused = True

            self.review_current_frame = 0
            self.view.update_review_slider_range(0)
            self.view.review_frame_slider.set(0)
            self.view.update_review_frame_info(0, 0)

            ret, frame_bgr = self.cap.read()
            if ret:
                self.current_cv_frame = frame_bgr.copy()
                self.current_frame_idx_conceptual = 0
                self._display_cv_frame_on_view(frame_bgr)
                self.update_status(f"Image loaded. Detect objects then click 'Confirm Labeling'.")
                self._update_interaction_status_and_label()
            else:
                messagebox.showerror("Error", "Failed to read image frame.", parent=self.root)
                self._release_video_capture()

        except Exception as e:
            logger.exception(f"Error loading image: {e}")
            messagebox.showerror("Error", f"Error loading image: {e}", parent=self.root)

    def load_label_file(self):
        import json

        if self.current_cv_frame is None:
            messagebox.showwarning("Notice", "Load video or image source first.", parent=self.root)
            return

        label_file_path = filedialog.askopenfilename(
            title="Select Label File (JSON or YOLO txt)",
            filetypes=(
                ("LabelMe JSON", "*.json"),
                ("YOLO txt", "*.txt"),
                ("All Files", "*.*")
            ),
            parent=self.root
        )

        if not label_file_path:
            return

        file_ext = os.path.splitext(label_file_path)[1].lower()

        try:
            frame_height, frame_width = self.current_cv_frame.shape[:2]
            if file_ext == '.json':
                loaded_objects = autolabel_workflow.parse_labelme_json(label_file_path, frame_width, frame_height)
            elif file_ext == '.txt':
                loaded_objects = autolabel_workflow.parse_yolo_txt(label_file_path, frame_width, frame_height)
            else:
                messagebox.showerror("Error", "Unsupported file format.\nOnly JSON or txt files are supported.", parent=self.root)
                return

            if not loaded_objects:
                messagebox.showinfo("Info", "No objects to load.", parent=self.root)
                return

            if self.low_level_api_enabled_var.get():
                self._apply_loaded_labels_as_polygon_masks(loaded_objects)
                return

            current_mode = self.prompt_mode_var.get()

            if current_mode == "PVS" or current_mode == "PVS_CHUNK":
                self._apply_loaded_labels_as_prompts(loaded_objects)
            elif current_mode in ("PCS", "PCS_IMAGE"):
                self._apply_loaded_labels_for_pcs(loaded_objects)

            self.update_status(f"{len(loaded_objects)} objects loaded.")
            logger.info(f"Label file loaded: {label_file_path}, objects: {len(loaded_objects)}")

        except Exception as e:
            logger.exception(f"Label file load error: {e}")
            messagebox.showerror("Error", f"Error loading label file:\n{e}", parent=self.root)

    def _apply_loaded_labels_as_polygon_masks(self, loaded_objects):
        if not loaded_objects:
            return

        if self.current_cv_frame is None:
            messagebox.showerror("Error", "No current frame available.", parent=self.root)
            return

        try:
            frame_height, frame_width = self.current_cv_frame.shape[:2]

            self.tracked_objects.clear()
            self.next_obj_id_to_propose = 1
            self.polygon_objects.clear()

            for obj in loaded_objects:
                label = obj.get('label', self.default_object_label_var.get())

                polygon = obj.get('polygon')
                if polygon and len(polygon) >= 3:
                    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts], 255)
                    mask_bool = mask > 0
                elif obj.get('bbox'):
                    x1, y1, x2, y2 = obj['bbox']
                    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
                    mask[int(y1):int(y2), int(x1):int(x2)] = 255
                    mask_bool = mask > 0
                else:
                    continue

                if not mask_bool.any():
                    continue

                new_obj_id = self.next_obj_id_to_propose
                self.next_obj_id_to_propose += 1

                self.tracked_objects[new_obj_id] = {
                    'custom_label': label,
                    'last_mask': mask_bool,
                    'is_polygon_object': False,
                }

                logger.info(f"Label load (Low API): object '{label}' (ID: {new_obj_id}) mask created")

            if not self.tracked_objects:
                messagebox.showwarning("Info", "No objects to convert.", parent=self.root)
                return

            self._display_cv_frame_on_view(self.current_cv_frame, self._get_current_masks_for_display())
            self._update_obj_id_info_label()

            self.update_status(f"{len(self.tracked_objects)} object masks created. Injecting Low data...")
            self.root.update_idletasks()

            self.inject_low_level_mask_prompt()

            self.update_status(f"Low-level API: {len(self.tracked_objects)} objects injected to SAM3.")
            logger.info(f"Label load + Low data injection complete: {len(self.tracked_objects)} objects")

        except Exception as e:
            logger.exception(f"Low-level API label load failed: {e}")
            messagebox.showerror("Error", f"Error loading labels via Low-level API:\n{e}", parent=self.root)

    def _apply_loaded_labels_as_prompts(self, loaded_objects):
        if not loaded_objects:
            return

        if self.inference_session is None:
            if not self._init_inference_session():
                messagebox.showerror("Error", "SAM3 session initialization failed.", parent=self.root)
                return

        if self.tracked_objects:
            response = messagebox.askyesno(
                "Existing Object Handling",
                f"Currently {len(self.tracked_objects)} objects exist.\n"
                "Keep existing objects and add new ones?\n\n"
                "Yes: Keep existing and add\n"
                "No: Clear existing and add new",
                parent=self.root
            )
            if not response:
                self.tracked_objects.clear()
                self.next_obj_id_to_propose = 1
                self._reset_inference_session()

        self._suppress_mid_session_guard = True
        try:
            for obj in loaded_objects:
                bbox = obj.get('bbox')
                label = obj.get('label', self.default_object_label_var.get())

                if bbox is None:
                    continue

                x1, y1, x2, y2 = bbox
                coords = np.array([x1, y1, x2, y2])

                new_obj_id = self.next_obj_id_to_propose

                self._handle_sam_prompt_wrapper(
                    prompt_type='bbox',
                    coords=coords,
                    label=1,  # positive
                    proposed_obj_id_for_new=new_obj_id,
                    target_existing_obj_id=None,
                    custom_label=label
                )

                logger.info(f"Label load: object '{label}' (ID: {new_obj_id}) bbox prompt applied")
        finally:
            self._suppress_mid_session_guard = False

        if self.current_cv_frame is not None:
            self._display_cv_frame_on_view(self.current_cv_frame, self._get_current_masks_for_display())

        self._update_obj_id_info_label()

    def _apply_loaded_labels_for_pcs(self, loaded_objects):
        from util import pcs_controller
        return pcs_controller.apply_loaded_labels_for_pcs(self, loaded_objects)

    def _display_cv_frame_on_view(self, frame_bgr, masks_to_overlay=None, yolo_bboxes_to_draw=None):
        if frame_bgr is None: return
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.current_frame_pil_rgb_original = Image.fromarray(frame_rgb)

            img_arr = np.array(self.current_frame_pil_rgb_original.convert("RGBA"))
            h, w = img_arr.shape[:2]

            bbox_cache = {}
            special_focus_objs = []

            if masks_to_overlay:
                erosion_k = self.erosion_kernel_size.get()
                erosion_i = self.erosion_iterations.get()
                base_alpha = self.mask_alpha_var.get() if hasattr(self, 'mask_alpha_var') else ALPHA_NORMAL

                group_bbox_cache = {}
                processed_groups = set()
                for obj_id, mask_array_bool in masks_to_overlay.items():
                    if mask_array_bool is None: continue

                    if mask_array_bool.dtype != bool:
                        mask_array_bool = mask_array_bool > 0.5
                    if mask_array_bool.shape[0] != h or mask_array_bool.shape[1] != w:
                        continue

                    bbox_cache[obj_id] = get_bbox_from_mask(mask_array_bool, erosion_k, erosion_i, 1)
                    group_id = self.sam_id_to_group.get(obj_id)
                    if group_id is not None:
                        first_member = min(self.object_groups.get(group_id, {obj_id}))
                        rgb_color = self._get_object_color(first_member)
                        current_bbox = bbox_cache[obj_id]
                        if current_bbox is not None:
                            if group_id not in group_bbox_cache:
                                group_bbox_cache[group_id] = list(current_bbox)
                            else:
                                x1, y1, x2, y2 = group_bbox_cache[group_id]
                                nx1, ny1, nx2, ny2 = current_bbox
                                group_bbox_cache[group_id] = [min(x1, nx1), min(y1, ny1), max(x2, nx2), max(y2, ny2)]
                    else:
                        rgb_color = self._get_object_color(obj_id)

                    alpha = base_alpha
                    is_multi_selected = obj_id in self.selected_objects_sam_ids

                    is_group_selected = False
                    if group_id is not None:
                        group_members = self.object_groups.get(group_id, set())
                        is_group_selected = any(
                            m in self.selected_objects_sam_ids or m == self.selected_object_sam_id
                            for m in group_members
                        )

                    is_special = (is_multi_selected or is_group_selected or
                                  obj_id == self.selected_object_sam_id or
                                  obj_id == self.interaction_correction_pending or
                                  obj_id == self.problematic_highlight_active_sam_id or
                                  obj_id == self.reassign_bbox_mode_active_sam_id)

                    if is_special:
                        alpha = min(255, alpha + 50)
                        special_focus_objs.append(obj_id)
                    if obj_id == self.problematic_highlight_active_sam_id:
                        alpha = max(30, alpha - 50)
                    elif obj_id == self.interaction_correction_pending or obj_id == self.reassign_bbox_mode_active_sam_id:
                        alpha = max(20, alpha - 80)

                    alpha_ratio = alpha / 255.0
                    inv_alpha = 1.0 - alpha_ratio
                    color_arr = np.array(rgb_color, dtype=np.float32)
                    img_arr[mask_array_bool, :3] = (
                        img_arr[mask_array_bool, :3] * inv_alpha + color_arr * alpha_ratio
                    ).astype(np.uint8)

            pil_image_to_draw_on = Image.fromarray(img_arr, "RGBA")
            draw_final = ImageDraw.Draw(pil_image_to_draw_on)

            drawn_group_ids = set()
            for obj_id in special_focus_objs:
                group_id = self.sam_id_to_group.get(obj_id)

                if group_id is not None:
                    if group_id in drawn_group_ids:
                        continue
                    drawn_group_ids.add(group_id)
                    bbox = group_bbox_cache.get(group_id) if 'group_bbox_cache' in dir() else bbox_cache.get(obj_id)
                else:
                    bbox = bbox_cache.get(obj_id)

                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    if x1 < x2 and y1 < y2:
                        draw_final.rectangle([x1, y1, x2, y2], outline="yellow", width=3)

            if masks_to_overlay:
                labeled_groups = set()
                for obj_id in masks_to_overlay.keys():
                    group_id = self.sam_id_to_group.get(obj_id)

                    if group_id is not None:
                        if group_id in labeled_groups:
                            continue
                        labeled_groups.add(group_id)
                        first_member = min(self.object_groups.get(group_id, {obj_id}))
                        obj_data = self.tracked_objects.get(first_member, {})
                        member_count = len(self.object_groups.get(group_id, set()))
                        display_label = f"Group-{group_id} ({member_count}): {obj_data.get('custom_label', 'object')}"
                        bbox_for_label = group_bbox_cache.get(group_id) if 'group_bbox_cache' in dir() else bbox_cache.get(obj_id)
                    else:
                        obj_data = self.tracked_objects.get(obj_id, {})
                        display_label = obj_data.get("custom_label", f"Obj-{obj_id}")
                        if obj_id == self.reassign_bbox_mode_active_sam_id: display_label += " (BBox Reassign)"
                        elif obj_id == self.problematic_highlight_active_sam_id: display_label += " (Check Required!)"
                        elif obj_id == self.interaction_correction_pending: display_label += " (Auto Correction...)"
                        bbox_for_label = bbox_cache.get(obj_id)

                    if bbox_for_label is not None:
                        x1, y1, x2, y2 = bbox_for_label
                        import math
                        diagonal = math.sqrt(w**2 + h**2)
                        font_size_percent = self.label_font_size_percent_var.get()
                        dynamic_font_size = max(8, int(diagonal * font_size_percent / 100))
                        dynamic_font = None
                        font_paths = [
                            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                            "arial.ttf",
                            "/System/Library/Fonts/Helvetica.ttc"
                        ]
                        for font_path in font_paths:
                            try:
                                dynamic_font = ImageFont.truetype(font_path, dynamic_font_size)
                                break
                            except:
                                continue
                        if dynamic_font is None:
                            dynamic_font = ImageFont.load_default()
                        text_pos = (int(x1 + 5), int(y1 - dynamic_font_size - 5 if y1 > dynamic_font_size + 5 else y1 + 5))
                        try: draw_final.text(text_pos, display_label, fill="yellow", font=dynamic_font)
                        except Exception: pass
                        if obj_id in (self.problematic_highlight_active_sam_id, self.interaction_correction_pending, self.reassign_bbox_mode_active_sam_id):
                            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                            marker_base_size = min(w, h) / 80
                            marker_color = "orange" if obj_id in (self.interaction_correction_pending, self.reassign_bbox_mode_active_sam_id) else "red"
                            marker_size = marker_base_size * 0.7 if marker_color == "orange" else marker_base_size
                            draw_star_marker(draw_final, center_x, center_y, marker_size, color=marker_color)

            if getattr(self, 'polygon_mode_active', False):
                polygon_color = (0, 255, 255)
                import math
                diagonal = math.sqrt(w**2 + h**2)
                point_size_percent = self.polygon_point_size_percent_var.get()
                point_radius = max(2, int(diagonal * point_size_percent / 100))

                if hasattr(self, 'polygon_points') and self.polygon_points:
                    for i, (px, py) in enumerate(self.polygon_points):
                        draw_final.ellipse(
                            [px - point_radius, py - point_radius, px + point_radius, py + point_radius],
                            fill="cyan", outline="white"
                        )
                        draw_final.text((px + point_radius + 2, py - point_radius), str(i + 1), fill="yellow")

                    if len(self.polygon_points) >= 2:
                        for i in range(len(self.polygon_points) - 1):
                            p1 = self.polygon_points[i]
                            p2 = self.polygon_points[i + 1]
                            draw_final.line([p1[0], p1[1], p2[0], p2[1]], fill="cyan", width=2)

                if hasattr(self, 'polygon_objects') and self.polygon_objects:
                    for obj_idx, poly_obj in enumerate(self.polygon_objects):
                        points = poly_obj.get('points', [])
                        if len(points) >= 3:
                            flat_points = [(p[0], p[1]) for p in points]
                            draw_final.polygon(flat_points, outline="lime", width=2)
                            first_pt = points[0]
                            draw_final.text((first_pt[0] + 5, first_pt[1] - 15), f"Poly-{obj_idx + 1}", fill="lime")

            if (hasattr(self, 'show_prompt_visualization_var') and self.show_prompt_visualization_var.get()
                and hasattr(self, 'object_prompt_history') and self.object_prompt_history):
                import math
                diagonal = math.sqrt(w**2 + h**2)
                point_size_percent = self.polygon_point_size_percent_var.get()
                line_width = max(1, int(diagonal * point_size_percent / 100))
                point_radius = max(2, int(diagonal * point_size_percent / 100))

                show_per_object = hasattr(self, 'show_prompt_per_object_var') and self.show_prompt_per_object_var.get()
                selected_id = getattr(self, 'selected_object_sam_id', None)

                for obj_id, prompts in self.object_prompt_history.items():
                    if show_per_object and selected_id is not None and obj_id != selected_id:
                        continue
                    for box in prompts.get('boxes', []):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        draw_final.rectangle([x1, y1, x2, y2], outline="black", width=line_width)

                    for pt in prompts.get('positive_points', []):
                        px, py = int(pt[0]), int(pt[1])
                        draw_final.ellipse(
                            [px - point_radius, py - point_radius, px + point_radius, py + point_radius],
                            fill="lime", outline="white"
                        )

                    for pt in prompts.get('negative_points', []):
                        px, py = int(pt[0]), int(pt[1])
                        draw_final.ellipse(
                            [px - point_radius, py - point_radius, px + point_radius, py + point_radius],
                            fill="red", outline="white"
                        )

                    for contour in prompts.get('mask_contours', []):
                        if len(contour) >= 2:
                            flat_pts = [(int(p[0][0]), int(p[0][1])) for p in contour if len(p) > 0]
                            if len(flat_pts) >= 2:
                                draw_final.line(flat_pts + [flat_pts[0]], fill="lime", width=line_width)

                    for box in prompts.get('exemplar_positive', []):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        draw_final.rectangle([x1, y1, x2, y2], outline="lime", width=max(1, line_width // 2))

                    for box in prompts.get('exemplar_negative', []):
                        x1, y1, x2, y2 = [int(v) for v in box]
                        draw_final.rectangle([x1, y1, x2, y2], outline="red", width=max(1, line_width // 2))

            if yolo_bboxes_to_draw:
                for yolo_id, data in yolo_bboxes_to_draw.items():
                    if data["bbox"] is None: continue
                    x1, y1, x2, y2 = data["bbox"]
                    label = f"Y-{yolo_id}: {data['class_name']}"
                    draw_final.rectangle([x1, y1, x2, y2], outline="cyan", width=1)
                    draw_final.text((x1, y1 - 10 if y1 > 10 else y1 + 2), label, fill="cyan", font=self.label_font)

            if (hasattr(self, 'show_object_border_var') and self.show_object_border_var.get() and masks_to_overlay):
                border_color = (0, 0, 0, 255)
                for obj_id, mask_arr in masks_to_overlay.items():
                    if mask_arr is None or not mask_arr.any(): continue
                    mask_uint8 = (mask_arr.astype(np.uint8) if mask_arr.dtype == bool else (mask_arr > 0.5).astype(np.uint8)) * 255
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if len(contour) >= 3:
                            points = [tuple(pt[0]) for pt in contour]
                            if len(points) >= 3:
                                draw_final.polygon(points, outline=border_color, width=2)

            if self.view: self.view.display_image(pil_image_to_draw_on)
            if self._pose_ui is not None and self.view and hasattr(self.view, 'canvas'):
                try:
                    self._pose_ui.render_pose_on_canvas(
                        self.view.canvas, self,
                        selected_pose_set=self.selected_pose_points
                    )
                except Exception as _pose_render_err:
                    logger.debug(f"pose overlay render skipped: {_pose_render_err}")
        except Exception as e: logger.exception(f"_display_cv_frame_on_view error: {e}")

    def _get_current_masks_for_display(self):
        current_keys = list(self.tracked_objects.keys())
        masks = {}

        filter_enabled = self.filter_small_objects_var.get()
        threshold_ratio = self.small_object_threshold_var.get()

        min_bbox_width = 0
        if filter_enabled and self.current_cv_frame is not None:
            frame_width = self.current_cv_frame.shape[1]
            min_bbox_width = frame_width * threshold_ratio

        for obj_id in current_keys:
            data = self.tracked_objects.get(obj_id)
            if data and "last_mask" in data and data["last_mask"] is not None:
                mask = data["last_mask"]

                if filter_enabled and min_bbox_width > 0:
                    bbox = get_bbox_from_mask(mask, min_bbox_area_val=1)
                    if bbox is not None:
                        bbox_width = bbox[2] - bbox[0]
                        if bbox_width < min_bbox_width:
                            logger.debug(f"Small object filter: ObjID {obj_id} (width {bbox_width:.1f} < threshold {min_bbox_width:.1f})")
                            continue

                masks[obj_id] = mask
        return masks

    def _update_ui_for_autolabel_state(self, is_autolabeling_active_or_resuming):
        is_paused_or_stopped = not is_autolabeling_active_or_resuming
        is_special_mode = self._is_any_special_mode_active()

        general_state = tk.NORMAL if is_paused_or_stopped and not is_special_mode else tk.DISABLED
        special_mode_override = is_paused_or_stopped

        self.view.set_ui_element_state("btn_select_source", general_state)
        self.view.set_ui_element_state("btn_clear_tracked", general_state if self.predictor else tk.DISABLED)
        self.view.set_ui_element_state("notebook_tabs", general_state)
        self.view.set_ui_element_state("entry_default_label", general_state)

        selected_obj_exists = self.selected_object_sam_id is not None
        self.view.set_ui_element_state("btn_set_custom_label", tk.NORMAL if selected_obj_exists and special_mode_override and not is_special_mode else tk.DISABLED)
        self.view.set_ui_element_state("btn_reassign_bbox_selected", tk.NORMAL if selected_obj_exists and special_mode_override else tk.DISABLED)

        can_delete = (selected_obj_exists and special_mode_override and not self.reassign_bbox_mode_active_sam_id and not self.problematic_highlight_active_sam_id)
        self.view.set_ui_element_state("btn_delete_selected", tk.NORMAL if can_delete else tk.DISABLED)

        can_load_label = self.current_cv_frame is not None and is_paused_or_stopped
        self.view.set_ui_element_state("btn_load_label", tk.NORMAL if can_load_label else tk.DISABLED)

        can_skip_batch = self.is_batch_running and is_paused_or_stopped
        if hasattr(self.view, 'btn_skip_batch'):
            self.view.set_ui_element_state("btn_skip_batch", tk.NORMAL if can_skip_batch else tk.DISABLED)

    def _reopen_capture(self):
        if self.video_source_path is None: return False
        logger.info(f"Attempting to reopen video capture: {self.video_source_path}")
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(self.video_source_path)
        if not self.cap.isOpened(): err_msg=f"Failed to reopen video source {self.video_source_path}"; logger.error(err_msg); messagebox.showerror("Error", err_msg, parent=self.root); return False
        if isinstance(self.video_source_path, str) and self.current_frame_idx_conceptual == 0:
            try: self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); logger.info("Moved to start after video file reopen.")
            except Exception as e: logger.warning(f"Failed to reset video frame position: {e} (ignored)")
        self.is_predictor_loaded_first_frame = False
        if self.predictor and hasattr(self.predictor, 'reset_state'):
            try: self.predictor.reset_state()
            except Exception as e:
                if 'point_inputs_per_obj' in str(e): logger.warning(f"SAM3 predictor.reset_state() known issue: {e} (ignored)")
                else: logger.warning(f"SAM reset failed during capture reopen: {e}")
        self.is_tracking_ever_started = False; self.last_active_tracked_sam_ids.clear()
        self.just_reset_sam = False; self.sam_operation_in_progress = False
        self.problematic_objects_flagged.clear(); self.interaction_correction_pending = None
        self.problematic_highlight_active_sam_id = None; self.reassign_bbox_mode_active_sam_id = None
        return True

    def clear_all_tracked_objects(self):
        if self.autolabel_active or self._is_any_special_mode_active():
            messagebox.showwarning("Clear Failed", "Stop auto-labeling or interaction before clearing objects.", parent=self.root); return
        logger.info("Clearing all tracked objects/prompts and resetting SAM3 predictor.")
        self._remove_dlmi_persistent_hooks()
        self.tracked_objects.clear(); self.next_obj_id_to_propose = 1
        self.is_predictor_loaded_first_frame = False; self.is_tracking_ever_started = False
        self.last_active_tracked_sam_ids.clear(); self.selected_object_sam_id = None
        self.just_reset_sam = False; self.sam_operation_in_progress = False
        self.problematic_objects_flagged.clear()
        self.interaction_correction_pending = None; self.problematic_highlight_active_sam_id = None; self.reassign_bbox_mode_active_sam_id = None
        self.is_restoration_pending = False
        self.suppressed_sam_ids.clear()
        if hasattr(self, 'object_prompt_history'):
            self.object_prompt_history.clear()
            logger.info("object_prompt_history cleared.")

        self._reset_group_and_polygon_state()

        if hasattr(self, 'inference_session') and self.inference_session is not None:
            try:
                if hasattr(self.inference_session, 'obj_ids'):
                    self.inference_session.obj_ids.clear()
                if hasattr(self.inference_session, 'point_inputs_per_obj'):
                    self.inference_session.point_inputs_per_obj.clear()
                if hasattr(self.inference_session, 'mask_inputs_per_obj'):
                    self.inference_session.mask_inputs_per_obj.clear()
                logger.info("SAM3 inference_session internal state cleared.")
            except Exception as e:
                logger.warning(f"Error during inference_session clear (ignored): {e}")
            self.inference_session = None
            logger.info("SAM3 inference_session set to None.")

        if self.predictor and hasattr(self.predictor, 'reset_state'):
            try: self.predictor.reset_state(); logger.info("SAM3 predictor.reset_state() called.")
            except Exception as e:
                if 'point_inputs_per_obj' in str(e): logger.warning(f"SAM3 predictor.reset_state() known issue: {e} (ignored)")
                else: logger.error(f"SAM3 predictor.reset_state() error: {e} (ignored)")

        if hasattr(self, 'propagated_results'):
            self.propagated_results.clear()
            logger.info("propagated_results cleared.")

        if hasattr(self, 'discarded_frames'):
            self.discarded_frames.clear()
            if hasattr(self, 'view') and hasattr(self.view, 'update_discarded_frames_display'):
                self.view.update_discarded_frames_display(set())
            logger.info("discarded_frames cleared.")

        self.object_colors.clear()
        logger.info("object_colors cache cleared.")

        # Reset review slider position and cut offset so subsequent pose/bbox
        # clicks anchor at frame 0 of a fresh propagation session (prevents a
        # stale mid-frame review position from making the next propagate
        # anchor pose queries to the wrong frame).
        self.review_current_frame = 0
        self.cut_start_frame = 0
        if hasattr(self, 'view') and hasattr(self.view, 'review_frame_slider'):
            try:
                self.view.review_frame_slider.set(0)
            except Exception:
                pass
        # Seek video to frame 0 so the display matches the reset state.
        if self.cap is not None:
            try:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, first_frame = self.cap.read()
                if ret:
                    self.current_cv_frame = first_frame
                    self.current_frame_idx_conceptual = 0
            except Exception as _seek_err:
                logger.debug(f"Clear: video seek to 0 failed: {_seek_err}")

        self.app_state = "IDLE"
        if hasattr(self, 'view') and hasattr(self.view, 'enable_review_controls'):
            self.view.enable_review_controls(False)
        logger.info("app_state reset to IDLE, review controls disabled.")

        self._update_obj_id_info_label()
        if self.current_cv_frame is not None: self._display_cv_frame_on_view(self.current_cv_frame)
        self.update_status("All objects cleared.")
        self._update_interaction_status_and_label()

    def _get_object_color(self, obj_id, for_tkinter_hex=False):
        obj_id_key = get_hashable_obj_id(obj_id)
        if obj_id_key not in self.object_colors:
            import matplotlib.pyplot as _plt  # lazy import: only needed when a new color is allocated
            cmap = _plt.get_cmap("tab10")
            color_idx = (obj_id_key - 1) % cmap.N if obj_id_key > 0 else abs(obj_id_key) % cmap.N
            self.object_colors[obj_id_key] = tuple(int(c * 255) for c in cmap(color_idx)[:3])

        color_tuple = self.object_colors.get(obj_id_key, (128, 128, 128))
        return rgb_to_tkinter_hex(color_tuple) if for_tkinter_hex else color_tuple

    def _perform_sam_tracking_for_frame(self, frame_bgr, frame_num):
        logger.debug(f"SAM3 tracking start. frame: {frame_num}")
        current_sam_masks_for_display = {}
        current_sam_masks_for_labeling = {}

        if self.tracker_model is None or self.inference_session is None:
            logger.debug(f"Frame {frame_num}: SAM3 tracker not ready. Skipping tracking.")
            return self._get_current_masks_for_display(), {
                k: v.copy() for k, v in self.tracked_objects.items()
                if "last_mask" in v and v["last_mask"] is not None and k not in self.suppressed_sam_ids
            }

        if self.just_reset_sam:
            logger.info(f"Frame {frame_num}: Right after SAM3 reset. Skipping tracking.")
            self.just_reset_sam = False
            return self._get_current_masks_for_display(), {
                k: v.copy() for k, v in self.tracked_objects.items()
                if "last_mask" in v and v["last_mask"] is not None and k not in self.suppressed_sam_ids
            }

        if not self.tracked_objects:
            logger.debug(f"Frame {frame_num}: tracked_objects empty. Skipping tracking.")
            return current_sam_masks_for_display, current_sam_masks_for_labeling

        track_successful_this_frame = False
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            inputs = self.tracker_processor(images=frame_pil, device=self.device, return_tensors="pt")

            frame_tensor = inputs.pixel_values[0]
            if self.model_dtype == torch.float32 and frame_tensor.dtype != torch.float32:
                frame_tensor = frame_tensor.to(dtype=torch.float32)

            with torch.inference_mode():
                model_outputs = self.tracker_model(
                    inference_session=self.inference_session,
                    frame=frame_tensor,
                )

            processed_outputs = self.tracker_processor.post_process_masks(
                [model_outputs.pred_masks],
                original_sizes=inputs.original_sizes,
                binarize=False
            )[0]

            track_successful_this_frame = True

            if not self.is_tracking_ever_started:
                self.is_tracking_ever_started = True
                logger.info(f"Frame {frame_num}: First SAM3 tracking success.")

            self.last_active_tracked_sam_ids.clear()
            target_h, target_w = frame_bgr.shape[:2]
            pil_size_for_mask = (target_w, target_h)
            target_shape = (pil_size_for_mask[1], pil_size_for_mask[0])

            apply_closing = self.sam_apply_closing_var.get()
            closing_kernel = self.sam_closing_kernel_size_var.get()
            erosion_k = self.erosion_kernel_size.get()
            erosion_i = self.erosion_iterations.get()
            min_bbox_area = self.min_bbox_area_for_reprompt.get()

            if not hasattr(self, 'current_confidence_masks'):
                self.current_confidence_masks = {}

            tracked_obj_ids = list(self.inference_session.obj_ids) if hasattr(self.inference_session, 'obj_ids') else list(self.tracked_objects.keys())

            for i, obj_id in enumerate(tracked_obj_ids):
                if obj_id in self.suppressed_sam_ids:
                    continue

                self.last_active_tracked_sam_ids.add(obj_id)

                if i < processed_outputs.shape[0]:
                    mask_tensor = processed_outputs[i]
                    mask_np = mask_tensor.cpu().numpy()

                    conf_mask_raw = np.squeeze(mask_np)
                    if conf_mask_raw.ndim > 2:
                        conf_mask_raw = conf_mask_raw[0]
                    if conf_mask_raw.shape != target_shape:
                        conf_mask_resized = cv2.resize(conf_mask_raw, pil_size_for_mask, interpolation=cv2.INTER_LINEAR)
                    else:
                        conf_mask_resized = conf_mask_raw
                    self.current_confidence_masks[obj_id] = conf_mask_resized

                    mask_processed = process_sam_mask(
                        mask_np,
                        pil_size_for_mask,
                        apply_closing=apply_closing,
                        closing_kernel_size=closing_kernel
                    )

                    if mask_processed is not None:
                        current_sam_masks_for_display[obj_id] = mask_processed
                        current_tracked_data = self.tracked_objects.get(obj_id)

                        if current_tracked_data:
                            current_tracked_data["last_mask"] = mask_processed
                            current_sam_masks_for_labeling[obj_id] = {
                                'last_mask': mask_processed.copy(),
                                'custom_label': current_tracked_data.get('custom_label', ''),
                                'points_for_reprompt': list(current_tracked_data.get('points_for_reprompt', [])),
                                'initial_bbox_prompt': current_tracked_data.get('initial_bbox_prompt'),
                            }
                        else:
                            logger.warning(f"SAM3 tracking result ID {obj_id} not in tracked_objects. Adding temp.")
                            initial_bbox = get_bbox_from_mask(mask_processed, erosion_k, erosion_i, min_bbox_area)
                            temp_data = {
                                "last_mask": mask_processed,
                                "custom_label": f"Tracked_{obj_id}",
                                "points_for_reprompt": [],
                                "initial_bbox_prompt": initial_bbox
                            }
                            self.tracked_objects[obj_id] = temp_data
                            current_sam_masks_for_labeling[obj_id] = {
                                'last_mask': mask_processed.copy(),
                                'custom_label': temp_data.get('custom_label', ''),
                                'points_for_reprompt': list(temp_data.get('points_for_reprompt', [])),
                                'initial_bbox_prompt': temp_data.get('initial_bbox_prompt'),
                            }
                    else:
                        logger.warning(f"SAM3 tracking ObjID {obj_id} mask processing failed.")

        except RuntimeError as e:
            track_successful_this_frame = False
            error_msg = str(e)
            logger.error(f"SAM3 tracking RuntimeError (frame {frame_num}): {e}")

            if "CUDA out of memory" in error_msg or "out of memory" in error_msg.lower():
                self._tracking_fatal_error = f"GPU OOM error - frame {frame_num}"
            else:
                self._tracking_fatal_error = f"Tracking error (frame {frame_num}): {error_msg[:100]}"

        except Exception as e:
            track_successful_this_frame = False
            logger.error(f"SAM3 tracking exception (frame {frame_num}): {e}")
            self._tracking_fatal_error = f"Tracking error (frame {frame_num}): {str(e)[:100]}"

        if not track_successful_this_frame and self.is_tracking_ever_started:
            logger.warning(f"Frame {frame_num}: SAM3 tracking failed. Stopping propagation.")
            if not hasattr(self, '_tracking_fatal_error') or not self._tracking_fatal_error:
                self._tracking_fatal_error = f"SAM3 tracking failed (frame {frame_num})"

        logger.debug(f"SAM3 tracking complete. frame: {frame_num}")
        return current_sam_masks_for_display, current_sam_masks_for_labeling

    def init_pcs_streaming_session(self, text_prompt):
        from util import pcs_controller
        return pcs_controller.init_pcs_streaming_session(self, text_prompt)

    def _perform_pcs_streaming_tracking(self, frame_bgr, frame_num):
        from util import pcs_controller
        return pcs_controller.perform_pcs_streaming_tracking(self, frame_bgr, frame_num)

    def init_pcs_session_with_single_frame(self):
        from util import pcs_controller
        return pcs_controller.init_pcs_session_with_single_frame(self)

    def detect_objects_with_pcs(self, text_prompt, frame_idx=0):
        from util import pcs_controller
        return pcs_controller.detect_objects_with_pcs(self, text_prompt, frame_idx=frame_idx)

    def _on_custom_save_toggle(self):
        self.view.update_custom_save_options_state()

    def _on_batch_mode_toggle(self):
        self.view.update_batch_options_state()

    def select_custom_save_dir(self):
        dir_path = filedialog.askdirectory(title="Select Save Folder", parent=self.root)
        if dir_path:
            self.custom_save_dir_var.set(dir_path)

    def select_batch_source_dir(self):
        dir_path = filedialog.askdirectory(title="Select Batch Video Folder", parent=self.root)
        if dir_path:
            self.batch_source_dir_var.set(dir_path)

    def _load_next_batch_video(self):
        if not self.batch_processing_mode_var.get() or not self.batch_video_files:
            logger.info("Batch list is empty or mode is disabled.")
            self.is_batch_running = False
            self.view.set_ui_element_state("btn_start_batch", tk.NORMAL)
            return False

        self.batch_current_index += 1
        if self.batch_current_index >= len(self.batch_video_files):
            logger.info("Batch processing for all video files completed.")
            messagebox.showinfo("Complete", "Batch processing for all video files completed.", parent=self.root)
            self.batch_processing_mode_var.set(False)
            self._on_batch_mode_toggle()
            self.is_batch_running = False
            self.view.set_ui_element_state("btn_start_batch", tk.NORMAL)
            return False

        next_video_path = self.batch_video_files[self.batch_current_index]
        logger.info(f"Batch: Loading next video... ({self.batch_current_index + 1}/{len(self.batch_video_files)}) - {next_video_path}")

        self._release_video_capture()
        self._reset_internal_states_for_new_source()

        self.video_source_path = next_video_path
        self.cap = cv2.VideoCapture(self.video_source_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open next video: {self.video_source_path}", parent=self.root)
            return self._load_next_batch_video()

        self.video_total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_resolution = f"{width}x{height}"
        self.video_display_name = os.path.basename(self.video_source_path)

        self.info_video_name_var.set(self.video_display_name)
        self.info_video_resolution_var.set(self.video_resolution)
        self.info_video_total_frames_var.set(f"{self.video_total_frames} frames")
        self.info_video_fps_var.set(f"{self.video_fps:.2f} FPS")
        self.info_batch_progress_var.set(f"{self.batch_current_index + 1} / {len(self.batch_video_files)}")
        self.update_status(f"Batch: {os.path.basename(next_video_path)} loaded. Detect objects then press 'Start Propagation'.")
        self.playback_paused = True

        max_slider_frame = self.video_total_frames - 1 if self.video_total_frames > 0 else 0
        self.review_current_frame = 0
        self.view.update_review_slider_range(max_slider_frame)
        self.view.review_frame_slider.set(0)
        self.view.update_review_frame_info(0, max_slider_frame)

        ret, frame_bgr = self.cap.read()
        if ret:
            self.current_cv_frame = frame_bgr.copy()
            self.current_frame_idx_conceptual = 0
            self._display_cv_frame_on_view(frame_bgr)
            self._update_interaction_status_and_label()
        else:
            messagebox.showerror("Error", f"Failed to read first frame from video: {self.video_source_path}", parent=self.root)
            self._release_video_capture()
            return self._load_next_batch_video()

        return True

    def _execute_pcs_with_exemplars(self):
        from util import pcs_controller
        return pcs_controller.execute_pcs_with_exemplars(self)

    def execute_pcs_detection(self):
        from util import pcs_controller
        return pcs_controller.execute_pcs_detection(self)

    def _auto_disable_polygon_mode(self):
        """Auto-disable polygon mode if active. Called before propagation start/resume."""
        if self.polygon_mode_active:
            self.polygon_mode_active = False
            self.polygon_points = []
            if self.view and hasattr(self.view, 'update_polygon_mode_ui'):
                self.view.update_polygon_mode_ui(False)
            logger.info("Polygon mode auto-disabled before propagation")

    def start_propagation(self):
        # Auto-disable polygon mode before starting/resuming
        self._auto_disable_polygon_mode()

        # Resume from pause
        if self.propagation_paused:
            self.propagation_paused = False
            self.propagation_pause_event.set()
            self.app_state = "PROPAGATING"
            self.view.set_propagate_button_states(is_propagating=True)
            if self.dlmi_pending_injection:
                self.update_status("Propagation resumed. DLMI injection will apply on next frame.")
            else:
                self.update_status("Propagation resumed.")
            logger.info("Propagation resumed from pause.")
            return

        current_mode = self.prompt_mode_var.get()

        if current_mode == "PCS_IMAGE":
            if not self.pcs_text_prompt_var.get().strip():
                messagebox.showwarning("Info", "In PCS(per-image) mode, please enter a text prompt.", parent=self.root)
                return
        elif not self.tracked_objects:
            messagebox.showwarning("Info", "Define objects first (PCS or PVS mode).", parent=self.root)
            return

        if current_mode == "PCS":
            if self.pcs_inference_session is None:
                messagebox.showwarning("Info", "PCS session not initialized. Perform text detection first.", parent=self.root)
                return
        elif current_mode in ("PVS", "PVS_CHUNK"):
            if self.inference_session is None:
                if not self._init_inference_session():
                    messagebox.showwarning("Info", "SAM3 session initialization failed.", parent=self.root)
                    return

        self.app_state = "PROPAGATING"
        self.propagation_stop_requested = False
        self.propagation_paused = False
        self.propagation_pause_event.set()
        self.propagated_results = {}
        self.propagation_progress = 0

        if hasattr(self, 'object_prompt_history'):
            self.object_prompt_history.clear()
            logger.info("Propagation start: object_prompt_history cleared.")

        self._snapshot_pose_queries_and_hide()

        self.view.set_propagate_button_states(is_propagating=True)
        self.view.update_propagate_progress(0, "Starting propagation...")

        self.processing_thread = threading.Thread(target=self._propagate_thread, daemon=True)
        self.processing_thread.start()

    def _snapshot_pose_queries_and_hide(self):
        from util import pose_controller
        return pose_controller.snapshot_pose_queries_and_hide(self)

    def _propagate_thread(self):
        try:
            total_frames = int(self.video_total_frames)
            if total_frames <= 0:
                total_frames = 1000

            start_frame = getattr(self, 'cut_start_frame', 0)
            actual_total_frames = total_frames - start_frame

            current_mode = self.prompt_mode_var.get()

            if self.sam2_enabled_var.get() and self.sam2_model is not None:
                if not self.sam2_tracking_enabled_var.get():
                    logger.info("SAM2 enabled state starting SAM3 propagation - transferring SAM2 masks to SAM3")
                    self.transfer_sam2_masks_to_sam3_and_unload()
                    self.sam2_enabled_var.set(False)
                    self.root.after(0, lambda: self.view._update_sam2_ui_state(enabled=False))

            if self.sam2_tracking_enabled_var.get() and self.sam2_model is not None:
                logger.info("Starting propagation in SAM2 tracking mode")
                if not self.sam2_masks and self.tracked_objects:
                    self._transfer_sam3_masks_to_sam2()
                propagation_controller.propagate_sam2_mode(self, start_frame, actual_total_frames)
            elif current_mode == "PCS":
                propagation_controller.propagate_pcs_mode(self, start_frame, actual_total_frames)
            elif current_mode == "PCS_IMAGE":
                propagation_controller.propagate_pcs_image_mode(self, start_frame, actual_total_frames)
            elif current_mode == "PVS_CHUNK":
                propagation_controller.propagate_pvs_chunk_mode(self, start_frame, actual_total_frames)
            else:
                propagation_controller.propagate_pvs_mode(self, start_frame, actual_total_frames)

            self.root.after(0, self._on_propagation_finished)

        except Exception as e:
            logger.exception("Propagation error:")
            self.root.after(0, self.update_status, f"Propagation error: {e}")
            self.root.after(0, self._on_propagation_finished)

    def _perform_pcs_single_image_detection(self, frame_bgr, text_prompt, frame_idx):
        from util import pcs_controller
        return pcs_controller.perform_pcs_single_image_detection(self, frame_bgr, text_prompt, frame_idx)

    def toggle_discard_current_frame(self):
        frame_idx = self.review_current_frame

        if frame_idx in self.discarded_frames:
            self.discarded_frames.remove(frame_idx)
            self.update_status(f"Frame {frame_idx} discard cancelled")
            self.view.update_discard_button_state(False)
        else:
            self.discarded_frames.add(frame_idx)
            self.update_status(f"Frame {frame_idx} marked for discard (excluded from save)")
            self.view.update_discard_button_state(True)

        self.view.update_discarded_frames_display(self.discarded_frames)

    def _on_propagation_finished(self):
        self.propagation_paused = False
        self.propagation_pause_event.set()
        self.dlmi_pending_injection = False
        self.dlmi_pending_masks = {}
        self.app_state = "REVIEWING"
        self.view.set_propagate_button_states(is_propagating=False)
        self.view.enable_review_controls(True)

        # Auto-disable "add" toggles so user goes into plain review mode,
        # not an accidental add-pose or add-polygon state.
        self._auto_disable_polygon_mode()
        try:
            if hasattr(self, 'pose_add_mode_var') and self.pose_add_mode_var.get():
                self.pose_add_mode_var.set(False)
            if hasattr(self, 'pose_chain_mode_var') and self.pose_chain_mode_var.get():
                self.pose_chain_mode_var.set(False)
        except Exception:
            pass

        cut_offset = getattr(self, 'cut_start_frame', 0)
        remaining_frames = self.video_total_frames - cut_offset
        max_slider_frame = remaining_frames - 1 if remaining_frames > 0 else 0
        propagated_count = len(self.propagated_results) if self.propagated_results else 0

        if propagated_count > 0:
            max_propagated_frame = max(self.propagated_results.keys())
            self.view.update_review_slider_range(max_slider_frame)
            self.view.update_review_frame_info(cut_offset, max_slider_frame)
            self.view.update_propagate_progress(100, f"Propagation complete: {propagated_count} frames (total: {remaining_frames})")
            self.update_status(f"Propagation complete. {propagated_count} frames processed. {remaining_frames} frames available for review.")
        else:
            self.view.update_review_slider_range(max_slider_frame)
            self.view.update_propagate_progress(0, "No propagation results")
            self.update_status("No propagation results. All frames can be reviewed.")

        try:
            tap_on = bool(self.pose_tapnext_enabled_var.get()) if hasattr(self, 'pose_tapnext_enabled_var') else False
            has_pose_queries = (
                any(obj.get('pose_points') for obj in self.tracked_objects.values())
                or bool(getattr(self, '_pose_query_seeds', None))
            )
            if propagated_count > 0 and tap_on and has_pose_queries:
                logger.info("TAPNext++ auto-run after propagation (toggle on).")
                self.update_status("Propagation done. Preparing TAPNext++ pose tracking...")
                self.root.update_idletasks()
                self.run_tapnext_post_process()
            elif propagated_count > 0 and tap_on and not has_pose_queries:
                logger.info("TAPNext++ toggle on but no pose points in tracked_objects; skipped.")
                self.update_status("TAPNext++ on, but no initial pose points were set \u2014 nothing to track.")
        except Exception as _tap_auto_err:
            logger.debug(f"TAPNext auto-run skipped: {_tap_auto_err}")

        try:
            if getattr(self, 'pose_automatch_var', None) and self.pose_automatch_var.get():
                merged = self._try_automatch_pose_to_segments()
                if merged > 0:
                    logger.info(f"Auto-match after propagation: {merged} merges.")
        except Exception as _am_err:
            logger.debug(f"Auto-match after propagation skipped: {_am_err}")

        # Restore pose seeds into frame 0 if TAPNext did NOT populate per-frame
        # pose data. Otherwise the user has no way to see/select pose points
        # after propagation (they were hidden at start_propagation).
        try:
            seeds = getattr(self, '_pose_query_seeds', None)
            if seeds and propagated_count > 0:
                has_pose_anywhere = any(
                    any(d.get('pose_points') for d in r.get('masks', {}).values() if isinstance(d, dict))
                    for r in self.propagated_results.values()
                )
                if not has_pose_anywhere:
                    first_key = min(self.propagated_results.keys())
                    frame0_masks = self.propagated_results[first_key].setdefault('masks', {})
                    for oid, seed in seeds.items():
                        slot = frame0_masks.setdefault(int(oid), {})
                        slot['pose_points'] = [dict(p) for p in seed.get('points', [])]
                        slot['pose_edges'] = [list(e) for e in seed.get('edges', [])]
                        if seed.get('pose_class'):
                            slot['pose_class'] = seed['pose_class']
                        if seed.get('custom_label'):
                            slot['custom_label'] = seed['custom_label']
                    logger.info(f"Pose seeds restored into frame {first_key} for review.")
                # Refresh current slider frame so tracked_objects reflects pose
                self.on_review_frame_change(getattr(self, 'review_current_frame', 0))
        except Exception as _seed_restore_err:
            logger.debug(f"Pose seed restore skipped: {_seed_restore_err}")

    def _run_pcs_review_mode_async(self):
        import threading
        from util.autolabel_workflow import run_pcs_review_mode

        def progress_callback(current, total):
            progress = int(current / total * 100)
            self.root.after(0, self.view.update_propagate_progress, progress,
                            f"Review mode: {current}/{total} frames analyzing...")

        def run_review():
            self.root.after(0, self.update_status, "PCS review mode running...")
            run_pcs_review_mode(self, progress_callback)
            self.root.after(0, self.update_status, "PCS review mode complete")
            self.root.after(0, self.view.update_propagate_progress, 100, "Review mode complete")

        thread = threading.Thread(target=run_review, daemon=True)
        thread.start()

    def pause_propagation(self):
        if not self.propagation_paused and self.app_state == "PROPAGATING":
            self.propagation_paused = True
            self.propagation_pause_event.clear()
            self.app_state = "PAUSED"
            self.view.set_propagate_button_states_paused()
            self.update_status("Propagation paused. Modify objects and/or inject DLMI, then resume.")
            logger.info(f"Propagation paused at frame {self.propagation_current_frame_idx}.")

    def stop_propagation(self):
        self.propagation_stop_requested = True
        if self.propagation_paused:
            self.propagation_paused = False
            self.propagation_pause_event.set()  # Unblock thread so it can check stop flag
        self.update_status("Propagation stop requested...")

    def on_review_frame_change(self, frame_idx):
        self.review_current_frame = frame_idx

        cut_offset = getattr(self, 'cut_start_frame', 0)
        actual_frame_idx = frame_idx + cut_offset

        remaining_frames = self.video_total_frames - cut_offset
        max_slider_frame = remaining_frames - 1 if remaining_frames > 0 else 0

        if frame_idx in self.propagated_results:
            result = self.propagated_results[frame_idx]
            frame_bgr = result['frame']
            masks = result['masks']

            self.current_cv_frame = frame_bgr

            masks_for_display = {obj_id: data['last_mask'] for obj_id, data in masks.items() if 'last_mask' in data}

            # Sync BOTH last_mask AND pose info into tracked_objects so that
            # any later redraw (e.g. notebook tab-change handler that calls
            # _get_current_masks_for_display) reflects THIS frame, not the last
            # propagated frame. Objects not present in this frame's masks get
            # their last_mask/pose fields removed to avoid stale display.
            frame_obj_ids = set(masks.keys())
            for _oid in list(self.tracked_objects.keys()):
                if _oid in frame_obj_ids:
                    _data = masks[_oid]
                    if 'last_mask' in _data and _data['last_mask'] is not None:
                        self.tracked_objects[_oid]['last_mask'] = _data['last_mask']
                    else:
                        self.tracked_objects[_oid].pop('last_mask', None)
                    _pts = _data.get('pose_points')
                    if _pts is not None:
                        self.tracked_objects[_oid]['pose_points'] = _pts
                        if 'pose_edges' in _data:
                            self.tracked_objects[_oid]['pose_edges'] = _data['pose_edges']
                    else:
                        self.tracked_objects[_oid].pop('pose_points', None)
                        self.tracked_objects[_oid].pop('pose_edges', None)
                elif self.tracked_objects[_oid].get('mid_session_added'):
                    # Preserve mid-session-added objects. Their last_mask is
                    # only registered at the frame where they were created;
                    # keep it intact so the next DLMI/propagation cycle can
                    # still use it.
                    continue
                else:
                    # Object doesn't appear in this frame's masks; strip its
                    # visual state so it doesn't ghost across frames.
                    self.tracked_objects[_oid].pop('last_mask', None)
                    self.tracked_objects[_oid].pop('pose_points', None)
                    self.tracked_objects[_oid].pop('pose_edges', None)
            for _oid in frame_obj_ids:
                if _oid not in self.tracked_objects:
                    self.tracked_objects[_oid] = {}
                    _data = masks[_oid]
                    if 'last_mask' in _data and _data['last_mask'] is not None:
                        self.tracked_objects[_oid]['last_mask'] = _data['last_mask']
                    if _data.get('pose_points') is not None:
                        self.tracked_objects[_oid]['pose_points'] = _data['pose_points']
                        if 'pose_edges' in _data:
                            self.tracked_objects[_oid]['pose_edges'] = _data['pose_edges']
                    if _data.get('custom_label'):
                        self.tracked_objects[_oid]['custom_label'] = _data['custom_label']

            self._display_cv_frame_on_view(frame_bgr, masks_for_display)
        else:
            # During propagation/pause, propagation thread owns self.cap.
            # Seeking/reading from this thread races with ffmpeg and triggers
            # "Assertion fctx->async_lock failed" crashes. Show status only.
            if self.app_state in ("PAUSED", "PROPAGATING"):
                self.update_status(
                    f"Frame {actual_frame_idx} not yet processed (propagation in progress). "
                    f"Reviewable once that frame is reached."
                )
            elif self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_idx)
                ret, frame_bgr = self.cap.read()
                if ret:
                    self.current_cv_frame = frame_bgr
                    self._display_cv_frame_on_view(frame_bgr, {})
                else:
                    logger.warning(f"Review: Frame {actual_frame_idx} read failed")
            else:
                logger.warning(f"Review: Video capture not open (frame {actual_frame_idx})")

        self.view.update_review_frame_info(actual_frame_idx, max_slider_frame)

        is_discarded = frame_idx in self.discarded_frames
        self.view.update_discard_button_state(is_discarded)

    def cut_and_repropagate(self):
        from util import cut_workflow
        return cut_workflow.cut_and_repropagate(self)

    def _cut_save_frames_0_to_n(self, slider_idx, save_labels, current_offset):
        from util import cut_workflow
        return cut_workflow.save_frames_0_to_n(self, slider_idx, save_labels, current_offset)

    def _pose_labels_subdir(self):
        from util import cut_workflow
        return cut_workflow.pose_labels_subdir(self)

    def _cut_reset_state_and_seek(self, next_start_frame):
        from util import cut_workflow
        return cut_workflow.reset_state_and_seek(self, next_start_frame)

    def _dlmi_mini_propagate_n_to_n1(self, frame_n_bgr, frame_n_plus_1_bgr, obj_id_to_mask_label):
        from util import cut_workflow
        return cut_workflow.dlmi_mini_propagate_n_to_n1(
            self, frame_n_bgr, frame_n_plus_1_bgr, obj_id_to_mask_label
        )

    def open_pose_settings(self):
        from util import pose_controller
        return pose_controller.open_pose_settings(self)

    def _default_pose_class_name(self):
        from util import pose_controller
        return pose_controller.default_pose_class_name(self)

    def add_pose_point_at(self, img_x, img_y):
        from util import pose_controller
        return pose_controller.add_pose_point_at(self, img_x, img_y)

    def toggle_pose_point_selection(self, obj_id, kpt_idx):
        from util import pose_controller
        return pose_controller.toggle_pose_point_selection(self, obj_id, kpt_idx)

    def clear_pose_selection(self):
        from util import pose_controller
        return pose_controller.clear_pose_selection(self)

    def _update_pose_action_button_states(self):
        from util import pose_controller
        return pose_controller.update_pose_action_button_states(self)

    def new_pose_object(self):
        from util import pose_controller
        return pose_controller.new_pose_object(self)

    def _refresh_pose_class_menu(self):
        from util import pose_controller
        return pose_controller.refresh_pose_class_menu(self)

    def _update_pose_class_display(self):
        from util import pose_controller
        return pose_controller.update_pose_class_display(self)

    def _on_pose_class_selected(self):
        from util import pose_controller
        return pose_controller.on_pose_class_selected(self)

    def delete_selected_object_pose(self):
        from util import pose_controller
        return pose_controller.delete_selected_object_pose(self)

    def select_pose_chain_at(self, img_x, img_y):
        from util import pose_controller
        return pose_controller.select_pose_chain_at(self, img_x, img_y)

    def toggle_selected_pose_visibility(self):
        from util import pose_controller
        return pose_controller.toggle_selected_pose_visibility(self)

    def _automatch_classify_pose_object(self, oid, force=False):
        from util import pose_controller
        return pose_controller.automatch_classify_pose_object(self, oid, force=force)

    def _automatch_all_new_pose_objects(self):
        from util import pose_controller
        return pose_controller.automatch_all_new_pose_objects(self)

    def _try_automatch_pose_to_segments(self, min_ratio=0.7):
        from util import pose_controller
        return pose_controller.try_automatch_pose_to_segments(self, min_ratio=min_ratio)

    def reassign_selected_pose_idx(self):
        from util import pose_controller
        return pose_controller.reassign_selected_pose_idx(self)

    def connect_selected_pose_points(self):
        from util import pose_controller
        return pose_controller.connect_selected_pose_points(self)

    def _default_pose_models_dir(self):
        from util import pose_controller
        return pose_controller.default_pose_models_dir(self)

    def _ensure_tapnext_ckpt(self, interactive=True):
        from util import pose_controller
        return pose_controller.ensure_tapnext_ckpt(self, interactive=interactive)

    def _open_download_dialog(self, url, dest):
        from util import ui_dialogs
        return ui_dialogs.open_download_dialog(self, url, dest)

    def _open_loading_dialog(self, title, subtitle):
        from util import ui_dialogs
        return ui_dialogs.open_loading_dialog(self, title, subtitle)

    def _get_pose_tracker(self):
        from util import pose_controller
        return pose_controller.get_pose_tracker(self)

    def _offer_pose_fallback(self, reason=""):
        from util import pose_controller
        return pose_controller.offer_pose_fallback(self, reason=reason)

    def _get_yolo_pose_detector(self):
        from util import pose_controller
        return pose_controller.get_yolo_pose_detector(self)

    def run_yolo_pose_detect(self):
        from util import pose_controller
        return pose_controller.run_yolo_pose_detect(self)

    def run_tapnext_post_process(self):
        from util import pose_controller
        return pose_controller.run_tapnext_post_process(self)

    def delete_selected_pose_points(self):
        from util import pose_controller
        return pose_controller.delete_selected_pose_points(self)

    def _seed_session_with_masks(self, oid_to_mask):
        from util import cut_workflow
        return cut_workflow.seed_session_with_masks(self, oid_to_mask)

    def cut_and_dlmi_propagate(self):
        from util import cut_workflow
        return cut_workflow.cut_and_dlmi_propagate(self)

    def cut_and_load_labels(self):
        from util import cut_workflow
        return cut_workflow.cut_and_load_labels(self)

    def repropagate_all(self):
        from util import cut_workflow
        return cut_workflow.repropagate_all(self)

    def confirm_and_save_labels(self):
        from util import save_controller
        return save_controller.confirm_and_save_labels(self)

    def _on_save_finished(self, total_frames):
        from util import save_controller
        return save_controller.on_save_finished(self, total_frames)

    def _handle_sam_reset(self):
        from util import sam2_manager
        return sam2_manager.handle_sam_reset(self)

    def load_sam2_model_async(self):
        from util import sam2_manager
        return sam2_manager.load_sam2_model_async(self)

    def unload_sam2_model(self):
        from util import sam2_manager
        return sam2_manager.unload_sam2_model(self)

    def _transfer_sam3_masks_to_sam2(self):
        from util import sam2_manager
        return sam2_manager.transfer_sam3_masks_to_sam2(self)

    def transfer_sam2_masks_to_sam3_and_unload(self):
        from util import sam2_manager
        return sam2_manager.transfer_sam2_masks_to_sam3_and_unload(self)

    def _get_current_frame_masks(self):
        from util import sam2_manager
        return sam2_manager.get_current_frame_masks(self)

    def _reinit_sam3_session_with_masks(self):
        from util import sam2_manager
        return sam2_manager.reinit_sam3_session_with_masks(self)

    def _handle_sam2_prompt(self, prompt_type, coords, label=None,
                            proposed_obj_id_for_new=None, target_existing_obj_id=None,
                            custom_label=None):
        from util import sam2_manager
        return sam2_manager.handle_sam2_prompt(
            self, prompt_type, coords, label=label,
            proposed_obj_id_for_new=proposed_obj_id_for_new,
            target_existing_obj_id=target_existing_obj_id,
            custom_label=custom_label,
        )

    def refine_mask_with_sam2(self, obj_id: int, input_points=None, input_labels=None, input_boxes=None):
        from util import sam2_manager
        return sam2_manager.refine_mask_with_sam2(
            self, obj_id, input_points=input_points,
            input_labels=input_labels, input_boxes=input_boxes,
        )

    def _on_closing_window_confirm(self):
        if self._is_any_special_mode_active():
            if messagebox.askokcancel("Exit Confirmation", f"Currently interacting with objects. Do you really want to exit?", parent=self.root): self._perform_cleanup_and_destroy()
            return
        if self.autolabel_active :
            if messagebox.askokcancel("Exit Confirmation", "Auto labeling is in progress. Do you really want to exit?", parent=self.root): self._perform_cleanup_and_destroy()
        else: self._perform_cleanup_and_destroy()

    def _perform_cleanup_and_destroy(self):
        logger.info("Performing application shutdown procedure..."); self.autolabel_active = False; self.playback_paused = True
        self.interaction_correction_pending = None; self.problematic_highlight_active_sam_id = None; self.reassign_bbox_mode_active_sam_id = None
        if self.processing_thread and self.processing_thread.is_alive():
            logger.info("Waiting for background processing thread join (max 1 second)..."); self.processing_thread.join(timeout=1.0)
            if self.processing_thread and self.processing_thread.is_alive(): logger.warning("Processing thread join timeout.")
        self.processing_thread = None; self._release_video_capture(); logger.info("Destroying root window.")
        if self.root:
            try: self.root.destroy()
            except tk.TclError as e: logger.warning(f"TclError while destroying root window: {e}")
            self.root = None

if __name__ == "__main__":
    root = tk.Tk()
    app = SAM3AutolabelApp(root)
    root.geometry("1100x850")
    root.minsize(960, 700)
    root.mainloop()
