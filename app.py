import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os
import threading
import time
import numpy as np

import torch
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import logging
import contextlib
import shutil
from collections import defaultdict
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
from util import sam2_controller
from util import dlmi_controller
from util import pcs_controller
from util import polygon_controller
from util import frame_renderer

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
AUTOLABEL_FOLDER = "results"
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
        self.root.title("DLMI-SAM labeler v1.0")

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
        self.low_level_mask_injected = False
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
        
        self.root.after(100, self._init_models)
        logger.info("SAM3AutolabelApp initialized.")

    def _canvas_to_image_coords(self, canvas_x, canvas_y):
        if not self.current_frame_pil_rgb_original: 
            logger.warning("_canvas_to_image_coords: current_frame_pil_rgb_original is None")
            return canvas_x, canvas_y 

        if self.scale_x == 0 or self.scale_y == 0: 
             logger.warning("_canvas_to_image_coords: scale_x/y not set or zero")
             return canvas_x, canvas_y

        orig_w, orig_h = self.current_frame_pil_rgb_original.size
        img_x = (canvas_x - self.offset_x) * self.scale_x
        img_y = (canvas_y - self.offset_y) * self.scale_y
        return int(max(0, min(img_x, orig_w -1))), int(max(0, min(img_y, orig_h -1)))

    def _handle_sam_prompt_wrapper(self, prompt_type, coords, label=None,
                                  proposed_obj_id_for_new=None, target_existing_obj_id=None,
                                  custom_label=None):
        if self.app_state == "PAUSED":
            self.update_status("Paused: BBox/Point prompts blocked (session protection). Use polygon mode + DLMI injection instead.")
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
        dlmi_controller.prepare_dlmi_mid_propagation(self)

    def inject_low_level_mask_prompt(self):
        dlmi_controller.inject_low_level_mask_prompt(self)

    def _install_dlmi_persistent_hooks(self):
        dlmi_controller._install_dlmi_persistent_hooks(self)

    def _remove_dlmi_persistent_hooks(self):
        dlmi_controller._remove_dlmi_persistent_hooks(self)

    def toggle_polygon_mode(self):
        polygon_controller.toggle_polygon_mode(self)

    def add_polygon_point(self, x, y):
        return polygon_controller.add_polygon_point(self, x, y)

    def undo_last_polygon_point(self):
        polygon_controller.undo_last_polygon_point(self)

    def complete_polygon_object(self):
        polygon_controller.complete_polygon_object(self)

    def input_polygon_to_sam3(self):
        polygon_controller.input_polygon_to_sam3(self)

    def cancel_polygon_mode(self):
        polygon_controller.cancel_polygon_mode(self)

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
        autolabel_workflow.load_label_file(self)

    def _apply_loaded_labels_as_polygon_masks(self, loaded_objects):
        autolabel_workflow._apply_loaded_labels_as_polygon_masks(self, loaded_objects)

    def _apply_loaded_labels_as_prompts(self, loaded_objects):
        autolabel_workflow._apply_loaded_labels_as_prompts(self, loaded_objects)

    def _apply_loaded_labels_for_pcs(self, loaded_objects):
        autolabel_workflow._apply_loaded_labels_for_pcs(self, loaded_objects)

    def _display_cv_frame_on_view(self, frame_bgr, masks_to_overlay=None, yolo_bboxes_to_draw=None):
        frame_renderer._display_cv_frame_on_view(self, frame_bgr, masks_to_overlay, yolo_bboxes_to_draw)

    def _get_current_masks_for_display(self):
        return frame_renderer._get_current_masks_for_display(self)

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
            cmap = plt.get_cmap("tab10") 
            color_idx = (obj_id_key -1) % cmap.N if obj_id_key > 0 else abs(obj_id_key) % cmap.N
            self.object_colors[obj_id_key] = tuple(int(c*255) for c in cmap(color_idx)[:3]) 
        
        color_tuple = self.object_colors.get(obj_id_key, (128,128,128))
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

            tracked_obj_ids = list(self.inference_session.obj_ids) if hasattr(self.inference_session, 'obj_ids') else list(self.tracked_objects.keys())

            for i, obj_id in enumerate(tracked_obj_ids):
                if obj_id in self.suppressed_sam_ids:
                    continue

                self.last_active_tracked_sam_ids.add(obj_id)

                if i < processed_outputs.shape[0]:
                    mask_tensor = processed_outputs[i]
                    mask_np = mask_tensor.cpu().numpy()

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
        return pcs_controller.init_pcs_streaming_session(self, text_prompt)

    def _perform_pcs_streaming_tracking(self, frame_bgr, frame_num):
        return pcs_controller._perform_pcs_streaming_tracking(self, frame_bgr, frame_num)

    def init_pcs_session_with_single_frame(self):
        return pcs_controller.init_pcs_session_with_single_frame(self)

    def detect_objects_with_pcs(self, text_prompt, frame_idx=0):
        return pcs_controller.detect_objects_with_pcs(self, text_prompt, frame_idx)

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
            batch_save_dir = os.path.abspath(self.custom_save_dir_var.get() if self.use_custom_save_path_var.get() else self.AUTOLABEL_FOLDER_val)
            messagebox.showinfo("Complete", f"Batch processing for all video files completed.\n\nSave path: {batch_save_dir}", parent=self.root)
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
        pcs_controller._execute_pcs_with_exemplars(self)

    def execute_pcs_detection(self):
        pcs_controller.execute_pcs_detection(self)

    def _auto_disable_polygon_mode(self):
        polygon_controller._auto_disable_polygon_mode(self)

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

        self.view.set_propagate_button_states(is_propagating=True)
        self.view.update_propagate_progress(0, "Starting propagation...")

        self.processing_thread = threading.Thread(target=self._propagate_thread, daemon=True)
        self.processing_thread.start()

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
        return pcs_controller._perform_pcs_single_image_detection(self, frame_bgr, text_prompt, frame_idx)

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

    def _run_pcs_review_mode_async(self):
        pcs_controller._run_pcs_review_mode_async(self)

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
            self._display_cv_frame_on_view(frame_bgr, masks_for_display)
        else:
            # During paused propagation, do NOT seek cap (propagation thread needs it)
            if self.app_state == "PAUSED":
                self.update_status(f"Paused: Frame {actual_frame_idx} not yet processed. Review only processed frames.")
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
        slider_idx = self.review_current_frame
        if slider_idx < 0:
            messagebox.showwarning("Info", "Please select a valid frame.", parent=self.root)
            return

        current_offset = getattr(self, 'cut_start_frame', 0)

        absolute_cut_frame = current_offset + slider_idx
        next_start_frame = absolute_cut_frame + 1

        frames_to_save = [f for f in self.propagated_results.keys() if f <= slider_idx]

        if next_start_frame >= self.video_total_frames:
            messagebox.showwarning("Info", "This is the last frame. No next video to cut.\nUse normal label save.", parent=self.root)
            return

        response = messagebox.askyesnocancel(
            "Cut Here",
            f"Save labels for frames 0~{slider_idx} ({len(frames_to_save)} frames),\n"
            f"and treat original video frames {next_start_frame} onwards as new video.\n\n"
            f"(Original frame numbers: {current_offset}~{absolute_cut_frame} saved, start from {next_start_frame})\n\n"
            f"Yes: Save labels then cut\n"
            f"No: Cut without saving labels\n"
            f"Cancel: Abort operation",
            parent=self.root
        )

        if response is None:
            return

        save_labels = response

        if save_labels and self.save_format_var.get() in ["yolo", "both"]:
            save_dir = self._get_save_directory()

            check_result = self._check_existing_yolo_dataset(save_dir)

            if check_result is None:
                return
            elif check_result == "new_setup":
                if not self._prompt_yolo_class_info():
                    return

                if not self._init_yolo_dataset_structure(save_dir):
                    messagebox.showerror("Error", "Failed to create YOLO dataset structure.", parent=self.root)
                    return

        saved_count = 0
        if save_labels:
            for frame_idx in sorted(frames_to_save):
                if frame_idx in self.propagated_results:
                    result = self.propagated_results[frame_idx]
                    frame_bgr = result['frame']
                    masks = result['masks']

                    if frame_bgr is not None and masks:
                        frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

                        save_format = self.save_format_var.get()
                        video_name = os.path.splitext(os.path.basename(self.video_source_path))[0] if isinstance(self.video_source_path, str) else "video"
                        actual_save_frame = current_offset + frame_idx

                        if save_format in ["yolo", "both"]:
                            from util.autolabel_workflow import save_yolo_format
                            save_yolo_format(self, frame_pil, actual_save_frame, masks, video_name)
                        if save_format in ["labelme", "both"]:
                            from util.autolabel_workflow import save_labelme_json
                            save_labelme_json(self, frame_pil, actual_save_frame, masks, video_name, is_both_mode=(save_format == "both"))

                        saved_count += 1

            logger.info(f"Cut: Saved {saved_count} frames for slider index 0~{slider_idx}. (Original frames {current_offset}~{absolute_cut_frame})")
        else:
            logger.info(f"Cut: Label saving skipped (user choice)")

        self.propagated_results = {}

        self.tracked_objects.clear()
        self.next_obj_id_to_propose = 1
        self._update_obj_id_info_label()
        self.is_tracking_ever_started = False
        self.inference_session = None
        self.pcs_inference_session = None
        self.pcs_streaming_session = None
        self.video_frames_cache = []

        self.discarded_frames.clear()
        if hasattr(self.view, 'update_discarded_frames_display'):
            self.view.update_discarded_frames_display(set())
        if hasattr(self, 'object_prompt_history'):
            self.object_prompt_history.clear()
            logger.info("Cut: object_prompt_history cleared")

        self.cut_start_frame = next_start_frame
        logger.info(f"Cut: New cut_start_frame = {self.cut_start_frame} (based on original video, saved up to {absolute_cut_frame})")

        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, next_start_frame)
            ret, frame_bgr = self.cap.read()
            if ret:
                self.current_cv_frame = frame_bgr
                self.current_frame_idx_conceptual = 0
                self._display_cv_frame_on_view(frame_bgr, {})

                if not self._init_inference_session():
                    messagebox.showerror("Error", "Failed to initialize new session", parent=self.root)
                    return

        remaining_frames = self.video_total_frames - next_start_frame
        max_slider_frame = remaining_frames - 1 if remaining_frames > 0 else 0

        self.view.update_review_slider_range(max_slider_frame)
        self.view.review_frame_slider.set(0)
        self.review_current_frame = 0

        self.view.update_review_frame_info(next_start_frame, max_slider_frame)

        self.view.btn_start_propagate.config(state=tk.NORMAL)
        self.view.enable_review_controls(False)
        self.app_state = "IDLE"
        self.view.update_propagate_progress(0, f"Cut complete (new video from original frame {next_start_frame})")

        if save_labels:
            self.update_status(
                f"Cut complete. {saved_count} frames saved (0~{absolute_cut_frame}).\n"
                f"Starting from original frame {next_start_frame} ({remaining_frames} frames remaining). Add objects and start propagation."
            )
        else:
            self.update_status(
                f"Cut complete. (Label saving skipped)\n"
                f"Starting from original frame {next_start_frame} ({remaining_frames} frames remaining). Add objects and start propagation."
            )

        self.info_video_total_frames_var.set(f"{remaining_frames} (original {absolute_cut_frame}~{self.video_total_frames-1})")

        if save_labels:
            messagebox.showinfo(
                "Cut Complete",
                f"Labels for {saved_count} frames (slider index 0~{slider_idx-1}) have been saved.\n"
                f"(Original video frames {current_offset}~{absolute_cut_frame-1})\n\n"
                f"From original frame {absolute_cut_frame}, it will be treated as a new video.\n"
                f"Add new objects and click 'Start Propagation' to proceed.",
                parent=self.root
            )
        else:
            messagebox.showinfo(
                "Cut Complete",
                f"Cut performed without saving labels.\n\n"
                f"From original frame {absolute_cut_frame}, it will be treated as a new video.\n"
                f"Add new objects and click 'Start Propagation' to proceed.",
                parent=self.root
            )

    def repropagate_all(self):
        response = messagebox.askyesno(
            "Confirm Full Re-propagation",
            "Do you want to delete all results and re-propagate from the beginning?\n\n"
            "Currently defined objects will be maintained.",
            parent=self.root
        )

        if not response:
            return

        self.propagated_results = {}
        self.cut_point_frame = None

        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame_bgr = self.cap.read()
            if ret:
                self.current_cv_frame = frame_bgr
                self._display_cv_frame_on_view(frame_bgr, self._get_current_masks_for_display())

        self.view.enable_review_controls(False)
        self.app_state = "IDLE"
        self.view.update_propagate_progress(0, "Ready for re-propagation")
        self.update_status("Re-propagation ready. Click 'Start Propagation' to begin.")

    def confirm_and_save_labels(self):
        if not self.propagated_results:
            messagebox.showwarning("Notice", "No propagation results to save.", parent=self.root)
            return

        cut_offset = getattr(self, 'cut_start_frame', 0)
        frame_indices = sorted(self.propagated_results.keys())
        actual_start = cut_offset + min(frame_indices) if frame_indices else cut_offset
        actual_end = cut_offset + max(frame_indices) if frame_indices else cut_offset

        response = messagebox.askyesno(
            "Confirm Labeling",
            f"Do you want to save labels for {len(self.propagated_results)} frames?\n\n"
            f"Original video frame range: {actual_start} ~ {actual_end}",
            parent=self.root
        )

        if not response:
            return

        if self.save_format_var.get() in ["yolo", "both"]:
            save_dir = self._get_save_directory()

            check_result = self._check_existing_yolo_dataset(save_dir)

            if check_result is None:
                return
            elif check_result == "new_setup":
                if not self._prompt_yolo_class_info():
                    return

                if os.path.exists(save_dir):
                    yaml_path = os.path.join(save_dir, "data.yaml")
                    images_dir = os.path.join(save_dir, "images")
                    labels_dir = os.path.join(save_dir, "labels")

                    if not (os.path.exists(yaml_path) and os.path.exists(images_dir) and os.path.exists(labels_dir)):
                        folder_response = messagebox.askyesnocancel(
                            "Existing Folder Found",
                            f"Folder '{save_dir}' already exists.\n\n"
                            f"Yes: Delete folder contents and create YOLO structure\n"
                            f"No: Keep existing contents and add YOLO structure\n"
                            f"Cancel: Abort operation",
                            parent=self.root
                        )

                        if folder_response is None:
                            return
                        elif folder_response:
                            try:
                                shutil.rmtree(save_dir)
                                logger.info(f"Existing folder deleted: {save_dir}")
                            except Exception as e:
                                logger.error(f"Folder deletion failed: {e}")
                                messagebox.showerror("Error", f"Folder deletion failed:\n{e}", parent=self.root)
                                return

                if not self._init_yolo_dataset_structure(save_dir):
                    messagebox.showerror("Error", "Failed to create YOLO dataset structure.", parent=self.root)
                    return

        self.app_state = "LABELING"
        self.update_status("Saving labels...")

        save_thread = threading.Thread(target=self._save_labels_thread, daemon=True)
        save_thread.start()

    def _save_labels_thread(self):
        try:
            from util.autolabel_workflow import save_yolo_format, save_labelme_json

            frame_base_name = "frame"
            if self.video_source_path and isinstance(self.video_source_path, str):
                frame_base_name = os.path.splitext(os.path.basename(self.video_source_path))[0]

            cut_offset = getattr(self, 'cut_start_frame', 0)

            frames_to_save = {
                frame_idx: result
                for frame_idx, result in self.propagated_results.items()
                if frame_idx not in self.discarded_frames
            }
            total_frames = len(frames_to_save)
            skipped_count = len(self.discarded_frames)

            if skipped_count > 0:
                logger.info(f"Label saving: {skipped_count} frames excluded by discard marking")

            saved_count = 0
            for i, (frame_idx, result) in enumerate(sorted(frames_to_save.items())):
                frame_bgr = result['frame']
                masks_data = result['masks']

                if not masks_data:
                    continue

                actual_frame_num = cut_offset + frame_idx

                frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                save_format = self.save_format_var.get()

                if save_format == "yolo":
                    save_yolo_format(self, frame_pil, actual_frame_num, masks_data, frame_base_name)
                elif save_format == "labelme":
                    save_labelme_json(self, frame_pil, actual_frame_num, masks_data, frame_base_name)
                elif save_format == "both":
                    save_yolo_format(self, frame_pil, actual_frame_num, masks_data, frame_base_name)
                    save_labelme_json(self, frame_pil, actual_frame_num, masks_data, frame_base_name, is_both_mode=True)

                saved_count += 1

                progress = int((i + 1) / total_frames * 100) if total_frames > 0 else 100
                self.root.after(0, self.view.update_propagate_progress, progress, f"Saving: {i + 1}/{total_frames} (frame {actual_frame_num})")

            self.discarded_frames.clear()
            self.root.after(0, self.view.update_discarded_frames_display, set())

            save_dir = self.AUTOLABEL_FOLDER_val
            if self.use_custom_save_path_var.get():
                save_dir = self.custom_save_dir_var.get()
            self.root.after(0, self._on_save_finished, saved_count, os.path.abspath(save_dir))

        except Exception as e:
            logger.exception("Error during label saving:")
            self.root.after(0, self.update_status, f"Error during saving: {e}")

    def _on_save_finished(self, total_frames, save_path=""):
        self.app_state = "IDLE"
        self.view.update_propagate_progress(100, f"Save complete: {total_frames} frames")
        self.update_status(f"Label saving complete! {total_frames} frames saved.")
        if hasattr(self, 'object_prompt_history'):
            self.object_prompt_history.clear()
            logger.info("Label saving complete: object_prompt_history cleared")
        messagebox.showinfo("Complete", f"Labels for {total_frames} frames have been saved.\n\nSave path: {save_path}", parent=self.root)

    def _handle_sam_reset(self):
        logger.info("SAM reset request detected.")

        try:
            if self.predictor:
                self.predictor.reset_state()
        except Exception as e:
            if 'point_inputs_per_obj' in str(e):
                logger.warning(f"SAM3 predictor.reset_state() known issue: {e} (ignored)")
            else:
                logger.warning(f"SAM3 predictor.reset_state() error: {e} (ignored)")
        return True

    def load_sam2_model_async(self):
        sam2_controller.load_sam2_model_async(self)

    def _load_sam2_model_thread(self):
        sam2_controller._load_sam2_model_thread(self)

    def _on_sam2_load_complete(self, success, error_msg=None):
        sam2_controller._on_sam2_load_complete(self, success, error_msg)

    def unload_sam2_model(self):
        sam2_controller.unload_sam2_model(self)

    def _transfer_sam3_masks_to_sam2(self):
        sam2_controller._transfer_sam3_masks_to_sam2(self)

    def transfer_sam2_masks_to_sam3_and_unload(self):
        sam2_controller.transfer_sam2_masks_to_sam3_and_unload(self)

    def _get_current_frame_masks(self):
        current_frame_idx = getattr(self, 'review_current_frame', 0)
        sam2_active = self.sam2_enabled_var.get() and self.sam2_model is not None

        if sam2_active and self.sam2_masks:
            logger.info(f"SAM2 active - Loading {len(self.sam2_masks)} masks from sam2_masks")
            return self.sam2_masks.copy()

        if current_frame_idx in self.propagated_results:
            result = self.propagated_results[current_frame_idx]
            masks_data = result.get('masks', {})
            masks = {}
            for obj_id, obj_data in masks_data.items():
                if 'last_mask' in obj_data and obj_data['last_mask'] is not None:
                    masks[obj_id] = obj_data['last_mask']
            if masks:
                logger.info(f"Loaded {len(masks)} masks from propagated_results[{current_frame_idx}]")
                return masks

        if self.sam2_masks:
            logger.info(f"Loaded {len(self.sam2_masks)} masks from sam2_masks")
            return self.sam2_masks.copy()

        masks = {}
        for obj_id, obj_data in self.tracked_objects.items():
            if 'last_mask' in obj_data and obj_data['last_mask'] is not None:
                masks[obj_id] = obj_data['last_mask']
        if masks:
            logger.info(f"Loaded {len(masks)} masks from tracked_objects")
        return masks

    def _reinit_sam3_session_with_masks(self):
        current_mode = self.prompt_mode_var.get()
        is_pcs_mode = current_mode in ("PCS", "PCS_IMAGE")

        if is_pcs_mode:
            logger.info("SAM2 deactivation in PCS mode - Auto switching to PVS mode")
            self.prompt_mode_var.set("PVS")
            self._previous_prompt_mode = "PVS"
            self.root.after(0, lambda: self.view._update_pcs_mode_ui())
            self.root.after(0, lambda: self.update_status("Switched to PVS mode due to SAM2 deactivation."))

        if self.tracker_processor is None or self.tracker_model is None:
            logger.error("SAM3 Tracker model not available.")
            return

        current_masks = {}
        for obj_id, obj_data in self.tracked_objects.items():
            if 'last_mask' in obj_data and obj_data['last_mask'] is not None:
                mask = obj_data['last_mask']
                if isinstance(mask, np.ndarray) and np.sum(mask > 0) > 0:
                    current_masks[obj_id] = mask

        if not current_masks:
            logger.warning("No masks to transfer in tracked_objects.")
            return

        logger.info(f"SAM3 session re-initialization - {len(current_masks)} masks")

        self._init_inference_session(for_pcs_mode=False)
        if self.inference_session is None:
            logger.error("SAM3 inference session initialization failed")
            return

        if self.current_cv_frame is None:
            logger.warning("Cannot add prompt to SAM3 session - no current frame")
            return

        frame_rgb = cv2.cvtColor(self.current_cv_frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        inputs = self.tracker_processor(images=frame_pil, device=self.device, return_tensors="pt")
        original_size = inputs.original_sizes[0]

        frame_tensor = inputs.pixel_values[0]
        if hasattr(self, 'model_dtype') and self.model_dtype == torch.float32:
            frame_tensor = frame_tensor.to(dtype=torch.float32)

        first_object = True
        for obj_id, mask in current_masks.items():
            if mask is None or not isinstance(mask, np.ndarray):
                continue

            if np.sum(mask > 0) == 0:
                continue

            try:
                mask_float = mask.astype(np.float32)
                input_masks = [[[mask_float]]]

                self.tracker_processor.add_inputs_to_inference_session(
                    inference_session=self.inference_session,
                    frame_idx=0,
                    obj_ids=obj_id,
                    input_masks=input_masks,
                    original_size=original_size,
                )

                with torch.inference_mode():
                    if first_object:
                        self.tracker_model(
                            inference_session=self.inference_session,
                            frame=frame_tensor,
                        )
                        first_object = False
                    else:
                        self.tracker_model(
                            inference_session=self.inference_session,
                            frame_idx=0,
                        )

                logger.info(f"SAM3 session: Object {obj_id} mask prompt applied successfully")

            except Exception as e:
                logger.warning(f"SAM3 mask prompt setting failed (obj_id={obj_id}): {e}")
                try:
                    y_coords, x_coords = np.where(mask > 0)
                    if len(x_coords) > 0:
                        x_min, x_max = float(x_coords.min()), float(x_coords.max())
                        y_min, y_max = float(y_coords.min()), float(y_coords.max())
                        input_boxes = [[[x_min, y_min, x_max, y_max]]]

                        self.tracker_processor.add_inputs_to_inference_session(
                            inference_session=self.inference_session,
                            frame_idx=0,
                            obj_ids=obj_id,
                            input_boxes=input_boxes,
                            original_size=original_size,
                        )
                        logger.info(f"SAM3 session: Object {obj_id} fallback to bbox prompt")

                        with torch.inference_mode():
                            if first_object:
                                self.tracker_model(inference_session=self.inference_session, frame=frame_tensor)
                                first_object = False
                            else:
                                self.tracker_model(inference_session=self.inference_session, frame_idx=0)
                except Exception as e2:
                    logger.error(f"SAM3 bbox fallback also failed (obj_id={obj_id}): {e2}")

        for obj_id, mask in current_masks.items():
            if obj_id not in self.tracked_objects:
                self.tracked_objects[obj_id] = {
                    'last_mask': mask.copy() if isinstance(mask, np.ndarray) else mask,
                    'custom_label': '',
                    'points_for_reprompt': [],
                    'initial_bbox_prompt': None,
                }
            else:
                self.tracked_objects[obj_id]['last_mask'] = mask.copy() if isinstance(mask, np.ndarray) else mask

        logger.info(f"SAM3 session re-initialization complete - {len(current_masks)} object mask prompts applied")

    def _handle_sam2_prompt(self, prompt_type, coords, label=None,
                            proposed_obj_id_for_new=None, target_existing_obj_id=None,
                            custom_label=None):
        sam2_controller._handle_sam2_prompt(self, prompt_type, coords, label,
                                            proposed_obj_id_for_new, target_existing_obj_id,
                                            custom_label)

    def refine_mask_with_sam2(self, obj_id, input_points=None, input_labels=None, input_boxes=None):
        return sam2_controller.refine_mask_with_sam2(self, obj_id, input_points, input_labels, input_boxes)

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
