import tkinter as tk
from tkinter import ttk
import logging
from PIL import Image, ImageTk

logger = logging.getLogger("DLMI_SAM_LABELER.AppView")

class AppView:
    def __init__(self, root, app_controller):
        self.root = root
        self.app = app_controller

        self.tk_image = None
        self.canvas_image_item = None
        self.temp_bbox_on_canvas_id = None

        self.original_frame_tab = None
        self.original_canvas = None
        self.original_tk_image = None

        self.guide_label = None
        self.button_frame = None

        self._setup_widgets()
        self._layout_widgets()
        self._bind_events()
        logger.info("AppView initialized.")

    def _setup_widgets(self):
        logger.debug("Setting up widgets.")
        self.top_frame = tk.Frame(self.root)
        self.main_content_frame = tk.Frame(self.root)
        self.video_frame = tk.Frame(self.main_content_frame, bd=2, relief=tk.SUNKEN)
        self.controls_and_config_frame = tk.Frame(self.main_content_frame)
        self.status_bar_frame = tk.Frame(self.root)

        self.canvas = tk.Canvas(self.video_frame, bg="black")

        self.btn_select_source = tk.Button(self.top_frame, text="Select Source", command=self.app.select_video_source, state=tk.DISABLED)
        self.btn_clear_tracked = tk.Button(self.top_frame, text="Clear All Objects", command=self.app.clear_all_tracked_objects, state=tk.DISABLED)
        self.btn_load_label = tk.Button(self.top_frame, text="Load Labels", command=self.app.load_label_file, state=tk.DISABLED, bg="#e3f2fd")
        self.btn_toggle_tabs = tk.Button(
            self.top_frame, text="Hide Tabs",
            command=self._toggle_tabs_visibility, bg="#fff9c4"
        )

        self.notebook = ttk.Notebook(self.controls_and_config_frame)

        self.obj_control_tab = self._make_scrollable_tab_frame("  Object Control  ")
        self.prompt_frame = tk.LabelFrame(self.obj_control_tab, text="Object Prompt")
        self.selected_obj_action_frame = tk.LabelFrame(self.obj_control_tab, text="Selected Object Actions")
        self.batch_control_frame = tk.LabelFrame(self.obj_control_tab, text="Batch Control")

        self.name_label = tk.Label(self.prompt_frame, text="Default Name:")
        self.entry_default_label = tk.Entry(self.prompt_frame, textvariable=self.app.default_object_label_var, width=12)
        self.modes_label = tk.Label(self.prompt_frame, text="Label Format:")
        self.modes_frame = tk.Frame(self.prompt_frame)
        modes = [("Instance", "Instance"), ("BBox", "Bounding Box"), ("Semantic", "Semantic")]
        self.mode_radio_buttons = [tk.Radiobutton(self.modes_frame, text=text, variable=self.app.labeling_mode_var, value=mode) for text, mode in modes]

        self.prompt_mode_label = tk.Label(self.prompt_frame, text="Detection Mode:")
        self.prompt_mode_frame = tk.Frame(self.prompt_frame)
        self.prompt_mode_row1 = tk.Frame(self.prompt_mode_frame)
        self.rb_pvs_mode = tk.Radiobutton(
            self.prompt_mode_row1, text="PVS (Point/Box)",
            variable=self.app.prompt_mode_var, value="PVS",
            command=self._on_prompt_mode_change
        )
        self.rb_pcs_mode = tk.Radiobutton(
            self.prompt_mode_row1, text="PCS (Text)",
            variable=self.app.prompt_mode_var, value="PCS",
            command=self._on_prompt_mode_change
        )
        self.prompt_mode_row2 = tk.Frame(self.prompt_mode_frame)
        self.rb_pcs_image_mode = tk.Radiobutton(
            self.prompt_mode_row2, text="PCS (Per-Image)",
            variable=self.app.prompt_mode_var, value="PCS_IMAGE",
            command=self._on_prompt_mode_change
        )
        self.rb_pvs_chunk_mode = tk.Radiobutton(
            self.prompt_mode_row2, text="PVS (Chunk)",
            variable=self.app.prompt_mode_var, value="PVS_CHUNK",
            command=self._on_prompt_mode_change
        )

        self.pcs_prompt_frame = tk.Frame(self.prompt_frame)
        self.pcs_prompt_label = tk.Label(self.pcs_prompt_frame, text="Text Prompt:")
        self.pcs_prompt_entry = tk.Entry(self.pcs_prompt_frame, textvariable=self.app.pcs_text_prompt_var, width=12)
        self.btn_pcs_detect = tk.Button(
            self.pcs_prompt_frame, text="Detect",
            command=self.app.execute_pcs_detection,
            state=tk.DISABLED, bg="#e3f2fd"
        )

        self.save_format_label = tk.Label(self.prompt_frame, text="Save Format:")
        self.save_format_frame = tk.Frame(self.prompt_frame)
        self.rb_labelme_format = tk.Radiobutton(self.save_format_frame, text="LabelMe", variable=self.app.save_format_var, value="labelme")
        self.rb_yolo_format = tk.Radiobutton(self.save_format_frame, text="YOLO", variable=self.app.save_format_var, value="yolo")
        self.rb_both_format = tk.Radiobutton(self.save_format_frame, text="Both", variable=self.app.save_format_var, value="both")


        selection_guide_text = ("- New object: Drag (box) / Ctrl+middle-click (point)\n"
                                "- Select object: Ctrl + left-click\n"
                                "- Add positive: Ctrl + middle-click (selected or new)\n"
                                "- Add negative: Ctrl + right-click (selected or new)\n"
                                "- Positive box: Alt + right-drag (add/modify object)\n"
                                "- Negative point: Alt + left-drag (exclude center)")
        self.guide_toggle_frame = tk.Frame(self.selected_obj_action_frame)
        self.guide_visible = False
        self.btn_guide_toggle = tk.Button(self.guide_toggle_frame, text="▶ Guide", font=("TkDefaultFont", 8), command=self._toggle_guide_visibility)
        self.guide_label = tk.Label(self.selected_obj_action_frame, text=selection_guide_text, justify=tk.LEFT, fg="gray50", font=("TkDefaultFont", 8))
        self.button_frame = tk.Frame(self.selected_obj_action_frame)
        self.btn_set_custom_label = tk.Button(self.button_frame, text="Set Name", command=self.app._set_custom_label_for_selected, state=tk.DISABLED)
        self.btn_reassign_bbox_selected = tk.Button(self.button_frame, text="Reset Box", command=self.app.toggle_reassign_bbox_mode, state=tk.DISABLED)
        self.btn_delete_selected = tk.Button(self.button_frame, text="Delete", command=self.app.delete_selected_object, state=tk.DISABLED)

        self.polygon_frame = tk.LabelFrame(self.obj_control_tab, text="Add Polygon")
        self.polygon_btn_frame = tk.Frame(self.polygon_frame)
        self.btn_polygon_mode = tk.Button(
            self.polygon_btn_frame, text="Add Polygon",
            command=self.app.toggle_polygon_mode if hasattr(self.app, 'toggle_polygon_mode') else None,
            bg="#c8e6c9"
        )
        self.btn_polygon_complete = tk.Button(
            self.polygon_btn_frame, text="Complete",
            command=self.app.complete_polygon_object if hasattr(self.app, 'complete_polygon_object') else None,
            state=tk.DISABLED
        )
        self.btn_polygon_undo = tk.Button(
            self.polygon_btn_frame, text="Undo Point",
            command=self.app.undo_last_polygon_point if hasattr(self.app, 'undo_last_polygon_point') else None,
            state=tk.DISABLED
        )
        self.btn_polygon_to_sam3 = tk.Button(
            self.polygon_btn_frame, text="SAM Input",
            command=self.app.input_polygon_to_sam3 if hasattr(self.app, 'input_polygon_to_sam3') else None,
            state=tk.DISABLED, bg="#bbdefb"
        )
        self.btn_polygon_cancel = tk.Button(
            self.polygon_btn_frame, text="Cancel",
            command=self.app.cancel_polygon_mode if hasattr(self.app, 'cancel_polygon_mode') else None,
            state=tk.DISABLED
        )
        self.polygon_point_size_frame = tk.Frame(self.polygon_frame)
        self.polygon_point_size_label = tk.Label(self.polygon_point_size_frame, text="Point/Prompt Size(%):", font=("TkDefaultFont", 8))
        self.polygon_point_size_entry = tk.Entry(self.polygon_point_size_frame, width=6, justify=tk.CENTER)
        self.polygon_point_size_entry.insert(0, "0.40")
        self.polygon_point_size_entry.bind("<Return>", self._on_point_size_change)
        self.polygon_point_size_entry.bind("<FocusOut>", self._on_point_size_change)
        self.polygon_status_label = tk.Label(self.polygon_point_size_frame, text="Left-click to add points", fg="gray50", font=("TkDefaultFont", 8))

        self.pose_control_frame = tk.LabelFrame(self.obj_control_tab, text="Pose Points")
        self.pose_btn_row1 = tk.Frame(self.pose_control_frame)
        self.btn_pose_add = tk.Checkbutton(
            self.pose_btn_row1, text="\u2795 Add Pose", indicatoron=False,
            variable=self.app.pose_add_mode_var if hasattr(self.app, 'pose_add_mode_var') else None,
            command=self._on_pose_add_mode_toggle if hasattr(self, '_on_pose_add_mode_toggle') else None,
            width=10, bg="#e3f2fd", selectcolor="#64b5f6"
        )
        self.btn_pose_chain = tk.Checkbutton(
            self.pose_btn_row1, text="\u26d3 Chain", indicatoron=False,
            variable=self.app.pose_chain_mode_var if hasattr(self.app, 'pose_chain_mode_var') else None,
            width=7, bg="#f3e5f5", selectcolor="#ba68c8"
        )
        self.btn_pose_new_obj = tk.Button(
            self.pose_btn_row1, text="\u2795 New Obj",
            command=self.app.new_pose_object if hasattr(self.app, 'new_pose_object') else None,
            width=10, bg="#c8e6c9"
        )

        self.pose_btn_row2 = tk.Frame(self.pose_control_frame)
        self.btn_pose_connect = tk.Button(
            self.pose_btn_row2, text="\U0001F517 Connect",
            command=self.app.connect_selected_pose_points if hasattr(self.app, 'connect_selected_pose_points') else None,
            state=tk.DISABLED, width=9
        )
        self.btn_pose_delete = tk.Button(
            self.pose_btn_row2, text="\U0001F5D1 Delete",
            command=self.app.delete_selected_pose_points if hasattr(self.app, 'delete_selected_pose_points') else None,
            state=tk.DISABLED, width=8, bg="#ffebee"
        )
        self.pose_idx_label = tk.Label(self.pose_btn_row2, text="idx:", font=("TkDefaultFont", 8))
        self.pose_idx_var = tk.StringVar(value="")
        self.pose_idx_entry = tk.Entry(
            self.pose_btn_row2, textvariable=self.pose_idx_var,
            width=4, justify=tk.CENTER, state=tk.DISABLED
        )
        self.btn_pose_set_idx = tk.Button(
            self.pose_btn_row2, text="Set",
            command=self.app.reassign_selected_pose_idx if hasattr(self.app, 'reassign_selected_pose_idx') else None,
            state=tk.DISABLED, width=4
        )
        self.btn_pose_toggle_vis = tk.Button(
            self.pose_btn_row2, text="Vis/Occ",
            command=self.app.toggle_selected_pose_visibility if hasattr(self.app, 'toggle_selected_pose_visibility') else None,
            state=tk.DISABLED, width=7, bg="#fff3e0"
        )

        self.pose_btn_row3 = tk.Frame(self.pose_control_frame)
        self.pose_class_label = tk.Label(self.pose_btn_row3, text="Class:", font=("TkDefaultFont", 8))
        self.pose_class_var = tk.StringVar(value="")
        self.pose_class_menu = tk.OptionMenu(self.pose_btn_row3, self.pose_class_var, "")
        self.pose_class_menu.config(width=12, font=("TkDefaultFont", 8))
        if hasattr(self.app, '_on_pose_class_selected'):
            self.pose_class_var.trace_add('write', lambda *_: self.app._on_pose_class_selected())
        self.btn_pose_delete_obj = tk.Button(
            self.pose_btn_row3, text="\U0001F5D1 Obj Pose",
            command=self.app.delete_selected_object_pose if hasattr(self.app, 'delete_selected_object_pose') else None,
            state=tk.DISABLED, width=11, bg="#ffcdd2"
        )

        self.pose_status_label = tk.Label(
            self.pose_control_frame,
            text=("Add Pose: click canvas to drop a keypoint (works mid-video too \u2014\n"
                  "points added while reviewing frame N are anchored at N and TAPNext\n"
                  "propagates them both forward and backward from that frame).\n"
                  "Chain: auto-connect consecutive clicks.\n"
                  "Shift+click: select single point \u2022 Shift+right-click: select entire chain.\n"
                  "Vis/Occ: toggle v=2 (visible, filled) \u2194 v=1 (occluded, dashed ring).\n"
                  "idx/Set: renumber the selected point \u2022 Obj Pose: delete the whole pose\n"
                  "of the currently selected object \u2022 Class: assigns schema."),
            fg="gray50", font=("TkDefaultFont", 7), justify=tk.LEFT
        )

        self.ui_display_frame = tk.LabelFrame(self.obj_control_tab, text="UI Display Settings")
        self.label_font_size_frame = tk.Frame(self.ui_display_frame)
        self.label_font_size_label = tk.Label(self.label_font_size_frame, text="Font(%):", font=("TkDefaultFont", 8))
        self.label_font_size_entry = tk.Entry(self.label_font_size_frame, width=5, justify=tk.CENTER)
        self.label_font_size_entry.insert(0, "0.70")
        self.label_font_size_entry.bind("<Return>", self._on_font_size_change)
        self.label_font_size_entry.bind("<FocusOut>", self._on_font_size_change)
        self.show_border_check = tk.Checkbutton(
            self.label_font_size_frame, text="Border",
            variable=self.app.show_object_border_var if hasattr(self.app, 'show_object_border_var') else None,
            command=self._on_ui_display_change, font=("TkDefaultFont", 8)
        )
        self.show_prompt_viz_check = tk.Checkbutton(
            self.label_font_size_frame, text="Show Prompts",
            variable=self.app.show_prompt_visualization_var if hasattr(self.app, 'show_prompt_visualization_var') else None,
            command=self._on_ui_display_change, font=("TkDefaultFont", 8)
        )
        self.show_prompt_per_object_check = tk.Checkbutton(
            self.label_font_size_frame, text="Per-Object",
            variable=self.app.show_prompt_per_object_var if hasattr(self.app, 'show_prompt_per_object_var') else None,
            command=self._on_ui_display_change, font=("TkDefaultFont", 8)
        )

        self.small_obj_filter_frame = tk.Frame(self.batch_control_frame)
        self.check_filter_small_obj = tk.Checkbutton(
            self.small_obj_filter_frame, text="Filter Small Objects",
            variable=self.app.filter_small_objects_var,
            command=self._on_small_obj_filter_change
        )
        self.small_obj_threshold_label = tk.Label(self.small_obj_filter_frame, text="Threshold(%):", font=("TkDefaultFont", 8))
        self.small_obj_threshold_entry = tk.Entry(
            self.small_obj_filter_frame, width=6, justify=tk.CENTER
        )
        self.small_obj_threshold_entry.insert(0, "0.1")
        self.small_obj_threshold_entry.bind("<Return>", self._on_small_obj_threshold_change)
        self.small_obj_threshold_entry.bind("<FocusOut>", self._on_small_obj_threshold_change)

        self.small_contour_filter_frame = tk.Frame(self.batch_control_frame)
        self.check_filter_small_contour = tk.Checkbutton(
            self.small_contour_filter_frame, text="Filter Small Contours",
            variable=self.app.filter_small_contours_var,
            command=self._on_small_contour_filter_change
        )
        self.small_contour_threshold_label = tk.Label(self.small_contour_filter_frame, text="Threshold(%):", font=("TkDefaultFont", 8))
        self.small_contour_threshold_entry = tk.Entry(
            self.small_contour_filter_frame, width=6, justify=tk.CENTER
        )
        self.small_contour_threshold_entry.insert(0, "0.01")
        self.small_contour_threshold_entry.bind("<Return>", self._on_small_contour_threshold_change)
        self.small_contour_threshold_entry.bind("<FocusOut>", self._on_small_contour_threshold_change)
        self.small_contour_base_frame = tk.Frame(self.small_contour_filter_frame)
        self.rb_contour_base_image = tk.Radiobutton(
            self.small_contour_base_frame, text="Image",
            variable=self.app.small_contour_base_var, value="image",
            font=("TkDefaultFont", 8)
        )
        self.rb_contour_base_object = tk.Radiobutton(
            self.small_contour_base_frame, text="Object",
            variable=self.app.small_contour_base_var, value="object",
            font=("TkDefaultFont", 8)
        )

        self.skip_video_frame = tk.Frame(self.obj_control_tab)
        self.btn_skip_batch = tk.Button(self.skip_video_frame, text="Next Video ⏭", command=self.app.skip_current_batch_video, state=tk.DISABLED, bg="#fff9c4", height=2)

        self.sam2_control_frame = tk.LabelFrame(self.obj_control_tab, text="SAM2 Refinement")
        self.sam2_toggle_frame = tk.Frame(self.sam2_control_frame)
        self.check_sam2_enabled = tk.Checkbutton(
            self.sam2_toggle_frame, text="Enable SAM2",
            variable=self.app.sam2_enabled_var,
            command=self._on_sam2_toggle
        )
        self.sam2_status_label = tk.Label(self.sam2_toggle_frame, text="(Inactive)", fg="gray50", font=("TkDefaultFont", 8))
        self.check_sam2_tracking = tk.Checkbutton(
            self.sam2_control_frame, text="Use SAM2 Tracking",
            variable=self.app.sam2_tracking_enabled_var,
            state=tk.DISABLED,
            command=self._on_sam2_tracking_toggle
        )
        self.sam2_guide_label = tk.Label(
            self.sam2_control_frame,
            text="Uses SAM2 when enabled",
            fg="gray50", font=("TkDefaultFont", 8)
        )

        self.save_batch_tab = self._make_scrollable_tab_frame("  Save/Batch  ")
        self.save_options_frame = tk.LabelFrame(self.save_batch_tab, text="Save Options")
        self.batch_options_frame = tk.LabelFrame(self.save_batch_tab, text="Batch Processing Options")

        self.check_custom_save = tk.Checkbutton(self.save_options_frame, text="Use Custom Save Path", variable=self.app.use_custom_save_path_var, command=self.app._on_custom_save_toggle)
        self.custom_save_widgets_frame = tk.Frame(self.save_options_frame)
        self.custom_save_dir_label = tk.Label(self.custom_save_widgets_frame, text="Save Location:")
        self.entry_custom_save_dir = tk.Entry(self.custom_save_widgets_frame, textvariable=self.app.custom_save_dir_var, width=18)
        self.btn_select_save_dir = tk.Button(self.custom_save_widgets_frame, text="...", command=self.app.select_custom_save_dir, width=3)
        self.custom_folder_name_label = tk.Label(self.custom_save_widgets_frame, text="Folder Format:")
        self.entry_custom_folder_name = tk.Entry(self.custom_save_widgets_frame, textvariable=self.app.custom_folder_name_var)
        self.custom_file_name_label = tk.Label(self.custom_save_widgets_frame, text="File Format:")
        self.entry_custom_file_name = tk.Entry(self.custom_save_widgets_frame, textvariable=self.app.custom_file_name_var)
        self.custom_format_label = tk.Label(self.custom_save_widgets_frame, text="* Format: {video_name} available", fg="gray50")

        self.check_batch_mode = tk.Checkbutton(self.batch_options_frame, text="Enable Batch Processing Mode", variable=self.app.batch_processing_mode_var, command=self.app._on_batch_mode_toggle)
        self.batch_widgets_frame = tk.Frame(self.batch_options_frame)
        self.batch_dir_label = tk.Label(self.batch_widgets_frame, text="Video Folder:")
        self.entry_batch_dir = tk.Entry(self.batch_widgets_frame, textvariable=self.app.batch_source_dir_var, width=18)
        self.btn_select_batch_dir = tk.Button(self.batch_widgets_frame, text="...", command=self.app.select_batch_source_dir, width=3)
        self.btn_start_batch = tk.Button(self.batch_widgets_frame, text="Start Batch", command=self.app.start_batch_processing)
        self.batch_save_label = tk.Label(self.batch_widgets_frame, text="Save Method:")
        self.batch_save_rb_frame = tk.Frame(self.batch_widgets_frame)
        self.rb_subfolder = tk.Radiobutton(self.batch_save_rb_frame, text="Create Subfolder", variable=self.app.batch_save_option_var, value="subfolder")
        self.rb_singlefolder = tk.Radiobutton(self.batch_save_rb_frame, text="Single Folder", variable=self.app.batch_save_option_var, value="singlefolder")
        self.batch_filename_label = tk.Label(self.batch_widgets_frame, text="Filename Method:")
        self.batch_filename_rb_frame = tk.Frame(self.batch_widgets_frame)
        self.rb_fname_video = tk.Radiobutton(self.batch_filename_rb_frame, text="Use Video Name", variable=self.app.batch_filename_option_var, value="video_name")
        self.rb_fname_custom = tk.Radiobutton(self.batch_filename_rb_frame, text="Use Custom Name", variable=self.app.batch_filename_option_var, value="custom")

        self.batch_move_frame = tk.LabelFrame(self.batch_options_frame, text="Completed Video Handling")
        self.check_batch_move = tk.Checkbutton(self.batch_move_frame, text="Auto-move Completed/Skipped Videos", variable=self.app.batch_move_completed_var)
        self.batch_move_widgets_frame = tk.Frame(self.batch_move_frame)
        self.batch_move_dir_label = tk.Label(self.batch_move_widgets_frame, text="Move to Folder:")
        self.entry_batch_move_dir = tk.Entry(self.batch_move_widgets_frame, textvariable=self.app.batch_completed_dir_var, width=15)
        self.btn_select_move_dir = tk.Button(self.batch_move_widgets_frame, text="...", command=self.app.select_batch_completed_dir, width=3)

        self.advanced_config_tab = self._make_scrollable_tab_frame("  Advanced  ")
        self.sam_adv_config_frame = tk.LabelFrame(self.advanced_config_tab, text="SAM Mask/Prompt")
        self.misc_settings_frame = tk.LabelFrame(self.advanced_config_tab, text="Other Settings")

        self.erosion_frame = tk.LabelFrame(self.misc_settings_frame, text="BBox Generation Options")
        self.edge_obj_frame = tk.LabelFrame(self.misc_settings_frame, text="Edge Object Handling")

        self.source_options_frame = tk.LabelFrame(self.save_options_frame, text="Source Options")

        self.mask_display_frame = tk.LabelFrame(self.sam_adv_config_frame, text="Mask Display")

        self.check_ignore_edge_labels = tk.Checkbutton(self.edge_obj_frame, text="Don't Save", variable=self.app.ignore_edge_labels_var)
        self.edge_margin_label = tk.Label(self.edge_obj_frame, text="Margin(px):")
        self.scale_edge_margin = tk.Scale(self.edge_obj_frame, from_=0, to=50, orient=tk.HORIZONTAL, variable=self.app.edge_margin_var, length=63)

        self.check_allow_image = tk.Checkbutton(self.source_options_frame, text="Allow Images (when selecting source)", variable=self.app.allow_image_source_var)

        self.mask_alpha_label = tk.Label(self.mask_display_frame, text="Mask Opacity:")
        self.scale_mask_alpha = tk.Scale(
            self.mask_display_frame, from_=0, to=255, orient=tk.HORIZONTAL,
            variable=self.app.mask_alpha_var, length=72,
            command=self._on_mask_alpha_change
        )

        self.sam_closing_row_frame = tk.Frame(self.sam_adv_config_frame)
        self.check_apply_closing = tk.Checkbutton(self.sam_closing_row_frame, text="Closing", variable=self.app.sam_apply_closing_var)
        self.sam_closing_kernel_label = tk.Label(self.sam_closing_row_frame, text="Kernel:", font=("TkDefaultFont", 8))
        self.scale_sam_closing_kernel = tk.Scale(self.sam_closing_row_frame, from_=1, to=11, orient=tk.HORIZONTAL, variable=self.app.sam_closing_kernel_size_var, length=100, showvalue=True)

        self.erosion_kernel_label = tk.Label(self.erosion_frame, text="Erosion Kernel:")
        self.scale_erosion_kernel = tk.Scale(self.erosion_frame, from_=0, to=7, orient=tk.HORIZONTAL, variable=self.app.erosion_kernel_size, length=63)
        self.erosion_iter_label = tk.Label(self.erosion_frame, text="Iterations:")
        self.scale_erosion_iterations = tk.Scale(self.erosion_frame, from_=0, to=5, orient=tk.HORIZONTAL, variable=self.app.erosion_iterations, length=63)

        self.low_level_api_frame = tk.LabelFrame(self.advanced_config_tab, text="Low-level API (Advanced)")
        self.check_low_level_api = tk.Checkbutton(
            self.low_level_api_frame, text="Use Low-level API (direct mask prompt input)",
            variable=self.app.low_level_api_enabled_var if hasattr(self.app, 'low_level_api_enabled_var') else None,
            command=self._on_low_level_api_toggle if hasattr(self, '_on_low_level_api_toggle') else None
        )
        self.low_level_api_guide = tk.Label(
            self.low_level_api_frame,
            text="* When enabled, directly accesses inference session without memory bank.\n"
                 "  Use 'Inject Data' button to inject current mask as mask prompt.",
            fg="gray50", font=("TkDefaultFont", 8), justify=tk.LEFT
        )
        self.dlmi_alpha_frame = tk.Frame(self.low_level_api_frame)
        self.dlmi_alpha_label = tk.Label(self.dlmi_alpha_frame, text="DLMI Alpha:")
        self.dlmi_alpha_slider = tk.Scale(
            self.dlmi_alpha_frame, from_=0.5, to=30.0, orient=tk.HORIZONTAL,
            variable=self.app.dlmi_alpha_var if hasattr(self.app, 'dlmi_alpha_var') else None,
            length=120, resolution=0.5
        )
        self.dlmi_alpha_info = tk.Label(
            self.low_level_api_frame,
            text="Default=10 (99.995% confidence) | Lower = more uncertain",
            fg="gray50", font=("TkDefaultFont", 7)
        )
        self.dlmi_boundary_frame = tk.Frame(self.low_level_api_frame)
        self.dlmi_boundary_label = tk.Label(self.dlmi_boundary_frame, text="Boundary Mode:")
        self.dlmi_boundary_combo = tk.OptionMenu(
            self.dlmi_boundary_frame,
            self.app.dlmi_boundary_mode_var if hasattr(self.app, 'dlmi_boundary_mode_var') else tk.StringVar(value="Fixed"),
            "Fixed", "Gradient"
        )
        self.dlmi_falloff_slider = tk.Scale(
            self.dlmi_boundary_frame, from_=1, to=100, orient=tk.HORIZONTAL,
            variable=self.app.dlmi_gradient_falloff_var if hasattr(self.app, 'dlmi_gradient_falloff_var') else None,
            length=100, resolution=1
        )
        self.dlmi_boundary_info = tk.Label(
            self.low_level_api_frame,
            text="Fixed: uniform logits | Gradient: distance-based continuous logits (slider=falloff px)",
            fg="gray50", font=("TkDefaultFont", 7)
        )
        self.dlmi_preserve_check = tk.Checkbutton(
            self.low_level_api_frame, text="Preserve DLMI Memory (permanent conditioning)",
            variable=self.app.dlmi_preserve_memory_var if hasattr(self.app, 'dlmi_preserve_memory_var') else None,
        )
        self.dlmi_boost_check = tk.Checkbutton(
            self.low_level_api_frame, text="Boost Conditioning Memory (3x weight in attention)",
            variable=self.app.dlmi_boost_cond_var if hasattr(self.app, 'dlmi_boost_cond_var') else None,
        )

        self.tapnext_frame = tk.LabelFrame(self.advanced_config_tab, text="TAPNext++ Pose (Advanced)")
        self.tapnext_row_frame = tk.Frame(self.tapnext_frame)
        self.btn_tapnext_settings = tk.Button(
            self.tapnext_row_frame, text="\u2699",
            command=self.app.open_pose_settings if hasattr(self.app, 'open_pose_settings') else None,
            width=3, font=("TkDefaultFont", 10)
        )
        self.check_tapnext_enabled = tk.Checkbutton(
            self.tapnext_row_frame, text="TAPNext++ On",
            variable=self.app.pose_tapnext_enabled_var if hasattr(self.app, 'pose_tapnext_enabled_var') else None,
            font=("TkDefaultFont", 9)
        )
        self.check_pose_automatch = tk.Checkbutton(
            self.tapnext_row_frame, text="Auto-match",
            variable=self.app.pose_automatch_var if hasattr(self.app, 'pose_automatch_var') else None,
            font=("TkDefaultFont", 9)
        )
        self.tapnext_info_label = tk.Label(
            self.tapnext_frame,
            text="* \u2699: configure. Auto-match: merge pose-only objects into segments when \u226570% of points fall inside a mask.",
            fg="gray50", font=("TkDefaultFont", 7), justify=tk.LEFT
        )

        self.propagate_review_tab = self._make_scrollable_tab_frame("  Propagate/Review  ")

        self.propagate_control_frame = tk.LabelFrame(self.propagate_review_tab, text="Propagate Control")
        self.btn_start_propagate = tk.Button(
            self.propagate_control_frame, text="Start Propagate ▶",
            command=self.app.start_propagation if hasattr(self.app, 'start_propagation') else None,
            state=tk.DISABLED, bg="#c8e6c9", width=15
        )
        self.btn_pause_propagate = tk.Button(
            self.propagate_control_frame, text="Pause ⏸",
            command=self.app.pause_propagation if hasattr(self.app, 'pause_propagation') else None,
            state=tk.DISABLED, bg="#fff9c4", width=8
        )
        self.btn_stop_propagate = tk.Button(
            self.propagate_control_frame, text="Stop ■",
            command=self.app.stop_propagation if hasattr(self.app, 'stop_propagation') else None,
            state=tk.DISABLED, bg="#ffcdd2", width=8
        )
        self.propagate_progress_label = tk.Label(self.propagate_control_frame, text="Progress:")
        self.propagate_progressbar = ttk.Progressbar(
            self.propagate_control_frame, orient=tk.HORIZONTAL,
            length=180, mode='determinate'
        )
        self.propagate_status_label = tk.Label(
            self.propagate_control_frame, text="Waiting",
            fg="gray50"
        )
        self.discarded_frames_label = tk.Label(
            self.propagate_control_frame, text="",
            fg="#d84315", font=("TkDefaultFont", 8)
        )

        self.mid_new_object_frame = tk.Frame(self.propagate_control_frame)
        self.mid_new_object_label = tk.Label(
            self.mid_new_object_frame,
            text="Mid-session new object:",
            font=("TkDefaultFont", 8), fg="gray30"
        )
        mid_var = getattr(self.app, 'mid_new_object_method_var', None)
        self.mid_new_object_radios = []
        if mid_var is not None:
            for label_text, val in (("Off (block)", "off"), ("DLMI inject", "dlmi"), ("Load labels", "load")):
                rb = tk.Radiobutton(
                    self.mid_new_object_frame, text=label_text,
                    variable=mid_var, value=val,
                    font=("TkDefaultFont", 8)
                )
                self.mid_new_object_radios.append(rb)

        self.review_mode_frame = tk.LabelFrame(self.propagate_review_tab, text="Review & Refine")
        self.review_slider_label = tk.Label(self.review_mode_frame, text="Frame:")
        self.review_frame_slider = tk.Scale(
            self.review_mode_frame, from_=0, to=100,
            orient=tk.HORIZONTAL, length=180,
            command=self._on_review_slider_change if hasattr(self, '_on_review_slider_change') else None
        )
        self.review_frame_info_label = tk.Label(self.review_mode_frame, text="0 / 0")

        self.review_frame_input_frame = tk.Frame(self.review_mode_frame)
        self.review_frame_input_label = tk.Label(self.review_frame_input_frame, text="Go to Frame:")
        self.review_frame_input_var = tk.StringVar(value="0")
        self.review_frame_input_entry = tk.Entry(
            self.review_frame_input_frame, textvariable=self.review_frame_input_var,
            width=8, justify=tk.CENTER
        )
        self.review_frame_input_entry.bind("<Return>", self._on_review_frame_input_enter)
        self.btn_goto_frame = tk.Button(
            self.review_frame_input_frame, text="Go",
            command=self._on_review_frame_goto_click, width=5
        )

        self.review_buttons_frame = tk.Frame(self.review_mode_frame)
        self.btn_cut_from_here = tk.Button(
            self.review_buttons_frame, text="Cut Here ✂",
            command=self.app.cut_and_repropagate if hasattr(self.app, 'cut_and_repropagate') else None,
            state=tk.DISABLED, width=12
        )
        self.btn_discard_frame = tk.Button(
            self.review_buttons_frame, text="Discard 🗑",
            command=self.app.toggle_discard_current_frame if hasattr(self.app, 'toggle_discard_current_frame') else None,
            state=tk.DISABLED, width=6, bg="#fff3e0"
        )
        self.btn_confirm_labels = tk.Button(
            self.review_buttons_frame, text="Confirm Labels ✓",
            command=self.app.confirm_and_save_labels if hasattr(self.app, 'confirm_and_save_labels') else None,
            state=tk.DISABLED, bg="#c8e6c9", width=12
        )

        self.review_buttons_frame2 = tk.Frame(self.review_mode_frame)
        self.btn_cut_dlmi = tk.Button(
            self.review_buttons_frame2, text="Cut + DLMI ✂▶",
            command=self.app.cut_and_dlmi_propagate if hasattr(self.app, 'cut_and_dlmi_propagate') else None,
            state=tk.DISABLED, width=14, bg="#e1bee7"
        )
        self.btn_cut_load = tk.Button(
            self.review_buttons_frame2, text="Cut + Load ✂📂",
            command=self.app.cut_and_load_labels if hasattr(self.app, 'cut_and_load_labels') else None,
            state=tk.DISABLED, width=14, bg="#bbdefb"
        )

        self.review_guide_collapsed = True
        review_guide_text = (
            "1. 'Start Propagate' to process entire video\n"
            "2. Check results with slider\n"
            "3. If issues found:\n"
            "   - 'Cut Here': Reprocess from this point\n"
            "   - 'Discard': Exclude current frame from save\n"
            "4. When satisfied, 'Confirm Labels' to save"
        )
        self.review_guide_frame = tk.Frame(self.review_mode_frame)
        self.btn_toggle_review_guide = tk.Button(
            self.review_guide_frame, text="▶ View Guide",
            command=self._toggle_review_guide, relief=tk.FLAT,
            fg="gray50", cursor="hand2"
        )
        self.review_guide_label = tk.Label(
            self.review_guide_frame, text=review_guide_text,
            justify=tk.LEFT, fg="gray50"
        )
        self.chunk_settings_frame = tk.Frame(self.review_guide_frame)
        self.chunk_threshold_label = tk.Label(
            self.chunk_settings_frame, text="Chunk Mode Error Tolerance:",
            fg="gray50", font=("TkDefaultFont", 8)
        )
        self.chunk_threshold_scale = tk.Scale(
            self.chunk_settings_frame, from_=0.05, to=0.5,
            resolution=0.01, orient=tk.HORIZONTAL,
            variable=self.app.chunk_error_threshold_var if hasattr(self.app, 'chunk_error_threshold_var') else None,
            length=100
        )

        self.pose_tool_frame = tk.LabelFrame(self.propagate_review_tab, text="Pose Tools (manual)")
        self.btn_pose_detect_yolo = tk.Button(
            self.pose_tool_frame, text="Detect Pose (YOLO)",
            command=self.app.run_yolo_pose_detect if hasattr(self.app, 'run_yolo_pose_detect') else None,
            width=18, bg="#fff3e0"
        )
        self.btn_pose_run_tapnext = tk.Button(
            self.pose_tool_frame, text="Run TAPNext Post",
            command=self.app.run_tapnext_post_process if hasattr(self.app, 'run_tapnext_post_process') else None,
            width=18, bg="#e1bee7"
        )
        self.pose_tool_status_label = tk.Label(
            self.pose_tool_frame,
            text=("\u2022 Detect Pose (YOLO): run a YOLO-pose model on the CURRENT frame to add keypoints.\n"
                  "\u2022 Run TAPNext Post: propagate existing pose points across all propagated frames.\n"
                  "  (auto-runs after SAM3 propagation when TAPNext++ toggle is ON)"),
            fg="gray50", font=("TkDefaultFont", 8), justify=tk.LEFT
        )

        self.progress_label = tk.Label(self.status_bar_frame, text="Status: Loading model...")
        self.obj_id_info_label = tk.Label(self.status_bar_frame, text=f"Next manual BBox ObjID (proposed): {self.app.next_obj_id_to_propose}")
        logger.debug("Widgets setup complete.")
        
             
        
    def _layout_widgets(self):
        logger.debug("Laying out widgets.")
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.top_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(5,0))
        self.main_content_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.status_bar_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=(2, 5))

        self.main_content_frame.grid_rowconfigure(0, weight=1)
        self.main_content_frame.grid_columnconfigure(0, weight=1)
        self.main_content_frame.grid_columnconfigure(1, weight=0)

        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.controls_and_config_frame.grid(row=0, column=1, sticky="ns")

        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.btn_select_source.pack(side=tk.LEFT, padx=2)
        self.btn_clear_tracked.pack(side=tk.LEFT, padx=2)
        self.btn_load_label.pack(side=tk.RIGHT, padx=2)
        self.btn_toggle_tabs.pack(side=tk.RIGHT, padx=2)

        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.obj_control_tab.columnconfigure(0, weight=1)
        self.prompt_frame.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        self.prompt_frame.columnconfigure(1, weight=1)
        self.name_label.grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.entry_default_label.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        self.modes_label.grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.modes_frame.grid(row=1, column=1, sticky='w')
        for radio_button in self.mode_radio_buttons:
            radio_button.pack(side=tk.LEFT, padx=2)

        self.save_format_label.grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.save_format_frame.grid(row=2, column=1, sticky='w')
        self.rb_labelme_format.pack(side=tk.LEFT, padx=2)
        self.rb_yolo_format.pack(side=tk.LEFT, padx=2)
        self.rb_both_format.pack(side=tk.LEFT, padx=2)

        self.prompt_mode_label.grid(row=3, column=0, sticky='nw', padx=5, pady=2)
        self.prompt_mode_frame.grid(row=3, column=1, sticky='w')
        self.prompt_mode_row1.pack(anchor='w')
        self.rb_pvs_mode.pack(side=tk.LEFT, padx=2)
        self.rb_pcs_mode.pack(side=tk.LEFT, padx=2)
        self.prompt_mode_row2.pack(anchor='w')
        self.rb_pcs_image_mode.pack(side=tk.LEFT, padx=2)
        self.rb_pvs_chunk_mode.pack(side=tk.LEFT, padx=2)
        self.pcs_prompt_frame.grid(row=4, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
        self.pcs_prompt_label.pack(side=tk.LEFT, padx=2)
        self.pcs_prompt_entry.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        self.btn_pcs_detect.pack(side=tk.LEFT, padx=2)

        self.selected_obj_action_frame.grid(row=1, column=0, padx=5, pady=5, sticky='ew')
        self.guide_toggle_frame.pack(side=tk.TOP, anchor='w', padx=5, pady=2)
        self.btn_guide_toggle.pack(side=tk.LEFT)
        self.button_frame.pack(side=tk.TOP, fill=tk.X)
        self.btn_low_data_inject = tk.Button(self.button_frame, text="Inject Data", command=self.app.inject_low_level_mask_prompt if hasattr(self.app, 'inject_low_level_mask_prompt') else None, state=tk.DISABLED, bg="#e1bee7")
        self.btn_low_data_inject.pack(side=tk.LEFT, padx=3)
        self.btn_set_custom_label.pack(side=tk.LEFT, padx=3)
        self.btn_merge_objects = tk.Button(self.button_frame, text="Merge Objects", command=self.app.merge_selected_objects if hasattr(self.app, 'merge_selected_objects') else None, state=tk.DISABLED)
        self.btn_merge_objects.pack(side=tk.LEFT, padx=3)
        self.btn_reassign_bbox_selected.pack(side=tk.LEFT, padx=3)
        self.btn_delete_selected.pack(side=tk.LEFT, padx=3)

        self.polygon_frame.grid(row=2, column=0, padx=5, pady=5, sticky='ew')
        self.polygon_btn_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        self.btn_polygon_mode.pack(side=tk.LEFT, padx=2)
        self.btn_polygon_complete.pack(side=tk.LEFT, padx=2)
        self.btn_polygon_undo.pack(side=tk.LEFT, padx=2)
        self.btn_polygon_to_sam3.pack(side=tk.LEFT, padx=2)
        self.btn_polygon_cancel.pack(side=tk.LEFT, padx=2)
        self.polygon_point_size_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        self.polygon_point_size_label.pack(side=tk.LEFT)
        self.polygon_point_size_entry.pack(side=tk.LEFT, padx=5)
        self.polygon_status_label.pack(side=tk.LEFT, padx=(10, 0))

        self.pose_control_frame.grid(row=3, column=0, padx=5, pady=5, sticky='ew')
        self.pose_btn_row1.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        self.btn_pose_add.pack(side=tk.LEFT, padx=2)
        self.btn_pose_chain.pack(side=tk.LEFT, padx=2)
        self.btn_pose_new_obj.pack(side=tk.LEFT, padx=2)
        self.pose_btn_row2.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        self.btn_pose_connect.pack(side=tk.LEFT, padx=2)
        self.btn_pose_delete.pack(side=tk.LEFT, padx=2)
        self.btn_pose_toggle_vis.pack(side=tk.LEFT, padx=2)
        self.pose_idx_label.pack(side=tk.LEFT, padx=(10, 2))
        self.pose_idx_entry.pack(side=tk.LEFT, padx=2)
        self.btn_pose_set_idx.pack(side=tk.LEFT, padx=2)
        self.pose_btn_row3.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        self.pose_class_label.pack(side=tk.LEFT, padx=2)
        self.pose_class_menu.pack(side=tk.LEFT, padx=2)
        self.btn_pose_delete_obj.pack(side=tk.LEFT, padx=(10, 2))
        self.pose_status_label.pack(side=tk.TOP, anchor='w', padx=5, pady=(0, 3))

        self.ui_display_frame.grid(row=4, column=0, padx=5, pady=5, sticky='ew')
        self.label_font_size_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        self.label_font_size_label.pack(side=tk.LEFT)
        self.label_font_size_entry.pack(side=tk.LEFT, padx=(2, 8))
        self.show_border_check.pack(side=tk.LEFT, padx=2)
        self.show_prompt_viz_check.pack(side=tk.LEFT, padx=2)
        self.show_prompt_per_object_check.pack(side=tk.LEFT, padx=2)

        self.batch_control_frame.grid(row=5, column=0, padx=5, pady=5, sticky='ew')
        self.small_obj_filter_frame.pack(pady=2, padx=5, fill=tk.X)
        self.check_filter_small_obj.pack(side=tk.LEFT)
        self.small_obj_threshold_label.pack(side=tk.LEFT, padx=(5, 2))
        self.small_obj_threshold_entry.pack(side=tk.LEFT)
        self.small_contour_filter_frame.pack(pady=2, padx=5, fill=tk.X)
        self.check_filter_small_contour.pack(side=tk.LEFT)
        self.small_contour_threshold_label.pack(side=tk.LEFT, padx=(5, 2))
        self.small_contour_threshold_entry.pack(side=tk.LEFT)
        self.small_contour_base_frame.pack(side=tk.LEFT, padx=(5, 0))
        self.rb_contour_base_image.pack(side=tk.LEFT)
        self.rb_contour_base_object.pack(side=tk.LEFT)

        self.skip_video_frame.grid(row=6, column=0, padx=5, pady=5, sticky='ew')
        self.btn_skip_batch.pack(expand=True, fill=tk.BOTH)

        self.sam2_control_frame.grid(row=7, column=0, padx=5, pady=5, sticky='ew')
        self.sam2_toggle_frame.pack(side=tk.LEFT, padx=5, pady=2)
        self.check_sam2_enabled.pack(side=tk.LEFT)
        self.sam2_status_label.pack(side=tk.LEFT, padx=(2, 0))
        self.check_sam2_tracking.pack(side=tk.LEFT, padx=10)
        self.sam2_guide_label.pack(side=tk.LEFT, padx=10)
        self.save_batch_tab.columnconfigure(0, weight=1)
        self.save_options_frame.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        self.batch_options_frame.grid(row=1, column=0, padx=5, pady=5, sticky='ew')
        
        self.check_custom_save.pack(anchor='w', padx=5)
        self.custom_save_widgets_frame.pack(anchor='w', padx=5, fill='x')
        self.custom_save_widgets_frame.columnconfigure(1, weight=1)
        self.custom_save_dir_label.grid(row=0, column=0, sticky='e', pady=2, padx=2)
        self.entry_custom_save_dir.grid(row=0, column=1, sticky='ew')
        self.btn_select_save_dir.grid(row=0, column=2, padx=(2,0))
        self.custom_folder_name_label.grid(row=1, column=0, sticky='e', pady=2, padx=2)
        self.entry_custom_folder_name.grid(row=1, column=1, columnspan=2, sticky='ew')
        self.custom_file_name_label.grid(row=2, column=0, sticky='e', pady=2, padx=2)
        self.entry_custom_file_name.grid(row=2, column=1, columnspan=2, sticky='ew')
        self.custom_format_label.grid(row=3, column=1, sticky='w')
        
        self.check_batch_mode.pack(anchor='w', padx=5)
        self.batch_widgets_frame.pack(anchor='w', padx=5, fill='x')
        self.batch_widgets_frame.columnconfigure(1, weight=1)
        self.batch_dir_label.grid(row=0, column=0, sticky='e', pady=2, padx=2)
        self.entry_batch_dir.grid(row=0, column=1, sticky='ew')
        self.btn_select_batch_dir.grid(row=0, column=2, padx=(2,0))
        self.btn_start_batch.grid(row=0, column=3, padx=(5,0), rowspan=3, sticky='ns')
        self.batch_save_label.grid(row=1, column=0, sticky='e', pady=2, padx=2)
        self.batch_save_rb_frame.grid(row=1, column=1, columnspan=2, sticky='w')
        self.rb_subfolder.pack(side=tk.LEFT)
        self.rb_singlefolder.pack(side=tk.LEFT)
        self.batch_filename_label.grid(row=2, column=0, sticky='e', pady=2, padx=2)
        self.batch_filename_rb_frame.grid(row=2, column=1, columnspan=2, sticky='w')
        self.rb_fname_video.pack(side=tk.LEFT)
        self.rb_fname_custom.pack(side=tk.LEFT)

        self.batch_move_frame.pack(anchor='w', padx=5, pady=(10,5), fill='x')
        self.check_batch_move.pack(anchor='w', padx=5)
        self.batch_move_widgets_frame.pack(anchor='w', padx=5, pady=2, fill='x')
        self.batch_move_widgets_frame.columnconfigure(1, weight=1)
        self.batch_move_dir_label.grid(row=0, column=0, sticky='e', padx=2)
        self.entry_batch_move_dir.grid(row=0, column=1, sticky='ew', padx=2)
        self.btn_select_move_dir.grid(row=0, column=2)

        self.source_options_frame.pack(anchor='w', padx=5, pady=(10,5), fill='x')
        self.check_allow_image.pack(anchor='w', padx=5, pady=2)

        self.advanced_config_tab.columnconfigure(0, weight=1)

        self.sam_adv_config_frame.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        self.misc_settings_frame.grid(row=1, column=0, padx=5, pady=5, sticky='ew')
        self.low_level_api_frame.grid(row=2, column=0, padx=5, pady=5, sticky='ew')
        self.tapnext_frame.grid(row=3, column=0, padx=5, pady=5, sticky='ew')
        self.tapnext_row_frame.pack(anchor='w', fill=tk.X, padx=5, pady=2)
        self.btn_tapnext_settings.pack(side=tk.LEFT, padx=2)
        self.check_tapnext_enabled.pack(side=tk.LEFT, padx=(8, 2))
        self.check_pose_automatch.pack(side=tk.LEFT, padx=(8, 2))
        self.tapnext_info_label.pack(anchor='w', padx=5, pady=(0, 3))
        self.check_low_level_api.pack(anchor='w', padx=5, pady=2)
        self.low_level_api_guide.pack(anchor='w', padx=5, pady=2)
        self.dlmi_alpha_frame.pack(anchor='w', padx=5, pady=(5, 0), fill='x')
        self.dlmi_alpha_label.pack(side=tk.LEFT)
        self.dlmi_alpha_slider.pack(side=tk.LEFT, padx=5)
        self.dlmi_alpha_info.pack(anchor='w', padx=10, pady=(0, 2))
        self.dlmi_boundary_frame.pack(anchor='w', padx=5, pady=(2, 0), fill='x')
        self.dlmi_boundary_label.pack(side=tk.LEFT)
        self.dlmi_boundary_combo.pack(side=tk.LEFT, padx=5)
        self.dlmi_falloff_slider.pack(side=tk.LEFT, padx=5)
        self.dlmi_boundary_info.pack(anchor='w', padx=10, pady=(0, 2))
        self.dlmi_preserve_check.pack(anchor='w', padx=5, pady=(2, 0))
        self.dlmi_boost_check.pack(anchor='w', padx=5, pady=(0, 5))

        self.sam_closing_row_frame.pack(anchor='w', fill=tk.X, padx=5, pady=(2, 2))
        self.check_apply_closing.pack(side=tk.LEFT)
        self.sam_closing_kernel_label.pack(side=tk.LEFT, padx=(8, 2))
        self.scale_sam_closing_kernel.pack(side=tk.LEFT)

        self.mask_display_frame.pack(anchor='w', fill=tk.X, padx=5, pady=(5,0))
        self.mask_alpha_label.pack(side=tk.LEFT, padx=5)
        self.scale_mask_alpha.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        self.erosion_frame.pack(anchor='w', fill=tk.X, padx=5, pady=5)
        self.erosion_frame.columnconfigure(1, weight=1)
        self.erosion_frame.columnconfigure(3, weight=1)
        self.erosion_kernel_label.grid(row=0, column=0, padx=(5,2), sticky='w')
        self.scale_erosion_kernel.grid(row=0, column=1, sticky='ew')
        self.erosion_iter_label.grid(row=0, column=2, padx=(10, 2), sticky='w')
        self.scale_erosion_iterations.grid(row=0, column=3, sticky='ew')
        
        self.edge_obj_frame.pack(anchor='w', fill=tk.X, padx=5, pady=5)
        self.check_ignore_edge_labels.pack(side=tk.LEFT, padx=5)
        self.edge_margin_label.pack(side=tk.LEFT, padx=(10,2))
        self.scale_edge_margin.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        # Tabs are registered during _make_scrollable_tab_frame()

        self.propagate_review_tab.columnconfigure(0, weight=1)

        self.propagate_control_frame.grid(row=0, column=0, padx=5, pady=5, sticky='ew')
        self.btn_start_propagate.grid(row=0, column=0, padx=3, pady=5)
        self.btn_pause_propagate.grid(row=0, column=1, padx=3, pady=5)
        self.btn_stop_propagate.grid(row=0, column=2, padx=3, pady=5)
        self.propagate_progress_label.grid(row=1, column=0, sticky='w', padx=5)
        self.propagate_progressbar.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky='ew')
        self.propagate_status_label.grid(row=2, column=0, columnspan=3, sticky='w', padx=5, pady=2)
        self.discarded_frames_label.grid(row=3, column=0, columnspan=3, sticky='w', padx=5, pady=2)
        self.mid_new_object_frame.grid(row=4, column=0, columnspan=3, sticky='w', padx=5, pady=(2, 4))
        self.mid_new_object_label.pack(side=tk.LEFT, padx=(0, 4))
        for rb in self.mid_new_object_radios:
            rb.pack(side=tk.LEFT, padx=2)

        self.review_mode_frame.grid(row=1, column=0, padx=5, pady=5, sticky='ew')
        self.review_slider_label.grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.review_frame_slider.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        self.review_frame_info_label.grid(row=0, column=2, padx=5, pady=2)

        self.review_frame_input_frame.grid(row=1, column=0, columnspan=3, pady=2)
        self.review_frame_input_label.pack(side=tk.LEFT, padx=5)
        self.review_frame_input_entry.pack(side=tk.LEFT, padx=2)
        self.btn_goto_frame.pack(side=tk.LEFT, padx=5)

        self.review_buttons_frame.grid(row=2, column=0, columnspan=3, pady=5)
        self.btn_cut_from_here.pack(side=tk.LEFT, padx=3)
        self.btn_discard_frame.pack(side=tk.LEFT, padx=3)
        self.btn_confirm_labels.pack(side=tk.LEFT, padx=3)
        self.review_buttons_frame2.grid(row=3, column=0, columnspan=3, pady=(0, 5))
        self.btn_cut_dlmi.pack(side=tk.LEFT, padx=3)
        self.btn_cut_load.pack(side=tk.LEFT, padx=3)
        self.review_guide_frame.grid(row=4, column=0, columnspan=3, sticky='w', padx=5, pady=2)
        self.btn_toggle_review_guide.pack(anchor='w')

        self.pose_tool_frame.grid(row=2, column=0, padx=5, pady=5, sticky='ew')
        self.btn_pose_detect_yolo.grid(row=0, column=0, padx=3, pady=3)
        self.btn_pose_run_tapnext.grid(row=0, column=1, padx=3, pady=3)
        self.pose_tool_status_label.grid(row=1, column=0, columnspan=2, sticky='w', padx=5)

        self.progress_label.pack(side=tk.LEFT, padx=10)
        self.obj_id_info_label.pack(side=tk.LEFT, padx=10)
        logger.debug("Widget layout complete.")

        self.update_custom_save_options_state()
        self.update_batch_options_state()
        self._update_pcs_mode_ui()

    def _on_prompt_mode_change(self):
        self._update_pcs_mode_ui()
        new_mode = self.app.prompt_mode_var.get()
        previous_mode = getattr(self.app, '_previous_prompt_mode', 'PVS')
        logger.info(f"Detection mode changed: {previous_mode} → {new_mode}")

        is_from_pcs = previous_mode in ("PCS", "PCS_IMAGE")
        is_to_pvs = new_mode in ("PVS", "PVS_CHUNK")
        has_masks = (
            (hasattr(self.app, 'tracked_objects') and len(self.app.tracked_objects) > 0) or
            (hasattr(self.app, 'propagated_results') and len(self.app.propagated_results) > 0) or
            (hasattr(self.app, 'sam2_masks') and len(self.app.sam2_masks) > 0)
        )

        if is_from_pcs and is_to_pvs and has_masks:
            logger.info(f"PCS → {new_mode} transition: masks retained")
            if hasattr(self.app, 'inference_session') and self.app.inference_session is not None:
                self.app.inference_session = None
            if hasattr(self.app, 'is_tracking_ever_started'):
                self.app.is_tracking_ever_started = False
            try:
                self.app._reinit_sam3_session_with_masks()
                mask_count = len(self.app._get_current_frame_masks()) if hasattr(self.app, '_get_current_frame_masks') else 0
                if hasattr(self.app, 'update_status'):
                    self.app.update_status(f"Switched to {new_mode} mode. {mask_count} masks retained.")
            except Exception as e:
                logger.exception(f"SAM3 session reinitialization failed: {e}")
                if hasattr(self.app, 'update_status'):
                    self.app.update_status(f"Error during {new_mode} mode transition")
        else:
            if hasattr(self.app, 'inference_session') and self.app.inference_session is not None:
                logger.info(f"Detection mode change: resetting inference_session")
                self.app.inference_session = None
            if hasattr(self.app, 'tracked_objects'):
                self.app.tracked_objects.clear()
            if hasattr(self.app, 'is_tracking_ever_started'):
                self.app.is_tracking_ever_started = False
            if hasattr(self.app, 'propagated_results'):
                self.app.propagated_results = {}
            if hasattr(self.app, 'update_status'):
                self.app.update_status(f"Detection mode changed to {new_mode}. Please redefine objects.")

        if hasattr(self.app, '_update_obj_id_info_label'):
            self.app._update_obj_id_info_label()

        self.app._previous_prompt_mode = new_mode

    def _update_pcs_mode_ui(self):
        current_mode = self.app.prompt_mode_var.get()
        is_pcs_text_mode = current_mode in ("PCS", "PCS_IMAGE")
        pcs_state = tk.NORMAL if is_pcs_text_mode else tk.DISABLED

        try:
            self.pcs_prompt_entry.config(state=pcs_state)
            if current_mode == "PCS" and self.app.current_cv_frame is not None:
                self.btn_pcs_detect.config(state=tk.NORMAL)
            else:
                self.btn_pcs_detect.config(state=tk.DISABLED)
        except tk.TclError:
            pass

    def enable_pcs_detect_button(self, enable=True):
        if self.app.prompt_mode_var.get() == "PCS":
            self.btn_pcs_detect.config(state=tk.NORMAL if enable else tk.DISABLED)

    def update_propagate_progress(self, progress, status_text=""):
        try:
            self.propagate_progressbar['value'] = progress
            if status_text:
                self.propagate_status_label.config(text=status_text)
        except tk.TclError:
            pass

    def set_propagate_button_states(self, is_propagating):
        try:
            if is_propagating:
                self.btn_start_propagate.config(state=tk.DISABLED, text="Start Propagate \u25b6")
                self.btn_pause_propagate.config(state=tk.NORMAL)
                self.btn_stop_propagate.config(state=tk.NORMAL)
                self.btn_cut_from_here.config(state=tk.DISABLED)
                self.btn_confirm_labels.config(state=tk.DISABLED)
                try:
                    self.btn_cut_dlmi.config(state=tk.DISABLED)
                    self.btn_cut_load.config(state=tk.DISABLED)
                except (tk.TclError, AttributeError):
                    pass
            else:
                self.btn_start_propagate.config(state=tk.NORMAL, text="Start Propagate \u25b6")
                self.btn_pause_propagate.config(state=tk.DISABLED)
                self.btn_stop_propagate.config(state=tk.DISABLED)
        except tk.TclError:
            pass

    def set_propagate_button_states_paused(self):
        try:
            self.btn_start_propagate.config(state=tk.NORMAL, text="Resume \u25b6")
            self.btn_pause_propagate.config(state=tk.DISABLED)
            self.btn_stop_propagate.config(state=tk.NORMAL)

            # During pause: allow polygon creation, object deletion, clear all
            self.btn_polygon_mode.config(state=tk.NORMAL)
            self.btn_delete_selected.config(state=tk.NORMAL)
            self.btn_clear_tracked.config(state=tk.NORMAL)

            self.btn_reassign_bbox_selected.config(state=tk.DISABLED)
            self.btn_pcs_detect.config(state=tk.DISABLED)
            # Enable review slider for frame inspection
            self.review_frame_slider.config(state=tk.NORMAL)
            # Keep name setting enabled (renaming allowed during pause)
            self.btn_set_custom_label.config(state=tk.NORMAL)
        except tk.TclError:
            pass

    def enable_review_controls(self, enable=True):
        state = tk.NORMAL if enable else tk.DISABLED
        try:
            self.btn_cut_from_here.config(state=state)
            self.btn_discard_frame.config(state=state)
            self.btn_confirm_labels.config(state=state)
            self.review_frame_slider.config(state=tk.NORMAL)
        except tk.TclError:
            pass
        try:
            self.btn_cut_dlmi.config(state=state)
            self.btn_cut_load.config(state=state)
        except (tk.TclError, AttributeError):
            pass

    def update_review_slider_range(self, max_frame):
        try:
            self.review_frame_slider.config(to=max_frame)
        except tk.TclError:
            pass

    def _on_mask_alpha_change(self, value=None):
        if hasattr(self.app, 'current_cv_frame') and self.app.current_cv_frame is not None:
            if hasattr(self.app, '_get_current_masks_for_display'):
                self.app._display_cv_frame_on_view(
                    self.app.current_cv_frame,
                    self.app._get_current_masks_for_display()
                )

    def _toggle_review_guide(self):
        if self.review_guide_collapsed:
            self.review_guide_label.pack(anchor='w', padx=10, pady=(0, 5))
            self.chunk_settings_frame.pack(anchor='w', padx=10, pady=(0, 5))
            self.chunk_threshold_label.pack(side=tk.LEFT, padx=2)
            self.chunk_threshold_scale.pack(side=tk.LEFT, padx=2)
            self.btn_toggle_review_guide.config(text="▼ Hide Guide")
            self.review_guide_collapsed = False
        else:
            self.review_guide_label.pack_forget()
            self.chunk_settings_frame.pack_forget()
            self.btn_toggle_review_guide.config(text="▶ View Guide")
            self.review_guide_collapsed = True

    def update_review_frame_info(self, current_frame, total_frames):
        try:
            self.review_frame_info_label.config(text=f"{current_frame} / {total_frames}")
        except tk.TclError:
            pass

    def update_discarded_frames_display(self, discarded_frames_set):
        try:
            if discarded_frames_set:
                sorted_frames = sorted(discarded_frames_set)
                if len(sorted_frames) > 10:
                    display_text = f"Discarded: {sorted_frames[:10]}... (total {len(sorted_frames)})"
                else:
                    display_text = f"Discarded: {sorted_frames}"
                self.discarded_frames_label.config(text=display_text)
            else:
                self.discarded_frames_label.config(text="")
        except tk.TclError:
            pass

    def update_discard_button_state(self, is_discarded):
        try:
            if is_discarded:
                self.btn_discard_frame.config(text="Restore ↩", bg="#ffcdd2")
            else:
                self.btn_discard_frame.config(text="Discard 🗑", bg="#fff3e0")
        except tk.TclError:
            pass

    def _on_review_slider_change(self, value):
        frame_idx = int(float(value))
        self.review_frame_input_var.set(str(frame_idx))
        if hasattr(self.app, 'on_review_frame_change'):
            self.app.on_review_frame_change(frame_idx)

    def _on_review_frame_input_enter(self, event=None):
        self._goto_review_frame()

    def _on_review_frame_goto_click(self):
        self._goto_review_frame()

    def _goto_review_frame(self):
        try:
            frame_idx = int(self.review_frame_input_var.get())
            max_frame = int(self.review_frame_slider.cget('to'))
            min_frame = int(self.review_frame_slider.cget('from'))

            if frame_idx < min_frame:
                frame_idx = min_frame
            elif frame_idx > max_frame:
                frame_idx = max_frame

            self.review_frame_slider.set(frame_idx)
            self.review_frame_input_var.set(str(frame_idx))

        except ValueError:
            current_val = int(self.review_frame_slider.get())
            self.review_frame_input_var.set(str(current_val))

    def update_custom_save_options_state(self):
        state = tk.NORMAL if self.app.use_custom_save_path_var.get() else tk.DISABLED
        for widget in self.custom_save_widgets_frame.winfo_children():
            try:
                if isinstance(widget, tk.Entry) and state == tk.DISABLED:
                    widget.config(state=tk.NORMAL)
                    widget.config(state='readonly')
                else:
                    widget.config(state=state)
            except tk.TclError:
                pass

    def update_batch_options_state(self):
        state = tk.NORMAL if self.app.batch_processing_mode_var.get() else tk.DISABLED
        for widget in self.batch_widgets_frame.winfo_children():
            try:
                if isinstance(widget, tk.Entry) and state == tk.DISABLED:
                    widget.config(state=tk.NORMAL)
                    widget.config(state='readonly')
                else:
                    widget.config(state=state)
            except tk.TclError:
                pass

    def _on_notebook_tab_change_attempt(self, event):
        if self.app.autolabel_active:
            return "break"
        return None

    def _make_scrollable_tab_frame(self, tab_text):
        """Create a scrollable container frame for a notebook tab. Returns the
        inner frame that widgets should use as their parent. The canvas keeps
        the inner frame at its natural requested width so the vertical
        scrollbar does NOT eat horizontal space from the content; the whole
        notebook simply grows wider by the scrollbar width.
        Mouse-wheel scrolls only while the cursor is over the tab."""
        wrapper = tk.Frame(self.notebook)
        canvas = tk.Canvas(wrapper, highlightthickness=0, borderwidth=0)
        scrollbar = tk.Scrollbar(wrapper, orient=tk.VERTICAL, command=canvas.yview)
        inner = tk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_inner_configure(_event=None):
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
                req_w = inner.winfo_reqwidth()
                if req_w > 1:
                    cur = canvas.cget('width')
                    try:
                        cur_int = int(float(cur))
                    except (TypeError, ValueError):
                        cur_int = 0
                    if req_w != cur_int:
                        canvas.configure(width=req_w)
                    canvas.itemconfig(inner_id, width=req_w)
            except tk.TclError:
                pass

        def _on_canvas_configure(event):
            try:
                req_w = inner.winfo_reqwidth()
                target = max(event.width, req_w)
                canvas.itemconfig(inner_id, width=target)
            except tk.TclError:
                pass

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_wheel(event):
            try:
                if getattr(event, 'num', 0) == 4 or getattr(event, 'delta', 0) > 0:
                    canvas.yview_scroll(-3, "units")
                elif getattr(event, 'num', 0) == 5 or getattr(event, 'delta', 0) < 0:
                    canvas.yview_scroll(3, "units")
            except tk.TclError:
                pass

        def _enter(_event=None):
            canvas.bind_all("<MouseWheel>", _on_wheel)
            canvas.bind_all("<Button-4>", _on_wheel)
            canvas.bind_all("<Button-5>", _on_wheel)

        def _leave(_event=None):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        for w in (canvas, inner):
            w.bind("<Enter>", _enter)
            w.bind("<Leave>", _leave)

        self.notebook.add(wrapper, text=tab_text)
        return inner

    def _on_tab_changed(self, event=None):
        if hasattr(self.app, 'current_cv_frame') and self.app.current_cv_frame is not None:
            if hasattr(self.app, '_get_current_masks_for_display'):
                self.root.after(50, lambda: self.app._display_cv_frame_on_view(
                    self.app.current_cv_frame,
                    self.app._get_current_masks_for_display()
                ))

    def update_original_frame_display(self, pil_image):
        pass

    def clear_original_canvas(self):
        pass

    def _bind_events(self):
        logger.debug("Binding events.")
        self.canvas.bind("<Configure>", self.app._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self.app._on_left_mouse_press)
        self.canvas.bind("<B1-Motion>", self.app._on_left_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.app._on_left_mouse_release)
        self.canvas.bind("<Control-Button-2>", self.app._on_ctrl_middle_click_for_point)
        self.canvas.bind("<Control-Button-3>", self.app._on_ctrl_right_click_for_point)
        self.canvas.bind("<ButtonPress-3>", self.app._on_right_mouse_press)
        self.canvas.bind("<B3-Motion>", self.app._on_right_mouse_drag)
        self.canvas.bind("<ButtonRelease-3>", self.app._on_right_mouse_release) 
        self.root.bind("<KeyPress-Control_L>", self.app._on_ctrl_press)
        self.root.bind("<KeyPress-Control_R>", self.app._on_ctrl_press)
        self.root.bind("<KeyRelease-Control_L>", self.app._on_ctrl_release)
        self.root.bind("<KeyRelease-Control_R>", self.app._on_ctrl_release)
        self.root.bind("<KeyPress-Shift_L>", self.app._on_shift_press)
        self.root.bind("<KeyPress-Shift_R>", self.app._on_shift_press)
        self.root.bind("<KeyRelease-Shift_L>", self.app._on_shift_release)
        self.root.bind("<KeyRelease-Shift_R>", self.app._on_shift_release)
        self.root.bind("<KeyPress-Alt_L>", self.app._on_alt_press)
        self.root.bind("<KeyPress-Alt_R>", self.app._on_alt_press)
        self.root.bind("<KeyRelease-Alt_L>", self.app._on_alt_release)
        self.root.bind("<KeyRelease-Alt_R>", self.app._on_alt_release)
        self.root.bind("<space>", self.app._on_spacebar_press)

        self.notebook.bind("<ButtonPress-1>", self._on_notebook_tab_change_attempt)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self.root.protocol("WM_DELETE_WINDOW", self.app._on_closing_window_confirm)
        logger.debug("Event binding complete.")

    def update_status(self, message):
        self.progress_label.config(text=f"Status: {message}")

    def update_obj_id_info_label(self):
        current_mode = "Normal (Draw BBox)"; next_action_prefix = "New manual BBox"
        if self.app.reassign_bbox_mode_active_sam_id is not None:
            current_mode = f"Object {self.app.reassign_bbox_mode_active_sam_id} BBox reassign"
            next_action_prefix = ""
        elif self.app.problematic_highlight_active_sam_id is not None:
            current_mode = f"Object {self.app.problematic_highlight_active_sam_id} issue check"
            next_action_prefix = ""
        elif self.app.interaction_correction_pending is not None:
            current_mode = f"Object {self.app.interaction_correction_pending} auto-correction (Draw BBox)"
            next_action_prefix = f"Correction for object {self.app.interaction_correction_pending}"
        elif self.app.is_ctrl_pressed and self.app.is_shift_pressed:
            current_mode = "Object delete mode (Ctrl+Shift+Click)"
            next_action_prefix = ""
        elif self.app.is_ctrl_pressed and self.app.selected_object_sam_id is not None:
            current_mode = f"Object {self.app.selected_object_sam_id} selected (add point or reassign BBox)"
            next_action_prefix = ""
        elif self.app.is_ctrl_pressed:
            current_mode = "Object select (Ctrl+Left-click)"
            next_action_prefix = ""
        id_proposal_text = f"ObjID (proposed): {self.app.next_obj_id_to_propose}" if next_action_prefix and not self.app._is_any_special_mode_active() else ""
        self.obj_id_info_label.config(text=f"Mode: {current_mode} | {next_action_prefix} {id_proposal_text}")

    def display_image(self, pil_image_to_draw_on):
        if pil_image_to_draw_on is None: return
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width < 2 or canvas_height < 2: canvas_width, canvas_height = 800, 600
        img_w, img_h = pil_image_to_draw_on.size
        if img_h == 0: logger.error("Image height 0."); return
        aspect_ratio = img_w / img_h
        if canvas_width / aspect_ratio <= canvas_height:
            new_w = canvas_width; new_h = int(canvas_width / aspect_ratio)
        else:
            new_h = canvas_height; new_w = int(canvas_height * aspect_ratio)
        if new_w <= 0 or new_h <= 0: logger.error(f"Invalid resize: {new_w}x{new_h}"); return
        resized_image = pil_image_to_draw_on.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image) 
        self.app.scale_x = img_w / new_w; self.app.scale_y = img_h / new_h
        self.app.offset_x = (canvas_width - new_w) // 2
        self.app.offset_y = (canvas_height - new_h) // 2
        if self.canvas_image_item: self.canvas.delete(self.canvas_image_item)
        self.canvas_image_item = self.canvas.create_image(
            self.app.offset_x, self.app.offset_y, anchor=tk.NW, image=self.tk_image
        )

    def draw_temp_bbox(self, x1, y1, x2, y2):
        if self.temp_bbox_on_canvas_id:
            self.canvas.coords(self.temp_bbox_on_canvas_id, x1, y1, x2, y2)
        else:
            self.temp_bbox_on_canvas_id = self.canvas.create_rectangle(
                x1, y1, x2, y2, outline="cyan", width=2, tags="temp_bbox"
            )
    
    def delete_temp_bbox(self):
        if self.temp_bbox_on_canvas_id:
            self.canvas.delete(self.temp_bbox_on_canvas_id)
            self.temp_bbox_on_canvas_id = None
            
    def get_temp_bbox_coords(self):
        if self.temp_bbox_on_canvas_id:
            return self.canvas.coords(self.temp_bbox_on_canvas_id)
        return None

    def clear_canvas_image(self):
        if self.canvas_image_item:
            self.canvas.delete(self.canvas_image_item)
            self.canvas_image_item = None
            self.tk_image = None 

    def set_ui_element_state(self, element_name, state):
        widget_map = {
            "btn_select_source": self.btn_select_source,
            "btn_clear_tracked": self.btn_clear_tracked,
            "btn_set_custom_label": self.btn_set_custom_label,
            "entry_default_label": self.entry_default_label,
            "btn_reassign_bbox_selected": self.btn_reassign_bbox_selected,
            "btn_delete_selected": self.btn_delete_selected,
            "notebook": self.notebook,
            "btn_start_batch": self.btn_start_batch,
            "btn_skip_batch": self.btn_skip_batch,
            "btn_load_label": self.btn_load_label
        }
        if element_name == "notebook_tabs":
            for child_widget in self.notebook.winfo_children():
                self._set_widget_state_recursively(child_widget, state)

        elif element_name in widget_map:
            widget = widget_map[element_name]
            if isinstance(widget, (tk.LabelFrame, ttk.Notebook)):
                self._set_widget_state_recursively(widget, state)
            else:
                try: widget.config(state=state)
                except tk.TclError: pass 
        else:
            logger.warning(f"Unknown UI element name for state change: {element_name}")

    def _set_widget_state_recursively(self, widget, state):
        entry_state = state if state in ('normal', 'disabled') else 'disabled'

        if isinstance(widget, tk.Entry):
            try: widget.config(state=entry_state)
            except tk.TclError: pass
        elif hasattr(widget, 'config'):
            try:
                if 'state' in widget.config():
                    widget.config(state=state)
            except tk.TclError: pass

        # tk.Canvas is needed so the scrollable-tab wrappers (Frame→Canvas→Frame)
        # propagate state to their inner-frame children. Without Canvas in this
        # tuple, every widget inside a scrollable notebook tab becomes
        # unreachable from set_ui_element_state("notebook_tabs", ...).
        if isinstance(widget, (tk.Frame, tk.LabelFrame, tk.Canvas)):
            for child in widget.winfo_children():
                self._set_widget_state_recursively(child, state)

    def _on_ui_display_change(self, event=None):
        if hasattr(self.app, 'current_cv_frame') and self.app.current_cv_frame is not None:
            if hasattr(self.app, '_get_current_masks_for_display'):
                self.app._display_cv_frame_on_view(
                    self.app.current_cv_frame,
                    self.app._get_current_masks_for_display()
                )

    def _on_pose_add_mode_toggle(self):
        if not hasattr(self.app, 'pose_add_mode_var'):
            return
        if self.app.pose_add_mode_var.get():
            if hasattr(self.app, 'clear_pose_selection'):
                self.app.clear_pose_selection()
            if hasattr(self, 'delete_temp_bbox'):
                self.delete_temp_bbox()
        else:
            classified = 0
            if hasattr(self.app, '_automatch_all_new_pose_objects'):
                try:
                    classified = self.app._automatch_all_new_pose_objects()
                except Exception as _e:
                    logger.debug(f"automatch classify failed: {_e}")
            if classified > 0 and hasattr(self.app, 'update_status'):
                self.app.update_status(f"Auto-matched class for {classified} pose object(s).")

    def _toggle_tabs_visibility(self):
        if hasattr(self.app, 'tabs_visible_var'):
            is_visible = self.app.tabs_visible_var.get()
            if is_visible:
                self.controls_and_config_frame.grid_remove()
                self.app.tabs_visible_var.set(False)
                self.btn_toggle_tabs.config(text="Show Tabs")
            else:
                self.controls_and_config_frame.grid(row=0, column=1, sticky="ns")
                self.app.tabs_visible_var.set(True)
                self.btn_toggle_tabs.config(text="Hide Tabs")

    def _toggle_guide_visibility(self):
        if self.guide_visible:
            self.guide_label.pack_forget()
            self.btn_guide_toggle.config(text="▶ Guide")
            self.guide_visible = False
        else:
            self.guide_label.pack(side=tk.TOP, anchor='w', padx=5, pady=(2, 5), after=self.guide_toggle_frame)
            self.btn_guide_toggle.config(text="▼ Guide")
            self.guide_visible = True

    def _on_small_obj_filter_change(self):
        if hasattr(self.app, 'current_cv_frame') and self.app.current_cv_frame is not None:
            if hasattr(self.app, '_get_current_masks_for_display'):
                self.app._display_cv_frame_on_view(
                    self.app.current_cv_frame,
                    self.app._get_current_masks_for_display()
                )
        logger.info(f"Small object filter: {'enabled' if self.app.filter_small_objects_var.get() else 'disabled'}")

    def _on_small_obj_threshold_change(self, event=None):
        try:
            percent_value = float(self.small_obj_threshold_entry.get())
            ratio_value = percent_value / 100.0
            self.app.small_object_threshold_var.set(ratio_value)
            logger.info(f"Small object filter threshold changed: {percent_value}% (ratio: {ratio_value})")
            self._on_small_obj_filter_change()
        except ValueError:
            logger.warning("Small object threshold must be a number.")
            current_ratio = self.app.small_object_threshold_var.get()
            self.small_obj_threshold_entry.delete(0, tk.END)
            self.small_obj_threshold_entry.insert(0, f"{current_ratio * 100:.4f}")

    def _on_small_contour_filter_change(self):
        if hasattr(self.app, 'current_cv_frame') and self.app.current_cv_frame is not None:
            if hasattr(self.app, '_get_current_masks_for_display'):
                self.app._display_cv_frame_on_view(
                    self.app.current_cv_frame,
                    self.app._get_current_masks_for_display()
                )
        logger.info(f"Small contour filter: {'enabled' if self.app.filter_small_contours_var.get() else 'disabled'}")

    def _on_small_contour_threshold_change(self, event=None):
        try:
            percent_value = float(self.small_contour_threshold_entry.get())
            ratio_value = percent_value / 100.0
            self.app.small_contour_threshold_var.set(ratio_value)
            logger.info(f"Small contour filter threshold changed: {percent_value}% (ratio: {ratio_value})")
            self._on_small_contour_filter_change()
        except ValueError:
            logger.warning("Small contour threshold must be a number.")
            current_ratio = self.app.small_contour_threshold_var.get()
            self.small_contour_threshold_entry.delete(0, tk.END)
            self.small_contour_threshold_entry.insert(0, f"{current_ratio * 100:.4f}")

    def _on_point_size_change(self, event=None):
        try:
            value = float(self.polygon_point_size_entry.get())
            value = max(0.1, min(5.0, value))
            self.polygon_point_size_entry.delete(0, tk.END)
            self.polygon_point_size_entry.insert(0, f"{value:.2f}")
            if hasattr(self.app, 'polygon_point_size_percent_var'):
                self.app.polygon_point_size_percent_var.set(value)
            self._on_ui_display_change()
            logger.info(f"Point/prompt size changed: {value}%")
        except ValueError:
            logger.warning("Point size must be a number.")
            self.polygon_point_size_entry.delete(0, tk.END)
            self.polygon_point_size_entry.insert(0, "0.40")

    def _on_font_size_change(self, event=None):
        try:
            value = float(self.label_font_size_entry.get())
            value = max(0.1, min(5.0, value))
            self.label_font_size_entry.delete(0, tk.END)
            self.label_font_size_entry.insert(0, f"{value:.2f}")
            if hasattr(self.app, 'label_font_size_percent_var'):
                self.app.label_font_size_percent_var.set(value)
            self._on_ui_display_change()
            logger.info(f"Font size changed: {value}%")
        except ValueError:
            logger.warning("Font size must be a number.")
            self.label_font_size_entry.delete(0, tk.END)
            self.label_font_size_entry.insert(0, "0.70")

    def _on_sam2_toggle(self):
        is_enabled = self.app.sam2_enabled_var.get()
        if is_enabled:
            self.sam2_status_label.config(text="(Loading...)", fg="orange")
            self.check_sam2_enabled.config(state=tk.DISABLED)
            self.root.update_idletasks()
            self.app.load_sam2_model_async()
        else:
            self.app.transfer_sam2_masks_to_sam3_and_unload()
            self._update_sam2_ui_state(enabled=False)
        logger.info(f"SAM2 toggle: {'enabled' if is_enabled else 'disabled'}")

    def _on_sam2_tracking_toggle(self):
        is_tracking = self.app.sam2_tracking_enabled_var.get()
        logger.info(f"SAM2 tracking mode: {'enabled' if is_tracking else 'disabled'}")

    def _update_sam2_ui_state(self, enabled: bool, loading: bool = False):
        if loading:
            self.sam2_status_label.config(text="(Loading...)", fg="orange")
            self.check_sam2_enabled.config(state=tk.DISABLED)
            self.check_sam2_tracking.config(state=tk.DISABLED)
        elif enabled:
            self.sam2_status_label.config(text="(Active)", fg="green")
            self.check_sam2_enabled.config(state=tk.NORMAL)
            self.check_sam2_tracking.config(state=tk.NORMAL)
        else:
            self.sam2_status_label.config(text="(Inactive)", fg="gray50")
            self.check_sam2_enabled.config(state=tk.NORMAL)
            self.check_sam2_tracking.config(state=tk.DISABLED)
            self.app.sam2_tracking_enabled_var.set(False)

    def update_sam2_loading_complete(self, success: bool):
        if success:
            self._update_sam2_ui_state(enabled=True)
            logger.info("SAM2 model loading complete - UI enabled")
        else:
            self._update_sam2_ui_state(enabled=False)
            self.app.sam2_enabled_var.set(False)
            logger.error("SAM2 model loading failed - UI disabled")

    def _on_low_level_api_toggle(self):
        is_enabled = self.app.low_level_api_enabled_var.get() if hasattr(self.app, 'low_level_api_enabled_var') else False

        if hasattr(self, 'btn_low_data_inject'):
            if is_enabled and hasattr(self.app, 'tracked_objects') and len(self.app.tracked_objects) > 0:
                self.btn_low_data_inject.config(state='normal')
            else:
                self.btn_low_data_inject.config(state='disabled')

        logger.info(f"Low-level API: {'enabled' if is_enabled else 'disabled'}")
        if hasattr(self.app, 'update_status'):
            if is_enabled:
                self.app.update_status("Low-level API enabled. Use 'Inject Data' button to inject masks.")
            else:
                self.app.update_status("Low-level API disabled.")

    def update_low_data_inject_button_state(self):
        if not hasattr(self, 'btn_low_data_inject'):
            return

        is_api_enabled = self.app.low_level_api_enabled_var.get() if hasattr(self.app, 'low_level_api_enabled_var') else False
        has_masks = hasattr(self.app, 'tracked_objects') and len(self.app.tracked_objects) > 0

        if is_api_enabled and has_masks:
            self.btn_low_data_inject.config(state='normal')
        else:
            self.btn_low_data_inject.config(state='disabled')

    def update_polygon_mode_ui(self, is_active):
        is_paused = getattr(self.app, 'app_state', '') == "PAUSED"

        if is_active:
            self.btn_polygon_mode.config(text="Polygon Mode ON", bg="#a5d6a7")
            self.btn_polygon_complete.config(state='normal')
            self.btn_polygon_undo.config(state='normal')
            self.btn_polygon_cancel.config(state='normal')
            if is_paused:
                self.polygon_status_label.config(text="DLMI: Left-click to add points | Complete → DLMI Inject", fg="blue")
            else:
                self.polygon_status_label.config(text="Left-click to add points | Right-click to undo", fg="blue")

            has_polygon_obj = any(
                data.get('is_polygon_object', False)
                for data in self.app.tracked_objects.values()
            ) if hasattr(self.app, 'tracked_objects') else False
            # During pause, disable SAM Input (use DLMI injection instead)
            self.btn_polygon_to_sam3.config(state='disabled' if is_paused else ('normal' if has_polygon_obj else 'disabled'))
        else:
            self.btn_polygon_mode.config(text="Add Polygon", bg="#c8e6c9")
            self.btn_polygon_complete.config(state='disabled')
            self.btn_polygon_undo.config(state='disabled')
            self.btn_polygon_cancel.config(state='disabled')
            self.polygon_status_label.config(text="Left-click to add polygon points.", fg="gray50")

            has_polygon_obj = any(
                data.get('is_polygon_object', False)
                for data in self.app.tracked_objects.values()
            ) if hasattr(self.app, 'tracked_objects') else False
            # During pause, disable SAM Input (use DLMI injection instead)
            self.btn_polygon_to_sam3.config(state='disabled' if is_paused else ('normal' if has_polygon_obj else 'disabled'))
