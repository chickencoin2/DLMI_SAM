import os
import logging
import yaml
import shutil
import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
import time

logger = logging.getLogger("DLMI_SAM_LABELER.BatchController")


def prompt_yolo_class_info(app):
    dialog = tk.Toplevel(app.root)
    dialog.title("YOLO Class Information Input")
    dialog.geometry("400x300")
    dialog.transient(app.root)
    dialog.grab_set()

    tk.Label(dialog, text="Number of classes (nc):").pack(pady=5)
    nc_entry = tk.Entry(dialog)
    nc_entry.pack(pady=5)
    nc_entry.insert(0, "1")

    tk.Label(dialog, text="Class name list (comma separated):").pack(pady=5)
    names_text = tk.Text(dialog, height=5, width=40)
    names_text.pack(pady=5)
    names_text.insert("1.0", "object")

    result = {"confirmed": False}

    def on_ok():
        try:
            nc = int(nc_entry.get())
            names_str = names_text.get("1.0", tk.END).strip()
            names = [name.strip() for name in names_str.split(',') if name.strip()]

            if nc != len(names):
                messagebox.showerror("Error", f"Number of classes ({nc}) does not match the number of names ({len(names)}).", parent=dialog)
                return

            app.yolo_nc = nc
            app.yolo_class_names_for_save = names
            result["confirmed"] = True
            dialog.destroy()
        except ValueError:
            messagebox.showerror("Error", "Number of classes must be an integer.", parent=dialog)

    def on_cancel():
        dialog.destroy()

    tk.Button(dialog, text="OK", command=on_ok).pack(side=tk.LEFT, padx=50, pady=10)
    tk.Button(dialog, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=50, pady=10)

    dialog.wait_window()
    return result["confirmed"]


def init_yolo_dataset_structure(app, save_dir):
    try:
        images_dir = os.path.join(save_dir, "images")
        labels_dir = os.path.join(save_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        if app.save_format_var.get() == "both":
            labelme_dir = os.path.join(save_dir, "labelme")
            os.makedirs(labelme_dir, exist_ok=True)
            logger.info(f"LabelMe folder created: {labelme_dir}")

        yaml_path = os.path.join(save_dir, "data.yaml")
        yaml_data = {
            'path': '',
            'train': '',
            'val': '',
            'test': '',
            'nc': app.yolo_nc,
            'names': app.yolo_class_names_for_save
        }

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False)

        logger.info(f"YOLO dataset structure created: {save_dir}")
        app.yolo_dataset_initialized = True
        return True
    except Exception as e:
        logger.error(f"YOLO dataset structure creation failed: {e}")
        return False


def update_yolo_yaml(app):
    try:
        save_dir = app._get_save_directory()
        yaml_path = os.path.join(save_dir, "data.yaml")

        yaml_data = {
            'path': '',
            'train': '',
            'val': '',
            'test': '',
            'nc': app.yolo_nc,
            'names': app.yolo_class_names_for_save
        }

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, allow_unicode=True, default_flow_style=False)

        logger.info(f"data.yaml updated")
    except Exception as e:
        logger.error(f"data.yaml update failed: {e}")


def check_existing_yolo_dataset(app, save_dir):
    yaml_path = os.path.join(save_dir, "data.yaml")
    images_dir = os.path.join(save_dir, "images")
    labels_dir = os.path.join(save_dir, "labels")

    if os.path.exists(yaml_path) and os.path.exists(images_dir) and os.path.exists(labels_dir):
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)

            nc = yaml_data.get('nc', 0)
            names = yaml_data.get('names', [])

            response = messagebox.askyesnocancel(
                "Existing YOLO Dataset Found",
                f"Existing YOLO dataset found.\n"
                f"Number of classes: {nc}\n"
                f"Class list: {', '.join(names)}\n\n"
                f"Yes: Use existing settings\n"
                f"No: Setup new settings\n"
                f"Cancel: Abort operation",
                parent=app.root
            )

            if response is None:
                return None
            elif response:
                app.yolo_nc = nc
                app.yolo_class_names_for_save = names
                app.yolo_dataset_initialized = True
                logger.info(f"Existing YOLO settings loaded: nc={nc}, names={names}")
                return "use_existing"
            else:
                return "new_setup"
        except Exception as e:
            logger.error(f"Failed to load existing YOLO data: {e}")
            return "new_setup"

    return "new_setup"


def get_save_directory(app):
    if app.use_custom_save_path_var.get():
        base_dir = app.custom_save_dir_var.get()
        if app.batch_processing_mode_var.get() and app.batch_save_option_var.get() == "subfolder":
            video_name = os.path.splitext(os.path.basename(app.video_source_path))[0] if isinstance(app.video_source_path, str) else "camera"
            folder_template = app.custom_folder_name_var.get()
            folder_name = folder_template.format(video_name=video_name)
            return os.path.join(base_dir, folder_name)
        return base_dir
    return app.AUTOLABEL_FOLDER_val


def start_batch_processing(app):
    if not app.batch_processing_mode_var.get():
        messagebox.showerror("Error", "Batch processing mode is not activated.", parent=app.root)
        return

    source_dir = app.batch_source_dir_var.get()
    if not source_dir or not os.path.isdir(source_dir):
        messagebox.showerror("Error", "Please specify a valid video source folder first.", parent=app.root)
        return

    supported_formats = ('.mp4', '.avi', '.mov', '.mkv')
    app.batch_video_files = sorted([os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.lower().endswith(supported_formats) and not f.startswith('.')])

    if not app.batch_video_files:
        messagebox.showwarning("Warning", f"No supported video files found in the selected folder.\n({', '.join(supported_formats)})", parent=app.root)
        return

    if messagebox.askokcancel("Batch Processing Start Confirmation", f"Starting batch processing for {len(app.batch_video_files)} videos.", parent=app.root):
        app.is_batch_running = True
        app.batch_current_index = -1
        app.view.set_ui_element_state("btn_start_batch", tk.DISABLED)

        if app.batch_move_completed_var.get():
            completed_dir = app.batch_completed_dir_var.get()
            if not os.path.exists(completed_dir):
                try:
                    os.makedirs(completed_dir)
                    logger.info(f"Completed video folder created: {completed_dir}")
                except Exception as e:
                    logger.error(f"Failed to create completed video folder: {e}")
                    messagebox.showerror("Error", f"Failed to create completed video folder:\n{e}", parent=app.root)
                    app.is_batch_running = False
                    return

        app._load_next_batch_video()


def skip_current_batch_video(app):
    if not app.is_batch_running:
        messagebox.showinfo("Info", "Batch processing is not running.", parent=app.root)
        return

    if messagebox.askyesno("Skip Video Confirmation",
                           f"Do you want to skip the current video ({app.video_display_name})?",
                           parent=app.root):
        logger.info(f"Batch processing: Skipping current video - {app.video_source_path}")

        if app.batch_move_completed_var.get() and isinstance(app.video_source_path, str):
            move_completed_video(app, app.video_source_path, skipped=True)

        app.autolabel_active = False
        app.playback_paused = True

        app._load_next_batch_video()


def move_completed_video(app, video_path, skipped=False):
    if not app.batch_move_completed_var.get() or not isinstance(video_path, str):
        return

    try:
        if app.cap and app.cap.isOpened():
            app.cap.release()
            app.cap = None
            time.sleep(0.1)

        completed_dir = app.batch_completed_dir_var.get()
        if not os.path.exists(completed_dir):
            os.makedirs(completed_dir)

        if skipped:
            completed_dir = os.path.join(completed_dir, "skipped")
            if not os.path.exists(completed_dir):
                os.makedirs(completed_dir)

        filename = os.path.basename(video_path)
        dest_path = os.path.join(completed_dir, filename)

        if os.path.exists(dest_path):
            base, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(completed_dir, f"{base}_{counter}{ext}")
                counter += 1

        shutil.move(video_path, dest_path)
        status = "skipped" if skipped else "completed"
        logger.info(f"Video moved ({status}): {video_path} -> {dest_path}")
    except Exception as e:
        logger.error(f"Failed to move video: {e}")


def select_batch_completed_dir(app):
    dir_path = filedialog.askdirectory(title="Select Completed Video Folder", parent=app.root)
    if dir_path:
        app.batch_completed_dir_var.set(dir_path)
