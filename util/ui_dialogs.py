"""Reusable tkinter progress dialog helpers.

`open_download_dialog(app, url, dest)` returns a dict with a determinate
progress bar suitable for URL downloads (bar['value'] 0..100).
`open_loading_dialog(app, title, subtitle)` returns a dict with an
indeterminate spinner bar for "waiting for a background op" scenarios
(pip install, model load, etc.).

Both dialogs:
  - are Toplevel grabbing modal focus
  - centre themselves over `app.root`
  - return a dict exposing the `window`, `bar`, `status` label and a
    `close()` callable (loading) / `cancel_flag` dict (download).
"""
import tkinter as tk
import tkinter.ttk as ttk


def _centre_over_root(app, win):
    try:
        win.update_idletasks()
        app.root.update_idletasks()
        rx = app.root.winfo_rootx() + app.root.winfo_width() // 2 - win.winfo_width() // 2
        ry = app.root.winfo_rooty() + app.root.winfo_height() // 2 - win.winfo_height() // 2
        win.geometry(f"+{max(0, rx)}+{max(0, ry)}")
    except Exception:
        pass


def open_download_dialog(app, url, dest):
    """Modal download progress dialog with a cancel flag.
    Returns {'window', 'bar', 'status', 'cancel_flag': {'flag': bool}}."""
    win = tk.Toplevel(app.root)
    win.title("Downloading TAPNext++")
    win.transient(app.root)
    win.resizable(False, False)
    try:
        win.grab_set()
    except tk.TclError:
        pass
    tk.Label(win, text="Downloading TAPNext++ checkpoint",
             font=("TkDefaultFont", 10, "bold")).pack(padx=20, pady=(12, 4))
    tk.Label(win, text=f"From: {url}", fg="gray40",
             font=("TkDefaultFont", 8)).pack(padx=20, pady=2)
    tk.Label(win, text=f"To:   {dest}", fg="gray40",
             font=("TkDefaultFont", 8)).pack(padx=20, pady=2)
    bar = ttk.Progressbar(win, orient=tk.HORIZONTAL, length=420,
                          mode='determinate', maximum=100)
    bar.pack(padx=20, pady=(8, 4))
    status = tk.Label(win, text="Starting...", font=("TkDefaultFont", 9))
    status.pack(padx=20, pady=2)
    cancel_flag = {"flag": False}

    def _cancel():
        cancel_flag["flag"] = True
        status.config(text="Cancelling...")

    tk.Button(win, text="Cancel", command=_cancel, width=10).pack(pady=(4, 12))
    win.protocol("WM_DELETE_WINDOW", _cancel)
    _centre_over_root(app, win)
    return {"window": win, "bar": bar, "status": status, "cancel_flag": cancel_flag}


def open_loading_dialog(app, title, subtitle):
    """Modal indeterminate-progress dialog. The status label is writeable
    to reflect sub-phase progress (e.g. 'Installing einops...').
    Returns {'window', 'bar', 'status', 'close': callable}."""
    win = tk.Toplevel(app.root)
    win.title(title)
    win.transient(app.root)
    try:
        win.grab_set()
    except tk.TclError:
        pass
    win.resizable(False, False)
    tk.Label(win, text=title, font=("TkDefaultFont", 10, "bold")
             ).pack(padx=20, pady=(12, 4))
    tk.Label(win, text=subtitle, fg='gray40',
             font=('TkDefaultFont', 8), justify=tk.LEFT
             ).pack(padx=20, pady=2)
    bar = ttk.Progressbar(win, orient=tk.HORIZONTAL, length=420, mode='indeterminate')
    bar.pack(padx=20, pady=(8, 4))
    bar.start(50)
    status = tk.Label(win, text="Starting...", font=('TkDefaultFont', 9),
                      wraplength=420, justify=tk.LEFT)
    status.pack(padx=20, pady=(2, 14))
    _centre_over_root(app, win)

    def _close():
        try:
            bar.stop()
            win.destroy()
        except Exception:
            pass

    return {"window": win, "status": status, "bar": bar, "close": _close}
