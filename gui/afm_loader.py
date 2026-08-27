*** Begin Patch
*** Update File: gui/afm_loader.py
@@
-        self.label_preview = QLabel("Preview")
-        self.label_preview.setAlignment(Qt.AlignCenter)
-        self.label_preview.setScaledContents(False)            # no escalar el contenido para forzar cambio de tamaño del label
-        self.label_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
-        self.label_preview.setMinimumSize(640, 480)
-        self.label_preview.setMaximumSize(1280, 960)
+        self.label_preview = QLabel("Preview")
+        self.label_preview.setAlignment(Qt.AlignCenter)
+        self.label_preview.setScaledContents(False)            # no escalar el contenido para forzar cambio de tamaño del label
+        self.label_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
+        # Reduced preview size to fit advanced filters panel
+        self.label_preview.setMinimumSize(480, 360)
+        self.label_preview.setMaximumSize(1280, 960)
@@
-        # --- Advanced Leveling / Flattening Controls ---
-       
-        layout_adv = QVBoxLayout()
+        # --- Advanced Leveling / Flattening Controls ---
+       
+        layout_adv = QVBoxLayout()
@@
-        # Iterations
-        self.slider_iterations = QSlider(Qt.Horizontal)
-        self.slider_iterations.setMinimum(1)
-        self.slider_iterations.setMaximum(5)
-        self.slider_iterations.setValue(1)
-        layout_adv.addWidget(QLabel("Iterations"))
-        layout_adv.addWidget(self.slider_iterations)
-
-        # Apply advanced pipeline button, accept and restart
-        self.btn_apply_advanced = QPushButton("Apply Advanced Leveling")
-        self.btn_accept = QPushButton("Accept preview")
-        self.btn_accept.setMinimumHeight(36)
-        self.btn_accept.setFont(self.btn_font)
-        self.btn_restart = QPushButton("Restart editing")
-        self.btn_restart.setMinimumHeight(36)
-        self.btn_restart.setFont(self.btn_font)
-        row = QHBoxLayout()
-        row.addWidget(self.btn_apply_advanced)
-        row.addWidget(self.btn_accept)
-        row.addWidget(self.btn_restart)
-        layout_adv.addLayout(row)
+        # Iterations
+        self.slider_iterations = QSlider(Qt.Horizontal)
+        self.slider_iterations.setMinimum(1)
+        self.slider_iterations.setMaximum(5)
+        self.slider_iterations.setValue(1)
+        layout_adv.addWidget(QLabel("Iterations"))
+        layout_adv.addWidget(self.slider_iterations)
+
+        # --- New dynamic filters panel (Plane / Line / Median / Outlier) ---
+        self.combo_adv_filter = QComboBox()
+        self.combo_adv_filter.addItems(["None", "Plane", "Line", "Median", "Outlier Removal"])
+        layout_adv.addWidget(QLabel("Advanced filter"))
+        layout_adv.addWidget(self.combo_adv_filter)
+
+        # Parameter widgets (we'll show/hide according to selection)
+        # Plane params
+        self.spin_plane_order = QSpinBox()
+        self.spin_plane_order.setRange(0, 3)
+        self.spin_plane_order.setValue(1)
+        self.chk_plane_robust = QCheckBox("Robust (iterative)")
+        self.chk_plane_robust.setChecked(True)
+        self.spin_plane_iters = QSpinBox()
+        self.spin_plane_iters.setRange(1, 10)
+        self.spin_plane_iters.setValue(3)
+        self.chk_plane_ransac = QCheckBox("Use RANSAC if available")
+
+        # Line params
+        self.combo_line_method = QComboBox()
+        self.combo_line_method.addItems(["median", "mean", "linear"]) 
+        self.combo_line_fit = QComboBox()
+        self.combo_line_fit.addItems(["offset", "slope"]) 
+        self.spin_line_clip = QSpinBox()
+        self.spin_line_clip.setRange(0, 10)
+        self.spin_line_clip.setValue(0)
+
+        # Median params
+        self.spin_median_size = QSpinBox()
+        self.spin_median_size.setRange(1, 11)
+        self.spin_median_size.setSingleStep(2)
+        self.spin_median_size.setValue(3)
+
+        # Outlier params
+        self.slider_outlier_k = QSlider(Qt.Horizontal)
+        self.slider_outlier_k.setMinimum(10)
+        self.slider_outlier_k.setMaximum(60)
+        self.slider_outlier_k.setValue(30)
+        self.slider_outlier_k.setToolTip("k * sigma (scale 10x)")
+        self.spin_outlier_neigh = QSpinBox()
+        self.spin_outlier_neigh.setRange(1, 5)
+        self.spin_outlier_neigh.setValue(1)
+
+        # Buttons: Preview / Apply / Undo
+        self.btn_preview_filter = QPushButton("Preview")
+        self.btn_apply_filter = QPushButton("Apply")
+        self.btn_undo = QPushButton("Undo")
+        self.btn_preview_filter.setMinimumHeight(28)
+        self.btn_apply_filter.setMinimumHeight(28)
+        self.btn_undo.setMinimumHeight(28)
+
+        # Param layout
+        params_grid = QGridLayout()
+        params_grid.addWidget(QLabel("Plane order"), 0, 0)
+        params_grid.addWidget(self.spin_plane_order, 0, 1)
+        params_grid.addWidget(self.chk_plane_robust, 1, 0, 1, 2)
+        params_grid.addWidget(QLabel("Plane iters"), 2, 0)
+        params_grid.addWidget(self.spin_plane_iters, 2, 1)
+        params_grid.addWidget(self.chk_plane_ransac, 3, 0, 1, 2)
+
+        params_grid.addWidget(QLabel("Line method"), 4, 0)
+        params_grid.addWidget(self.combo_line_method, 4, 1)
+        params_grid.addWidget(QLabel("Line fit"), 5, 0)
+        params_grid.addWidget(self.combo_line_fit, 5, 1)
+        params_grid.addWidget(QLabel("Line clip sigma (0=off)"), 6, 0)
+        params_grid.addWidget(self.spin_line_clip, 6, 1)
+
+        params_grid.addWidget(QLabel("Median size (odd)"), 7, 0)
+        params_grid.addWidget(self.spin_median_size, 7, 1)
+
+        params_grid.addWidget(QLabel("Outlier k (sigma) *10"), 8, 0)
+        params_grid.addWidget(self.slider_outlier_k, 8, 1)
+        params_grid.addWidget(QLabel("Outlier neigh radius"), 9, 0)
+        params_grid.addWidget(self.spin_outlier_neigh, 9, 1)
+
+        params_grid.addWidget(self.btn_preview_filter, 10, 0)
+        params_grid.addWidget(self.btn_apply_filter, 10, 1)
+        params_grid.addWidget(self.btn_undo, 11, 0, 1, 2)
+
+        layout_adv.addLayout(params_grid)
@@
-        self.group_advanced.setLayout(layout_adv)
+        self.group_advanced.setLayout(layout_adv)
         left_col.addWidget(self.group_advanced)
@@
-        # Explorer connections
-        self.btn_refresh_files.clicked.connect(lambda: self.populate_parent_combo(getattr(self, "current_file_or_folder", os.getcwd())))
-        self.btn_apply_advanced.clicked.connect(lambda: self.apply_advanced_pipeline(self.processed_stack))
-        self.btn_open_in_explorer.clicked.connect(self.open_selected_folder_in_explorer)
-        self.combo_parent_files.currentIndexChanged.connect(lambda idx: self.refresh_file_preview())
-        self.btn_accept.clicked.connect(self.accept_preview)
-        self.btn_restart.clicked.connect(self.restart_editing)
+        # Explorer connections
+        self.btn_refresh_files.clicked.connect(lambda: self.populate_parent_combo(getattr(self, "current_file_or_folder", os.getcwd())))
+        self.btn_open_in_explorer.clicked.connect(self.open_selected_folder_in_explorer)
+        self.combo_parent_files.currentIndexChanged.connect(lambda idx: self.refresh_file_preview())
+        self.btn_accept.clicked.connect(self.accept_preview)
+        self.btn_restart.clicked.connect(self.restart_editing)
+
+        # Advanced filter connections
+        from core.afm_filters import (
+            plane_fit_subtract,
+            line_level,
+            median_filter,
+            despike_outliers,
+            apply_to_stack,
+        )
+
+        # Undo history (store up to 3 full copies)
+        self._undo_history = []  # list of stacks (numpy arrays)
+        self._undo_max = 3
+
+        def push_undo(stack_copy):
+            try:
+                self._undo_history.insert(0, stack_copy.copy())
+                # Trim
+                if len(self._undo_history) > self._undo_max:
+                    self._undo_history = self._undo_history[: self._undo_max]
+            except Exception:
+                pass
+
+        self._push_undo = push_undo
+
+        def do_preview():
+            if self.current_stack is None:
+                self.status_label.setText("No stack loaded for preview")
+                return
+            idx = max(0, min(self.current_frame, len(self.current_stack) - 1))
+            frame = self.current_stack[idx]
+            sel = self.combo_adv_filter.currentText()
+            try:
+                if sel == "Plane":
+                    out = plane_fit_subtract(frame, order=self.spin_plane_order.value(), robust=self.chk_plane_robust.isChecked(), iters=self.spin_plane_iters.value(), ransac=self.chk_plane_ransac.isChecked())
+                elif sel == "Line":
+                    clip = self.spin_line_clip.value() or None
+                    out = line_level(frame, method=self.combo_line_method.currentText(), fit=self.combo_line_fit.currentText(), clip_sigma=clip)
+                elif sel == "Median":
+                    out = median_filter(frame, size=self.spin_median_size.value())
+                elif sel == "Outlier Removal":
+                    k = self.slider_outlier_k.value() / 10.0
+                    neigh = self.spin_outlier_neigh.value()
+                    out, mask = despike_outliers(frame, k_sigma=k, neigh=neigh)
+                else:
+                    self.status_label.setText("No advanced filter selected for preview")
+                    return
+
+                # Show preview without mutating stack
+                qimg = frame_to_qimage_safe(out)
+                pix = QPixmap.fromImage(qimg)
+                pix = pix.scaled(self.label_preview.width(), self.label_preview.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
+                self.label_preview.setPixmap(pix)
+                self.status_label.setText(f"Preview: {sel}")
+            except Exception as e:
+                self.status_label.setText(f"Preview error: {e}")
+
+        def do_apply():
+            if self.current_stack is None:
+                self.status_label.setText("No stack loaded to apply filter")
+                return
+            sel = self.combo_adv_filter.currentText()
+            try:
+                # push undo (full copy)
+                self._push_undo(self.current_stack)
+                if sel == "Plane":
+                    new_stack = apply_to_stack(self.current_stack, plane_fit_subtract, progress_callback=lambda i, n: None, order=self.spin_plane_order.value(), robust=self.chk_plane_robust.isChecked(), iters=self.spin_plane_iters.value(), ransac=self.chk_plane_ransac.isChecked())
+                elif sel == "Line":
+                    clip = self.spin_line_clip.value() or None
+                    new_stack = apply_to_stack(self.current_stack, line_level, progress_callback=lambda i, n: None, method=self.combo_line_method.currentText(), fit=self.combo_line_fit.currentText(), clip_sigma=clip)
+                elif sel == "Median":
+                    new_stack = apply_to_stack(self.current_stack, median_filter, progress_callback=lambda i, n: None, size=self.spin_median_size.value())
+                elif sel == "Outlier Removal":
+                    k = self.slider_outlier_k.value() / 10.0
+                    neigh = self.spin_outlier_neigh.value()
+                    # apply despike and ignore mask
+                    new_stack = apply_to_stack(self.current_stack, lambda fr, k_sigma=k, neigh=neigh: despike_outliers(fr, k_sigma=k, neigh=neigh)[0])
+                else:
+                    self.status_label.setText("No advanced filter selected to apply")
+                    return
+
+                self.current_stack = np.asarray(new_stack, dtype=np.float32)
+                self.processed_stack = self.current_stack.copy()
+                self.update_preview()
+                self.status_label.setText(f"Applied: {sel}")
+            except Exception as e:
+                self.status_label.setText(f"Apply error: {e}")
+
+        def do_undo():
+            if not self._undo_history:
+                self.status_label.setText("Nothing to undo")
+                return
+            last = self._undo_history.pop(0)
+            self.current_stack = last.copy()
+            self.processed_stack = self.current_stack.copy()
+            self.update_preview()
+            self.status_label.setText("Undo: reverted last change")
+
+        self.btn_preview_filter.clicked.connect(do_preview)
+        self.btn_apply_filter.clicked.connect(do_apply)
+        self.btn_undo.clicked.connect(do_undo)
*** End Patch
