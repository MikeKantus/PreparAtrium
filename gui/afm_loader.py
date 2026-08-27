*** Begin Patch
*** Update File: gui/afm_loader.py
@@
     def save_metadata_and_video(self):
-        if self.current_stack is None:
-            self.status_label.setText("No processed stack to save")
-            return
-        path, _ = QFileDialog.getSaveFileName(
-            self,
-            "Save video",
-            "",
-            "AVI Files (*.avi);;MP4 Files (*.mp4)"
-        )
-
-        if not path:
-            return
-        folder = os.path.dirname(path)
-        meta_path = os.path.splitext(path)[0] + "_metadata.json"
-        try:
-            with open(meta_path, "w") as f:
-                json.dump(self.meta, f, indent=2, default=str)
-        except Exception as e:
-            self.status_label.setText(f"Error saving metadata: {e}")
-            return
-        try:
-            stack = np.asarray(self.current_stack)
+        if self.current_stack is None:
+            self.status_label.setText("No processed stack to save")
+            return
+
+        path, _ = QFileDialog.getSaveFileName(
+            self,
+            "Save session / video",
+            "",
+            "AVI Files (*.avi);;MP4 Files (*.mp4);;Session files (*.npz)"
+        )
+
+        if not path:
+            return
+
+        base = os.path.splitext(path)[0]
+
+        try:
+            # Always save a session .npz containing the stack and profiles
+            session_npz = base + ".session.npz"
+            np.savez_compressed(session_npz,
+                                 stack=np.asarray(self.current_stack),
+                                 profiles=np.asarray(self.profiles) if hasattr(self, 'profiles') else np.array([]))
+        except Exception as e:
+            self.status_label.setText(f"Error saving session npz: {e}")
+            return
+
+        # Save metadata JSON with basename.json (no suffix)
+        try:
+            meta_out = {}
+            if self.meta is not None:
+                # convert ndarrays to lists and handle NaN
+                for k, v in self.meta.items():
+                    try:
+                        if isinstance(v, np.ndarray):
+                            meta_out[k] = v.tolist()
+                        else:
+                            meta_out[k] = v
+                    except Exception:
+                        meta_out[k] = str(v)
+
+            # add required fields
+            first_frame_name = None
+            try:
+                first_frame = self.current_stack[0]
+                first_frame_name = getattr(first_frame, 'name', None) or getattr(self, 'first_frame_name', None)
+            except Exception:
+                first_frame_name = getattr(self, 'first_frame_name', None)
+
+            meta_out.update({
+                'first_frame_name': first_frame_name,
+                'timestamp': datetime.datetime.now().isoformat(),
+                'n_frames': len(self.current_stack),
+                'saved_by': 'afm_loader.save_metadata_and_video'
+            })
+
+            json_path = base + ".json"
+            with open(json_path, 'w') as f:
+                json.dump(meta_out, f, indent=2)
+        except Exception as e:
+            self.status_label.setText(f"Error saving metadata JSON: {e}")
+            return
+
+        # If the user chose an AVI/MP4 path, also write video
+        if path.lower().endswith(('.avi', '.mp4')):
+            try:
+                stack = np.asarray(self.current_stack)
*** End Patch