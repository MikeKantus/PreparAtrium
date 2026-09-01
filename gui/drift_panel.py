*** Begin Patch
*** Update File: gui/drift_panel.py
@@
     def save_fine_aligned_video(self):
         if self.current_stack is None:
             self.status_label.setText("No current stack to save")
             return
@@
-        for f in self.current_stack:
-            out.write(f.astype(np.uint8))
-
-        out.release()
-        self.status_label.setText("Aligned video saved")
+        for f in self.current_stack:
+            out.write(f.astype(np.uint8))
+
+        out.release()
+
+        # Save metadata JSON next to the AVI
+        try:
+            import json
+            meta = dict(self.meta) if self.meta is not None else {}
+            meta.update({
+                "n_frames": len(self.current_stack),
+                "saved_by": "drift_panel.save_fine_aligned_video",
+            })
+            if hasattr(self, "tm_drifts") and self.tm_drifts is not None:
+                try:
+                    meta["tm_drifts"] = np.asarray(self.tm_drifts).tolist()
+                except Exception:
+                    meta["tm_drifts"] = None
+            meta_path = os.path.splitext(path)[0] + "_metadata.json"
+            with open(meta_path, "w") as f:
+                json.dump(meta, f, indent=2)
+        except Exception as e:
+            # Non-fatal: report in status
+            self.status_label.setText(f"Video saved (metadata save failed: {e})")
+            return
+
+        self.status_label.setText("Aligned video and metadata saved")
*** End Patch
