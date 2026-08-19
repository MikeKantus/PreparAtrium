import sys
import os
import traceback
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

try:
    from playnano.io.loader import load_afm_stack
    print("PlayNano importado correctamente.")
except Exception as e:
    print("ERROR importando PlayNano:", e)
    sys.exit(1)


def test_playnano_folder(folder):
    print("\n=== TEST PLAYNANO HS-AFM ===")
    print("Carpeta seleccionada:", folder)

    try:
        afm = load_afm_stack(folder)
        print("OK: load_afm_stack funcionó")
        print("Shape:", afm.data.shape)
        print("Pixel size:", afm.pixel_size_nm)
        print("Channel:", afm.channel)
        print("Frames:", len(afm.frame_metadata))
    except Exception as e:
        print("\nERROR en load_afm_stack:")
        traceback.print_exc()


def main():
    app = QApplication(sys.argv)

    dialog = QFileDialog()
    dialog.setWindowTitle("Selecciona la carpeta HS-AFM (contiene muchos .jpk)")
    dialog.setFileMode(QFileDialog.Directory)

    if dialog.exec():
        folder = dialog.selectedFiles()[0]

        # Si el usuario selecciona un archivo por error
        if os.path.isfile(folder):
            QMessageBox.warning(
                None,
                "Error",
                "Has seleccionado un archivo .jpk.\n\n"
                "Para HS-AFM debes seleccionar la carpeta que contiene todos los .jpk."
            )
            return

        test_playnano_folder(folder)

    else:
        QMessageBox.information(None, "Cancelado", "No se seleccionó ninguna carpeta.")

    sys.exit()


if __name__ == "__main__":
    main()
