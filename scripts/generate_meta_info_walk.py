#!/usr/bin/env python3
import os
import argparse

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def generate_meta_info(root_dir: str, resolution: str, output_file: str, to_csv: bool):
    dir_count = 0
    line_count = 0

    # Si queremos CSV, prepararnos:
    if to_csv:
        import csv
        f_out = open(output_file, "w", newline="")
        writer = csv.writer(f_out)
        writer.writerow(["image_path"])  # cabecera, o añade más columnas si quieres
    else:
        f_out = open(output_file, "w")
        writer = None

    # Recorremos todo el árbol
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) == resolution:
            dir_count += 1
            # *** Aquí mostramos en qué carpeta estamos ***
            print(f"[{dir_count:>3}] Procesando carpeta: {dirpath}")

            # Recorremos subdirectorios de esa carpeta de resolución
            for subpath, _, files in os.walk(dirpath):
                for fname in files:
                    if os.path.splitext(fname)[1].lower() in IMG_EXTS:
                        full_path = os.path.join(subpath, fname)
                        if to_csv:
                            writer.writerow([full_path])
                        else:
                            f_out.write(full_path + "\n")
                        line_count += 1

    f_out.close()
    print(f"\n➡️  Carpetas encontradas: {dir_count}")
    print(f"➡️  Rutas escritas   : {line_count} en {output_file!r}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Genera meta_info (TXT o CSV) listando imágenes de una resolución.")
    p.add_argument("-r","--root_dir", required=True,
                   help="Raíz con todas las carpetas de muestras")
    p.add_argument("-s","--resolution", required=True,
                   help="Carpeta de resolución, p.ej. '1024px'")
    p.add_argument("-o","--output", default="meta_info.txt",
                   help="Archivo de salida (.txt o .csv)")
    p.add_argument(      "--csv", action="store_true",
                   help="Si se pasa, genera CSV en lugar de TXT")
    args = p.parse_args()
    generate_meta_info(args.root_dir, args.resolution, args.output, args.csv)
