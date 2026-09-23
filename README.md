# local-file-organizer (V1 — inventario read-only)

Organizador inteligente de archivos personales, 100% local. Esta V1
**únicamente inventaría y analiza** una carpeta: no mueve, renombra, borra
ni modifica ningún archivo original, y no realiza ninguna llamada a
servicios en la nube ni usa IA generativa.

## Qué hace

1. Recorre recursivamente una carpeta objetivo (ignorando directorios como
   `.git`, `.venv`, `venv`, `node_modules`, `__pycache__`, `.cache`, `.npm`,
   `.docker` — configurable).
2. Clasifica cada archivo por extensión: `video`, `image`, `document`,
   `audio`, `archive`, `software`, `other`.
3. Extrae metadatos generales (ruta, nombre, extensión, categoría, tamaño,
   fecha de modificación, tipo MIME).
4. Para videos, extrae metadata técnica vía `ffprobe` (duración, resolución,
   codecs, FPS, bitrate) si `ffprobe` está disponible en el sistema.
5. Detecta duplicados exactos de forma eficiente: agrupa por tamaño y solo
   calcula SHA-256 para los grupos con tamaños repetidos.
6. Guarda todo en `data/catalog.sqlite` (re-escanear no crea filas
   duplicadas: usa UPSERT sobre `path`).
7. Exporta `reports/inventory.csv`.
8. Imprime un resumen estadístico al finalizar.

## Qué NO hace (por diseño, en esta V1)

No mueve, copia, renombra ni borra archivos; no modifica metadatos ni
timestamps originales; no usa IA generativa ni APIs externas; no sube nada
a la nube; no hace reconocimiento facial ni extrae frames de video; no
reorganiza carpetas.

## Requisitos

- Python 3.9+ (solo biblioteca estándar; no hay dependencias externas).
- `ffmpeg`/`ffprobe` instalado en el sistema para metadata de video (si no
  está disponible, el escaneo continúa y solo se omite esa metadata,
  registrando el motivo por archivo).

## Uso

Desde la raíz del proyecto:

```bash
python -m local_file_organizer scan /ruta/a/analizar
python -m local_file_organizer summary
python -m local_file_organizer duplicates
```

Opciones de `scan`:

- `--db RUTA` — ruta del catálogo SQLite (por defecto `data/catalog.sqlite`).
- `--csv RUTA` — ruta del reporte CSV (por defecto `reports/inventory.csv`).
- `--ignore-dir NOMBRE` — directorio adicional a ignorar (repetible).
- `--quiet` — reduce mensajes de progreso.
- `--verify-readonly` — captura un snapshot antes/después del escaneo y
  reporta si algún archivo original cambió (added/removed/size_changed/
  mtime_changed). Recomendado para el primer piloto.
- `--snapshot-dir RUTA` — dónde guardar `before.json`/`after.json` (por
  defecto `data/snapshots/`).

Necesitas tener el paquete en el `PYTHONPATH`, por ejemplo ejecutando desde
la raíz del repo con:

```bash
PYTHONPATH=src python -m local_file_organizer scan /ruta/a/analizar
```

o instalando el proyecto en modo editable:

```bash
pip install -e .
python -m local_file_organizer scan /ruta/a/analizar
```

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Los tests solo operan sobre directorios temporales; nunca tocan archivos
personales reales.

## Garantías de seguridad

- Todo lo que el programa escribe vive bajo `data/` y `reports/` dentro de
  este repositorio.
- El código fuente no contiene llamadas a `os.remove`, `os.unlink`,
  `os.rmdir`, `os.rename`, `os.replace`, `shutil.move`, `shutil.copy` ni
  `shutil.copy2` dirigidas a la carpeta analizada — ver
  `tests/test_safety_gate.py`.
- Nunca se invoca `sudo` ni se cambian permisos automáticamente; los
  errores de lectura se registran y el escaneo continúa.
- `local_file_organizer.snapshot` permite capturar y comparar snapshots
  (`path`, `size`, `mtime_ns`) de la carpeta escaneada antes/después de un
  escaneo, como prueba en tiempo de ejecución (no solo estática) de que los
  originales no cambiaron. Actívalo con `scan --verify-readonly`; los
  snapshots se guardan en `data/snapshots/` (nunca en la carpeta escaneada).

## Modelo de errores por archivo

Cada fila en `files` distingue dos fallos independientes:

- `scan_status` / `scan_error`: si el inventario básico del archivo (stat,
  tamaño, mtime, y su lectura para hashing de duplicados) se pudo hacer.
  Un fallo aquí sí implica que la fila puede estar incompleta.
- `metadata_status` / `metadata_error`: si la extracción de metadata
  especializada por categoría (hoy: `ffprobe` para video) tuvo éxito.
  Valores: `ok` (extraída), `error` (se intentó y falló, p. ej. un video
  corrupto), `skipped` (no aplica para esa categoría, o el escaneo básico
  ya había fallado). Un fallo aquí **nunca** convierte `scan_status` en
  `error`: el archivo sigue correctamente inventariado, solo le falta esa
  metadata adicional.

## Frontera de privacidad (futuro)

Esta V1 no se conecta a ningún proveedor externo. Se documenta aquí, de
forma temprana, la frontera que deberá respetar cualquier módulo futuro
que hable con un orquestador remoto (p. ej. Luna vía API o AGY), para que
el catálogo nunca se acople directamente a un proveedor:

```text
LOCAL ONLY (nunca sale de esta máquina)
- archivos originales (imágenes, videos, documentos)
- frames extraídos
- embeddings faciales
- hashes de contenido (sha256, perceptuales)
- rutas absolutas / rutas sensibles

REMOTE-ELIGIBLE (lo único que un futuro módulo podría enviar, ya sanitizado)
- file_id opaco (no la ruta real)
- categoría, tamaño, duración, resolución
- identificadores relativos (P001/P002, PAIR_001, DUP_00001)
- tags no sensibles, estadísticas agregadas
- estructura lógica propuesta (plan.json)
```

Un futuro módulo tipo `ai_gateway` sería responsable de sanitizar el
catálogo antes de enviarlo; no existe todavía ninguna abstracción de ese
tipo en esta V1, y no debe añadirse hasta que sea necesaria.

## Roadmap (no implementado en esta V1)

- V2: análisis de imagen/video, detección facial, embeddings, clustering
  de personas, hashing perceptual, extracción inteligente de frames.
- V3: parsing de documentos, OCR local, clasificación documental.
- V4: búsqueda local, consultas SQLite, embeddings locales.
- V5: `plan.json`, dry-run de movimientos, apply, rollback.
