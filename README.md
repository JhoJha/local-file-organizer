# local-file-organizer (V2.3 — inventario read-only + similitud visual + muestreo de frames + detección facial)

Organizador inteligente de archivos personales, 100% local. El sistema
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
5. Para imágenes, calcula un **hash perceptual** (dHash) y agrupa imágenes
   visualmente similares — ver [Similitud visual de imágenes (V2.1)](#similitud-visual-de-imágenes-v21).
6. Detecta duplicados exactos de forma eficiente: agrupa por tamaño y solo
   calcula SHA-256 para los grupos con tamaños repetidos.
7. Guarda todo en `data/catalog.sqlite` (re-escanear no crea filas
   duplicadas: usa UPSERT sobre `path`).
8. Exporta `reports/inventory.csv`.
9. Imprime un resumen estadístico al finalizar.
10. (V2.2) `sample-frames` muestrea frames de video de forma efímera con
    `ffmpeg` y guarda solo los resultados (índice, timestamp, estado) en
    `video_frame_samples` — ver
    [Muestreo de frames de video (V2.2)](#muestreo-de-frames-de-video-v22).
11. (V2.3) `detect-faces` detecta regiones con forma de cara en imágenes y
    en frames muestreados de video, y guarda solo números (bbox +
    confianza) en `face_observations` — ver
    [Detección facial local (V2.3)](#detección-facial-local-v23).

## Qué NO hace (por diseño)

No mueve, copia, renombra ni borra archivos; no modifica metadatos ni
timestamps originales; no usa IA generativa ni APIs externas; no sube nada
a la nube; no hace reconocimiento facial: V2.3 *detecta* regiones con
forma de cara (que *hay* una cara en tal bbox con tal confianza) pero
sigue sin *reconocer* ni identificar quién es nadie — no hay nombres,
embeddings ni IDs de persona (eso queda para V2.4); no reorganiza
carpetas.

## Requisitos

- Python 3.9+.
- [Pillow](https://python-pillow.org/) (`pillow>=9.1,<13`) — usada solo
  para decodificar imágenes y calcular su hash perceptual (ver más abajo
  por qué es necesaria).
- [opencv-python-headless](https://opencv.org/) (`opencv-python-headless>=4.8,<5.1`) —
  necesario solo para la detección facial local (V2.3: YuNet vía
  `cv2.FaceDetectorYN`); variante `-headless` porque es una herramienta
  CLI sin GUI (instalación más pequeña, sin bindings X11/Qt).
- `ffmpeg`/`ffprobe` instalado en el sistema para metadata de video (si no
  está disponible, el escaneo continúa y solo se omite esa metadata,
  registrando el motivo por archivo).

Instala las dependencias del proyecto (idealmente en un entorno virtual):

```bash
pip install -e .
```

## Uso

Desde la raíz del proyecto:

```bash
python -m local_file_organizer scan /ruta/a/analizar
python -m local_file_organizer summary
python -m local_file_organizer duplicates
python -m local_file_organizer similar-images
python -m local_file_organizer sample-frames /ruta/a/analizar
python -m local_file_organizer detect-faces /ruta/a/analizar
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
- `--similarity-threshold N` — distancia Hamming máxima (0-64) para
  agrupar dos imágenes como visualmente similares (por defecto `8`; ver
  [Similitud visual de imágenes (V2.1)](#similitud-visual-de-imágenes-v21)).

`similar-images` acepta `--db RUTA` y lista los grupos ya calculados por el
último `scan` (no vuelve a leer las imágenes).

Opciones de `sample-frames`:

- `--db RUTA` — ruta del catálogo SQLite (por defecto `data/catalog.sqlite`).
- `--max-samples N` — tope de frames por video (por defecto `30`).
- `--keep-debug-frames` — guarda los frames extraídos bajo `--debug-dir`
  en vez de borrarlos (solo para depuración manual).
- `--debug-dir RUTA` — dónde guardar los frames de depuración (por defecto
  `data/debug_frames/`).
- `--redact-paths` — no imprime rutas por video, solo progreso agregado.
- `--verify-readonly` — igual que en `scan`: snapshot antes/después y
  bloque `=== VERIFICACIÓN READ-ONLY ===`.
- `--snapshot-dir RUTA` — dónde guardar `before.json`/`after.json` (por
  defecto `data/snapshots/`).
- `--quiet` — reduce mensajes de progreso.

Opciones de `detect-faces`:

- `--db RUTA` — ruta del catálogo SQLite (por defecto `data/catalog.sqlite`).
- `--confidence-threshold F` — confianza mínima del detector (por defecto
  `0.6`).
- `--min-face-size N` — lado mínimo en píxeles del bbox para conservar una
  detección (por defecto `20`).
- `--max-video-samples N` — tope de frames por video, pasado a la
  estrategia de muestreo de V2.2 (por defecto `30`).
- `--redact-paths` — no imprime rutas por archivo, solo el bloque agregado
  final.
- `--verify-readonly` — igual que en `scan`: snapshot antes/después y
  bloque `=== VERIFICACIÓN READ-ONLY ===`.
- `--snapshot-dir RUTA` — dónde guardar `before.json`/`after.json` (por
  defecto `data/snapshots/`).
- `--quiet` — reduce mensajes de progreso.

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
- El hash perceptual de imágenes (V2.1) solo **abre y decodifica en
  memoria** cada imagen con Pillow para calcular su hash; nunca la
  reescribe, recomprime ni guarda una copia. No se envía ninguna imagen,
  hash ni ruta a ningún servicio externo.

## Similitud visual de imágenes (V2.1)

Para cada archivo de categoría `image` (`.png`, `.jpg`/`.jpeg`, y el resto
de extensiones ya soportadas por el clasificador), `scan` calcula un hash
perceptual y agrupa imágenes visualmente parecidas — sin enviar ninguna
imagen, hash ni ruta a ningún servicio externo.

### Algoritmo: dHash (difference hash)

Se eligió **dHash** en vez de un pHash clásico (basado en DCT) porque
resuelve el objetivo de esta iteración (similitud perceptual aproximada,
no exacta) con una implementación mucho más simple: solo requiere reducir
la imagen a escala de grises de `9x8` píxeles y comparar cada píxel con su
vecino de la derecha, sin ninguna transformada de frecuencia. El resultado
son 64 bits (16 caracteres hex) guardados en `perceptual_hash`.

Trade-off explícito: dHash es más sensible a rotaciones y recortes fuertes
que un pHash clásico. Para una primera iteración conservadora (imágenes
recomprimidas, reescaladas o guardadas en otro formato) es suficiente.

### Distancia y umbral

La similitud entre dos hashes se mide con **distancia de Hamming** (bits
distintos entre los 64 bits de cada hash). El umbral por defecto es
**8 bits sobre 64** (`--similarity-threshold`, configurable en `scan`):
distancias de 0-5 corresponden casi siempre a la misma imagen
recomprimida o reescalada, 6-10 siguen siendo en la práctica imágenes
relacionadas, y por encima de eso empiezan a aparecer falsos positivos
(imágenes distintas que comparten estructura clara/oscura similar). Se
optó deliberadamente por un umbral conservador (menos falsos positivos)
en vez de uno agresivo.

### Agrupación

Las imágenes cuya distancia es `<= umbral` se agrupan de forma transitiva
(unión de conjuntos: si A~B y B~C, las tres quedan en el mismo grupo) en
identificadores `SIM_00001`, `SIM_00002`, etc., guardados en
`similar_group`. Una imagen sin ninguna otra suficientemente parecida
queda con `similar_group = NULL`, igual que un archivo único no entra en
ningún `duplicate_group`.

### `similar_group` vs. `duplicate_group`: no son el mismo concepto

`duplicate_group` (de V1) exige **bytes idénticos** (SHA-256).
`similar_group` (V2.1) exige solo **parecido visual aproximado**: dos
imágenes con contenido visualmente igual pero guardadas en formatos o
compresiones distintas (p. ej. el mismo original exportado a `.png` y a
`.jpg`) nunca comparten `duplicate_group`, pero sí pueden compartir
`similar_group`. Son columnas independientes, calculadas por mecanismos
distintos.

### Limitaciones de esta iteración

- Solo imágenes (`category = 'image'`); **no** hay hashing perceptual de
  video, ni nada relacionado con detección facial,
  embeddings, clustering de personas u OCR — eso queda para iteraciones
  futuras. (La extracción efímera de frames vive en V2.2 como paso previo,
  sin análisis facial.)
- El umbral es global por escaneo, no ajustable por grupo.
- El algoritmo no es robusto a rotaciones grandes, espejados ni recortes
  agresivos — solo a recompresión, reescalado moderado y pequeños cambios
  de formato/color.

### Consultar resultados

```bash
python -m local_file_organizer similar-images
```

Imprime cada grupo con su cantidad de imágenes, la distancia Hamming
máxima dentro del grupo, y la ruta/tamaño/extensión de cada archivo. El
resumen impreso por `scan` y `summary` también incluye cuántas imágenes
tienen hash calculado, cuántos grupos se detectaron y cuántas imágenes
quedaron agrupadas vs. únicas.

## Muestreo de frames de video (V2.2)

Para cada archivo de categoría `video`, `sample-frames` estima la duración
con `ffprobe` y extrae una serie de frames con `ffmpeg` — sin enviar ningún
frame, hash ni ruta a ningún servicio externo.

### Estrategia de muestreo

El número objetivo de muestras depende de la duración (antes de aplicar el
tope `--max-samples`): hasta 60 s → 8 frames; hasta 300 s → 12; hasta
900 s → 20; más → 30. Los timestamps se reparten uniformemente sobre la
ventana útil `[margen, duración - margen]`, donde el margen es
`max(duración * 0.05, 0.2)` segundos desde cada extremo (nunca más de la
mitad de la duración), para evitar los negros de intro/outro. Nunca se
incluyen `t=0` ni `t=duración` exactos, y el resultado siempre sale
ordenado y sin duplicados.

### Efímero por defecto

Los frames se extraen a un directorio temporal que **siempre se elimina**
al terminar cada video (incluso si el consumidor del frame lanza una
excepción), y nunca se escribe nada dentro del árbol escaneado. El resumen
final incluye `temporary_frames_remaining` (verificado de verdad: se
comprueba que los directorios temporales ya no existen), que siempre debe
ser `0`. La única excepción es `--keep-debug-frames`, un escape hatch solo
para depuración manual que conserva los JPG bajo `data/debug_frames/`
(ignorado por git) en subcarpetas `0000/`, `0001/`, … numeradas
secuencialmente para no filtrar nombres de archivo.

Cada video deja sus resultados (índice, timestamp, `generated`/`error`) en
la tabla `video_frame_samples` del catálogo; re-ejecutar reemplaza las
filas de ese video en vez de acumularlas. Un video corrupto no aborta el
resto: sus muestras quedan en `error` y el programa sigue con el siguiente.

Todavía **no** hay embeddings ni clustering de personas:
V2.2 solo demuestra que el pipeline de muestreo funciona y que los frames
se limpian correctamente. La detección facial vive en V2.3 (ver abajo);
los embeddings y el clustering quedan explícitamente para V2.4+.

```bash
python -m local_file_organizer sample-frames /ruta/a/analizar
```

## Detección facial local (V2.3)

Para cada archivo de categoría `image`, y para cada frame muestreado
(efímeramente, con la misma estrategia de V2.2) de cada archivo de
categoría `video`, `detect-faces` detecta regiones con forma de cara —
sin enviar ninguna imagen, frame, hash ni ruta a ningún servicio externo.

### Backend: YuNet vía OpenCV, 100% local

El detector es `cv2.FaceDetectorYN` (OpenCV) con el modelo YuNet
`face_detection_yunet_2023mar.onnx`, vendoreado en `models/` (232589
bytes, SHA-256
`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`,
origen: `https://github.com/opencv/opencv_zoo`,
`models/face_detection_yunet/face_detection_yunet_2023mar.onnx`). El
modelo viaja dentro del repo: en tiempo de detección no hay ninguna
llamada de red.

### Qué se guarda (y qué no)

Nunca se persiste ningún recorte de cara ni ningún byte de imagen/frame:
cada detección guarda solo números en la tabla `face_observations`
(bbox `x/y/width/height`, `detection_confidence`, dimensiones de la
imagen/frame origen, más `source_type` — `'image'` o `'video_frame'` — y
`timestamp_ms` para frames de video). Re-ejecutar reemplaza las filas de
cada archivo en vez de acumularlas, y `detect-faces` deja además las
mismas filas de diagnóstico en `video_frame_samples` que `sample-frames`.

### Filtros de calidad

Dos filtros independientes: `--confidence-threshold` (confianza mínima
del detector, por defecto `0.6`) y `--min-face-size` (lado mínimo en
píxeles del bbox, por defecto `20`; una cara exactamente en el umbral se
conserva).

```bash
python -m local_file_organizer detect-faces /ruta/a/analizar
```

El resumen final (`=== FACE DETECTION ===`) solo muestra agregados, nunca
rutas: imágenes procesadas, frames de video procesados, observaciones
totales, archivos con/sin caras, errores del detector y segundos
transcurridos.

Este hito **no** identifica ni nombra a nadie: solo detecta que *existe*
una región con forma de cara. Los embeddings y el clustering de personas
(quién es quién) quedan explícitamente para V2.4.

## Modelo de errores por archivo

Cada fila en `files` distingue dos fallos independientes:

- `scan_status` / `scan_error`: si el inventario básico del archivo (stat,
  tamaño, mtime, y su lectura para hashing de duplicados) se pudo hacer.
  Un fallo aquí sí implica que la fila puede estar incompleta.
- `metadata_status` / `metadata_error`: si la extracción de metadata
  especializada por categoría (`ffprobe` para video, hash perceptual para
  imagen) tuvo éxito. Valores: `ok` (extraída), `error` (se intentó y
  falló, p. ej. un video o una imagen corrupta), `skipped` (no aplica para
  esa categoría, o el escaneo básico ya había fallado). Un fallo aquí
  **nunca** convierte `scan_status` en `error`: el archivo sigue
  correctamente inventariado, solo le falta esa metadata adicional. Una
  imagen que no pudo decodificarse (bytes corruptos, extensión engañosa)
  queda con `scan_status='ok'` (se pudo inventariar) y
  `metadata_status='error'` con el motivo de Pillow en `metadata_error`.

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

## Roadmap (no implementado todavía)

- V2.1 (implementado): hashing perceptual de imágenes y agrupación por
  similitud visual.
- V2.2 (implementado): muestreo efímero de frames de video (`sample-frames`,
  tabla `video_frame_samples`); sin detección facial todavía.
- V2.3 (implementado): detección facial local (`detect-faces`, tabla
  `face_observations`); detecta regiones, no identifica personas.
- V2.4+: embeddings faciales, clustering de personas.
- V3: parsing de documentos, OCR local, clasificación documental.
- V4: búsqueda local, consultas SQLite, embeddings locales.
- V5: `plan.json`, dry-run de movimientos, apply, rollback.
