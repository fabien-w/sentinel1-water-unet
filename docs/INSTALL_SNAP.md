# Installing ESA SNAP and `esa_snappy`

SNAP is needed only for **step d** (preprocessing a new Sentinel-1 scene). Prediction and
post-processing run without it.

Tested with SNAP 14.0.0 and `esa-snappy` 1.1.3 on Ubuntu 22.04/24.04 (WSL2 and AWS EC2), Python 3.11 and 3.12.

## 1. Install SNAP (Sentinel toolboxes, ~1 GB)

Linux, unattended:

```bash
wget https://download.esa.int/step/snap/14.0/installers/esa-snap_sentinel_linux-14.0.0.sh
sh esa-snap_sentinel_linux-14.0.0.sh -q -dir $HOME/esa-snap
```

Windows and macOS: download the installer from <https://step.esa.int/main/download/snap-download/>
and keep the default options.

## 2. Install and configure `esa_snappy` in your Python environment

```bash
conda activate s1water                       # the environment created from environment.yml
pip install esa-snappy
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
$HOME/esa-snap/bin/snappy-conf "$(which python)" "$SITE"
```

On Windows: `"C:\Program Files\esa-snap\bin\snappy-conf.bat" <path-to-python.exe> <site-packages>`.

## 3. Memory

`snappy-conf` sets `java_max_mem` to 70 % of the RAM in `<site-packages>/esa_snappy/esa_snappy.ini`.
Lower it if other programs need memory, e.g. `java_max_mem: 8G` (enough for the default 0.5° area
of interest). A full scene needs about 16-24 GB.

## 4. Check

```bash
python -c "import esa_snappy; print('esa_snappy OK')"
```

## Notes

- Auxiliary data (precise orbit files, GETASSE30 DEM) are downloaded automatically by SNAP the first
  time and cached in `~/.snap/auxdata` (~100 MB for one scene).
- Precise orbit files are published ~20 days after the acquisition. For more recent scenes the package
  falls back to restituted orbits and prints a warning.
- The original study used SNAP 9.0.0. SNAP 14 gives water masks that agree on 99.9 % of pixels.
