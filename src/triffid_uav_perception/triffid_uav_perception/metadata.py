"""
DJI XMP Metadata Extractor
===========================
Extracts drone/camera/GPS metadata from the XMP block embedded in JPEG/TIFF
images produced by DJI drones (tested on M30T).

The XMP is stored as an XML string inside the JPEG APP1 marker. We pull it
out with a regex (no need for a full EXIF library) and parse the drone-dji
namespace attributes.

Returned dataclass has typed fields ready for geo-projection.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# Namespace used by DJI in XMP blocks
_DJI_NS = 'http://www.dji.com/drone-dji/1.0/'

# Regex to find the XMP packet inside raw file bytes.
# XMP is always UTF-8 text bracketed by <x:xmpmeta> tags.
_XMP_RE = re.compile(
    b'<x:xmpmeta[^>]*>.*?</x:xmpmeta>',
    re.DOTALL,
)


@dataclass
class DJIMetadata:
    """Parsed DJI drone image metadata."""

    # --- Drone position (WGS-84) ---
    lat: float               # GpsLatitude (degrees, + = North)
    lon: float               # GpsLongitude (degrees, + = East)
    abs_alt: float            # AbsoluteAltitude (m, ellipsoidal WGS-84)
    rel_alt: float            # RelativeAltitude (m, above takeoff)

    # --- Gimbal orientation (NED frame, degrees) ---
    gimbal_yaw: float         # Yaw from True North, clockwise positive
    gimbal_pitch: float       # Negative = looking downward
    gimbal_roll: float        # Usually ~0

    # --- Flight attitude (degrees) ---
    flight_yaw: float
    flight_pitch: float
    flight_roll: float

    # --- RTK quality ---
    gps_status: str           # e.g. "RTK"
    rtk_flag: int             # 0/15/16/34/50 (50 = best)

    # --- Laser rangefinder ---
    lrf_status: str           # "Normal" or other
    lrf_distance: float       # metres, camera → target
    lrf_target_lat: float     # target point latitude
    lrf_target_lon: float     # target point longitude
    lrf_target_abs_alt: float # target point ellipsoidal altitude

    # --- Camera ---
    camera_model: str         # e.g. "M30T"
    image_source: str         # e.g. "WideCamera", "ZoomCamera", "ThermalCamera"

    @property
    def rtk_is_fixed(self) -> bool:
        """True when RTK has integer ambiguity solution (cm-level)."""
        return self.rtk_flag == 50

    @property
    def lrf_valid(self) -> bool:
        """True when laser rangefinder reading is usable."""
        return self.lrf_status == 'Normal' and self.lrf_distance > 0


def extract_xmp_xml(image_path: str) -> Optional[str]:
    """Read an image file and return the raw XMP XML string, or None."""
    data = Path(image_path).read_bytes()
    match = _XMP_RE.search(data)
    if match is None:
        return None
    return match.group(0).decode('utf-8', errors='replace')


def parse_xmp(xmp_xml: str) -> DJIMetadata:
    """Parse a DJI XMP XML string into a DJIMetadata dataclass.

    The DJI attributes sit on the rdf:Description element as XML attributes
    in the drone-dji namespace. We register the namespace and pull them out.
    """
    root = ET.fromstring(xmp_xml)

    # Find the rdf:Description element that carries drone-dji attributes
    ns = {
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'drone-dji': _DJI_NS,
    }
    desc = root.find('.//rdf:Description', ns)
    if desc is None:
        raise ValueError('No rdf:Description element found in XMP')

    def _get(attr: str, default: str = '0') -> str:
        """Get a drone-dji attribute, stripping leading +/- whitespace."""
        key = f'{{{_DJI_NS}}}{attr}'
        val = desc.get(key, default)
        return val.strip()

    def _float(attr: str, default: float = 0.0) -> float:
        try:
            return float(_get(attr, str(default)))
        except (ValueError, TypeError):
            return default

    def _int(attr: str, default: int = 0) -> int:
        try:
            return int(_get(attr, str(default)))
        except (ValueError, TypeError):
            return default

    return DJIMetadata(
        lat=_float('GpsLatitude'),
        lon=_float('GpsLongitude'),
        abs_alt=_float('AbsoluteAltitude'),
        rel_alt=_float('RelativeAltitude'),
        gimbal_yaw=_float('GimbalYawDegree'),
        gimbal_pitch=_float('GimbalPitchDegree'),
        gimbal_roll=_float('GimbalRollDegree'),
        flight_yaw=_float('FlightYawDegree'),
        flight_pitch=_float('FlightPitchDegree'),
        flight_roll=_float('FlightRollDegree'),
        gps_status=_get('GpsStatus', 'Unknown'),
        rtk_flag=_int('RtkFlag'),
        lrf_status=_get('LRFStatus', 'Unknown'),
        lrf_distance=_float('LRFTargetDistance'),
        lrf_target_lat=_float('LRFTargetLat'),
        lrf_target_lon=_float('LRFTargetLon'),
        lrf_target_abs_alt=_float('LRFTargetAbsAlt'),
        camera_model=_get('DroneModel', 'Unknown'),
        image_source=_get('ImageSource', 'Unknown'),
    )


def extract_metadata(image_path: str) -> DJIMetadata:
    """One-shot: extract XMP from an image file and parse it.

    Raises ValueError if no XMP block is found.
    """
    xmp_xml = extract_xmp_xml(image_path)
    if xmp_xml is None:
        raise ValueError(f'No XMP metadata found in {image_path}')
    return parse_xmp(xmp_xml)
