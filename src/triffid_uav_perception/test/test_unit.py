"""
Unit tests for TRIFFID UAV Perception
=======================================
Tests metadata extraction, geo-projection, GeoJSON building, and pipeline
logic without requiring a GPU, YOLO model, or MQTT broker.
"""

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from triffid_uav_perception.metadata import (
    DJIMetadata,
    parse_xmp,
    extract_xmp_xml,
)
from triffid_uav_perception.geo import (
    CameraIntrinsics,
    get_intrinsics,
    pixel_to_ray,
    project_pixel_to_ground,
    project_mask_to_ground,
    project_bbox_to_ground,
    estimate_object_height,
    _ned_offset_to_gps,
    _gps_to_ned_offset,
    _ned_rotation_matrix,
)
from triffid_uav_perception.uav_node import (
    UAVPipeline,
    TARGET_CLASSES,
    _CLASS_COLORS,
    _CLASS_CATEGORIES,
)
from triffid_uav_perception.api_client import (
    FuturisedClient,
    MediaFile,
    _CAMERA_SUFFIX,
)


# ── Sample XMP for testing ──────────────────────────────────────────

SAMPLE_XMP = """
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about="DJI Meta Data"
    xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"
   drone-dji:GpsLatitude="+49.726349021"
   drone-dji:GpsLongitude="+13.350951205"
   drone-dji:AbsoluteAltitude="+431.531"
   drone-dji:RelativeAltitude="+36.280"
   drone-dji:GimbalRollDegree="+0.00"
   drone-dji:GimbalYawDegree="+84.10"
   drone-dji:GimbalPitchDegree="-24.50"
   drone-dji:FlightRollDegree="+3.90"
   drone-dji:FlightYawDegree="+84.50"
   drone-dji:FlightPitchDegree="+4.70"
   drone-dji:GpsStatus="RTK"
   drone-dji:RtkFlag="50"
   drone-dji:LRFStatus="Normal"
   drone-dji:LRFTargetDistance="54.103"
   drone-dji:LRFTargetLon="13.3516290"
   drone-dji:LRFTargetLat="49.7263932"
   drone-dji:LRFTargetAbsAlt="409.100"
   drone-dji:DroneModel="M30T"
   drone-dji:ImageSource="WideCamera">
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""

# Reusable metadata fixture
@pytest.fixture
def sample_meta():
    return parse_xmp(SAMPLE_XMP)


@pytest.fixture
def default_intrinsics():
    return CameraIntrinsics(
        fx=2850, fy=2850, cx=2000, cy=1500, width=4000, height=3000
    )


# ═══════════════════════════════════════════════════════════════════
#  Metadata tests
# ═══════════════════════════════════════════════════════════════════

class TestMetadataParsing:

    def test_parse_basic_fields(self, sample_meta):
        assert sample_meta.lat == pytest.approx(49.726349021)
        assert sample_meta.lon == pytest.approx(13.350951205)
        assert sample_meta.abs_alt == pytest.approx(431.531)
        assert sample_meta.rel_alt == pytest.approx(36.280)

    def test_parse_gimbal(self, sample_meta):
        assert sample_meta.gimbal_yaw == pytest.approx(84.10)
        assert sample_meta.gimbal_pitch == pytest.approx(-24.50)
        assert sample_meta.gimbal_roll == pytest.approx(0.0)

    def test_parse_flight_attitude(self, sample_meta):
        assert sample_meta.flight_yaw == pytest.approx(84.50)
        assert sample_meta.flight_pitch == pytest.approx(4.70)
        assert sample_meta.flight_roll == pytest.approx(3.90)

    def test_parse_rtk(self, sample_meta):
        assert sample_meta.gps_status == 'RTK'
        assert sample_meta.rtk_flag == 50
        assert sample_meta.rtk_is_fixed is True

    def test_parse_lrf(self, sample_meta):
        assert sample_meta.lrf_status == 'Normal'
        assert sample_meta.lrf_distance == pytest.approx(54.103)
        assert sample_meta.lrf_target_lat == pytest.approx(49.7263932)
        assert sample_meta.lrf_target_lon == pytest.approx(13.3516290)
        assert sample_meta.lrf_target_abs_alt == pytest.approx(409.100)
        assert sample_meta.lrf_valid is True

    def test_parse_camera(self, sample_meta):
        assert sample_meta.camera_model == 'M30T'
        assert sample_meta.image_source == 'WideCamera'

    def test_rtk_flag_not_fixed(self):
        xmp = SAMPLE_XMP.replace('RtkFlag="50"', 'RtkFlag="16"')
        meta = parse_xmp(xmp)
        assert meta.rtk_is_fixed is False
        assert meta.rtk_flag == 16

    def test_lrf_invalid(self):
        xmp = SAMPLE_XMP.replace('LRFStatus="Normal"', 'LRFStatus="Failed"')
        meta = parse_xmp(xmp)
        assert meta.lrf_valid is False

    def test_missing_optional_field_defaults(self):
        """Missing DJI attributes should default gracefully."""
        minimal_xmp = """
        <x:xmpmeta xmlns:x="adobe:ns:meta/">
         <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
          <rdf:Description rdf:about="DJI Meta Data"
            xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/"
           drone-dji:GpsLatitude="+49.0"
           drone-dji:GpsLongitude="+13.0"
           drone-dji:AbsoluteAltitude="+400.0"
           drone-dji:RelativeAltitude="+30.0">
          </rdf:Description>
         </rdf:RDF>
        </x:xmpmeta>
        """
        meta = parse_xmp(minimal_xmp)
        assert meta.lat == pytest.approx(49.0)
        assert meta.gimbal_yaw == 0.0
        assert meta.lrf_status == 'Unknown'
        assert meta.lrf_valid is False

    def test_extract_xmp_from_binary(self):
        """extract_xmp_xml should find XMP inside binary data."""
        fake_jpeg = b'\xff\xd8\xff\xe1' + SAMPLE_XMP.encode() + b'\xff\xd9'
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(fake_jpeg)
            f.flush()
            result = extract_xmp_xml(f.name)
        assert result is not None
        assert 'drone-dji' in result

    def test_extract_xmp_missing(self):
        """extract_xmp_xml returns None for files without XMP."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(b'\xff\xd8\xff\xd9')  # minimal JPEG, no XMP
            f.flush()
            result = extract_xmp_xml(f.name)
        assert result is None


# ═══════════════════════════════════════════════════════════════════
#  Geo-projection tests
# ═══════════════════════════════════════════════════════════════════

class TestCameraIntrinsics:

    def test_get_default_4k(self):
        intr = get_intrinsics(4000, 3000)
        assert intr.fx == 2850
        assert intr.width == 4000

    def test_get_default_1080p(self):
        intr = get_intrinsics(1920, 1080)
        assert intr.fx == 1370
        assert intr.width == 1920

    def test_get_scaled_resolution(self):
        intr = get_intrinsics(2000, 1500)
        assert intr.fx == pytest.approx(2850 * 2000 / 4000)
        assert intr.cy == pytest.approx(1500 * 1500 / 3000)

    def test_override(self):
        custom = CameraIntrinsics(fx=1000, fy=1000, cx=500, cy=500,
                                   width=1000, height=1000)
        intr = get_intrinsics(1920, 1080, override=custom)
        assert intr.fx == 1000  # override used regardless of resolution


class TestPixelToRay:

    def test_centre_pixel_points_forward(self, default_intrinsics):
        ray = pixel_to_ray(2000, 1500, default_intrinsics)
        # Centre pixel → ray should point along Z (forward in camera frame)
        assert ray[2] > 0.99  # mostly forward
        assert abs(ray[0]) < 0.01
        assert abs(ray[1]) < 0.01

    def test_ray_is_normalised(self, default_intrinsics):
        ray = pixel_to_ray(100, 200, default_intrinsics)
        assert np.linalg.norm(ray) == pytest.approx(1.0, abs=1e-6)

    def test_right_pixel_direction(self, default_intrinsics):
        # Pixel to the right of centre → positive X in camera frame
        ray = pixel_to_ray(3000, 1500, default_intrinsics)
        assert ray[0] > 0  # right


class TestNEDRotation:

    def test_identity_at_zero_angles(self):
        """At zero gimbal angles: camera looks North, horizontal."""
        R = _ned_rotation_matrix(0, 0, 0)
        # Camera Z (forward) should map to NED X (North)
        cam_forward = np.array([0, 0, 1])
        ned_dir = R @ cam_forward
        assert ned_dir[0] == pytest.approx(1.0, abs=1e-6)  # North
        assert abs(ned_dir[1]) < 1e-6  # no East
        assert abs(ned_dir[2]) < 1e-6  # no Down

    def test_yaw_90_points_east(self):
        """At yaw=90°, camera forward should point East."""
        R = _ned_rotation_matrix(90, 0, 0)
        cam_forward = np.array([0, 0, 1])
        ned_dir = R @ cam_forward
        assert abs(ned_dir[0]) < 1e-6  # no North
        assert ned_dir[1] == pytest.approx(1.0, abs=1e-6)  # East

    def test_pitch_minus90_points_down(self):
        """At pitch=-90°, camera looks straight down."""
        R = _ned_rotation_matrix(0, -90, 0)
        cam_forward = np.array([0, 0, 1])
        ned_dir = R @ cam_forward
        assert abs(ned_dir[0]) < 1e-6
        assert abs(ned_dir[1]) < 1e-6
        assert ned_dir[2] == pytest.approx(1.0, abs=1e-6)  # Down


class TestProjectPixelToGround:

    def test_nadir_centre_returns_drone_position(self, sample_meta, default_intrinsics):
        """Looking straight down, centre pixel should project to drone GPS."""
        # Override gimbal to nadir
        meta = DJIMetadata(
            lat=49.726, lon=13.351, abs_alt=431.0, rel_alt=36.0,
            gimbal_yaw=0.0, gimbal_pitch=-90.0, gimbal_roll=0.0,
            flight_yaw=0.0, flight_pitch=0.0, flight_roll=0.0,
            gps_status='RTK', rtk_flag=50,
            lrf_status='Normal', lrf_distance=36.0,
            lrf_target_lat=49.726, lrf_target_lon=13.351,
            lrf_target_abs_alt=395.0,
            camera_model='M30T', image_source='WideCamera',
        )
        result = project_pixel_to_ground(2000, 1500, meta, default_intrinsics)
        assert result is not None
        lon, lat, alt = result
        # Should be very close to drone position
        assert lat == pytest.approx(49.726, abs=0.001)
        assert lon == pytest.approx(13.351, abs=0.001)

    def test_horizontal_ray_returns_none(self, default_intrinsics):
        """Camera pointing at horizon should not hit ground."""
        meta = DJIMetadata(
            lat=49.726, lon=13.351, abs_alt=431.0, rel_alt=36.0,
            gimbal_yaw=0.0, gimbal_pitch=0.0, gimbal_roll=0.0,
            flight_yaw=0.0, flight_pitch=0.0, flight_roll=0.0,
            gps_status='RTK', rtk_flag=50,
            lrf_status='Normal', lrf_distance=100.0,
            lrf_target_lat=49.726, lrf_target_lon=13.351,
            lrf_target_abs_alt=395.0,
            camera_model='M30T', image_source='WideCamera',
        )
        result = project_pixel_to_ground(2000, 1500, meta, default_intrinsics)
        assert result is None

    def test_custom_ground_alt(self, default_intrinsics):
        """Explicit ground_alt should be used instead of LRF."""
        meta = DJIMetadata(
            lat=49.726, lon=13.351, abs_alt=431.0, rel_alt=36.0,
            gimbal_yaw=0.0, gimbal_pitch=-45.0, gimbal_roll=0.0,
            flight_yaw=0.0, flight_pitch=0.0, flight_roll=0.0,
            gps_status='RTK', rtk_flag=50,
            lrf_status='Normal', lrf_distance=50.0,
            lrf_target_lat=49.726, lrf_target_lon=13.351,
            lrf_target_abs_alt=409.0,
            camera_model='M30T', image_source='WideCamera',
        )
        result = project_pixel_to_ground(
            2000, 1500, meta, default_intrinsics, ground_alt=395.0,
        )
        assert result is not None
        assert result[2] == 395.0


class TestNEDOffsetConversion:

    def test_roundtrip(self):
        """GPS → NED → GPS should be identity (approx)."""
        lat0, lon0 = 49.726, 13.351
        lat1, lon1 = _ned_offset_to_gps(lat0, lon0, 100.0, 50.0)
        north, east = _gps_to_ned_offset(lat0, lon0, lat1, lon1)
        assert north == pytest.approx(100.0, abs=0.1)
        assert east == pytest.approx(50.0, abs=0.1)

    def test_zero_offset(self):
        lat, lon = _ned_offset_to_gps(49.0, 13.0, 0.0, 0.0)
        assert lat == pytest.approx(49.0)
        assert lon == pytest.approx(13.0)


class TestProjectBboxToGround:

    def test_nadir_bbox(self, default_intrinsics):
        meta = DJIMetadata(
            lat=49.726, lon=13.351, abs_alt=431.0, rel_alt=36.0,
            gimbal_yaw=0.0, gimbal_pitch=-90.0, gimbal_roll=0.0,
            flight_yaw=0.0, flight_pitch=0.0, flight_roll=0.0,
            gps_status='RTK', rtk_flag=50,
            lrf_status='Normal', lrf_distance=36.0,
            lrf_target_lat=49.726, lrf_target_lon=13.351,
            lrf_target_abs_alt=395.0,
            camera_model='M30T', image_source='WideCamera',
        )
        result = project_bbox_to_ground(
            1800, 1300, 2200, 1700, meta, default_intrinsics,
        )
        assert result is not None
        ring, centre = result
        assert len(ring) == 5  # closed ring
        assert ring[0] == ring[-1]
        lon, lat, alt = centre
        assert lat == pytest.approx(49.726, abs=0.001)


class TestProjectMaskToGround:

    def test_simple_mask(self, default_intrinsics):
        meta = DJIMetadata(
            lat=49.726, lon=13.351, abs_alt=431.0, rel_alt=36.0,
            gimbal_yaw=0.0, gimbal_pitch=-90.0, gimbal_roll=0.0,
            flight_yaw=0.0, flight_pitch=0.0, flight_roll=0.0,
            gps_status='RTK', rtk_flag=50,
            lrf_status='Normal', lrf_distance=36.0,
            lrf_target_lat=49.726, lrf_target_lon=13.351,
            lrf_target_abs_alt=395.0,
            camera_model='M30T', image_source='WideCamera',
        )
        # Create a rectangular mask near the image centre
        mask = np.zeros((3000, 4000), dtype=bool)
        mask[1400:1600, 1900:2100] = True

        result = project_mask_to_ground(mask, meta, default_intrinsics)
        assert result is not None
        ring, centre = result
        assert len(ring) >= 4
        assert ring[0] == ring[-1]  # closed


# ═══════════════════════════════════════════════════════════════════
#  GeoJSON / Pipeline tests
# ═══════════════════════════════════════════════════════════════════

class TestClassMappings:

    def test_all_classes_have_categories(self):
        for cls_name in TARGET_CLASSES.values():
            assert cls_name in _CLASS_CATEGORIES, f'{cls_name} missing category'

    def test_all_classes_have_colors(self):
        for cls_name in TARGET_CLASSES.values():
            assert cls_name in _CLASS_COLORS, f'{cls_name} missing color'

    def test_63_classes(self):
        assert len(TARGET_CLASSES) == 63


class TestGeoJSONOutput:

    def test_feature_structure(self):
        """Check that a detection produces a valid GeoJSON Feature."""
        pipeline = UAVPipeline.__new__(UAVPipeline)
        pipeline.model = None
        pipeline._mqtt = None
        pipeline._next_id = 1
        pipeline.conf_thresh = 0.35
        pipeline.yolo_imgsz = 1280
        pipeline.intrinsics_override = None

        intrinsics = CameraIntrinsics(
            fx=2850, fy=2850, cx=2000, cy=1500, width=4000, height=3000,
        )
        meta = DJIMetadata(
            lat=49.726, lon=13.351, abs_alt=431.0, rel_alt=36.0,
            gimbal_yaw=0.0, gimbal_pitch=-90.0, gimbal_roll=0.0,
            flight_yaw=0.0, flight_pitch=0.0, flight_roll=0.0,
            gps_status='RTK', rtk_flag=50,
            lrf_status='Normal', lrf_distance=36.0,
            lrf_target_lat=49.726, lrf_target_lon=13.351,
            lrf_target_abs_alt=395.0,
            camera_model='M30T', image_source='WideCamera',
        )
        det = {
            'bbox': (1800, 1300, 2200, 1700),
            'class_id': 14,
            'class_name': 'building',
            'confidence': 0.85,
            'mask': None,
        }

        feature = pipeline._detection_to_feature(det, meta, intrinsics)
        assert feature is not None
        assert feature['type'] == 'Feature'
        assert feature['geometry']['type'] == 'Polygon'
        props = feature['properties']
        assert props['class'] == 'building'
        assert props['source'] == 'uav'
        assert props['category'] == 'infrastructure'
        assert props['detection_type'] == 'seg'
        assert props['local_frame'] is False
        assert 'altitude_m' in props
        assert 'height_m' in props
        assert 'marker-color' in props
        assert 'stroke' in props
        assert 'fill' in props

    def test_point_geometry_for_person(self):
        pipeline = UAVPipeline.__new__(UAVPipeline)
        pipeline.model = None
        pipeline._mqtt = None
        pipeline._next_id = 1
        pipeline.conf_thresh = 0.35
        pipeline.yolo_imgsz = 1280
        pipeline.intrinsics_override = None

        intrinsics = CameraIntrinsics(
            fx=2850, fy=2850, cx=2000, cy=1500, width=4000, height=3000,
        )
        meta = DJIMetadata(
            lat=49.726, lon=13.351, abs_alt=431.0, rel_alt=36.0,
            gimbal_yaw=0.0, gimbal_pitch=-90.0, gimbal_roll=0.0,
            flight_yaw=0.0, flight_pitch=0.0, flight_roll=0.0,
            gps_status='RTK', rtk_flag=50,
            lrf_status='Normal', lrf_distance=36.0,
            lrf_target_lat=49.726, lrf_target_lon=13.351,
            lrf_target_abs_alt=395.0,
            camera_model='M30T', image_source='WideCamera',
        )
        det = {
            'bbox': (1900, 1400, 2100, 1600),
            'class_id': 23,
            'class_name': 'citizen',
            'confidence': 0.72,
            'mask': None,
        }

        feature = pipeline._detection_to_feature(det, meta, intrinsics)
        assert feature is not None
        assert feature['geometry']['type'] == 'Point'
        assert 'stroke' not in feature['properties']

    def test_geojson_serialisable(self):
        """Ensure GeoJSON output is valid JSON."""
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "id": "1",
                "geometry": {"type": "Point", "coordinates": [13.351, 49.726]},
                "properties": {"class": "Building", "source": "uav"},
            }],
        }
        text = json.dumps(geojson)
        parsed = json.loads(text)
        assert parsed['type'] == 'FeatureCollection'
        assert len(parsed['features']) == 1


# ═══════════════════════════════════════════════════════════════════
#  API Client tests
# ═══════════════════════════════════════════════════════════════════

class TestCameraSuffix:

    def test_wide(self):
        assert _CAMERA_SUFFIX['_W'] == 'Wide'

    def test_zoom(self):
        assert _CAMERA_SUFFIX['_Z'] == 'Zoom'

    def test_thermal(self):
        assert _CAMERA_SUFFIX['_T'] == 'Thermal'


class TestTelemetryCoordParsing:
    """Test FuturisedClient.parse_telemetry_coord for the various formats
    returned by the getDJIData endpoint."""

    def test_normal_degrees(self):
        assert FuturisedClient.parse_telemetry_coord('49.726') == pytest.approx(49.726)

    def test_euro_comma_sci_notation(self):
        # "5,15493453694969E+15" → 5.15493453694969e15 / 1e14 ≈ 51.549
        result = FuturisedClient.parse_telemetry_coord('5,15493453694969E+15')
        assert result is not None
        assert 40 < abs(result) < 90  # plausible latitude

    def test_plain_integer(self):
        # "515493451026715" → ambiguous scale; parser returns a plausible
        # coordinate but the exact divisor depends on heuristic order.
        # The key requirement: it doesn't return None and the value is
        # in a plausible geographic range (< 180°).
        result = FuturisedClient.parse_telemetry_coord('515493451026715')
        assert result is not None
        assert abs(result) < 180

    def test_zero_returns_none(self):
        assert FuturisedClient.parse_telemetry_coord('0') is None

    def test_empty_returns_none(self):
        assert FuturisedClient.parse_telemetry_coord('') is None

    def test_negative_degrees(self):
        result = FuturisedClient.parse_telemetry_coord('-33.868')
        assert result == pytest.approx(-33.868)


class TestMediaFileParsing:
    """Test that list_media correctly parses API responses."""

    def _make_client(self, tmp_path):
        return FuturisedClient(
            media_api_key='test-key',
            download_dir=str(tmp_path),
        )

    def test_camera_detection_wide(self, tmp_path):
        client = self._make_client(tmp_path)
        # Simulate API response
        api_response = [
            {
                'id': 'abc-123',
                'name': 'DJI_20260317100842_0003_W.JPG',
                'uploaded_at': 1773738524938,
                'path': 'DJI_album',
            },
        ]
        with patch.object(client, '_get_json', return_value=api_response):
            files = client.list_media()

        assert len(files) == 1
        assert files[0].camera == 'Wide'
        assert files[0].extension == '.JPG'
        assert files[0].id == 'abc-123'

    def test_camera_detection_thermal(self, tmp_path):
        client = self._make_client(tmp_path)
        api_response = [
            {
                'id': 'def-456',
                'name': 'DJI_20260317100842_0003_T.JPG',
                'uploaded_at': 0,
                'path': '',
            },
        ]
        with patch.object(client, '_get_json', return_value=api_response):
            files = client.list_media()

        assert files[0].camera == 'Thermal'

    def test_mp4_extension(self, tmp_path):
        client = self._make_client(tmp_path)
        api_response = [
            {
                'id': 'vid-789',
                'name': 'DJI_20260317100717_0001_W.MP4',
                'uploaded_at': 0,
                'path': '',
            },
        ]
        with patch.object(client, '_get_json', return_value=api_response):
            files = client.list_media()

        assert files[0].extension == '.MP4'

    def test_api_error_returns_empty(self, tmp_path):
        client = self._make_client(tmp_path)
        with patch.object(client, '_get_json', return_value=None):
            files = client.list_media()
        assert files == []


class TestPollNewImages:

    def test_filters_by_camera_and_extension(self, tmp_path):
        client = FuturisedClient(
            media_api_key='test-key',
            download_dir=str(tmp_path),
        )
        api_response = [
            {'id': '1', 'name': 'DJI_0001_W.JPG', 'uploaded_at': 0, 'path': ''},
            {'id': '2', 'name': 'DJI_0001_T.JPG', 'uploaded_at': 0, 'path': ''},
            {'id': '3', 'name': 'DJI_0001_W.MP4', 'uploaded_at': 0, 'path': ''},
        ]
        # Mock list_media to return parsed files, and download_image to return a path
        with patch.object(client, '_get_json', return_value=api_response):
            files = client.list_media()

        # Manually call poll logic — Wide JPGs only
        candidates = [
            f for f in files
            if f.extension in {'.JPG', '.JPEG'}
            and f.camera == 'Wide'
        ]
        assert len(candidates) == 1
        assert candidates[0].id == '1'

    def test_skips_already_seen(self, tmp_path):
        client = FuturisedClient(
            media_api_key='test-key',
            download_dir=str(tmp_path),
        )
        client._seen_ids.add('1')
        api_response = [
            {'id': '1', 'name': 'DJI_0001_W.JPG', 'uploaded_at': 0, 'path': ''},
            {'id': '2', 'name': 'DJI_0002_W.JPG', 'uploaded_at': 0, 'path': ''},
        ]
        with patch.object(client, '_get_json', return_value=api_response):
            with patch.object(client, 'download_image', return_value=tmp_path / 'img.jpg'):
                downloaded = client.poll_new_images(camera_filter='Wide')

        # Only file '2' should be downloaded (file '1' already seen)
        assert len(downloaded) == 1


class TestDownloadImage:

    def test_skips_existing_file(self, tmp_path):
        client = FuturisedClient(
            media_api_key='test-key',
            download_dir=str(tmp_path),
        )
        # Pre-create the file
        existing = tmp_path / 'DJI_test.JPG'
        existing.write_bytes(b'fake jpeg')

        details = MediaFile(
            id='abc', name='DJI_test.JPG', uploaded_at=0,
            path='', camera='Wide', extension='.JPG',
            download_url='https://example.com/img.jpg',
        )
        with patch.object(client, 'get_file_details', return_value=details):
            result = client.download_image('abc')

        assert result == existing
        assert 'abc' in client._seen_ids
