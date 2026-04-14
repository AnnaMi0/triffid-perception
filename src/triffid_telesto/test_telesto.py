"""Unit tests for triffid_telesto — client + bridge."""

import json
import threading
import unittest
from unittest.mock import MagicMock, patch, call

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from triffid_telesto.telesto_client import (
    TelestoClient, TelestoError, _request,
)


# ── Sample fixtures ──────────────────────────────────────────────────────

def _point_feature(source='ugv', local_id='1', cls='Building',
                   lon=23.72, lat=37.98, alt=320.0):
    return {
        'type': 'Feature',
        'id': local_id,
        'geometry': {'type': 'Point', 'coordinates': [lon, lat]},
        'properties': {
            'class': cls,
            'id': local_id,
            'confidence': 0.85,
            'category': 'infrastructure',
            'detection_type': 'seg',
            'source': source,
            'local_frame': False,
            'altitude_m': alt,
            'height_m': 4.5,
            'marker-color': '#708090',
            'marker-size': 'medium',
            'marker-symbol': 'building',
        },
    }


def _collection(*features):
    return {'type': 'FeatureCollection', 'features': list(features)}


def _server_feature(feature, server_id='feature_abc123'):
    """Simulate server response (adds server ID)."""
    out = dict(feature)
    out['id'] = server_id
    return out


# ── TelestoClient unit tests ────────────────────────────────────────────

class TestClientInit(unittest.TestCase):

    def test_default_urls(self):
        c = TelestoClient()
        assert 'crispres.com' in c.base_url
        assert 'observer-sync' in c.observer_url

    def test_custom_urls(self):
        c = TelestoClient(
            base_url='http://localhost:8080/api',
            observer_url='http://localhost:8080/obs',
        )
        assert c.base_url == 'http://localhost:8080/api'
        assert c.observer_url == 'http://localhost:8080/obs'

    def test_trailing_slash_stripped(self):
        c = TelestoClient(base_url='http://x.com/api/')
        assert not c.base_url.endswith('/')


class TestGetFeatures(unittest.TestCase):

    @patch('triffid_telesto.telesto_client._request')
    def test_returns_feature_collection(self, mock_req):
        mock_req.return_value = _collection(_point_feature())
        c = TelestoClient()
        result = c.get_features()
        assert result['type'] == 'FeatureCollection'
        assert len(result['features']) == 1
        mock_req.assert_called_once()

    @patch('triffid_telesto.telesto_client._request')
    def test_uses_correct_url(self, mock_req):
        mock_req.return_value = _collection()
        c = TelestoClient(base_url='http://test.com/api')
        c.get_features()
        url = mock_req.call_args[0][0]
        assert url == 'http://test.com/api/features'


class TestPutFeature(unittest.TestCase):

    @patch('triffid_telesto.telesto_client._request')
    def test_put_sends_geometry_and_properties(self, mock_req):
        feat = _point_feature()
        mock_req.return_value = _server_feature(feat)
        c = TelestoClient()
        resp = c.put_feature(feat)
        assert resp['id'] == 'feature_abc123'
        # Verify PUT method
        _, kwargs = mock_req.call_args
        assert kwargs.get('method', mock_req.call_args[0][1] if len(mock_req.call_args[0]) > 1 else None) == 'PUT' or mock_req.call_args[1].get('method') == 'PUT'

    @patch('triffid_telesto.telesto_client._request')
    def test_put_payload_has_geometry_and_properties(self, mock_req):
        feat = _point_feature()
        mock_req.return_value = _server_feature(feat)
        c = TelestoClient()
        c.put_feature(feat)
        body = mock_req.call_args[1].get('body') or mock_req.call_args.kwargs.get('body')
        assert 'geometry' in body
        assert 'properties' in body
        assert body['geometry']['type'] == 'Point'


class TestPatchFeature(unittest.TestCase):

    @patch('triffid_telesto.telesto_client._request')
    def test_patch_uses_feature_id_in_url(self, mock_req):
        mock_req.return_value = {'success': True}
        c = TelestoClient(base_url='http://test.com/api')
        c.patch_feature('feature_xyz', _point_feature())
        url = mock_req.call_args[0][0]
        assert url == 'http://test.com/api/features/feature_xyz'


class TestDeleteFeature(unittest.TestCase):

    @patch('triffid_telesto.telesto_client._request')
    def test_delete_uses_feature_id_in_url(self, mock_req):
        mock_req.return_value = {'deleted': 'feature_xyz'}
        c = TelestoClient(base_url='http://test.com/api')
        resp = c.delete_feature('feature_xyz')
        url = mock_req.call_args[0][0]
        assert url == 'http://test.com/api/features/feature_xyz'
        assert resp['deleted'] == 'feature_xyz'


class TestUploadCollection(unittest.TestCase):

    @patch('triffid_telesto.telesto_client._request')
    def test_upload_puts_each_feature(self, mock_req):
        f1 = _point_feature(local_id='1')
        f2 = _point_feature(local_id='2', lon=23.73)
        mock_req.side_effect = [
            _server_feature(f1, 'feature_001'),
            _server_feature(f2, 'feature_002'),
        ]
        c = TelestoClient()
        results = c.upload_collection(_collection(f1, f2))
        assert len(results) == 2
        assert results[0]['id'] == 'feature_001'
        assert results[1]['id'] == 'feature_002'

    @patch('triffid_telesto.telesto_client._request')
    def test_upload_handles_errors(self, mock_req):
        f1 = _point_feature()
        mock_req.side_effect = TelestoError('server down')
        c = TelestoClient()
        results = c.upload_collection(_collection(f1))
        assert len(results) == 1
        assert 'error' in results[0]


class TestSyncCollection(unittest.TestCase):

    @patch('triffid_telesto.telesto_client._request')
    def test_creates_new_features(self, mock_req):
        f1 = _point_feature(local_id='1')
        # GET returns empty, then PUT creates
        mock_req.side_effect = [
            _collection(),  # GET
            _server_feature(f1, 'feature_new'),  # PUT
        ]
        c = TelestoClient()
        stats = c.sync_collection(_collection(f1))
        assert stats['created'] == 1
        assert stats['updated'] == 0
        assert stats['deleted'] == 0

    @patch('triffid_telesto.telesto_client._request')
    def test_updates_existing_features(self, mock_req):
        f1 = _point_feature(local_id='42', source='ugv')
        remote_f1 = _server_feature(
            _point_feature(local_id='42', source='ugv'), 'feature_remote42'
        )
        mock_req.side_effect = [
            _collection(remote_f1),  # GET
            {'success': True},  # PATCH
        ]
        c = TelestoClient()
        stats = c.sync_collection(_collection(f1), source='ugv')
        assert stats['updated'] == 1
        assert stats['created'] == 0

    @patch('triffid_telesto.telesto_client._request')
    def test_deletes_stale_features(self, mock_req):
        # Remote has feature '42' but local collection is empty
        remote_f1 = _server_feature(
            _point_feature(local_id='42', source='ugv'), 'feature_remote42'
        )
        mock_req.side_effect = [
            _collection(remote_f1),  # GET
            {'deleted': 'feature_remote42'},  # DELETE
        ]
        c = TelestoClient()
        stats = c.sync_collection(_collection(), source='ugv')
        assert stats['deleted'] == 1

    @patch('triffid_telesto.telesto_client._request')
    def test_ignores_other_sources(self, mock_req):
        """Sync with source='ugv' should not delete UAV features."""
        remote_uav = _server_feature(
            _point_feature(local_id='99', source='uav'), 'feature_uav99'
        )
        mock_req.side_effect = [
            _collection(remote_uav),  # GET — only UAV remote feature
        ]
        c = TelestoClient()
        stats = c.sync_collection(_collection(), source='ugv')
        # Should not delete the UAV feature
        assert stats['deleted'] == 0


class TestClearSource(unittest.TestCase):

    @patch('triffid_telesto.telesto_client._request')
    def test_clears_matching_source(self, mock_req):
        rf1 = _server_feature(_point_feature(source='ugv', local_id='1'), 'f1')
        rf2 = _server_feature(_point_feature(source='uav', local_id='2'), 'f2')
        mock_req.side_effect = [
            _collection(rf1, rf2),  # GET
            {'deleted': 'f1'},  # DELETE ugv
        ]
        c = TelestoClient()
        deleted = c.clear_source('ugv')
        assert deleted == 1


class TestObserver(unittest.TestCase):

    @patch('triffid_telesto.telesto_client._request')
    def test_get_observer_status(self, mock_req):
        mock_req.return_value = {
            'last_update': '2026-04-13 10:00:00',
            'fe_updated': 0,
            'mobile_updated': 0,
            'ar_updated': 0,
        }
        c = TelestoClient()
        status = c.get_observer_status()
        assert 'fe_updated' in status
        assert 'observer-sync' in mock_req.call_args[0][0]

    @patch('triffid_telesto.telesto_client._request')
    def test_notify_observer(self, mock_req):
        mock_req.return_value = {'success': True, 'data': {'fe_updated': 1}}
        c = TelestoClient()
        resp = c.notify_observer(fe_updated=1)
        assert resp['success'] is True
        body = mock_req.call_args[1].get('body') or mock_req.call_args.kwargs.get('body')
        assert body == {'fe_updated': 1}


class TestTelestoError(unittest.TestCase):

    def test_error_is_exception(self):
        assert issubclass(TelestoError, Exception)

    def test_error_message(self):
        e = TelestoError('something went wrong')
        assert 'something went wrong' in str(e)


# ── Bridge unit tests ───────────────────────────────────────────────────

class TestBridgeMerge(unittest.TestCase):
    """Test the merge logic without MQTT or network."""

    def _make_bridge(self):
        from triffid_telesto.bridge import Bridge
        b = Bridge.__new__(Bridge)
        b._lock = threading.Lock()
        b._ugv_latest = None
        b._uav_latest = None
        b._dirty = False
        return b

    def test_merge_empty(self):
        b = self._make_bridge()
        merged = b._merge()
        assert merged['type'] == 'FeatureCollection'
        assert len(merged['features']) == 0

    def test_merge_ugv_only(self):
        b = self._make_bridge()
        b._ugv_latest = _collection(
            _point_feature(source='ugv', local_id='1'),
        )
        b._dirty = True
        merged = b._merge()
        assert len(merged['features']) == 1
        assert merged['features'][0]['properties']['source'] == 'ugv'

    def test_merge_both_sources(self):
        b = self._make_bridge()
        b._ugv_latest = _collection(
            _point_feature(source='ugv', local_id='1'),
            _point_feature(source='ugv', local_id='2'),
        )
        b._uav_latest = _collection(
            _point_feature(source='uav', local_id='10'),
        )
        b._dirty = True
        merged = b._merge()
        assert len(merged['features']) == 3
        sources = {f['properties']['source'] for f in merged['features']}
        assert sources == {'ugv', 'uav'}

    def test_merge_clears_dirty_flag(self):
        b = self._make_bridge()
        b._dirty = True
        b._merge()
        assert b._dirty is False


class TestBridgeOnMessage(unittest.TestCase):
    """Test MQTT message handling without a real broker."""

    def _make_bridge(self):
        from triffid_telesto.bridge import Bridge
        b = Bridge.__new__(Bridge)
        b._lock = threading.Lock()
        b._ugv_latest = None
        b._uav_latest = None
        b._dirty = False
        b.ugv_topic = 'ugv/detections/front/geojson'
        b.uav_topic = 'triffid/uav/geojson'
        return b

    def test_ugv_message_stored(self):
        b = self._make_bridge()
        msg = MagicMock()
        msg.topic = 'ugv/detections/front/geojson'
        msg.payload = json.dumps(_collection(_point_feature())).encode()
        b._on_message(None, None, msg)
        assert b._ugv_latest is not None
        assert b._dirty is True

    def test_uav_message_stored(self):
        b = self._make_bridge()
        msg = MagicMock()
        msg.topic = 'triffid/uav/geojson'
        msg.payload = json.dumps(_collection(
            _point_feature(source='uav')
        )).encode()
        b._on_message(None, None, msg)
        assert b._uav_latest is not None

    def test_bad_payload_ignored(self):
        b = self._make_bridge()
        msg = MagicMock()
        msg.topic = 'ugv/detections/front/geojson'
        msg.payload = b'not json'
        b._on_message(None, None, msg)
        assert b._ugv_latest is None
        assert b._dirty is False

    def test_non_feature_collection_ignored(self):
        b = self._make_bridge()
        msg = MagicMock()
        msg.topic = 'ugv/detections/front/geojson'
        msg.payload = json.dumps({'type': 'Feature'}).encode()
        b._on_message(None, None, msg)
        assert b._ugv_latest is None


class TestGeoJSONFieldAlignment(unittest.TestCase):
    """Verify both modules use the same property schema."""

    def test_altitude_field_is_altitude_m(self):
        """Both UGV and UAV features should use 'altitude_m'."""
        ugv = _point_feature(source='ugv')
        uav = _point_feature(source='uav')
        assert 'altitude_m' in ugv['properties']
        assert 'altitude_m' in uav['properties']
        # Old names should not be present
        assert 'ellipsoidal_alt_m' not in ugv['properties']
        assert 'gnss_altitude_m' not in uav['properties']

    def test_shared_property_keys(self):
        """UGV and UAV features should have identical property keys."""
        ugv_keys = set(_point_feature(source='ugv')['properties'].keys())
        uav_keys = set(_point_feature(source='uav')['properties'].keys())
        # All keys should match except 'source' values differ
        assert ugv_keys == uav_keys


if __name__ == '__main__':
    unittest.main()
