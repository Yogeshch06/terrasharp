import io
import math
import logging
from typing import Optional, Tuple
import requests
import numpy as np
import rasterio
from rasterio.transform import from_bounds

from config import settings

logger = logging.getLogger("terrasharp.copernicus")


class CopernicusClient:
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        token_url: str = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        process_url: str = "https://sh.dataspace.copernicus.eu/api/v1/process"
    ):
        self.client_id = client_id or settings.COPERNICUS_CLIENT_ID
        self.client_secret = client_secret or settings.COPERNICUS_CLIENT_SECRET
        self.token_url = token_url
        self.process_url = process_url
        self._access_token: Optional[str] = None

    def _get_token(self) -> Optional[str]:
        if not self.client_id or not self.client_secret:
            logger.warning("Copernicus client credentials not configured.")
            return None

        try:
            resp = requests.post(
                self.token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret
                },
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            self._access_token = data.get("access_token")
            return self._access_token
        except Exception as e:
            logger.error(f"Failed to acquire Copernicus access token: {e}")
            return None

    def _calculate_bbox(self, lat: float, lon: float, aoi_size_km: float) -> Tuple[float, float, float, float]:
        half_size = aoi_size_km / 2.0
        delta_lat = half_size / 111.32
        cos_lat = max(math.cos(math.radians(lat)), 1e-6)
        delta_lon = half_size / (111.32 * cos_lat)

        min_lon = lon - delta_lon
        max_lon = lon + delta_lon
        min_lat = lat - delta_lat
        max_lat = lat + delta_lat
        return min_lon, min_lat, max_lon, max_lat

    def _generate_synthetic_geotiff(
        self,
        lat: float,
        lon: float,
        aoi_size_km: float,
        pixels: int = 256
    ) -> bytes:
        min_lon, min_lat, max_lon, max_lat = self._calculate_bbox(lat, lon, aoi_size_km)
        transform = from_bounds(min_lon, min_lat, max_lon, max_lat, pixels, pixels)

        synthetic_data = (np.random.rand(4, pixels, pixels) * 4000.0 + 500.0).astype(np.uint16)

        mem = io.BytesIO()
        with rasterio.open(
            mem,
            "w",
            driver="GTiff",
            height=pixels,
            width=pixels,
            count=4,
            dtype="uint16",
            crs="EPSG:4326",
            transform=transform
        ) as dst:
            dst.write(synthetic_data)

        return mem.getvalue()

    def fetch_sentinel2(
        self,
        lat: float,
        lon: float,
        date: str = "2023-08-01",
        aoi_size_km: float = 2.56
    ) -> bytes:
        token = self._get_token()
        if not token:
            logger.warning("Falling back to synthetic Sentinel-2 GeoTIFF data (no active credentials).")
            return self._generate_synthetic_geotiff(lat, lon, aoi_size_km)

        min_lon, min_lat, max_lon, max_lat = self._calculate_bbox(lat, lon, aoi_size_km)
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04", "B08"]
                }],
                output: {
                    bands: 4,
                    sampleType: "UINT16"
                }
            };
        }
        function evaluatePixel(sample) {
            return [sample.B02 * 10000, sample.B03 * 10000, sample.B04 * 10000, sample.B08 * 10000];
        }
        """

        time_range = {
            "from": f"{date}T00:00:00Z",
            "to": f"{date}T23:59:59Z"
        }

        payload = {
            "input": {
                "bounds": {
                    "bbox": [min_lon, min_lat, max_lon, max_lat],
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                },
                "data": [{
                    "type": "sentinel-2-l2a",
                    "dataFilter": {
                        "timeRange": time_range,
                        "maxCloudCoverage": 20
                    }
                }]
            },
            "output": {
                "width": max(64, int((aoi_size_km * 1000) / 10)),
                "height": max(64, int((aoi_size_km * 1000) / 10)),
                "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
            },
            "evalscript": evalscript
        }

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        try:
            resp = requests.post(self.process_url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            logger.error(f"Sentinel Hub Process API request failed: {e}. Falling back to synthetic GeoTIFF.")
            return self._generate_synthetic_geotiff(lat, lon, aoi_size_km)
