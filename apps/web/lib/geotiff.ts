export interface GeoTiffInfo {
  bands: Float32Array[];
  width: number;
  height: number;
  crs: string | null;
  bounds: [number, number, number, number] | null;
  bandCount: number;
}

export async function parseGeoTiff(file: File): Promise<GeoTiffInfo> {
  const { fromBlob } = await import("geotiff");
  const tiff = await fromBlob(file);
  const image = await tiff.getImage();

  const width = image.getWidth();
  const height = image.getHeight();
  const bandCount = image.getSamplesPerPixel();

  const rasters = await image.readRasters({ interleave: false });

  const bands: Float32Array[] = [];
  for (let i = 0; i < bandCount; i++) {
    const raw = (rasters as unknown as ArrayLike<number>[])[i];
    const band = new Float32Array(raw.length);
    let maxVal = 0;
    for (let j = 0; j < raw.length; j++) {
      maxVal = Math.max(maxVal, raw[j]);
    }
    const scale = maxVal > 1 ? (maxVal > 255 ? 10000 : 255) : 1;
    for (let j = 0; j < raw.length; j++) {
      band[j] = raw[j] / scale;
    }
    bands.push(band);
  }

  let crs: string | null = null;
  let bounds: [number, number, number, number] | null = null;

  try {
    const geoKeys = image.getGeoKeys();
    if (geoKeys?.ProjectedCSTypeGeoKey) {
      crs = `EPSG:${geoKeys.ProjectedCSTypeGeoKey}`;
    } else if (geoKeys?.GeographicTypeGeoKey) {
      crs = `EPSG:${geoKeys.GeographicTypeGeoKey}`;
    }
  } catch {
  }

  try {
    const bbox = image.getBoundingBox();
    if (bbox) bounds = [bbox[0], bbox[1], bbox[2], bbox[3]];
  } catch {
  }

  return { bands, width, height, crs, bounds, bandCount };
}

export function bandsToRgbUint8(
  rBand: Float32Array,
  gBand: Float32Array,
  bBand: Float32Array,
  width: number,
  height: number
): Uint8ClampedArray {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    data[i * 4] = Math.min(255, Math.max(0, rBand[i] * 255));
    data[i * 4 + 1] = Math.min(255, Math.max(0, gBand[i] * 255));
    data[i * 4 + 2] = Math.min(255, Math.max(0, bBand[i] * 255));
    data[i * 4 + 3] = 255;
  }
  return data;
}
