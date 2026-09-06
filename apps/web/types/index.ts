export interface BandMetrics {
  [bandName: string]: number;
}

export interface EdgeMetrics {
  edge_cosine_similarity: number;
  edge_l1_diff: number;
  edge_enhancement_ratio: number;
}

export interface SpectralMetrics {
  psnr: BandMetrics & { mean_psnr: number };
  ssim: BandMetrics & { mean_ssim: number };
  spectral_angle_mapper_deg: number;
  ndvi_preservation: number;
  ndwi_preservation: number;
  edge_metrics: EdgeMetrics;
}

export interface DownloadUrls {
  enhanced: string;
  uncertainty: string;
}

export interface EnhanceJob {
  job_id: string;
  status: "pending" | "processing" | "completed" | "failed" | string;
  metrics?: SpectralMetrics;
  download_urls?: DownloadUrls;
  created_at?: string;
}

export interface CopernicusRequest {
  lat: number;
  lon: number;
  date: string;
  aoi_size_km: number;
}

export type BandMode = "RGB" | "NIR" | "NDVI" | "NDWI";
