import type { GripLineResponse, AnalyzeOptions } from '../types/gripline';
import { analyzeFileClientSide } from './clientAnalyzer';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function analyzeTrack(
  file: File,
  options: AnalyzeOptions = {}
): Promise<GripLineResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('frame_stride', String(options.frame_stride ?? 5));
  formData.append('max_frames', String(options.max_frames ?? 30));
  formData.append('demo_mode', String(options.demo_mode ?? false));

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      console.warn(`Backend returned ${response.status} — using client analyzer fallback`);
      return await analyzeFileClientSide(file);
    }

    const data: GripLineResponse = await response.json();
    return data;
  } catch (err: any) {
    console.warn('Backend server unreachable — seamlessly analyzing upload client-side:', err);
    return await analyzeFileClientSide(file);
  }
}

export async function checkHealth(): Promise<{ status: string; model_loaded: boolean }> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/health`);
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.status}`);
    }
    return await response.json();
  } catch {
    return { status: 'unreachable', model_loaded: false };
  }
}
