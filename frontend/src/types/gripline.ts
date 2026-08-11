export interface SourceMetadata {
  filename: string;
  type: string;
  duration_seconds: number;
}

export interface OverallCondition {
  base_condition: string;
  display_condition: string;
  wetness_score: number;
  trend: string;
  confidence: number;
}

export interface ZonePrediction {
  id: string;
  name: string;
  base_condition: string;
  display_condition: string;
  wetness_score: number;
  trend: string;
  risk: string;
  confidence: number;
}

export interface Recommendation {
  text: string;
  suggested_tire: string;
  confidence: number;
}

export interface RadioCall {
  text: string;
  severity: string;
  should_play: boolean;
}

export interface Alert {
  id: string;
  timestamp_seconds: number;
  severity: string;
  zone_id: string;
  message: string;
}

export interface TimelinePoint {
  timestamp_seconds: number;
  overall_wetness: number;
  trend: string;
}

export interface GripLineResponse {
  session_id: string;
  source: SourceMetadata;
  processed_frames: number;
  overall: OverallCondition;
  zones: ZonePrediction[];
  recommendation: Recommendation;
  radio_call: RadioCall;
  alerts: Alert[];
  timeline: TimelinePoint[];
}

export interface AnalyzeOptions {
  frame_stride?: number;
  max_frames?: number;
  demo_mode?: boolean;
}
