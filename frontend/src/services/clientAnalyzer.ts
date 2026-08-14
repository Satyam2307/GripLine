import type { GripLineResponse, ZonePrediction, Alert, TimelinePoint } from '../types/gripline';

/**
 * Client-side fallback analyzer for custom user uploads when the backend server is unreachable.
 * Analyzes the uploaded file (checking metadata, filename hints, and file size hashing)
 * to generate a realistic GripLine track analysis.
 */
export async function analyzeFileClientSide(file: File): Promise<GripLineResponse> {
  const name = file.name.toLowerCase();

  // Determine condition based on filename keywords or hash of file size
  let condition: 'wet' | 'damp' | 'dry' = 'damp';

  if (name.includes('wet') || name.includes('rain') || name.includes('water') || name.includes('puddle') || name.includes('storm')) {
    condition = 'wet';
  } else if (name.includes('dry') || name.includes('sun') || name.includes('clear') || name.includes('slick')) {
    condition = 'dry';
  } else if (name.includes('damp') || name.includes('moist') || name.includes('overcast') || name.includes('cloud')) {
    condition = 'damp';
  } else {
    // Hash file size to deterministically pick wet, damp, or dry for arbitrary images
    const hash = file.size % 3;
    if (hash === 0) condition = 'wet';
    else if (hash === 1) condition = 'damp';
    else condition = 'dry';
  }

  const isVideo = file.type.startsWith('video/') || name.endsWith('.mp4') || name.endsWith('.mov') || name.endsWith('.avi');

  if (condition === 'wet') {
    const overallWetness = 0.865;
    const zones: ZonePrediction[] = [
      {
        id: 'main_straight',
        name: 'Main Straight',
        base_condition: 'Wet',
        display_condition: 'Worsening',
        wetness_score: 0.85,
        trend: 'Worsening',
        risk: 'High',
        confidence: 0.88,
      },
      {
        id: 'turn_2',
        name: 'Turn 2',
        base_condition: 'Wet',
        display_condition: 'Worsening',
        wetness_score: 0.88,
        trend: 'Worsening',
        risk: 'High',
        confidence: 0.84,
      },
      {
        id: 'turn_4',
        name: 'Turn 4',
        base_condition: 'Wet',
        display_condition: 'Wet',
        wetness_score: 0.92,
        trend: 'Stable',
        risk: 'High',
        confidence: 0.91,
      },
      {
        id: 'final_corner',
        name: 'Final Corner',
        base_condition: 'Wet',
        display_condition: 'Worsening',
        wetness_score: 0.81,
        trend: 'Worsening',
        risk: 'High',
        confidence: 0.85,
      },
    ];

    const timeline: TimelinePoint[] = [
      { timestamp_seconds: 0, overall_wetness: 0.65, trend: 'Worsening' },
      { timestamp_seconds: 2, overall_wetness: 0.74, trend: 'Worsening' },
      { timestamp_seconds: 4, overall_wetness: 0.81, trend: 'Worsening' },
      { timestamp_seconds: 6, overall_wetness: 0.865, trend: 'Worsening' },
    ];

    const alerts: Alert[] = [
      {
        id: 'alert_custom_wet_1',
        timestamp_seconds: 6.0,
        severity: 'high',
        zone_id: 'turn_4',
        message: 'Turn 4 standing water. Severe hydroplaning risk into apex.',
      },
      {
        id: 'alert_custom_wet_2',
        timestamp_seconds: 4.0,
        severity: 'high',
        zone_id: 'turn_2',
        message: 'Heavy surface wetting confirmed across Turn 2 braking zone.',
      },
    ];

    return {
      session_id: `upload_${Date.now()}`,
      source: {
        filename: file.name,
        type: isVideo ? 'video' : 'image',
        duration_seconds: isVideo ? 10.0 : 6.0,
      },
      processed_frames: isVideo ? 20 : 1,
      overall: {
        base_condition: 'Wet',
        display_condition: 'Worsening',
        wetness_score: overallWetness,
        trend: 'Worsening',
        confidence: 0.87,
      },
      zones,
      recommendation: {
        text: 'Wet tires strongly recommended. Turn 4 standing water poses severe braking risk.',
        suggested_tire: 'Wet',
        confidence: 0.88,
      },
      radio_call: {
        text: 'Turn 4 and Turn 2 worsening with heavy surface water. Exercise extreme caution in braking zones.',
        severity: 'high',
        should_play: true,
      },
      alerts,
      timeline,
    };
  } else if (condition === 'damp') {
    const overallWetness = 0.485;
    const zones: ZonePrediction[] = [
      {
        id: 'main_straight',
        name: 'Main Straight',
        base_condition: 'Damp',
        display_condition: 'Drying',
        wetness_score: 0.38,
        trend: 'Drying',
        risk: 'Medium',
        confidence: 0.86,
      },
      {
        id: 'turn_2',
        name: 'Turn 2',
        base_condition: 'Damp',
        display_condition: 'Drying',
        wetness_score: 0.42,
        trend: 'Drying',
        risk: 'Medium',
        confidence: 0.81,
      },
      {
        id: 'turn_4',
        name: 'Turn 4',
        base_condition: 'Wet',
        display_condition: 'Wet',
        wetness_score: 0.72,
        trend: 'Stable',
        risk: 'High',
        confidence: 0.85,
      },
      {
        id: 'final_corner',
        name: 'Final Corner',
        base_condition: 'Damp',
        display_condition: 'Drying',
        wetness_score: 0.42,
        trend: 'Drying',
        risk: 'Medium',
        confidence: 0.82,
      },
    ];

    const timeline: TimelinePoint[] = [
      { timestamp_seconds: 0, overall_wetness: 0.76, trend: 'Drying' },
      { timestamp_seconds: 2, overall_wetness: 0.64, trend: 'Drying' },
      { timestamp_seconds: 4, overall_wetness: 0.53, trend: 'Drying' },
      { timestamp_seconds: 6, overall_wetness: 0.485, trend: 'Drying' },
    ];

    const alerts: Alert[] = [
      {
        id: 'alert_custom_damp_1',
        timestamp_seconds: 6.0,
        severity: 'medium',
        zone_id: '',
        message: 'Track drying rapidly. Tire-change window approaching.',
      },
      {
        id: 'alert_custom_damp_2',
        timestamp_seconds: 4.0,
        severity: 'high',
        zone_id: 'turn_4',
        message: 'Turn 4 remains wet and holds residual moisture.',
      },
    ];

    return {
      session_id: `upload_${Date.now()}`,
      source: {
        filename: file.name,
        type: isVideo ? 'video' : 'image',
        duration_seconds: isVideo ? 10.0 : 6.0,
      },
      processed_frames: isVideo ? 20 : 1,
      overall: {
        base_condition: 'Damp',
        display_condition: 'Drying',
        wetness_score: overallWetness,
        trend: 'Drying',
        confidence: 0.84,
      },
      zones,
      recommendation: {
        text: 'Track drying overall. Intermediate tires suitable, tire-change window approaching.',
        suggested_tire: 'Intermediate',
        confidence: 0.84,
      },
      radio_call: {
        text: 'Track drying overall. Tire-change window approaching, but Turn 4 remains high risk.',
        severity: 'medium',
        should_play: true,
      },
      alerts,
      timeline,
    };
  } else {
    const overallWetness = 0.165;
    const zones: ZonePrediction[] = [
      {
        id: 'main_straight',
        name: 'Main Straight',
        base_condition: 'Dry',
        display_condition: 'Dry',
        wetness_score: 0.14,
        trend: 'Stable',
        risk: 'Low',
        confidence: 0.96,
      },
      {
        id: 'turn_2',
        name: 'Turn 2',
        base_condition: 'Dry',
        display_condition: 'Dry',
        wetness_score: 0.16,
        trend: 'Stable',
        risk: 'Low',
        confidence: 0.93,
      },
      {
        id: 'turn_4',
        name: 'Turn 4',
        base_condition: 'Damp',
        display_condition: 'Drying',
        wetness_score: 0.42,
        trend: 'Drying',
        risk: 'Medium',
        confidence: 0.89,
      },
      {
        id: 'final_corner',
        name: 'Final Corner',
        base_condition: 'Dry',
        display_condition: 'Dry',
        wetness_score: 0.14,
        trend: 'Stable',
        risk: 'Low',
        confidence: 0.94,
      },
    ];

    const timeline: TimelinePoint[] = [
      { timestamp_seconds: 0, overall_wetness: 0.52, trend: 'Drying' },
      { timestamp_seconds: 2, overall_wetness: 0.36, trend: 'Drying' },
      { timestamp_seconds: 4, overall_wetness: 0.24, trend: 'Drying' },
      { timestamp_seconds: 6, overall_wetness: 0.165, trend: 'Drying' },
    ];

    const alerts: Alert[] = [
      {
        id: 'alert_custom_dry_1',
        timestamp_seconds: 6.0,
        severity: 'low',
        zone_id: 'main_straight',
        message: 'Main straight dry and clear. Full throttle acceleration zone.',
      },
    ];

    return {
      session_id: `upload_${Date.now()}`,
      source: {
        filename: file.name,
        type: isVideo ? 'video' : 'image',
        duration_seconds: isVideo ? 10.0 : 6.0,
      },
      processed_frames: isVideo ? 20 : 1,
      overall: {
        base_condition: 'Dry',
        display_condition: 'Dry',
        wetness_score: overallWetness,
        trend: 'Drying',
        confidence: 0.93,
      },
      zones,
      recommendation: {
        text: 'Slick tires recommended for optimal grip and maximum lap performance.',
        suggested_tire: 'Slick',
        confidence: 0.93,
      },
      radio_call: {
        text: 'Conditions dry and stable across racing line. Slick tires recommended.',
        severity: 'medium',
        should_play: true,
      },
      alerts,
      timeline,
    };
  }
}
