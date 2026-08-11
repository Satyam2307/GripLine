import type { GripLineResponse, ZonePrediction } from '../types/gripline';

export interface UIZoneCard {
  id: string;
  label: string;
  condition: string;
  wetness: number; // 0 - 100
  risk: string; // "Low", "Moderate", "High"
  confidence: number; // 0 - 100
  progress: number; // 0 - 100
  tone: 'good' | 'warn' | 'danger';
}

export interface UIAlert {
  time: string;
  title: string;
  detail: string;
}

export interface UIWetnessPoint {
  minute: string;
  wetness: number;
}

export function adaptResponseToUI(response: GripLineResponse) {
  // 1. Overall metrics
  const overallCondition = response.overall.display_condition || response.overall.base_condition;
  const overallWetnessPct = Math.round(response.overall.wetness_score * 100);
  const overallConfidencePct = response.overall.confidence > 1 
    ? Math.round(response.overall.confidence) 
    : Math.round(response.overall.confidence * 100);

  // 2. Zone Cards
  const zoneCards: UIZoneCard[] = response.zones.map((zone: ZonePrediction) => {
    const wetnessPct = Math.round(zone.wetness_score * 100);
    const confidencePct = zone.confidence > 1 ? Math.round(zone.confidence) : Math.round(zone.confidence * 100);
    
    // Determine tone & normalized risk
    let tone: 'good' | 'warn' | 'danger' = 'good';
    let displayRisk = zone.risk;

    if (zone.risk === 'High' || zone.wetness_score >= 0.75) {
      tone = 'danger';
      displayRisk = 'High';
    } else if (zone.risk === 'Medium' || zone.risk === 'Moderate' || zone.wetness_score >= 0.45 || zone.trend === 'Drying') {
      tone = 'warn';
      displayRisk = 'Moderate';
    } else {
      tone = 'good';
      displayRisk = 'Low';
    }

    return {
      id: zone.id,
      label: zone.name,
      condition: zone.display_condition || zone.base_condition,
      wetness: wetnessPct,
      risk: displayRisk,
      confidence: confidencePct,
      progress: wetnessPct,
      tone,
    };
  });

  // 3. Wetness Trend Chart Points
  const wetnessTrend: UIWetnessPoint[] = response.timeline.length > 0
    ? response.timeline.map((tp, idx) => {
        const wetnessPct = Math.round(tp.overall_wetness * 100);
        // Format timestamp_seconds to a clean time string (e.g., "00:05", "00:10")
        const mins = Math.floor(tp.timestamp_seconds / 60);
        const secs = Math.floor(tp.timestamp_seconds % 60);
        const timeStr = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
        return {
          minute: tp.timestamp_seconds === 0 ? `Frame ${idx + 1}` : timeStr,
          wetness: wetnessPct,
        };
      })
    : [
        { minute: 'Start', wetness: overallWetnessPct },
        { minute: 'Current', wetness: overallWetnessPct },
      ];

  // 4. Alerts
  const uiAlerts: UIAlert[] = response.alerts.map((alert, idx) => {
    const mins = Math.floor(alert.timestamp_seconds / 60);
    const secs = Math.floor(alert.timestamp_seconds % 60);
    const timeStr = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    
    // Extract title and detail from alert message if possible
    const messageParts = alert.message.split('. ');
    const title = messageParts[0] || 'Alert';
    const detail = messageParts.slice(1).join('. ') || alert.message;

    return {
      time: alert.timestamp_seconds === 0 ? `Alert ${idx + 1}` : timeStr,
      title: title.endsWith('.') ? title.slice(0, -1) : title,
      detail: detail || alert.message,
    };
  });

  return {
    overallCondition,
    overallWetnessPct,
    overallConfidencePct,
    zoneCards,
    wetnessTrend,
    alerts: uiAlerts,
    radioCall: response.radio_call.text.replace(/^GripLine to pit wall\.\s*/i, ''),
    radioSeverity: response.radio_call.severity,
    recommendationText: response.recommendation.text,
    suggestedTire: response.recommendation.suggested_tire,
  };
}
