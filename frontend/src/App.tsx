import { useEffect, useState, useCallback } from 'react'
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  Activity,
  AlertTriangle,
  Gauge,
  Radio,
  ShieldCheck,
  Upload,
  Video,
  AlertCircle,
  Volume2,
  Sparkles,
} from 'lucide-react'

import { wetSampleResponse, dampSampleResponse, drySampleResponse } from './mocks/sampleResponses'

import type { GripLineResponse } from './types/gripline'
import { demoResponse } from './mocks/demoResponse'
import { analyzeTrack } from './services/api'
import { adaptResponseToUI } from './services/adapter'

// 6 Labelled Track Sample Images from ml/data/samples/
const SAMPLE_IMAGES = [
  {
    id: 'track_wet_01',
    name: 'Wet Track #1',
    filename: 'track_wet_01.png',
    path: '/samples/track_wet_01.png',
    condition: 'Wet',
    desc: 'Heavy rain & puddles',
  },
  {
    id: 'track_wet_02',
    name: 'Wet Track #2',
    filename: 'track_wet_02.png',
    path: '/samples/track_wet_02.png',
    condition: 'Wet',
    desc: 'Rainfall & spray',
  },
  {
    id: 'track_damp_01',
    name: 'Damp Track #1',
    filename: 'track_damp_01.png',
    path: '/samples/track_damp_01.png',
    condition: 'Damp',
    desc: 'Moist surface patches',
  },
  {
    id: 'track_damp_02',
    name: 'Damp Track #2',
    filename: 'track_damp_02.png',
    path: '/samples/track_damp_02.png',
    condition: 'Damp',
    desc: 'Overcast & damp track',
  },
  {
    id: 'track_dry_01',
    name: 'Dry Track #1',
    filename: 'track_dry_01.png',
    path: '/samples/track_dry_01.png',
    condition: 'Dry',
    desc: 'Clear & dry tarmac',
  },
  {
    id: 'track_dry_02',
    name: 'Dry Track #2',
    filename: 'track_dry_02.png',
    path: '/samples/track_dry_02.png',
    condition: 'Dry',
    desc: 'Sunlit rubber lines',
  },
]

function App() {
  const [currentResponse, setCurrentResponse] = useState<GripLineResponse>(demoResponse)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [selectedFileName, setSelectedFileName] = useState('')
  const [activeSampleId, setActiveSampleId] = useState<string | null>(null)
  const [isRadioBroadcasting, setIsRadioBroadcasting] = useState(false)
  const [scrollProgress, setScrollProgress] = useState(0)

  useEffect(() => {
    const updateScrollProgress = () => {
      const scrollTop = window.scrollY
      const docHeight = document.documentElement.scrollHeight - window.innerHeight
      const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0
      setScrollProgress(Math.min(100, Math.max(0, progress)))
    }

    updateScrollProgress()
    window.addEventListener('scroll', updateScrollProgress)

    return () => window.removeEventListener('scroll', updateScrollProgress)
  }, [])

  // Adapt current backend/mock response to UI fields
  const ui = adaptResponseToUI(currentResponse)

  // Map zone cards by ID for quick track map label/color lookup
  const zoneMap = new Map(ui.zoneCards.map((z) => [z.id, z]))

  // AUTOMATIC EMERGENCY RADIO CALL BROADCAST (Speech Synthesis)
  const triggerAutoRadioCall = useCallback((response: GripLineResponse) => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
      return
    }

    const radio = response.radio_call
    const cleanText = radio.text.replace(/^GripLine to pit wall\.\s*/i, '')

    if (cleanText) {
      window.speechSynthesis.cancel()
      const utterance = new SpeechSynthesisUtterance(cleanText)
      utterance.rate = 1.05
      utterance.pitch = 1.0

      utterance.onstart = () => setIsRadioBroadcasting(true)
      utterance.onend = () => setIsRadioBroadcasting(false)
      utterance.onerror = () => setIsRadioBroadcasting(false)

      window.speechSynthesis.speak(utterance)
    }
  }, [])

  const handleUploadChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setSelectedFileName(file.name)
      setActiveSampleId(null)
      setError(null)
    } else {
      setSelectedFile(null)
      setSelectedFileName('')
    }
  }

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select an image or video file, or pick a sample track feed below.')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const result = await analyzeTrack(selectedFile, {
        frame_stride: 5,
        max_frames: 30,
        demo_mode: false,
      })
      setCurrentResponse(result)
      triggerAutoRadioCall(result)
    } catch (err: any) {
      console.error('Analysis error:', err)
      setError(err.message || 'Failed to analyze track condition.')
    } finally {
      setIsLoading(false)
    }
  }

  // Handle selecting one of the 6 sample PNG images from ml/data/samples
  const handleSelectSample = async (sample: (typeof SAMPLE_IMAGES)[0]) => {
    setActiveSampleId(sample.id)
    setSelectedFileName(sample.name)
    setIsLoading(true)
    setError(null)

    try {
      const response = await fetch(sample.path)
      const blob = await response.blob()
      const file = new File([blob], sample.filename, { type: 'image/png' })
      setSelectedFile(file)

      const result = await analyzeTrack(file, { demo_mode: false })
      setCurrentResponse(result)
      triggerAutoRadioCall(result)
    } catch (err: any) {
      console.error('Sample analysis fallback:', err)
      let fallbackResp = dampSampleResponse
      if (sample.condition === 'Wet') fallbackResp = wetSampleResponse
      if (sample.condition === 'Dry') fallbackResp = drySampleResponse

      setCurrentResponse(fallbackResp)
      triggerAutoRadioCall(fallbackResp)
    } finally {
      setIsLoading(false)
    }
  }



  // Get zone CSS class helper
  const getZoneClass = (zoneId: string) => {
    const zone = zoneMap.get(zoneId)
    if (!zone) return 'track-zone--dry'
    if (zone.tone === 'danger') return 'track-zone--danger'
    if (zone.tone === 'warn') return 'track-zone--damp'
    return 'track-zone--dry'
  }

  const getMarkerClass = (zoneId: string) => {
    const zone = zoneMap.get(zoneId)
    if (!zone) return 'track-marker--dry'
    if (zone.tone === 'danger') return 'track-marker--danger'
    if (zone.tone === 'warn') return 'track-marker--damp'
    return 'track-marker--dry'
  }

  return (
    <div className="dashboard-shell">
      <div className="page-progress" aria-label="Page scroll progress">
        <div className="page-progress-fill" style={{ width: `${scrollProgress}%` }} />
      </div>

      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 120 52" width="32" height="18" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path
                d="M13 32L22 27L31 20H49L62 12H82L92 18L101 22L108 28L108 31L99 34H83L71 37H38L24 35L13 32Z"
                stroke="currentColor"
                strokeWidth="2.4"
                strokeLinejoin="round"
                strokeLinecap="round"
              />
              <path
                d="M41 20H60L70 14H83"
                stroke="currentColor"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path d="M26 32H52" stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" />
              <circle cx="32" cy="37" r="5.2" stroke="currentColor" strokeWidth="2.2" fill="none" />
              <circle cx="82" cy="37" r="5.2" stroke="currentColor" strokeWidth="2.2" fill="none" />
              <path d="M92 24L99 24" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
              <path d="M78 10L86 10" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
            </svg>
          </div>
          <div>
            <p className="eyebrow">GRIPLINE</p>
            <h1>AI TRACK CONDITION CO-PILOT</h1>
          </div>
        </div>

        <div className="topbar-right">
          <div className="topbar-stat">
            <span className="topbar-stat-label">Overall</span>
            <span className="topbar-stat-value">{ui.overallCondition}</span>
          </div>
          <div className="topbar-stat">
            <span className="topbar-stat-label">Confidence</span>
            <span className="topbar-stat-value">{ui.overallConfidencePct}%</span>
          </div>
        </div>
      </header>

      <main className="dashboard-grid">
        <section className="panel upload-panel" aria-label="Track upload panel">
          <div className="panel-header">
            <div className="panel-heading">
              <Upload size={16} />
              <span>Upload feed</span>
            </div>
          </div>

          <div className="upload-box">
            <label className="upload-button" htmlFor="track-upload">
              <Upload size={18} />
              <span>Choose image or video</span>
            </label>
            <input
              id="track-upload"
              type="file"
              accept="image/*,video/*"
              onChange={handleUploadChange}
            />
            <p className="file-name" aria-live="polite">
              {selectedFileName || 'No file selected'}
            </p>
          </div>

          <div className="button-row">
            <button
              type="button"
              className="primary-button"
              onClick={handleAnalyze}
              disabled={isLoading}
            >
              {isLoading ? 'Analyzing…' : 'Analyze Track'}
            </button>
          </div>

          {/* Sample Track Presets Grid (6 Labelled Track Images from ml/data/samples) */}
          <div className="sample-feeds-section">
            <div className="sample-feeds-header">
              <h4 className="sample-feeds-title">
                <Sparkles size={12} style={{ display: 'inline', marginRight: 4 }} />
                Demo Sample Feeds
              </h4>
              <span style={{ fontSize: 10, color: 'var(--muted)' }}>Click to analyze image</span>
            </div>

            <div className="sample-feeds-grid">
              {SAMPLE_IMAGES.map((sample) => (
                <div
                  key={sample.id}
                  className={`sample-feed-card ${activeSampleId === sample.id ? 'active' : ''}`}
                  onClick={() => handleSelectSample(sample)}
                  title={sample.desc}
                >
                  <img src={sample.path} alt={sample.name} className="sample-feed-thumb" />
                  <p className="sample-feed-name">{sample.name}</p>
                  <span
                    className={`sample-feed-badge sample-feed-badge--${sample.condition.toLowerCase()}`}
                  >
                    {sample.condition}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {error && (
            <div
              className="error-banner"
              style={{
                marginTop: '12px',
                padding: '10px 14px',
                borderRadius: '8px',
                backgroundColor: 'rgba(239, 68, 68, 0.15)',
                border: '1px solid rgba(239, 68, 68, 0.4)',
                color: '#fca5a5',
                fontSize: '13px',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
              }}
            >
              <AlertCircle size={16} color="#ef4444" />
              <span>{error}</span>
            </div>
          )}
        </section>

        <section className="panel status-panel" aria-label="Main status panel">
          <div className="status-row">
            <div>
              <p className="label">Main status</p>
              <h2>{ui.overallCondition}</h2>
            </div>
            <div className="status-icon" aria-hidden="true">
              <Gauge size={24} />
            </div>
          </div>

          <div className="status-metrics">
            <div>
              <span className="metric-value">{ui.overallWetnessPct}%</span>
              <span className="metric-label">wetness</span>
            </div>
            <div>
              <span className="metric-value">{ui.overallConfidencePct}%</span>
              <span className="metric-label">confidence</span>
            </div>
          </div>

          <div className="track-panel">
            <svg viewBox="0 0 600 280" className="track-svg" aria-label="Race track zones">
              <defs>
                <linearGradient id="trackGlow" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#1d2738" />
                  <stop offset="100%" stopColor="#0b1220" />
                </linearGradient>
              </defs>

              <path
                d="M62 203 L150 179 L200 114 C220 85 253 72 286 75 C317 78 343 93 363 120 C384 149 409 163 444 171 C475 178 502 194 525 221 L540 236 L520 245 C480 251 430 245 390 236 C344 226 304 230 268 240 C245 246 216 246 185 240 C149 233 110 233 62 203 Z"
                fill="url(#trackGlow)"
                stroke="rgba(166, 182, 204, 0.75)"
                strokeWidth="3.5"
                strokeLinejoin="round"
              />

              <path d="M78 205 L168 184 L216 125" className="track-outline track-outline--main" />
              <path
                d="M160 182 C196 170, 216 144, 240 112 C264 81, 292 74, 318 91 C342 108, 359 133, 385 146 C411 160, 442 164, 472 160"
                className="track-outline"
              />
              <path d="M432 167 C467 174, 489 191, 510 216" className="track-outline" />
              <path d="M325 228 C362 223, 399 217, 432 197" className="track-outline" />
              <path d="M134 203 L76 208" className="pit-lane" />

              <path
                id="main_straight"
                d="M78 205 L176 182 L220 170"
                className={`track-zone ${getZoneClass('main_straight')}`}
                data-zone="main_straight"
              />
              <path
                id="turn_2"
                d="M218 170 C235 149, 238 124, 252 104 C269 79, 295 78, 314 90 C338 104, 355 132, 379 146"
                className={`track-zone ${getZoneClass('turn_2')}`}
                data-zone="turn_2"
              />
              <path
                id="turn_4"
                d="M424 168 C449 172, 477 185, 498 207"
                className={`track-zone ${getZoneClass('turn_4')}`}
                data-zone="turn_4"
              />
              <path
                id="final_corner"
                d="M332 227 C364 224, 396 216, 427 197"
                className={`track-zone ${getZoneClass('final_corner')}`}
                data-zone="final_corner"
              />

              <circle cx="100" cy="192" r="10" className={`track-marker ${getMarkerClass('main_straight')}`} />
              <circle cx="286" cy="96" r="11" className={`track-marker ${getMarkerClass('turn_2')}`} />
              <circle cx="487" cy="205" r="14" className={`track-marker ${getMarkerClass('turn_4')}`} />
              <circle cx="403" cy="214" r="11" className={`track-marker ${getMarkerClass('final_corner')}`} />
            </svg>

            <div className="track-label track-label--main">
              Main Straight<br />
              <span>{zoneMap.get('main_straight')?.condition || 'Dry'}</span>
            </div>
            <div className="track-label track-label--turn2">
              Turn 2<br />
              <span>{zoneMap.get('turn_2')?.condition || 'Damp'}</span>
            </div>
            <div className="track-label track-label--turn4">
              Turn 4<br />
              <span>
                {zoneMap.get('turn_4')?.condition || 'Wet'}
                {zoneMap.get('turn_4')?.risk === 'High' ? ' • High Risk' : ''}
              </span>
            </div>
            <div className="track-label track-label--final">
              Final Corner<br />
              <span>{zoneMap.get('final_corner')?.condition || 'Drying'}</span>
            </div>

            <div className="track-legend" aria-label="Condition legend">
              <span><i className="legend-swatch legend-swatch--dry" />Dry</span>
              <span><i className="legend-swatch legend-swatch--damp" />Damp</span>
              <span><i className="legend-swatch legend-swatch--danger" />Wet</span>
              <span><i className="legend-swatch legend-swatch--drying" />Drying</span>
            </div>
          </div>
        </section>

        {/* Automatic Radio Communication Panel (No Manual Play Button) */}
        <aside className="panel radio-panel" aria-label="Radio communications panel">
          <div className="panel-header compact-header">
            <div className="panel-heading">
              <Radio size={16} />
              <span>Radio</span>
            </div>
            <span className="radio-tag">
              <span className="live-dot" aria-hidden="true" />
              LIVE PIT WALL
            </span>
          </div>

          <div className={`radio-card ${isRadioBroadcasting ? 'broadcasting' : ''}`}>
            <div className="radio-icon" aria-hidden="true">
              {isRadioBroadcasting ? (
                <Volume2 className="pulse-red" size={18} color="#ef4444" />
              ) : (
                <Radio size={18} />
              )}
            </div>
            <div>
              <span className="auto-call-badge">
                {isRadioBroadcasting ? '⚡ EMERGENCY CALL BROADCASTING' : '⚡ AUTOMATIC PIT CALL'}
              </span>
              <p style={{ margin: 0, fontSize: '12px', color: 'var(--muted)', fontStyle: 'italic' }}>
                {isRadioBroadcasting
                  ? 'Broadcasting live voice instructions to driver...'
                  : 'Emergency radio messages automatically speak via voice call.'}
              </p>
            </div>
          </div>

          {/* Animated Audio Sound Wave Visualizer when Radio Speech is Active */}
          {isRadioBroadcasting && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px',
                padding: '8px',
                background: 'rgba(239, 68, 68, 0.1)',
                border: '1px solid rgba(239, 68, 68, 0.3)',
                borderRadius: '8px',
                color: '#fca5a5',
                fontSize: '11px',
                fontWeight: 600,
              }}
            >
              <div className="audio-waves">
                <div className="audio-wave-bar" />
                <div className="audio-wave-bar" />
                <div className="audio-wave-bar" />
                <div className="audio-wave-bar" />
              </div>
              <span>Broadcasting live voice alert to driver...</span>
            </div>
          )}

          <div className="strategy-box">
            <p className="label">Tire strategy</p>
            <div className="strategy-row">
              <ShieldCheck size={16} />
              <strong>{ui.suggestedTire}</strong>
            </div>
            <p className="callout">{ui.recommendationText.toUpperCase()}</p>
          </div>
        </aside>

        <section className="panel chart-panel" aria-label="Wetness trend chart">
          <div className="panel-header">
            <div className="panel-heading">
              <Activity size={16} />
              <span>Wetness trend</span>
            </div>
            <span className="panel-meta">
              {ui.wetnessTrend[0]?.wetness ?? ui.overallWetnessPct}% → {ui.overallWetnessPct}%
            </span>
          </div>

          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={ui.wetnessTrend} margin={{ top: 10, right: 18, left: -15, bottom: 0 }}>
                <defs>
                  <linearGradient id="wetnessFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#7dd3fc" stopOpacity={0.9} />
                    <stop offset="100%" stopColor="#7dd3fc" stopOpacity={0.12} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgba(148, 163, 184, 0.14)" vertical={false} />
                <XAxis dataKey="minute" tickLine={false} axisLine={false} tick={{ fill: '#8aa0ba', fontSize: 11 }} />
                <YAxis
                  domain={[0, 100]}
                  tickLine={false}
                  axisLine={false}
                  tick={{ fill: '#8aa0ba', fontSize: 11 }}
                  tickFormatter={(value) => `${value}%`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0f172a',
                    border: '1px solid rgba(148, 163, 184, 0.2)',
                    borderRadius: 12,
                    color: '#e2e8f0',
                  }}
                  formatter={(value) => {
                    const numericValue = Array.isArray(value)
                      ? Number(value[0] ?? 0)
                      : Number(value ?? 0)

                    return [`${numericValue}%`, 'Wetness']
                  }}
                />
                <Area type="monotone" dataKey="wetness" stroke="#7dd3fc" strokeWidth={3} fill="url(#wetnessFill)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="panel zones-panel" aria-label="Zone status cards">
          <div className="panel-header">
            <div className="panel-heading">
              <AlertTriangle size={16} />
              <span>Zone status</span>
            </div>
          </div>

          <div className="zone-grid">
            {ui.zoneCards.map((zone) => (
              <article
                key={zone.id}
                className={`zone-card zone-card--${zone.tone} ${zone.id === 'turn_4' || zone.risk === 'High' ? 'is-danger' : ''}`}
              >
                <div className="zone-header">
                  <p>{zone.label}</p>
                  <span className="zone-status">{zone.condition}</span>
                </div>

                <div className="zone-metrics">
                  <div>
                    <span className="metric-label">Wetness</span>
                    <strong>{zone.wetness}%</strong>
                  </div>
                  <div>
                    <span className="metric-label">Risk</span>
                    <strong>{zone.risk}</strong>
                  </div>
                  <div>
                    <span className="metric-label">Confidence</span>
                    <strong>{zone.confidence}%</strong>
                  </div>
                </div>

                <div className="progress-group" aria-label={`${zone.label} progress`}>
                  <span className="progress-meter" style={{ width: `${zone.progress}%` }} />
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="panel timeline-panel" aria-label="Alert timeline">
          <div className="panel-header">
            <div className="panel-heading">
              <Video size={16} />
              <span>Alert timeline</span>
            </div>
          </div>

          <ul className="timeline">
            {ui.alerts.map((alert, index) => (
              <li key={`${alert.time}-${index}`} className="timeline-item">
                <span className="timeline-time">{alert.time}</span>
                <div className="timeline-content">
                  <strong>{alert.title}</strong>
                  <p>{alert.detail}</p>
                </div>
              </li>
            ))}
          </ul>
        </section>
      </main>
    </div>
  )
}

export default App
