import React, { useState, useEffect, useCallback } from 'react';
import {
  Brain,
  Clock,
  Layers,
  AlertTriangle,
  CheckCircle2,
  RefreshCw,
  Info,
  Server,
  Database,
  AlertCircle,
  FileText,
  Activity,
} from 'lucide-react';
import { StatCard } from '../components/common/StatCard';
import { Badge } from '../components/common/Badge';
import { EmptyState } from '../components/common/EmptyState';
import { fetchActivityIntelligence, diagnoseEvents } from '../api/client';

// Built-in fallback demo dataset matching Python get_demo_events()
const FALLBACK_DEMO_DATA = {
  status: 'success',
  is_demo: true,
  data_source: 'client_fallback_demo',
  intelligence: {
    summary: {
      total_time_seconds: 4500.0,
      total_time_formatted: '1h 15m',
      total_sessions: 4,
      valid_events_count: 5,
      invalid_events_count: 0,
      top_application: 'Visual Studio Code',
      application_breakdown: [
        {
          application: 'Visual Studio Code',
          seconds: 2400.0,
          formatted: '40m',
          percentage: 53.3,
        },
        {
          application: 'Google Chrome',
          seconds: 1500.0,
          formatted: '25m',
          percentage: 33.3,
        },
        {
          application: 'Slack',
          seconds: 600.0,
          formatted: '10m',
          percentage: 13.3,
        },
      ],
      summary_text:
        'Recorded 4 sessions across 3 applications (total: 1h 15m). Top application was Visual Studio Code (40m, 53.3%).',
    },
    sessions: [
      {
        application: 'Visual Studio Code',
        activity: 'coding',
        start_time: Date.now() / 1000 - 4500,
        end_time: Date.now() / 1000 - 2700,
        duration_seconds: 1800.0,
        duration_formatted: '30m',
        event_count: 1,
        window_titles: ['ai.py — BLINDSPOT — Visual Studio Code'],
      },
      {
        application: 'Google Chrome',
        activity: 'researching in browser',
        start_time: Date.now() / 1000 - 2700,
        end_time: Date.now() / 1000 - 1500,
        duration_seconds: 1200.0,
        duration_formatted: '20m',
        event_count: 1,
        window_titles: ['FastAPI Documentation — Google Chrome'],
      },
      {
        application: 'Slack',
        activity: 'chatting',
        start_time: Date.now() / 1000 - 1500,
        end_time: Date.now() / 1000 - 900,
        duration_seconds: 600.0,
        duration_formatted: '10m',
        event_count: 1,
        window_titles: ['#engineering-standup — Slack'],
      },
      {
        application: 'Visual Studio Code',
        activity: 'coding',
        start_time: Date.now() / 1000 - 900,
        end_time: Date.now() / 1000,
        duration_seconds: 900.0,
        duration_formatted: '15m',
        event_count: 1,
        window_titles: ['activity_intelligence.py — BLINDSPOT'],
      },
    ],
    diagnostics: {
      warning_count: 1,
      warnings: [
        {
          severity: 'info',
          code: 'EMPTY_OCR',
          message: 'Event at index 4 (Google Chrome): In-memory OCR buffer was empty.',
          event_index: 4,
        },
      ],
    },
  },
};

// Stress test events with synthetic anomalies to test diagnostic warnings
const STRESS_TEST_EVENTS = [
  {
    application: 'Visual Studio Code',
    window_title: 'main.py - Visual Studio Code',
    activity: 'coding',
    timestamp: 1000.0,
    ocr_text: '', // EMPTY_OCR warning for text-heavy app
  },
  {
    application: 'Visual Studio Code',
    window_title: 'main.py - Visual Studio Code',
    activity: 'coding',
    timestamp: 1000.0, // SUSPICIOUS_DUPLICATE warning
  },
  {
    application: 'Command Prompt',
    window_title: 'cmd.exe',
    activity: 'terminal',
    timestamp: 900.0, // OUT_OF_ORDER_TIMESTAMPS warning (retrograde)
  },
  {
    // MISSING_IDENTIFIER warning & rejected
    timestamp: 1500.0,
    activity: 'unknown',
  },
  {
    application: 'Calculator',
    window_title: 'Calculator',
    activity: 'calculating',
    timestamp: -100.0, // INVALID_TIMESTAMP warning & rejected
  },
  {
    application: 'Google Chrome',
    window_title: 'Documentation - Google Chrome',
    activity: 'browsing',
    timestamp: 2000.0,
    end_time: 2600.0,
    ocr_text: 'Python FastAPI Guide',
  },
];

export const ActivityIntelligencePage = () => {
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('active'); // 'active' | 'demo' | 'stress' | 'empty'
  const [isBackendConnected, setIsBackendConnected] = useState(false);

  const fetchFromApi = useCallback(async (tabName = activeTab) => {
    setIsLoading(true);
    setError(null);

    if (tabName === 'empty') {
      setData({
        status: 'success',
        is_demo: false,
        data_source: 'empty_state',
        intelligence: {
          summary: {
            total_time_seconds: 0,
            total_time_formatted: '0s',
            total_sessions: 0,
            valid_events_count: 0,
            invalid_events_count: 0,
            top_application: null,
            application_breakdown: [],
            summary_text: 'No desktop activity recorded.',
          },
          sessions: [],
          diagnostics: { warning_count: 0, warnings: [] },
        },
      });
      setIsLoading(false);
      return;
    }

    if (tabName === 'stress') {
      try {
        const json = await diagnoseEvents(STRESS_TEST_EVENTS);
        setData({
          ...json,
          is_demo: true,
          data_source: 'synthetic_stress_dataset',
        });
        setIsBackendConnected(true);
      } catch {
        // Run locally synthesized response if backend is offline
        setData({
          status: 'success',
          is_demo: true,
          data_source: 'offline_stress_dataset',
          intelligence: {
            summary: {
              total_time_seconds: 605.0,
              total_time_formatted: '10m 5s',
              total_sessions: 3,
              valid_events_count: 4,
              invalid_events_count: 2,
              top_application: 'Google Chrome',
              application_breakdown: [
                { application: 'Google Chrome', seconds: 600.0, formatted: '10m', percentage: 99.2 },
                { application: 'Visual Studio Code', seconds: 5.0, formatted: '5s', percentage: 0.8 },
              ],
              summary_text:
                'Recorded 3 sessions across 2 applications (total: 10m 5s). Top application was Google Chrome.',
            },
            sessions: [
              {
                application: 'Command Prompt',
                activity: 'terminal',
                start_time: 900.0,
                end_time: 1000.0,
                duration_seconds: 100.0,
                duration_formatted: '1m 40s',
                event_count: 1,
                window_titles: ['cmd.exe'],
              },
              {
                application: 'Visual Studio Code',
                activity: 'coding',
                start_time: 1000.0,
                end_time: 1000.0,
                duration_seconds: 5.0,
                duration_formatted: '5s',
                event_count: 1,
                window_titles: ['main.py - Visual Studio Code'],
              },
              {
                application: 'Google Chrome',
                activity: 'browsing',
                start_time: 2000.0,
                end_time: 2600.0,
                duration_seconds: 600.0,
                duration_formatted: '10m',
                event_count: 1,
                window_titles: ['Documentation - Google Chrome'],
              },
            ],
            diagnostics: {
              warning_count: 5,
              warnings: [
                {
                  severity: 'info',
                  code: 'EMPTY_OCR',
                  message: 'Event at index 0 (Visual Studio Code): In-memory OCR buffer was empty.',
                  event_index: 0,
                },
                {
                  severity: 'warning',
                  code: 'OUT_OF_ORDER_TIMESTAMPS',
                  message: 'Timestamp retrograde: event at index 2 (900.0) is earlier than previous event.',
                  event_index: 2,
                },
                {
                  severity: 'info',
                  code: 'SUSPICIOUS_DUPLICATE',
                  message: "Duplicate event for 'Visual Studio Code' at timestamp 1000.00.",
                  event_index: 1,
                },
                {
                  severity: 'warning',
                  code: 'MISSING_IDENTIFIER',
                  message: "Event at index 3 is missing both 'application' and 'window_title'.",
                  event_index: 3,
                },
                {
                  severity: 'warning',
                  code: 'INVALID_TIMESTAMP',
                  message: 'Event at index 4 has a non-positive timestamp (-100.0).',
                  event_index: 4,
                },
              ],
            },
          },
        });
      }
      setIsLoading(false);
      return;
    }

    const useDemo = tabName === 'demo';
    try {
      const json = await fetchActivityIntelligence(useDemo);
      setData(json);
      setIsBackendConnected(true);
    } catch {
      setIsBackendConnected(false);
      // Fallback to client-side demonstration dataset
      setData(FALLBACK_DEMO_DATA);
    } finally {
      setIsLoading(false);
    }
  }, [activeTab]);

  useEffect(() => {
    const timer = setTimeout(() => {
      fetchFromApi(activeTab);
    }, 0);
    return () => clearTimeout(timer);
  }, [activeTab, fetchFromApi]);

  const summary = data?.intelligence?.summary || {};
  const sessions = data?.intelligence?.sessions || [];
  const diagnostics = data?.intelligence?.diagnostics || { warning_count: 0, warnings: [] };
  const isDemo = data?.is_demo ?? true;
  const dataSource = data?.data_source || 'unknown';

  return (
    <div className="space-y-6">
      {/* Top Source & Mode Switcher Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-4 rounded-xl bg-background-secondary border border-border/70">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shrink-0">
            <Brain className="w-5 h-5 text-indigo-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-semibold text-white tracking-wide">
                Activity Intelligence & Diagnostics
              </h2>
              {isDemo ? (
                <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20 flex items-center gap-1">
                  <Database className="w-3 h-3" />
                  DEMO DATASET (SIMULATED)
                </span>
              ) : (
                <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-1">
                  <Server className="w-3 h-3" />
                  LIVE FASTAPI BACKEND
                </span>
              )}
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Transform desktop events into structured session timelines, time aggregates, and health warnings.
            </p>
          </div>
        </div>

        {/* Action Controls & Tab Filters */}
        <div className="flex flex-wrap items-center gap-2">
          <div className="inline-flex rounded-lg bg-background-elevated p-1 border border-border/60">
            <button
              onClick={() => setActiveTab('active')}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                activeTab === 'active'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Active Stream
            </button>
            <button
              onClick={() => setActiveTab('demo')}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                activeTab === 'demo'
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Demo Dataset
            </button>
            <button
              onClick={() => setActiveTab('stress')}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                activeTab === 'stress'
                  ? 'bg-amber-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Stress Diagnostics
            </button>
            <button
              onClick={() => setActiveTab('empty')}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-all ${
                activeTab === 'empty'
                  ? 'bg-slate-700 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Empty State
            </button>
          </div>

          <button
            onClick={() => fetchFromApi(activeTab)}
            disabled={isLoading}
            className="p-1.5 rounded-lg bg-background-elevated hover:bg-background-subtle border border-border/60 text-slate-300 hover:text-white transition-colors"
            title="Refresh Intelligence Data"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin text-indigo-400' : ''}`} />
          </button>
        </div>
      </div>

      {/* Backend Status Notice */}
      <div className={`p-3 rounded-lg border text-xs flex items-start sm:items-center justify-between gap-3 ${
        isBackendConnected
          ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300'
          : 'bg-indigo-500/10 border-indigo-500/20 text-indigo-300'
      }`}>
        <div className="flex items-center gap-2">
          {isBackendConnected ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
          ) : (
            <Info className="w-4 h-4 text-indigo-400 shrink-0" />
          )}
          <span>
            {isBackendConnected ? (
              <>
                <strong>FastAPI Connected (127.0.0.1:8000):</strong> Serving real-time processed desktop activity records via read-only intelligence endpoints.
              </>
            ) : (
              <>
                <strong>Backend Standalone / Demo Mode:</strong> Python backend at <code className="bg-background-elevated px-1 py-0.5 rounded text-indigo-200">127.0.0.1:8000</code> is offline or unreachable. Displaying validated local desktop event dataset.
              </>
            )}
          </span>
        </div>
        <span className="text-[11px] font-mono shrink-0 px-2 py-0.5 rounded bg-background-elevated border border-border text-slate-300">
          Source: {dataSource}
        </span>
      </div>

      {/* Error Banner */}
      {error && (
        <div className="p-3.5 rounded-lg bg-rose-500/10 border border-rose-500/20 flex items-center justify-between text-xs text-rose-300">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
          <button
            onClick={() => fetchFromApi(activeTab)}
            className="text-xs font-medium px-2.5 py-1 rounded bg-rose-500/20 hover:bg-rose-500/30 text-rose-200 transition-colors"
          >
            Retry Connection
          </button>
        </div>
      )}

      {/* Loading Overlay / Content */}
      {isLoading ? (
        <div className="flex flex-col items-center justify-center p-16 text-center bg-background-secondary rounded-xl border border-border/70">
          <RefreshCw className="w-8 h-8 text-indigo-400 animate-spin mb-3" />
          <h3 className="text-sm font-semibold text-slate-200">Processing Activity Intelligence...</h3>
          <p className="text-xs text-slate-400 mt-1 max-w-sm">
            Validating desktop event structures, grouping consecutive app sessions, and checking OCR heuristics.
          </p>
        </div>
      ) : activeTab === 'empty' || summary.total_sessions === 0 ? (
        <EmptyState
          icon={Activity}
          title="No Desktop Activity Recorded"
          description="The event stream is currently clear. Switch to Demo Dataset or Stress Diagnostics to preview session calculation and health flags."
          actionText="Load Demonstration Dataset"
          onAction={() => setActiveTab('demo')}
        />
      ) : (
        <>
          {/* Top Metric Cards Row */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <StatCard
              title="Total Time Calculated"
              value={summary.total_time_formatted || '0s'}
              subtitle="Aggregated non-idle sessions"
              icon={Clock}
              badgeText={isDemo ? 'Simulated' : 'Engine Live'}
              badgeVariant={isDemo ? 'primary' : 'success'}
            />

            <StatCard
              title="Consecutive Sessions"
              value={`${summary.total_sessions || 0} Sessions`}
              subtitle={`${summary.valid_events_count || 0} events validated`}
              icon={Layers}
              trend={{
                value: `${summary.invalid_events_count || 0} Rejected`,
                isPositive: summary.invalid_events_count === 0,
                label: 'invalid events',
              }}
            />

            <StatCard
              title="Top Application"
              value={summary.top_application || 'None'}
              subtitle={
                summary.application_breakdown?.[0]
                  ? `${summary.application_breakdown[0].formatted} (${summary.application_breakdown[0].percentage}%)`
                  : 'No usage'
              }
              icon={Activity}
              badgeText="Dominant"
              badgeVariant="primary"
            />

            <StatCard
              title="Diagnostic Health"
              value={`${diagnostics.warning_count} Notice${diagnostics.warning_count === 1 ? '' : 's'}`}
              subtitle={diagnostics.warning_count === 0 ? 'Engine clean' : 'Anomalies flagged'}
              icon={AlertTriangle}
              badgeText={diagnostics.warning_count === 0 ? 'Clean' : 'Review'}
              badgeVariant={diagnostics.warning_count === 0 ? 'success' : 'warning'}
            />
          </div>

          {/* Human-Readable Summary Card */}
          <div className="p-4 rounded-xl bg-background-secondary border border-border/70 relative overflow-hidden">
            <div className="flex items-center gap-2 mb-2 text-xs font-semibold uppercase tracking-wider text-indigo-400">
              <FileText className="w-4 h-4" />
              Generated Activity Summary
            </div>
            <p className="text-sm text-slate-200 leading-relaxed font-sans">
              {summary.summary_text}
            </p>
            <div className="mt-3 pt-3 border-t border-border/40 flex flex-wrap items-center gap-3 text-xs text-slate-400">
              <span>
                <strong>Pipeline:</strong> Python <code className="text-slate-300">activity_intelligence.py</code>
              </span>
              <span>•</span>
              <span>
                <strong>Validated Events:</strong> {summary.valid_events_count || 0} accepted
              </span>
              <span>•</span>
              <span>
                <strong>Sanitization:</strong> Bad timestamps & missing headers safely filtered
              </span>
            </div>
          </div>

          {/* Application Breakdown & Diagnostics Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Application Time Breakdown */}
            <div className="p-5 rounded-xl bg-background-secondary border border-border/70 space-y-4">
              <div className="flex items-center justify-between border-b border-border/40 pb-3">
                <div className="flex items-center gap-2">
                  <Layers className="w-4 h-4 text-indigo-400" />
                  <h3 className="text-sm font-semibold text-slate-200">Application Time Breakdown</h3>
                </div>
                <span className="text-xs text-slate-400">
                  {summary.application_breakdown?.length || 0} Applications
                </span>
              </div>

              <div className="space-y-3.5">
                {summary.application_breakdown?.map((app, index) => {
                  const colors = [
                    'bg-indigo-500',
                    'bg-sky-500',
                    'bg-emerald-500',
                    'bg-amber-500',
                    'bg-purple-500',
                  ];
                  const barColor = colors[index % colors.length];

                  return (
                    <div key={app.application} className="space-y-1.5">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-medium text-slate-300">{app.application}</span>
                        <div className="flex items-center gap-2">
                          <span className="text-slate-400 font-mono">{app.formatted}</span>
                          <span className="text-slate-500 font-mono text-[11px] w-12 text-right">
                            {app.percentage}%
                          </span>
                        </div>
                      </div>
                      <div className="w-full h-2 rounded-full bg-background-elevated overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${barColor}`}
                          style={{ width: `${app.percentage}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Diagnostic Warnings Panel */}
            <div className="p-5 rounded-xl bg-background-secondary border border-border/70 space-y-4">
              <div className="flex items-center justify-between border-b border-border/40 pb-3">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-400" />
                  <h3 className="text-sm font-semibold text-slate-200">
                    Engine Diagnostics & Quality Flags
                  </h3>
                </div>
                <Badge
                  variant={diagnostics.warning_count === 0 ? 'success' : 'warning'}
                  size="sm"
                >
                  {diagnostics.warning_count} Notice{diagnostics.warning_count === 1 ? '' : 's'}
                </Badge>
              </div>

              {diagnostics.warnings?.length === 0 ? (
                <div className="py-8 flex flex-col items-center justify-center text-center text-slate-400">
                  <CheckCircle2 className="w-8 h-8 text-emerald-400 mb-2" />
                  <p className="text-xs font-medium text-slate-300">Clean Event Stream</p>
                  <p className="text-[11px] text-slate-500 max-w-xs mt-0.5">
                    No duplicate events, missing timestamps, or OCR dropouts detected.
                  </p>
                </div>
              ) : (
                <div className="space-y-2.5 max-h-[280px] overflow-y-auto pr-1">
                  {diagnostics.warnings.map((w, idx) => (
                    <div
                      key={idx}
                      className={`p-3 rounded-lg border text-xs flex items-start gap-2.5 ${
                        w.severity === 'warning'
                          ? 'bg-amber-500/10 border-amber-500/20 text-amber-200'
                          : 'bg-blue-500/10 border-blue-500/20 text-blue-200'
                      }`}
                    >
                      <AlertCircle className={`w-4 h-4 shrink-0 mt-0.5 ${
                        w.severity === 'warning' ? 'text-amber-400' : 'text-blue-400'
                      }`} />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-0.5">
                          <span className="font-mono font-semibold text-[11px] uppercase tracking-wider">
                            {w.code}
                          </span>
                          {w.event_index !== undefined && (
                            <span className="text-[10px] px-1.5 py-0.2 rounded bg-background-elevated text-slate-300 font-mono">
                              event #{w.event_index}
                            </span>
                          )}
                        </div>
                        <p className="text-[11px] text-slate-300 leading-snug">{w.message}</p>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <p className="text-[11px] text-slate-500 italic pt-1 border-t border-border/30">
                Notice: Diagnostic warnings identify transient OCR dropouts, temporal anomalies, or format issues; they do not mean the perception engine failed.
              </p>
            </div>
          </div>

          {/* Session Durations Timeline / Table */}
          <div className="p-5 rounded-xl bg-background-secondary border border-border/70 space-y-4">
            <div className="flex items-center justify-between border-b border-border/40 pb-3">
              <div className="flex items-center gap-2">
                <Clock className="w-4 h-4 text-indigo-400" />
                <h3 className="text-sm font-semibold text-slate-200">
                  Calculated Session Timeline
                </h3>
              </div>
              <span className="text-xs text-slate-400">
                Grouped consecutively by application
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-border/60 text-slate-400 uppercase tracking-wider text-[11px]">
                    <th className="py-2.5 px-3">Application</th>
                    <th className="py-2.5 px-3">Activity</th>
                    <th className="py-2.5 px-3">Observed Windows</th>
                    <th className="py-2.5 px-3 text-center">Events</th>
                    <th className="py-2.5 px-3 text-right">Duration</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/30 font-sans">
                  {sessions.map((sess, idx) => (
                    <tr key={idx} className="hover:bg-background-elevated/40 transition-colors">
                      <td className="py-3 px-3">
                        <div className="font-medium text-slate-200 flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-indigo-400 shrink-0" />
                          {sess.application}
                        </div>
                      </td>
                      <td className="py-3 px-3">
                        <span className="px-2 py-0.5 rounded bg-background-elevated text-slate-300 font-mono text-[11px]">
                          {sess.activity || 'using an app'}
                        </span>
                      </td>
                      <td className="py-3 px-3 max-w-xs truncate text-slate-400 font-mono text-[11px]">
                        {sess.window_titles?.length > 0
                          ? sess.window_titles.join(' • ')
                          : 'No window title recorded'}
                      </td>
                      <td className="py-3 px-3 text-center font-mono text-slate-400">
                        {sess.event_count || 1}
                      </td>
                      <td className="py-3 px-3 text-right">
                        <span className="font-mono font-semibold text-indigo-300 bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20">
                          {sess.duration_formatted}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default ActivityIntelligencePage;
