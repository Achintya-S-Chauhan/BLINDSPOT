import React from 'react';
import {
  Monitor,
  Clock,
  Layers,
  Sparkles,
  Info,
  Wifi,
  WifiOff,
} from 'lucide-react';
import { StatCard } from '../components/common/StatCard';
import { UsageChart } from '../components/overview/UsageChart';
import { CategoryBreakdown } from '../components/overview/CategoryBreakdown';
import { ActivityTimeline } from '../components/overview/ActivityTimeline';
import { useMonitoring } from '../context/MonitoringContext';

export const OverviewPage = () => {
  const {
    liveContext,
    overviewData,
    isMonitoring,
    connectionStatus,
    isDemoMode,
  } = useMonitoring();

  const isLive = connectionStatus === 'connected' && !isDemoMode;
  const isDisconnected = connectionStatus === 'disconnected' && !isDemoMode;

  return (
    <div className="space-y-6">
      {/* Live / Disconnected / Demo Banner */}
      <div
        className={`p-3.5 rounded-xl border flex items-start sm:items-center justify-between gap-3 text-xs ${
          isLive
            ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-300'
            : isDisconnected
            ? 'bg-rose-500/10 border-rose-500/20 text-rose-300'
            : 'bg-indigo-500/10 border-indigo-500/20 text-indigo-300'
        }`}
      >
        <div className="flex items-center gap-2.5">
          {isLive ? (
            <Wifi className="w-4 h-4 shrink-0 text-emerald-400" />
          ) : isDisconnected ? (
            <WifiOff className="w-4 h-4 shrink-0 text-rose-400" />
          ) : (
            <Info className="w-4 h-4 shrink-0 text-indigo-400" />
          )}
          <span>
            {isLive ? (
              <>
                <strong>Live Desktop Perception Active:</strong> Connected to BLINDSPOT engine (127.0.0.1:8000). Telemetry reflects actual Win32 window focus and in-memory OCR.
              </>
            ) : isDisconnected ? (
              <>
                <strong>Backend Disconnected:</strong> Cannot reach BLINDSPOT engine at <code className="bg-background-elevated px-1 py-0.5 rounded text-rose-200">http://127.0.0.1:8000</code>. Start engine via <code className="bg-background-elevated px-1 py-0.5 rounded text-rose-200">python api.py</code>.
              </>
            ) : (
              <>
                <strong>Simulated Demo Prototype:</strong> Operating on client demonstration dataset. Click <strong>Exit Demo</strong> in the sidebar to bind to live engine data.
              </>
            )}
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-background-elevated border border-border text-slate-300">
            {isLive ? 'Win32 Engine Live' : isDisconnected ? 'Service Offline' : 'Demo Mode'}
          </span>
        </div>
      </div>

      {/* Metric Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Active Application"
          value={liveContext.activeApplication || 'None'}
          subtitle={liveContext.processId ? `Process ID: ${liveContext.processId}` : 'No active PID'}
          icon={Monitor}
          badgeText={isLive ? (isMonitoring ? 'Active' : 'Paused') : 'Demo'}
          badgeVariant={isMonitoring ? 'success' : 'neutral'}
        />

        <StatCard
          title="Tracked Time Today"
          value={isLive ? (overviewData?.trackedTimeTodayFormatted || '0s') : '7h 24m'}
          subtitle={isLive ? 'Recorded desktop focus time' : 'Total recorded session'}
          icon={Clock}
          trend={isLive ? undefined : { value: '+14%', isPositive: true, label: 'vs yesterday' }}
        />

        <StatCard
          title="Applications Observed"
          value={isLive ? `${overviewData?.applicationsObservedCount || (liveContext.activeApplication !== 'None' ? 1 : 0)} Apps` : '12 Apps'}
          subtitle={isLive ? 'Distinct applications observed' : 'Across 5 active categories'}
          icon={Layers}
        />

        <StatCard
          title="Workflow Pattern"
          value={liveContext.workflowPattern ? liveContext.workflowPattern.replace(/_/g, ' ') : 'Focused'}
          subtitle={`Confidence: ${liveContext.confidence || 0}/3`}
          icon={Sparkles}
          badgeText={liveContext.confidence >= 2 ? 'High' : 'Moderate'}
          badgeVariant="primary"
        />
      </div>

      {/* Main Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <UsageChart />
        </div>
        <div className="lg:col-span-1">
          <CategoryBreakdown />
        </div>
      </div>

      {/* Activity Timeline */}
      <ActivityTimeline />
    </div>
  );
};
