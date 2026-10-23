import React from 'react';
import { ActivityTable } from '../components/history/ActivityTable';
import { Clock, Layers, Flame, History } from 'lucide-react';
import { useMonitoring } from '../context/MonitoringContext';

export const ActivityHistoryPage = () => {
  const { activityHistory, connectionStatus, isDemoMode } = useMonitoring();
  const isLive = connectionStatus === 'connected' && !isDemoMode;

  const summary = activityHistory?.summary || {};
  const totalRecords = isLive
    ? `${summary.totalRecords || 0} Records`
    : '12 Sessions';
  const totalTime = isLive
    ? (summary.totalTimeFormatted || '0s')
    : '9h 22m';
  const topApp = isLive
    ? (summary.topApplication || 'None')
    : 'VS Code';
  const deepWorkAvg = isLive
    ? (summary.deepWorkAvgFormatted || '0s')
    : '46m';

  return (
    <div className="space-y-6">
      {/* Summary KPI Strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-background-secondary border border-border">
          <div className="flex items-center gap-2 text-xs text-slate-400 mb-1">
            <History className="w-4 h-4 text-indigo-400" />
            <span>Total Records</span>
          </div>
          <div className="text-xl font-bold font-mono text-white">{totalRecords}</div>
          <span className="text-[11px] text-slate-500">
            {isLive ? 'Recorded in active engine memory' : 'Across 2 calendar days'}
          </span>
        </div>

        <div className="p-4 rounded-xl bg-background-secondary border border-border">
          <div className="flex items-center gap-2 text-xs text-slate-400 mb-1">
            <Clock className="w-4 h-4 text-emerald-400" />
            <span>Total Time Logged</span>
          </div>
          <div className="text-xl font-bold font-mono text-white">{totalTime}</div>
          <span className="text-[11px] text-slate-500">
            {isLive ? 'Cumulative context duration' : '88% active focus ratio'}
          </span>
        </div>

        <div className="p-4 rounded-xl bg-background-secondary border border-border">
          <div className="flex items-center gap-2 text-xs text-slate-400 mb-1">
            <Layers className="w-4 h-4 text-blue-400" />
            <span>Top Application</span>
          </div>
          <div className="text-xl font-bold font-mono text-white">{topApp}</div>
          <span className="text-[11px] text-slate-500">
            {isLive ? 'Most focused application' : '4h 36m total usage'}
          </span>
        </div>

        <div className="p-4 rounded-xl bg-background-secondary border border-border">
          <div className="flex items-center gap-2 text-xs text-slate-400 mb-1">
            <Flame className="w-4 h-4 text-amber-400" />
            <span>Avg Session Length</span>
          </div>
          <div className="text-xl font-bold font-mono text-white">{deepWorkAvg}</div>
          <span className="text-[11px] text-slate-500">
            {isLive ? 'Mean duration per episode' : 'Per task window'}
          </span>
        </div>
      </div>

      {/* Main Historical Table */}
      <ActivityTable />
    </div>
  );
};
