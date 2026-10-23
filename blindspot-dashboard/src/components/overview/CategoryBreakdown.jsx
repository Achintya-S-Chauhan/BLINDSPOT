import React from 'react';
import { Code2, Globe, MessageSquare, Terminal, FileText, ChevronRight } from 'lucide-react';
import { useMonitoring } from '../../context/MonitoringContext';

export const CategoryBreakdown = () => {
  const { overviewData, isDemoMode, connectionStatus } = useMonitoring();
  const isLive = connectionStatus === 'connected' && !isDemoMode;

  const demoCategoryTotals = {
    Development: { seconds: 16560, color: '#6366f1', count: 2 },
    Research: { seconds: 5880, color: '#3b82f6', count: 1 },
    Communication: { seconds: 2760, color: '#10b981', count: 1 },
    Documentation: { seconds: 1560, color: '#f59e0b', count: 1 },
    Design: { seconds: 3000, color: '#ec4899', count: 1 },
    System: { seconds: 400, color: '#64748b', count: 1 },
  };

  const categoryTotals = (isLive && overviewData?.categoryTotals && Object.keys(overviewData.categoryTotals).length > 0)
    ? overviewData.categoryTotals
    : demoCategoryTotals;

  const totalSeconds = Object.values(categoryTotals).reduce((acc, curr) => acc + (curr.seconds || 0), 0);

  const getCategoryIcon = (cat) => {
    switch (cat) {
      case 'Development':
        return <Code2 className="w-4 h-4 text-indigo-400" />;
      case 'Research':
        return <Globe className="w-4 h-4 text-blue-400" />;
      case 'Communication':
        return <MessageSquare className="w-4 h-4 text-emerald-400" />;
      case 'Documentation':
        return <FileText className="w-4 h-4 text-amber-400" />;
      default:
        return <Terminal className="w-4 h-4 text-slate-400" />;
    }
  };

  const formatHoursMinutes = (secs) => {
    const hours = Math.floor(secs / 3600);
    const mins = Math.floor((secs % 3600) / 60);
    if (hours > 0) return `${hours}h ${mins}m`;
    return `${mins}m`;
  };

  const activeCategories = Object.entries(categoryTotals)
    .filter(([, data]) => (data.seconds || 0) > 0)
    .sort((a, b) => (b[1].seconds || 0) - (a[1].seconds || 0));

  return (
    <div className="bg-background-secondary border border-border rounded-xl p-5 flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-semibold text-white">Category Distribution</h3>
            <p className="text-xs text-slate-400 mt-0.5">Time partitioned by activity taxonomy</p>
          </div>
          <span className="text-[11px] font-mono text-slate-400 bg-background-elevated px-2 py-1 rounded border border-border/50">
            {formatHoursMinutes(totalSeconds)} Logged
          </span>
        </div>

        {/* Multi-segment Progress Bar */}
        <div className="h-3 w-full rounded-full bg-background-elevated overflow-hidden flex gap-0.5 p-0.5 mb-5 border border-border/40">
          {activeCategories.map(([name, data]) => {
            const percentage = (data.seconds / totalSeconds) * 100;
            return (
              <div
                key={name}
                style={{
                  width: `${percentage}%`,
                  backgroundColor: data.color,
                }}
                className="h-full first:rounded-l-full last:rounded-r-full transition-all duration-500"
                title={`${name}: ${percentage.toFixed(1)}%`}
              />
            );
          })}
        </div>

        {/* Category List */}
        <div className="space-y-3">
          {activeCategories.map(([name, data]) => {
            const percentage = Math.round((data.seconds / totalSeconds) * 100);
            return (
              <div
                key={name}
                className="flex items-center justify-between p-2 rounded-lg hover:bg-background-elevated/40 transition-colors text-xs"
              >
                <div className="flex items-center gap-2.5">
                  <div
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: data.color }}
                  />
                  {getCategoryIcon(name)}
                  <span className="font-medium text-slate-200">{name}</span>
                </div>

                <div className="flex items-center gap-3">
                  <span className="font-mono text-slate-400 text-[11px]">
                    {formatHoursMinutes(data.seconds)}
                  </span>
                  <span className="font-mono text-slate-300 font-semibold w-9 text-right">
                    {percentage}%
                  </span>
                  <ChevronRight className="w-3.5 h-3.5 text-slate-600" />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="mt-5 pt-3 border-t border-border/40 text-[11px] text-slate-500 flex items-center justify-between">
        <span>Taxonomy Rules: BLINDSPOT Heuristics</span>
        <span>Confidence: High</span>
      </div>
    </div>
  );
};
