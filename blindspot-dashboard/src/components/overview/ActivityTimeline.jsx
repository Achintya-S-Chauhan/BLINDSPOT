import React from 'react';
import { mockActivityHistory } from '../../data/mockData';
import { Badge } from '../common/Badge';
import { Clock } from 'lucide-react';
import { useMonitoring } from '../../context/MonitoringContext';

export const ActivityTimeline = () => {
  const { overviewData, isDemoMode, connectionStatus } = useMonitoring();
  const isLive = connectionStatus === 'connected' && !isDemoMode;

  const recentEvents = isLive
    ? (overviewData?.recentTimeline || [])
    : mockActivityHistory.slice(0, 5);

  const getCategoryBadgeVariant = (cat) => {
    switch (cat) {
      case 'Development':
        return 'primary';
      case 'Research':
        return 'info';
      case 'Communication':
        return 'success';
      case 'Documentation':
        return 'warning';
      default:
        return 'neutral';
    }
  };

  return (
    <div className="bg-background-secondary border border-border rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-white">Recent Perception Timeline</h3>
          <p className="text-xs text-slate-400 mt-0.5">Sequential desktop focus transitions</p>
        </div>
        <span className="text-[11px] font-mono text-slate-400 bg-background-elevated px-2 py-1 rounded border border-border/50">
          {recentEvents.length} Recent Events
        </span>
      </div>

      {recentEvents.length === 0 ? (
        <div className="py-8 text-center text-xs text-slate-400 bg-background-elevated/30 rounded-lg border border-border/40">
          No focus transitions recorded yet. As you switch between applications, episodes will be recorded here.
        </div>
      ) : (
        <div className="relative pl-6 space-y-6 before:content-[''] before:absolute before:left-2 before:top-2 before:bottom-2 before:w-[2px] before:bg-border">
        {recentEvents.map((event, idx) => (
          <div key={event.id} className="relative group">
            {/* Timeline node bullet */}
            <div
              className={`absolute -left-[29px] top-1.5 w-3.5 h-3.5 rounded-full border-2 border-background-secondary transition-transform group-hover:scale-125 ${
                idx === 0 ? 'bg-indigo-500 ring-2 ring-indigo-500/30' : 'bg-slate-600'
              }`}
            />

            <div className="p-3.5 rounded-lg bg-background-elevated/50 hover:bg-background-elevated border border-border/40 hover:border-border transition-all">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1.5 mb-1.5">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-white">
                    {event.application}
                  </span>
                  <Badge variant={getCategoryBadgeVariant(event.category)} size="sm">
                    {event.category}
                  </Badge>
                </div>
                <div className="flex items-center gap-2 text-[11px] text-slate-400 font-mono">
                  <Clock className="w-3 h-3 text-slate-500" />
                  <span>{event.startTime} - {event.endTime}</span>
                  <span className="text-indigo-400 font-medium">({event.durationFormatted})</span>
                </div>
              </div>

              <div className="text-xs text-slate-300 font-mono bg-background-secondary/80 p-2 rounded border border-border/30 truncate">
                {event.windowTitle}
              </div>

              {event.ocrSnippet && (
                <div className="mt-2 text-[11px] text-slate-400 line-clamp-1 italic">
                  OCR: "{event.ocrSnippet}"
                </div>
              )}
            </div>
          </div>
        ))}
        </div>
      )}
    </div>
  );
};
