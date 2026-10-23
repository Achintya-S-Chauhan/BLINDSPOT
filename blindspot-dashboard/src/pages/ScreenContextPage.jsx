import React, { useState } from 'react';
import { ScreenshotPreview } from '../components/context/ScreenshotPreview';
import { OcrViewer } from '../components/context/OcrViewer';
import { mockScreenContext } from '../data/mockData';
import { EmptyState } from '../components/common/EmptyState';
import { Badge } from '../components/common/Badge';
import { useMonitoring } from '../context/MonitoringContext';
import {
  RefreshCw,
  ScanEye,
  ShieldCheck,
  ToggleLeft,
  ToggleRight,
} from 'lucide-react';

export const ScreenContextPage = () => {
  const {
    isContextEmpty,
    setIsContextEmpty,
    screenContext,
    fetchDetailedData,
    connectionStatus,
    isDemoMode,
  } = useMonitoring();

  const [isLoading, setIsLoading] = useState(false);
  const isLive = connectionStatus === 'connected' && !isDemoMode;

  const contextData = isLive
    ? (screenContext || {
        windowTitle: 'Waiting for foreground OCR scan...',
        rawOcrText: '',
        wordCount: 0,
        characterCount: 0,
        detectedKeywords: [],
        capturedAt: 'Polling active engine',
        confidence: 0,
      })
    : mockScreenContext;

  const handleRefresh = async () => {
    setIsLoading(true);
    if (fetchDetailedData) {
      await fetchDetailedData();
    }
    setTimeout(() => {
      setIsLoading(false);
    }, 400);
  };

  return (
    <div className="space-y-6">
      {/* Privacy Guarantee & Simulator Strip */}
      <div className="p-4 rounded-xl bg-background-secondary border border-border/70 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 shrink-0">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs font-semibold text-white flex items-center gap-2">
              <span>Zero-Screengrab Privacy Model</span>
              <Badge variant="success" size="sm">
                ENFORCED
              </Badge>
            </div>
            <p className="text-[11px] text-slate-400 mt-0.5">
              Visual pixels are processed transiently in memory. No screenshots are saved or uploaded to cloud storage.
            </p>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2.5 self-start sm:self-auto">
          {/* Toggle simulated empty state */}
          <button
            onClick={() => setIsContextEmpty(!isContextEmpty)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-background-elevated hover:bg-background-elevated/80 border border-border text-xs text-slate-300 transition-colors"
          >
            {isContextEmpty ? (
              <ToggleRight className="w-4 h-4 text-indigo-400" />
            ) : (
              <ToggleLeft className="w-4 h-4 text-slate-500" />
            )}
            <span>{isContextEmpty ? 'Simulating Empty' : 'Normal State'}</span>
          </button>

          {/* Refresh context button */}
          <button
            onClick={handleRefresh}
            disabled={isLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white text-xs font-medium transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            <span>Refresh Context</span>
          </button>
        </div>
      </div>

      {/* Screen Context Panels or Empty State */}
      {isContextEmpty ? (
        <EmptyState
          icon={ScanEye}
          title="No Foreground Window Detected"
          description="BLINDSPOT has detected an idle desktop, locked session, or minimized window state. Optical OCR analysis is paused to conserve system resources."
          actionText="Restore Context View"
          onAction={() => setIsContextEmpty(false)}
        />
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <ScreenshotPreview contextData={contextData} isLoading={isLoading} />
          <OcrViewer contextData={contextData} isLoading={isLoading} />
        </div>
      )}
    </div>
  );
};
