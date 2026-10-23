import React from 'react';
import { Badge } from '../common/Badge';
import { Shield, Lock } from 'lucide-react';
import { useMonitoring } from '../../context/MonitoringContext';

export const ScreenshotPreview = ({ contextData = {}, isLoading }) => {
  const { connectionStatus, isDemoMode } = useMonitoring();
  const isLive = connectionStatus === 'connected' && !isDemoMode;

  const rect = contextData?.windowRect || { left: 0, top: 0, width: 1920, height: 1080 };
  const previewText = contextData?.rawOcrText
    ? contextData.rawOcrText.slice(0, 320)
    : contextData?.mockCodeSnippet || 'Awaiting in-memory OCR buffer...';

  return (
    <div className="bg-background-secondary border border-border rounded-xl p-5 flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-semibold text-white">Visual Context Canvas</h3>
            <Badge variant={isLive ? 'primary' : 'warning'} size="sm">
              {isLive ? 'WIN32 HOOK' : 'MOCK CANVAS'}
            </Badge>
          </div>
          <span className="text-xs text-slate-400 font-mono flex items-center gap-1">
            <Shield className="w-3.5 h-3.5 text-emerald-400" />
            Zero Real Screengrabs
          </span>
        </div>

        {/* Window Frame representing desktop screen */}
        <div className="relative rounded-lg border border-border bg-[#0d0f18] overflow-hidden shadow-2xl">
          {/* OS Titlebar */}
          <div className="h-7 bg-[#161826] border-b border-border/70 px-3 flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500/80" />
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500/80" />
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500/80" />
              <span className="text-[11px] text-slate-400 font-mono ml-2 truncate max-w-[200px]">
                {contextData?.windowTitle || 'No active window'}
              </span>
            </div>
            <div className="text-[10px] text-slate-500 font-mono">
              {rect.width || 1920}×{rect.height || 1080} (Win32 DPI Scaled)
            </div>
          </div>

          {/* Canvas area (viewport) */}
          <div className="p-4 font-mono text-xs text-slate-300 relative min-h-[220px] bg-[#0c0d15] flex flex-col justify-between">
            {isLoading ? (
              <div className="absolute inset-0 bg-background/80 backdrop-blur-sm flex flex-col items-center justify-center gap-3 z-10">
                <div className="w-6 h-6 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
                <span className="text-xs text-indigo-300 font-sans">Reading in-memory OCR buffer...</span>
              </div>
            ) : null}

            <div>
              <div className="text-slate-500 text-[11px] mb-2 select-none">
                # Region: active_window_rect [x={rect.left || 0}, y={rect.top || 0}, w={rect.width || 1920}, h={rect.height || 1080}]
              </div>
              <pre className="text-indigo-300/90 whitespace-pre-wrap leading-relaxed overflow-x-auto text-[11px] font-mono">
                {previewText}
              </pre>
            </div>

            {/* Privacy watermark badge */}
            <div className="mt-4 pt-2 border-t border-border/30 flex items-center justify-between text-[11px] text-slate-500">
              <span className="flex items-center gap-1">
                <Lock className="w-3 h-3 text-slate-400" />
                Local in-memory buffer (No cloud upload)
              </span>
              <span>Optical OCR Region 100% Client-Side</span>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-4 pt-3 border-t border-border/40 text-xs text-slate-400 flex items-center justify-between">
        <span>Captured: {contextData?.capturedAt || 'Live polling'}</span>
        <span className="text-indigo-400 font-mono">Confidence: {contextData?.confidence || 0}/3</span>
      </div>
    </div>
  );
};
