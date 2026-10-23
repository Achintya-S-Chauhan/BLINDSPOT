import React, { useState, useEffect, useCallback, useRef } from 'react';
import { MonitoringContext } from './monitoringContextDef';
import {
  initialSettings,
  mockHourlyActivity,
  mockAppUsage,
  mockActivityHistory,
  mockScreenContext,
} from '../data/mockData';
import {
  fetchLiveContext,
  fetchScreenContext,
  fetchActivityHistory,
  fetchOverview,
  fetchSettings,
  toggleEngineMonitoring,
} from '../api/client';

const initialLiveContext = {
  isMonitoring: false,
  activeWindow: 'No active window detected',
  activeApplication: 'None',
  processId: 0,
  category: 'Productivity',
  confidence: 0,
  sessionStartTime: Date.now(),
  currentDurationSeconds: 0,
  currentDurationFormatted: '0s',
  windowBounds: {
    x: 0,
    y: 0,
    width: 1920,
    height: 1080,
  },
  workflowPattern: 'no_activity',
  workflowInterpretation: 'Connecting to BLINDSPOT perception engine...',
  recentSwitches: [],
};

const initialDemoLiveContext = {
  isMonitoring: true,
  activeWindow: 'ui.py — BLINDSPOT — Visual Studio Code',
  activeApplication: 'Visual Studio Code',
  processId: 14280,
  category: 'Development',
  confidence: 3,
  sessionStartTime: Date.now() - 1000 * 60 * 34,
  currentDurationSeconds: 2040,
  currentDurationFormatted: '34m 0s',
  windowBounds: {
    x: 0,
    y: 0,
    width: 1920,
    height: 1040,
  },
  workflowPattern: 'focused_session',
  workflowInterpretation: 'Deep uninterrupted coding session in frontend/backend Python logic.',
  recentSwitches: [
    {
      id: 'sw-1',
      fromApp: 'Windows Terminal',
      toApp: 'Visual Studio Code',
      timestamp: '18:14:20',
      durationInApp: '12m 40s',
    },
    {
      id: 'sw-2',
      fromApp: 'Google Chrome',
      toApp: 'Windows Terminal',
      timestamp: '18:01:40',
      durationInApp: '14m 10s',
    },
    {
      id: 'sw-3',
      fromApp: 'Slack',
      toApp: 'Google Chrome',
      timestamp: '17:47:30',
      durationInApp: '15m 00s',
    },
    {
      id: 'sw-4',
      fromApp: 'Visual Studio Code',
      toApp: 'Slack',
      timestamp: '17:32:30',
      durationInApp: '45m 20s',
    },
  ],
};

const initialDemoOverview = {
  activeApplication: 'Visual Studio Code',
  processId: 14280,
  isMonitoring: true,
  trackedTimeTodaySeconds: 26640,
  trackedTimeTodayFormatted: '7h 24m',
  applicationsObservedCount: 12,
  workflowPattern: 'focused_session',
  categoryTotals: {
    Development: { seconds: 16560, color: '#6366f1', count: 2 },
    Research: { seconds: 5880, color: '#3b82f6', count: 1 },
    Communication: { seconds: 2760, color: '#10b981', count: 1 },
    Documentation: { seconds: 1560, color: '#f59e0b', count: 1 },
  },
  appUsage: mockAppUsage,
  hourlyActivity: mockHourlyActivity,
  recentTimeline: mockActivityHistory.slice(0, 5),
};

const demoActivityHistory = {
  records: mockActivityHistory,
  summary: {
    totalRecords: mockActivityHistory.length,
    totalTimeFormatted: '9h 22m',
    topApplication: 'Visual Studio Code',
    deepWorkAvgFormatted: '46m',
  },
};

export const MonitoringProvider = ({ children }) => {
  const [currentPage, setCurrentPage] = useState('overview');
  const [connectionStatus, setConnectionStatus] = useState('connecting'); // 'connected' | 'disconnected' | 'connecting' | 'demo'
  const [isDemoMode, setIsDemoMode] = useState(false);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [lastSyncTime, setLastSyncTime] = useState(null);

  // Isolated demo state
  const [demoLiveContext, setDemoLiveContext] = useState(initialDemoLiveContext);
  const [isDemoMonitoring, setIsDemoMonitoring] = useState(true);

  // Real data state
  const [liveContext, setLiveContext] = useState(initialLiveContext);
  const [screenContext, setScreenContext] = useState(null);
  const [activityHistory, setActivityHistory] = useState({ records: [], summary: {} });
  const [overviewData, setOverviewData] = useState(null);
  const [settings, setSettings] = useState(initialSettings);
  const [isContextEmpty, setIsContextEmpty] = useState(false);

  const isPollingRef = useRef(false);

  // Poll real backend
  const syncWithBackend = useCallback(async () => {
    if (isPollingRef.current || isDemoMode) return;
    isPollingRef.current = true;

    try {
      // 1. Fetch live context & overview in parallel
      const [liveRes, overviewRes] = await Promise.all([
        fetchLiveContext(),
        fetchOverview(),
      ]);

      if (liveRes.status === 'success' && liveRes.context) {
        setLiveContext(liveRes.context);
        setIsMonitoring(Boolean(liveRes.context.isMonitoring));
      }

      if (overviewRes.status === 'success' && overviewRes.overview) {
        setOverviewData(overviewRes.overview);
      }

      setConnectionStatus('connected');
      setLastSyncTime(new Date());
    } catch {
      setConnectionStatus('disconnected');
    } finally {
      isPollingRef.current = false;
    }
  }, [isDemoMode]);

  // Deep fetch for detailed pages (screen context, history, settings)
  const fetchDetailedData = useCallback(async () => {
    if (isDemoMode) return;
    try {
      const [screenRes, historyRes, settingsRes] = await Promise.all([
        fetchScreenContext().catch(() => null),
        fetchActivityHistory().catch(() => null),
        fetchSettings().catch(() => null),
      ]);

      if (screenRes?.status === 'success' && screenRes.context) {
        setScreenContext(screenRes.context);
      }
      if (historyRes?.status === 'success') {
        setActivityHistory({
          records: historyRes.records || [],
          summary: historyRes.summary || {},
        });
      }
      if (settingsRes) {
        setSettings((prev) => ({ ...prev, ...settingsRes }));
      }
    } catch {
      // Handled silently during periodic refresh
    }
  }, [isDemoMode]);

  // Periodic polling timer (every 1.5s for live context)
  useEffect(() => {
    if (isDemoMode) return;

    // Initial sync
    syncWithBackend();
    fetchDetailedData();

    const interval = setInterval(() => {
      syncWithBackend();
    }, 1500);

    // Slower poll for detailed screen/history
    const detailedInterval = setInterval(() => {
      fetchDetailedData();
    }, 4000);

    return () => {
      clearInterval(interval);
      clearInterval(detailedInterval);
    };
  }, [isDemoMode, syncWithBackend, fetchDetailedData]);

  // Live seconds ticker when active session is running
  useEffect(() => {
    const isRunning = isDemoMode ? isDemoMonitoring : isMonitoring;
    if (!isRunning) return;

    const interval = setInterval(() => {
      if (isDemoMode) {
        setDemoLiveContext((prev) => ({
          ...prev,
          currentDurationSeconds: (prev.currentDurationSeconds || 0) + 1,
        }));
      } else {
        setLiveContext((prev) => ({
          ...prev,
          currentDurationSeconds: (prev.currentDurationSeconds || 0) + 1,
        }));
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [isDemoMode, isDemoMonitoring, isMonitoring]);

  const toggleMonitoring = async () => {
    if (isDemoMode) {
      setIsDemoMonitoring((prev) => !prev);
      setDemoLiveContext((prev) => ({ ...prev, isMonitoring: !prev.isMonitoring }));
      return;
    }

    try {
      const res = await toggleEngineMonitoring(!isMonitoring);
      if (res && res.status === 'success') {
        setIsMonitoring(res.is_running);
        setLiveContext((prev) => ({ ...prev, isMonitoring: res.is_running }));
      }
    } catch {
      // Toggle failed (backend unreachable)
    }
  };

  const switchSimulatedApp = (app, title, category) => {
    if (!isDemoMode) return; // Only permitted in Demo mode
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const formattedDuration = `${Math.floor(demoLiveContext.currentDurationSeconds / 60)}m ${demoLiveContext.currentDurationSeconds % 60}s`;

    setDemoLiveContext((prev) => ({
      ...prev,
      activeApplication: app,
      activeWindow: title,
      category: category,
      currentDurationSeconds: 0,
      sessionStartTime: Date.now(),
      recentSwitches: [
        {
          id: `sw-${Date.now()}`,
          fromApp: prev.activeApplication,
          toApp: app,
          timestamp: timeStr,
          durationInApp: formattedDuration,
        },
        ...prev.recentSwitches.slice(0, 7),
      ],
    }));
  };

  const updateSettings = (newSettings) => {
    setSettings((prev) => ({ ...prev, ...newSettings }));
  };

  const addExcludedApp = (app) => {
    const trimmed = app.trim();
    if (!trimmed || settings.excludedApps?.includes(trimmed)) return;
    setSettings((prev) => ({
      ...prev,
      excludedApps: [...(prev.excludedApps || []), trimmed],
    }));
  };

  const removeExcludedApp = (app) => {
    setSettings((prev) => ({
      ...prev,
      excludedApps: (prev.excludedApps || []).filter((a) => a !== app),
    }));
  };

  return (
    <MonitoringContext.Provider
      value={{
        currentPage,
        setCurrentPage,
        connectionStatus: isDemoMode ? 'demo' : connectionStatus,
        isDemoMode,
        setIsDemoMode,
        lastSyncTime,
        isMonitoring: isDemoMode ? isDemoMonitoring : isMonitoring,
        toggleMonitoring,
        liveContext: isDemoMode ? demoLiveContext : liveContext,
        screenContext: isDemoMode ? mockScreenContext : screenContext,
        activityHistory: isDemoMode ? demoActivityHistory : activityHistory,
        overviewData: isDemoMode ? initialDemoOverview : overviewData,
        refreshData: syncWithBackend,
        fetchDetailedData,
        switchSimulatedApp,
        settings,
        updateSettings,
        addExcludedApp,
        removeExcludedApp,
        isContextEmpty,
        setIsContextEmpty,
      }}
    >
      {children}
    </MonitoringContext.Provider>
  );
};

export { MonitoringContext } from './monitoringContextDef';
export { useMonitoring } from './useMonitoring';
