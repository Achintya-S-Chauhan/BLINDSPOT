/**
 * BLINDSPOT Dashboard — Shared Engine API Client
 *
 * Provides typed HTTP communication with the local BLINDSPOT Python backend (http://127.0.0.1:8000).
 * Handles connectivity checks, live context, OCR stream, history, and analytics.
 */

export const API_BASE_URL = 'http://127.0.0.1:8000';

const request = async (endpoint, options = {}) => {
  const url = `${API_BASE_URL}${endpoint}`;
  const res = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  });

  if (!res.ok) {
    const errorText = await res.text().catch(() => '');
    throw new Error(`API Error ${res.status}: ${errorText || res.statusText}`);
  }

  return res.json();
};

export const checkHealth = async () => {
  return request('/api/health');
};

export const fetchEngineStatus = async () => {
  return request('/api/engine/status');
};

export const toggleEngineMonitoring = async (running) => {
  return request('/api/engine/toggle', {
    method: 'POST',
    body: JSON.stringify(typeof running === 'boolean' ? { running } : {}),
  });
};

export const fetchLiveContext = async () => {
  return request('/api/context/live');
};

export const fetchScreenContext = async () => {
  return request('/api/context/screen');
};

export const fetchActivityHistory = async () => {
  return request('/api/activity/history');
};

export const fetchOverview = async () => {
  return request('/api/overview');
};

export const fetchActivityIntelligence = async (useDemo = false) => {
  return request(`/api/activity/intelligence?use_demo=${Boolean(useDemo)}`);
};

export const fetchSettings = async () => {
  return request('/api/settings');
};

export const diagnoseEvents = async (events) => {
  return request('/api/activity/diagnose', {
    method: 'POST',
    body: JSON.stringify(events),
  });
};

export default {
  API_BASE_URL,
  checkHealth,
  fetchEngineStatus,
  toggleEngineMonitoring,
  fetchLiveContext,
  fetchScreenContext,
  fetchActivityHistory,
  fetchOverview,
  fetchActivityIntelligence,
  fetchSettings,
  diagnoseEvents,
};
