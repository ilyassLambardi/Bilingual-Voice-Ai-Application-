import { useRef, useState, useCallback, useEffect } from "react";

/**
 * Manages the WebSocket connection to the FastAPI backend.
 *
 * Handles:
 *   - connection lifecycle
 *   - sending binary audio chunks
 *   - receiving JSON messages + binary PCM audio
 *   - exposing pipeline state (idle | listening | thinking | speaking)
 */
// Auto-reconnect state (module-level so it persists across re-renders)
let _reconnectAttempts = 0;
let _reconnectTimer = null;

export default function useWebSocket({ onAudio, onTranscript, onStateChange, onGhostText, onBackchannel }) {
  const wsRef = useRef(null);
  const [connected, setConnected] = useState(false);
  const [pipelineState, setPipelineState] = useState("idle");

  // ── Connect ──────────────────────────────────────────────
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const host = window.location.host;
    const url = `${protocol}://${host}/ws`;
    console.log("[WS] Connecting to:", url);

    const ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      console.log("[WS] Connected");
      setConnected(true);
      _reconnectAttempts = 0; // reset on successful connect
    };

    ws.onclose = (event) => {
      console.log("[WS] Disconnected, code:", event.code, "reason:", event.reason);
      setConnected(false);
      setPipelineState("idle");
      // Auto-reconnect after disconnect (exponential backoff)
      if (event.code !== 1000) {
        const delay = Math.min(2000 * Math.pow(1.5, _reconnectAttempts), 15000);
        _reconnectAttempts++;
        console.log(`[WS] Reconnecting in ${(delay/1000).toFixed(1)}s (attempt ${_reconnectAttempts})...`);
        _reconnectTimer = setTimeout(() => connect(), delay);
      }
    };

    ws.onerror = (e) => {
      console.error("[WS] Error:", e);
    };

    ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        // Binary = PCM audio from TTS
        onAudio?.(event.data);
      } else {
        // JSON control message
        try {
          const msg = JSON.parse(event.data);
          _handleMessage(msg);
        } catch {
          // ignore malformed
        }
      }
    };

    wsRef.current = ws;
  }, [onAudio, onTranscript, onStateChange, onGhostText]);

  function _handleMessage(msg) {
    switch (msg.type) {
      case "state":
        setPipelineState(msg.state);
        onStateChange?.(msg.state);
        break;
      case "transcript":
        onTranscript?.(msg);
        break;
      case "partial_transcript":
        onTranscript?.(msg);
        break;
      case "audio_config":
        // Store sample rate for playback
        window.__tts_sample_rate = msg.sample_rate || 24000;
        break;
      case "audio_end":
        // Could trigger end-of-playback UI
        break;
      case "ghost_text":
        onGhostText?.(msg.text);
        break;
      case "interrupt":
        // AI was interrupted by user speaking — stop audio playback
        onGhostText?.("");
        if (window.__stopPlayback) window.__stopPlayback();
        break;
      case "backchannel":
        // User made a short affirmation (mhm, yeah) — visual pulse, no interrupt
        onBackchannel?.();
        break;
      case "language_shift":
        // User switched languages (e.g. EN→DE after 3 EN messages)
        onStateChange?.("language_shift:" + msg.from + ":" + msg.to);
        break;
      default:
        break;
    }
  }

  // ── Send binary audio chunk ──────────────────────────────
  const sendAudio = useCallback((buffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(buffer);
    }
  }, []);

  // ── Send JSON command ────────────────────────────────────
  const sendCommand = useCallback((obj) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(obj));
    }
  }, []);

  // ── Disconnect ───────────────────────────────────────────
  const disconnect = useCallback(() => {
    clearTimeout(_reconnectTimer);
    _reconnectAttempts = 0;
    if (wsRef.current) {
      wsRef.current.close(1000); // clean close — no auto-reconnect
      wsRef.current = null;
    }
    setConnected(false);
  }, []);

  // Cleanup on unmount
  useEffect(() => () => disconnect(), [disconnect]);

  return {
    connected,
    pipelineState,
    connect,
    disconnect,
    sendAudio,
    sendCommand,
  };
}
