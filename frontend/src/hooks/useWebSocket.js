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

  // Store callbacks in refs so WebSocket handler always sees latest versions
  // (avoids stale closure bug where ws.onmessage captures old callbacks)
  const cbRef = useRef({ onAudio, onTranscript, onStateChange, onGhostText, onBackchannel });
  useEffect(() => {
    cbRef.current = { onAudio, onTranscript, onStateChange, onGhostText, onBackchannel };
  }, [onAudio, onTranscript, onStateChange, onGhostText, onBackchannel]);

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
      const cb = cbRef.current;
      if (event.data instanceof ArrayBuffer) {
        cb.onAudio?.(event.data);
      } else {
        try {
          const msg = JSON.parse(event.data);
          switch (msg.type) {
            case "state":
              setPipelineState(msg.state);
              cb.onStateChange?.(msg.state);
              break;
            case "transcript":
            case "partial_transcript":
              cb.onTranscript?.(msg);
              break;
            case "audio_config":
              window.__tts_sample_rate = msg.sample_rate || 24000;
              break;
            case "audio_end":
              break;
            case "ghost_text":
              cb.onGhostText?.(msg.text);
              break;
            case "interrupt":
              cb.onGhostText?.("");
              if (window.__stopPlayback) window.__stopPlayback();
              break;
            case "backchannel":
              cb.onBackchannel?.();
              break;
            case "error":
              console.warn("[WS] Server error:", msg.message);
              cb.onStateChange?.("error:" + (msg.message || "Unknown error"));
              break;
            default:
              break;
          }
        } catch {
          // ignore malformed
        }
      }
    };

    wsRef.current = ws;
  }, []);  // no deps needed — callbacks accessed via cbRef

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
