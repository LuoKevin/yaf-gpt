import { useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from "react";

import { DiscussionWorkspace } from "./components/DiscussionWorkspace";
import { MusicWorkspace } from "./components/MusicWorkspace";
import { StudyWorkspace } from "./components/StudyWorkspace";
import { TextChatWorkspace } from "./components/TextChatWorkspace";
import { ViewSwitcher } from "./components/ViewSwitcher";
import {
  requestJson,
  requestRealtimeAnswer,
  requestSse
} from "./lib/api";
import type {
  BiblePassageResponse,
  ChatMessage,
  MusicGenerateResponse,
  MusicJobResponse,
  PassageImageResponse,
  SseEventPayload,
  TranslationCode,
  ViewMode,
  VoiceRealtimeSessionResponse,
  StudyPlanResponse
} from "./types";

export default function App() {
  const [activeView, setActiveView] = useState<ViewMode>("chat");
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);

  const [reference, setReference] = useState("Luke 21:5-28");
  const [translation, setTranslation] = useState<TranslationCode>("WEB");
  const [goals, setGoals] = useState("");
  const [userNotes, setUserNotes] = useState("");
  const [includeQuestionNotes, setIncludeQuestionNotes] = useState(false);

  const [passage, setPassage] = useState<BiblePassageResponse | null>(null);
  const [studyPlan, setStudyPlan] = useState<StudyPlanResponse | null>(null);
  const [passageError, setPassageError] = useState("");
  const [studyPlanError, setStudyPlanError] = useState("");
  const [isLoadingPassage, setIsLoadingPassage] = useState(false);
  const [isLoadingStudyPlan, setIsLoadingStudyPlan] = useState(false);

  const [passageImage, setPassageImage] = useState<PassageImageResponse | null>(null);
  const [passageImageError, setPassageImageError] = useState("");
  const [isLoadingPassageImage, setIsLoadingPassageImage] = useState(false);

  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatModel, setChatModel] = useState<string | null>(null);
  const [chatError, setChatError] = useState("");
  const [isSendingChat, setIsSendingChat] = useState(false);

  const [discussionMessages, setDiscussionMessages] = useState<ChatMessage[]>([]);
  const [discussionError, setDiscussionError] = useState("");
  const [isConnectingDiscussion, setIsConnectingDiscussion] = useState(false);
  const [isDiscussionSessionActive, setIsDiscussionSessionActive] = useState(false);
  const [isPlayingDiscussionAudio, setIsPlayingDiscussionAudio] = useState(false);
  const [discussionVoiceStatus, setDiscussionVoiceStatus] = useState("");
  const [discussionVoiceVisualLevel, setDiscussionVoiceVisualLevel] = useState(0);

  const discussionDataChannelRef = useRef<RTCDataChannel | null>(null);
  const discussionPeerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const discussionMessageIndexByItemIdRef = useRef<Map<string, number>>(new Map());
  const discussionSessionActiveRef = useRef(false);
  const discussionAudioContextRef = useRef<AudioContext | null>(null);
  const discussionAudioAnalyserRef = useRef<AnalyserNode | null>(null);
  const discussionAudioSourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const discussionAudioAnimationFrameRef = useRef<number | null>(null);
  const discussionStreamRef = useRef<MediaStream | null>(null);
  const discussionAudioRef = useRef<HTMLAudioElement | null>(null);
  const discussionAudioUrlRef = useRef<string | null>(null);

  const [musicPrompt, setMusicPrompt] = useState("");
  const [musicStyle, setMusicStyle] = useState("modern worship, acoustic");
  const [musicMood, setMusicMood] = useState("hopeful");
  const [musicResult, setMusicResult] = useState<MusicGenerateResponse | null>(null);
  const [musicJob, setMusicJob] = useState<MusicJobResponse | null>(null);
  const [musicError, setMusicError] = useState("");
  const [isGeneratingMusic, setIsGeneratingMusic] = useState(false);

  const musicJobId = musicResult?.job_id ?? null;
  const musicJobStatus = musicJob?.status ?? musicResult?.status ?? null;

  useEffect(() => {
    discussionSessionActiveRef.current = isDiscussionSessionActive;
  }, [isDiscussionSessionActive]);

  useEffect(() => {
    if (!musicJobId) {
      return;
    }
    if (musicJobStatus === "completed" || musicJobStatus === "failed") {
      return;
    }

    let cancelled = false;
    const poll = async () => {
      try {
        const next = await requestJson<MusicJobResponse>(`/api/music/jobs/${musicJobId}`);
        if (!cancelled) {
          setMusicJob(next);
        }
      } catch (error) {
        if (!cancelled) {
          setMusicError(error instanceof Error ? error.message : "Unable to refresh music status.");
        }
      }
    };

    void poll();
    const timer = window.setInterval(() => {
      void poll();
    }, 2000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [musicJobId, musicJobStatus]);

  function releaseDiscussionMediaResources() {
    if (discussionStreamRef.current) {
      discussionStreamRef.current.getTracks().forEach((track) => track.stop());
    }
    discussionStreamRef.current = null;
  }

  function getDiscussionAudioElement() {
    if (!discussionAudioRef.current) {
      const audio = new Audio();
      audio.autoplay = true;
      audio.onplay = () => {
        setIsPlayingDiscussionAudio(true);
        setDiscussionVoiceStatus("Mentor is speaking...");
      };
      audio.onended = () => {
        setIsPlayingDiscussionAudio(false);
        setDiscussionVoiceStatus(discussionSessionActiveRef.current ? "Live session connected." : "Voice reply ready.");
        setDiscussionVoiceVisualLevel(discussionSessionActiveRef.current ? 0.6 : 0);
      };
      audio.onpause = () => {
        setIsPlayingDiscussionAudio(false);
        setDiscussionVoiceVisualLevel(discussionSessionActiveRef.current ? 0.6 : 0);
      };
      discussionAudioRef.current = audio;
    }
    return discussionAudioRef.current;
  }

  function stopDiscussionAudioPlayback() {
    const audio = discussionAudioRef.current;
    if (audio) {
      audio.pause();
      audio.currentTime = 0;
    }
    if (discussionAudioUrlRef.current) {
      URL.revokeObjectURL(discussionAudioUrlRef.current);
      discussionAudioUrlRef.current = null;
    }
  }

  function stopDiscussionVoiceVisualizer() {
    if (discussionAudioAnimationFrameRef.current !== null) {
      window.cancelAnimationFrame(discussionAudioAnimationFrameRef.current);
      discussionAudioAnimationFrameRef.current = null;
    }

    if (discussionAudioSourceNodeRef.current) {
      discussionAudioSourceNodeRef.current.disconnect();
      discussionAudioSourceNodeRef.current = null;
    }

    if (discussionAudioAnalyserRef.current) {
      discussionAudioAnalyserRef.current.disconnect();
      discussionAudioAnalyserRef.current = null;
    }

    if (discussionAudioContextRef.current) {
      void discussionAudioContextRef.current.close();
      discussionAudioContextRef.current = null;
    }
  }

  async function startDiscussionVoiceVisualizer(stream: MediaStream) {
    stopDiscussionVoiceVisualizer();

    const AudioContextCtor = window.AudioContext || (window as typeof window & {
      webkitAudioContext?: typeof AudioContext;
    }).webkitAudioContext;

    if (!AudioContextCtor) {
      return;
    }

    const context = new AudioContextCtor();
    discussionAudioContextRef.current = context;

    if (context.state === "suspended") {
      await context.resume();
    }

    const analyser = context.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.9;
    discussionAudioAnalyserRef.current = analyser;

    const sourceNode = context.createMediaStreamSource(stream);
    sourceNode.connect(analyser);
    discussionAudioSourceNodeRef.current = sourceNode;

    const sampleBuffer = new Uint8Array(analyser.fftSize);
    let smoothedLevel = 0;
    let warmupFrames = 10;

    const tick = () => {
      const activeSession = discussionSessionActiveRef.current;
      const activeAudio = !discussionAudioRef.current?.paused;

      analyser.getByteTimeDomainData(sampleBuffer);
      const rms = Math.sqrt(
        sampleBuffer.reduce((total, value) => {
          const centered = (value - 128) / 128;
          return total + centered * centered;
        }, 0) / Math.max(sampleBuffer.length, 1),
      );
      const normalizedLevel = Math.min(1, rms * 5.5);
      const attack = warmupFrames > 0 ? 0.08 : 0.18;
      const release = 0.9;
      smoothedLevel =
        normalizedLevel > smoothedLevel
          ? smoothedLevel + (normalizedLevel - smoothedLevel) * attack
          : smoothedLevel * release + normalizedLevel * (1 - release);

      if (warmupFrames > 0) {
        warmupFrames -= 1;
      }

      if (activeAudio || smoothedLevel > 0.05) {
        setDiscussionVoiceVisualLevel(Math.max(0.06, smoothedLevel));
      } else if (activeSession) {
        setDiscussionVoiceVisualLevel(0.6);
      } else {
        setDiscussionVoiceVisualLevel(0);
      }

      discussionAudioAnimationFrameRef.current = window.requestAnimationFrame(tick);
    };

    discussionAudioAnimationFrameRef.current = window.requestAnimationFrame(tick);
  }

  function closeDiscussionRealtimeSession() {
    const dataChannel = discussionDataChannelRef.current;
    if (dataChannel) {
      dataChannel.onopen = null;
      dataChannel.onmessage = null;
      dataChannel.onerror = null;
      dataChannel.onclose = null;
      dataChannel.close();
    }
    discussionDataChannelRef.current = null;

    const peerConnection = discussionPeerConnectionRef.current;
    if (peerConnection) {
      peerConnection.ontrack = null;
      peerConnection.onconnectionstatechange = null;
      peerConnection.close();
    }
    discussionPeerConnectionRef.current = null;

    const audio = discussionAudioRef.current;
    if (audio) {
      audio.pause();
      audio.srcObject = null;
      audio.removeAttribute("src");
      audio.load();
    }

    releaseDiscussionMediaResources();
    stopDiscussionVoiceVisualizer();
    setIsConnectingDiscussion(false);
    setIsDiscussionSessionActive(false);
    setIsPlayingDiscussionAudio(false);
    setDiscussionVoiceVisualLevel(0);
    discussionMessageIndexByItemIdRef.current.clear();
  }

  useEffect(() => {
    return () => {
      stopDiscussionAudioPlayback();
      closeDiscussionRealtimeSession();
    };
  }, []);

  async function handleStudyRequestSubmit() {
    const trimmedReference = reference.trim();
    if (!trimmedReference) {
      setPassageError("Enter a reference.");
      setStudyPlanError("Enter a reference.");
      return;
    }

    setIsLoadingPassage(true);
    setIsLoadingStudyPlan(true);
    setPassageError("");
    setStudyPlanError("");

    try {
      const [passageResponse, studyPlanResponse] = await Promise.all([
        requestJson<BiblePassageResponse>(
          "/api/bible/passage",
          undefined,
          {
            reference: trimmedReference,
            translation
          }
        ),
        requestJson<StudyPlanResponse>("/api/study-plan", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            reference: trimmedReference,
            translation,
            goals: goals.trim() || undefined,
            user_notes: userNotes.trim() || undefined,
            include_question_notes: includeQuestionNotes,
          })
        })
      ]);

      setPassage(passageResponse);
      setStudyPlan(studyPlanResponse);
    } catch (error) {
      setStudyPlan(null);
      setPassage(null);
      const detail = error instanceof Error ? error.message : "Unable to generate study.";
      setPassageError(detail);
      setStudyPlanError(detail);
    } finally {
      setIsLoadingPassage(false);
      setIsLoadingStudyPlan(false);
    }
  }

  async function handlePassageImageGeneration() {
    const trimmedReference = reference.trim();
    if (!trimmedReference) {
      setPassageImageError("Enter a reference.");
      return;
    }

    setIsLoadingPassageImage(true);
    setPassageImageError("");

    try {
      const response = await requestJson<PassageImageResponse>("/api/passage-image", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          reference: trimmedReference,
          translation,
          style: "modern_editorial_illustration"
        })
      });
      setPassageImage(response);
    } catch (error) {
      setPassageImage(null);
      setPassageImageError(error instanceof Error ? error.message : "Unable to generate image.");
    } finally {
      setIsLoadingPassageImage(false);
    }
  }

  async function streamWorkspaceMessage({
    rawMessage,
    currentMessages,
    setMessages,
    setError,
    setIsSending,
    setModel,
    onReplyFinished,
  }: {
    rawMessage: string;
    currentMessages: ChatMessage[];
    setMessages: Dispatch<SetStateAction<ChatMessage[]>>;
    setError: Dispatch<SetStateAction<string>>;
    setIsSending: Dispatch<SetStateAction<boolean>>;
    setModel: Dispatch<SetStateAction<string | null>>;
    onReplyFinished?: (replyText: string) => void;
  }) {
    const userMessage = rawMessage.trim();
    if (!userMessage) {
      setError("Type a message.");
      return;
    }

    const nextMessages = [...currentMessages, { role: "user", content: userMessage } as ChatMessage];
    setMessages([...nextMessages, { role: "assistant", content: "" }]);
    setError("");
    setIsSending(true);

    try {
      let streamedModel: string | null = null;
      let receivedChunk = false;
      let assistantReply = "";
      await requestSse(
        "/api/chat/stream",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            messages: nextMessages
          })
        },
        ({ event, data }: SseEventPayload) => {
          if (event === "meta") {
            if (
              typeof data === "object" &&
              data !== null &&
              "model" in data &&
              typeof data.model === "string"
            ) {
              streamedModel = data.model;
            }
            return;
          }

          if (event === "chunk") {
            const delta =
              typeof data === "object" &&
              data !== null &&
              "delta" in data &&
              typeof data.delta === "string"
                ? data.delta
                : null;
            if (delta && delta.length > 0) {
              assistantReply += delta;
              receivedChunk = true;
              setMessages((current) => {
                if (current.length === 0) {
                  return [{ role: "assistant", content: delta }];
                }
                const next = [...current];
                const last = next[next.length - 1];
                if (last.role !== "assistant") {
                  next.push({ role: "assistant", content: delta });
                  return next;
                }
                next[next.length - 1] = { role: "assistant", content: `${last.content}${delta}` };
                return next;
              });
            }
            return;
          }

          if (event === "error") {
            const detail =
              typeof data === "object" &&
              data !== null &&
              "detail" in data &&
              typeof data.detail === "string"
                ? data.detail
                : "Unable to stream response.";
            throw new Error(detail);
          }
        }
      );

      if (streamedModel) {
        setModel(streamedModel);
      }
      if (!receivedChunk) {
        setMessages((current) => {
          if (current.length === 0) {
            return current;
          }
          const last = current[current.length - 1];
          if (last.role === "assistant" && !last.content.trim()) {
            return current.slice(0, -1);
          }
          return current;
        });
      } else if (onReplyFinished) {
        onReplyFinished(assistantReply);
      }
    } catch (error) {
      setMessages((current) => {
        if (current.length === 0) {
          return current;
        }
        const last = current[current.length - 1];
        if (last.role === "assistant" && !last.content.trim()) {
          return current.slice(0, -1);
        }
        return current;
      });
      setError(error instanceof Error ? error.message : "Unable to send message.");
    } finally {
      setIsSending(false);
    }
  }

  async function handleTextChatSend() {
    const userMessage = chatInput.trim();
    if (!userMessage) {
      setChatError("Type a message.");
      return;
    }
    setChatInput("");
    await streamWorkspaceMessage({
      rawMessage: userMessage,
      currentMessages: chatMessages,
      setMessages: setChatMessages,
      setError: setChatError,
      setIsSending: setIsSendingChat,
      setModel: setChatModel,
    });
  }

  function upsertDiscussionMessage(itemId: string, role: ChatMessage["role"], content: string, append = false) {
    if (!itemId) {
      return;
    }

    setDiscussionMessages((current) => {
      const next = [...current];
      const existingIndex = discussionMessageIndexByItemIdRef.current.get(itemId);

      if (existingIndex === undefined) {
        discussionMessageIndexByItemIdRef.current.set(itemId, next.length);
        next.push({ role, content });
        return next;
      }

      const existingMessage = next[existingIndex];
      if (!existingMessage) {
        discussionMessageIndexByItemIdRef.current.set(itemId, next.length);
        next.push({ role, content });
        return next;
      }

      next[existingIndex] = {
        role,
        content: append ? `${existingMessage.content}${content}` : content,
      };
      return next;
    });
  }

  function sendDiscussionRealtimeEvent(event: Record<string, unknown>) {
    const dataChannel = discussionDataChannelRef.current;
    if (!dataChannel || dataChannel.readyState !== "open") {
      return false;
    }

    dataChannel.send(JSON.stringify(event));
    return true;
  }

  function requestInitialDiscussionGreeting() {
    return sendDiscussionRealtimeEvent({
      type: "response.create",
      response: {
        instructions:
          "Briefly introduce yourself out loud as YAF-GPT, welcome the user to this live voice discussion, and invite them to start speaking. Keep it warm and under two sentences.",
        output_modalities: ["audio"],
        metadata: {
          response_purpose: "initial_greeting",
        },
      },
    });
  }

  function handleDiscussionRealtimeEvent(event: unknown) {
    if (!event || typeof event !== "object" || !("type" in event) || typeof event.type !== "string") {
      return;
    }

    if (event.type === "conversation.item.input_audio_transcription.completed") {
      const itemId = "item_id" in event && typeof event.item_id === "string" ? event.item_id : "";
      const transcript = "transcript" in event && typeof event.transcript === "string" ? event.transcript : "";
      if (itemId && transcript) {
        upsertDiscussionMessage(itemId, "user", transcript);
        setDiscussionVoiceStatus("Heard you. Waiting for mentor reply...");
        setDiscussionVoiceVisualLevel(0.7);
      }
      return;
    }

    if (event.type === "response.output_audio_transcript.delta") {
      const itemId = "item_id" in event && typeof event.item_id === "string" ? event.item_id : "";
      const delta = "delta" in event && typeof event.delta === "string" ? event.delta : "";
      if (itemId && delta) {
        upsertDiscussionMessage(itemId, "assistant", delta, true);
        setDiscussionVoiceStatus("Mentor is responding...");
      }
      return;
    }

    if (event.type === "response.output_audio_transcript.done") {
      const itemId = "item_id" in event && typeof event.item_id === "string" ? event.item_id : "";
      const transcript = "transcript" in event && typeof event.transcript === "string" ? event.transcript : "";
      if (itemId && transcript) {
        upsertDiscussionMessage(itemId, "assistant", transcript);
        setDiscussionVoiceStatus("Live session connected.");
      }
      return;
    }

    if (event.type === "input_audio_buffer.speech_started") {
      setDiscussionVoiceStatus("Listening...");
      setDiscussionVoiceVisualLevel(0.95);
      return;
    }

    if (event.type === "input_audio_buffer.speech_stopped") {
      setDiscussionVoiceStatus("Processing your speech...");
      setDiscussionVoiceVisualLevel(0.72);
      return;
    }

    if (event.type === "response.done") {
      setDiscussionVoiceStatus("Live session connected.");
      setDiscussionVoiceVisualLevel(0.6);
      return;
    }

    if (event.type === "error") {
      const detail =
        "error" in event &&
        event.error &&
        typeof event.error === "object" &&
        "message" in event.error &&
        typeof event.error.message === "string"
          ? event.error.message
          : "Realtime voice session failed.";
      setDiscussionError(detail);
      setDiscussionVoiceStatus("");
      setDiscussionVoiceVisualLevel(0);
    }
  }

  async function startDiscussionRealtimeSession() {
    if (isConnectingDiscussion || isDiscussionSessionActive) {
      return;
    }
    if (typeof window === "undefined" || typeof RTCPeerConnection === "undefined") {
      setDiscussionError("Realtime voice is not supported in this browser.");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setDiscussionError("Microphone access is not available in this browser.");
      return;
    }

    setIsConnectingDiscussion(true);
    setDiscussionError("");
    setDiscussionVoiceStatus("Opening live session...");
    setDiscussionVoiceVisualLevel(0.55);

    try {
      stopDiscussionAudioPlayback();

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      discussionStreamRef.current = stream;

      const session = await requestJson<VoiceRealtimeSessionResponse>("/api/voice/realtime/session", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          reference_context: reference.trim() || undefined,
          translation,
          voice: "cedar"
        })
      });

      const peerConnection = new RTCPeerConnection();
      discussionPeerConnectionRef.current = peerConnection;

      stream.getTracks().forEach((track) => {
        peerConnection.addTrack(track, stream);
      });

      peerConnection.ontrack = (event) => {
        const audio = getDiscussionAudioElement();
        const [remoteStream] = event.streams;
        if (!remoteStream) {
          return;
        }
        audio.srcObject = remoteStream;
        void startDiscussionVoiceVisualizer(remoteStream).catch(() => {
          // Keep live audio working even if the browser blocks analysis setup.
        });
        void audio.play().catch(() => {
          setDiscussionVoiceStatus("Mentor audio is ready. Press play if your browser blocked autoplay.");
        });
      };

      peerConnection.onconnectionstatechange = () => {
        const state = peerConnection.connectionState;
        if (state === "connected") {
          setDiscussionVoiceStatus("Live session connected.");
          setDiscussionVoiceVisualLevel(0.6);
        } else if (state === "failed" || state === "disconnected" || state === "closed") {
          if (discussionSessionActiveRef.current || discussionDataChannelRef.current || discussionPeerConnectionRef.current) {
            closeDiscussionRealtimeSession();
            setDiscussionVoiceStatus("");
          }
        }
      };

      const dataChannel = peerConnection.createDataChannel("oai-events");
      discussionDataChannelRef.current = dataChannel;

      dataChannel.onopen = () => {
        setIsConnectingDiscussion(false);
        setIsDiscussionSessionActive(true);
        setDiscussionVoiceStatus("Live session connected. Starting introduction...");
        setDiscussionVoiceVisualLevel(0.6);
        requestInitialDiscussionGreeting();
      };

      dataChannel.onmessage = (messageEvent) => {
        try {
          handleDiscussionRealtimeEvent(JSON.parse(String(messageEvent.data)));
        } catch {
          // Ignore malformed data-channel events.
        }
      };

      dataChannel.onerror = () => {
        setDiscussionError("Realtime event channel failed.");
      };

      dataChannel.onclose = () => {
        if (discussionSessionActiveRef.current) {
          closeDiscussionRealtimeSession();
        }
      };

      const offer = await peerConnection.createOffer();
      await peerConnection.setLocalDescription(offer);

      const answerSdp = await requestRealtimeAnswer(
        session.webrtc_url,
        session.client_secret,
        offer.sdp ?? "",
      );

      await peerConnection.setRemoteDescription({
        type: "answer",
        sdp: answerSdp,
      });
    } catch (error) {
      closeDiscussionRealtimeSession();
      setDiscussionError(error instanceof Error ? error.message : "Unable to start live voice session.");
      setDiscussionVoiceStatus("");
      setDiscussionVoiceVisualLevel(0);
    } finally {
      setIsConnectingDiscussion(false);
    }
  }

  function endDiscussionRealtimeSession() {
    closeDiscussionRealtimeSession();
    setDiscussionVoiceStatus("");
  }

  async function handleDiscussionSessionToggle() {
    if (isDiscussionSessionActive || isConnectingDiscussion) {
      endDiscussionRealtimeSession();
      return;
    }

    await startDiscussionRealtimeSession();
  }

  function handleTextChatReset() {
    setChatMessages([]);
    setChatModel(null);
    setChatError("");
    setChatInput("");
    setIsSendingChat(false);
  }

  async function handleMusicGeneration() {
    setMusicError("");
    window.alert(
      "Sorry, music generation is currently limited to YAF-GPT Pro users. Looking to turn your thoughts into music? Try suno.com!"
    );
    if (musicResult || musicJob) {
      setMusicResult(null);
      setMusicJob(null);
    }
  }

  const musicStatusLabel = useMemo(() => {
    if (!musicJobStatus) {
      return "queued";
    }
    return musicJobStatus;
  }, [musicJobStatus]);
  const activeViewLabel = useMemo(() => {
    if (activeView === "chat") {
      return "Text Chat";
    }
    if (activeView === "study") {
      return "Study Guide";
    }
    if (activeView === "discussion") {
      return "Mentor Chat";
    }
    return "Music Draft";
  }, [activeView]);

  return (
    <div className={`shell ${isSidebarCollapsed ? "shell-collapsed" : ""}`}>
      <ViewSwitcher
        activeView={activeView}
        isCollapsed={isSidebarCollapsed}
        onChange={setActiveView}
        onToggleCollapse={() => setIsSidebarCollapsed((current) => !current)}
        onNewChat={handleTextChatReset}
      />

      <div className="shell-content">
        <header className="app-topbar">
          <div className="app-topbar-copy">
            <span className="app-brand">YAF-GPT</span>
            <p className="app-topbar-title">{activeViewLabel}</p>
          </div>

        </header>

        <main
          className={`workspace ${activeView === "discussion" || activeView === "chat" ? "workspace-single" : ""}`}
        >
          {activeView === "chat" ? (
            <TextChatWorkspace
              chatError={chatError}
              chatMessages={chatMessages}
              chatInput={chatInput}
              isSendingChat={isSendingChat}
              onChatInputChange={setChatInput}
              onChatSend={handleTextChatSend}
            />
          ) : null}

          {activeView === "study" ? (
            <StudyWorkspace
              reference={reference}
              goals={goals}
              passage={passage}
              studyPlan={studyPlan}
              passageError={passageError}
              studyPlanError={studyPlanError}
              passageImage={passageImage}
              passageImageError={passageImageError}
              isLoadingPassage={isLoadingPassage}
              isLoadingStudyPlan={isLoadingStudyPlan}
              isLoadingPassageImage={isLoadingPassageImage}
              onReferenceChange={setReference}
              onGoalsChange={setGoals}
              onSubmitStudyRequest={handleStudyRequestSubmit}
              onGenerateImage={handlePassageImageGeneration}
            />
          ) : null}

          {activeView === "discussion" ? (
            <DiscussionWorkspace
              discussionError={discussionError}
              discussionMessages={discussionMessages}
              isConnectingSession={isConnectingDiscussion}
              isSessionActive={isDiscussionSessionActive}
              isPlayingSessionAudio={isPlayingDiscussionAudio}
              voiceVisualLevel={discussionVoiceVisualLevel}
              voiceStatus={discussionVoiceStatus}
              onSessionToggle={handleDiscussionSessionToggle}
            />
          ) : null}

          {activeView === "music" ? (
            <MusicWorkspace
              musicPrompt={musicPrompt}
              musicStyle={musicStyle}
              musicMood={musicMood}
              musicResult={musicResult}
              musicJob={musicJob}
              musicError={musicError}
              musicStatusLabel={musicStatusLabel}
              isGeneratingMusic={isGeneratingMusic}
              onMusicPromptChange={setMusicPrompt}
              onMusicStyleChange={setMusicStyle}
              onMusicMoodChange={setMusicMood}
              onGenerateMusic={handleMusicGeneration}
            />
          ) : null}
        </main>
      </div>
    </div>
  );
}
