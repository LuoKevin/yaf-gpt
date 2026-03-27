import { useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from "react";

import { DiscussionWorkspace } from "./components/DiscussionWorkspace";
import { MusicWorkspace } from "./components/MusicWorkspace";
import { StudyWorkspace } from "./components/StudyWorkspace";
import { TextChatWorkspace } from "./components/TextChatWorkspace";
import { ViewSwitcher } from "./components/ViewSwitcher";
import {
  blobToDataUrl,
  normalizeReference,
  requestJson,
  requestRealtimeAnswer,
  requestSse
} from "./lib/api";
import type {
  BiblePassageResponse,
  MusicGenerateResponse,
  MusicJobResponse,
  PassageImageResponse,
  PersonaChatMessage,
  RealtimeVoice,
  SseEventPayload,
  TranslationCode,
  ViewMode,
  VoiceRealtimeSessionResponse,
  VoiceTranscriptionResponse,
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
  const [chatMessages, setChatMessages] = useState<PersonaChatMessage[]>([]);
  const [chatModel, setChatModel] = useState<string | null>(null);
  const [chatError, setChatError] = useState("");
  const [isSendingChat, setIsSendingChat] = useState(false);

  const [discussionMessages, setDiscussionMessages] = useState<PersonaChatMessage[]>([]);
  const [discussionModel, setDiscussionModel] = useState<string | null>(null);
  const [discussionError, setDiscussionError] = useState("");
  const [isSendingDiscussion, setIsSendingDiscussion] = useState(false);
  const [isRecordingDiscussion, setIsRecordingDiscussion] = useState(false);
  const [isTranscribingDiscussion, setIsTranscribingDiscussion] = useState(false);
  const [enableDiscussionVoiceReply, setEnableDiscussionVoiceReply] = useState(true);
  const [isRealtimeVoiceConnecting, setIsRealtimeVoiceConnecting] = useState(false);
  const [isRealtimeVoiceActive, setIsRealtimeVoiceActive] = useState(false);
  const [realtimeVoiceStatus, setRealtimeVoiceStatus] = useState("");

  const discussionRecorderRef = useRef<MediaRecorder | null>(null);
  const discussionStreamRef = useRef<MediaStream | null>(null);
  const discussionChunksRef = useRef<Blob[]>([]);
  const realtimePeerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const realtimeDataChannelRef = useRef<RTCDataChannel | null>(null);
  const realtimeLocalStreamRef = useRef<MediaStream | null>(null);
  const realtimeAudioRef = useRef<HTMLAudioElement | null>(null);
  const realtimeAssistantItemIdRef = useRef<string | null>(null);
  const realtimeUserTranscriptItemIdsRef = useRef<Set<string>>(new Set());
  const realtimeFinalizedAssistantItemIdsRef = useRef<Set<string>>(new Set());

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
    discussionRecorderRef.current = null;
    discussionChunksRef.current = [];
  }

  function getRealtimeAudioElement() {
    if (!realtimeAudioRef.current) {
      const audio = new Audio();
      audio.autoplay = true;
      realtimeAudioRef.current = audio;
    }
    return realtimeAudioRef.current;
  }

  function stopRealtimeVoiceSession() {
    realtimeDataChannelRef.current?.close();
    realtimeDataChannelRef.current = null;

    realtimePeerConnectionRef.current?.close();
    realtimePeerConnectionRef.current = null;

    if (realtimeLocalStreamRef.current) {
      realtimeLocalStreamRef.current.getTracks().forEach((track) => track.stop());
    }
    realtimeLocalStreamRef.current = null;

    const audio = realtimeAudioRef.current;
    if (audio) {
      audio.pause();
      audio.srcObject = null;
    }

    realtimeAssistantItemIdRef.current = null;
    realtimeUserTranscriptItemIdsRef.current.clear();
    realtimeFinalizedAssistantItemIdsRef.current.clear();
    setIsRealtimeVoiceConnecting(false);
    setIsRealtimeVoiceActive(false);
    setRealtimeVoiceStatus("");
  }

  function appendRealtimeUserTranscript(itemId: string, transcript: string) {
    const cleaned = transcript.trim();
    if (!cleaned || realtimeUserTranscriptItemIdsRef.current.has(itemId)) {
      return;
    }

    realtimeUserTranscriptItemIdsRef.current.add(itemId);
    setDiscussionMessages((current) => [...current, { role: "user", content: cleaned }]);
  }

  function appendRealtimeAssistantDelta(itemId: string, delta: string) {
    if (!delta || realtimeFinalizedAssistantItemIdsRef.current.has(itemId)) {
      return;
    }

    setDiscussionMessages((current) => {
      if (realtimeAssistantItemIdRef.current !== itemId) {
        realtimeAssistantItemIdRef.current = itemId;
        return [...current, { role: "assistant", content: delta }];
      }

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

  function finalizeRealtimeAssistantTranscript(itemId: string, transcript: string) {
    const cleaned = transcript.trim();
    if (!cleaned) {
      realtimeAssistantItemIdRef.current = null;
      return;
    }
    if (realtimeFinalizedAssistantItemIdsRef.current.has(itemId)) {
      return;
    }

    setDiscussionMessages((current) => {
      if (current.length === 0) {
        return [{ role: "assistant", content: cleaned }];
      }

      const next = [...current];
      const last = next[next.length - 1];
      if (realtimeAssistantItemIdRef.current === itemId && last.role === "assistant") {
        next[next.length - 1] = { role: "assistant", content: cleaned };
        return next;
      }
      if (last.role === "assistant" && last.content.trim() === cleaned) {
        return current;
      }

      next.push({ role: "assistant", content: cleaned });
      return next;
    });

    realtimeFinalizedAssistantItemIdsRef.current.add(itemId);
    realtimeAssistantItemIdRef.current = null;
  }

  function handleRealtimeServerEvent(event: unknown, model: string, voice: RealtimeVoice) {
    if (typeof event !== "object" || event === null || !("type" in event)) {
      return;
    }

    const eventType = typeof event.type === "string" ? event.type : "";
    if (!eventType) {
      return;
    }

    if (eventType === "session.created" || eventType === "session.updated") {
      setDiscussionModel(model);
      setRealtimeVoiceStatus(`Live voice connected on ${model} (${voice}).`);
      return;
    }

    if (eventType === "conversation.item.input_audio_transcription.completed") {
      const itemId = "item_id" in event && typeof event.item_id === "string" ? event.item_id : "";
      const transcript = "transcript" in event && typeof event.transcript === "string" ? event.transcript : "";
      if (itemId && transcript) {
        appendRealtimeUserTranscript(itemId, transcript);
      }
      return;
    }

    if (eventType === "response.output_audio_transcript.delta") {
      const itemId = "item_id" in event && typeof event.item_id === "string" ? event.item_id : "";
      const delta = "delta" in event && typeof event.delta === "string" ? event.delta : "";
      if (itemId && delta) {
        appendRealtimeAssistantDelta(itemId, delta);
      }
      return;
    }

    if (eventType === "response.output_audio_transcript.done") {
      const itemId = "item_id" in event && typeof event.item_id === "string" ? event.item_id : "";
      const transcript = "transcript" in event && typeof event.transcript === "string" ? event.transcript : "";
      if (itemId && transcript) {
        finalizeRealtimeAssistantTranscript(itemId, transcript);
      }
      return;
    }

    if (eventType === "error") {
      const detail =
        "error" in event &&
        typeof event.error === "object" &&
        event.error !== null &&
        "message" in event.error &&
        typeof event.error.message === "string"
          ? event.error.message
          : "Realtime voice session failed.";
      setDiscussionError(detail);
      stopRealtimeVoiceSession();
    }
  }

  useEffect(() => {
    return () => {
      if (discussionRecorderRef.current && discussionRecorderRef.current.state !== "inactive") {
        discussionRecorderRef.current.stop();
      }
      stopRealtimeVoiceSession();
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
      releaseDiscussionMediaResources();
    };
  }, []);

  async function handlePassageLookup() {
    const trimmedReference = reference.trim();
    if (!trimmedReference) {
      setPassageError("Enter a reference.");
      return;
    }

    setIsLoadingPassage(true);
    setPassageError("");

    try {
      const response = await requestJson<BiblePassageResponse>("/api/bible/passage", undefined, {
        reference: trimmedReference,
        translation
      });
      setPassage(response);
    } catch (error) {
      setPassage(null);
      setPassageError(error instanceof Error ? error.message : "Unable to load passage.");
    } finally {
      setIsLoadingPassage(false);
    }
  }

  async function handleStudyPlanGeneration() {
    const trimmedReference = reference.trim();
    if (!trimmedReference) {
      setStudyPlanError("Enter a reference.");
      return;
    }

    setIsLoadingStudyPlan(true);
    setStudyPlanError("");

    try {
      const canReusePassage =
        passage !== null &&
        normalizeReference(passage.reference) === normalizeReference(trimmedReference) &&
        passage.translation === translation;

      const response = await requestJson<StudyPlanResponse>("/api/study-plan", {
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
          passage_text: canReusePassage ? passage.text : undefined
        })
      });

      setStudyPlan(response);
      if (!canReusePassage) {
        setPassage({
          reference: response.reference,
          normalized_reference: response.normalized_reference,
          translation: response.translation,
          text: response.passage_text,
          verses: []
        });
      }
    } catch (error) {
      setStudyPlan(null);
      setStudyPlanError(error instanceof Error ? error.message : "Unable to generate study plan.");
    } finally {
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

  function speakDiscussionReply(replyText: string) {
    const cleaned = replyText.trim();
    if (!cleaned || !enableDiscussionVoiceReply) {
      return;
    }
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(cleaned);
    utterance.rate = 1;
    window.speechSynthesis.speak(utterance);
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
    currentMessages: PersonaChatMessage[];
    setMessages: Dispatch<SetStateAction<PersonaChatMessage[]>>;
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

    const nextMessages = [...currentMessages, { role: "user", content: userMessage } as PersonaChatMessage];
    setMessages([...nextMessages, { role: "assistant", content: "" }]);
    setError("");
    setIsSending(true);

    try {
      let streamedModel: string | null = null;
      let receivedChunk = false;
      let assistantReply = "";
      await requestSse(
        "/api/persona-chat/stream",
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

  async function sendDiscussionMessage(rawMessage: string) {
    await streamWorkspaceMessage({
      rawMessage,
      currentMessages: discussionMessages,
      setMessages: setDiscussionMessages,
      setError: setDiscussionError,
      setIsSending: setIsSendingDiscussion,
      setModel: setDiscussionModel,
      onReplyFinished: speakDiscussionReply,
    });
  }

  async function transcribeDiscussionRecording(blob: Blob) {
    setIsTranscribingDiscussion(true);
    setDiscussionError("");

    try {
      const audioBase64 = await blobToDataUrl(blob);
      const response = await requestJson<VoiceTranscriptionResponse>("/api/voice/transcribe", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          audio_base64: audioBase64,
          mime_type: blob.type || "audio/webm",
          file_name: "voice_input.webm"
        })
      });

      await sendDiscussionMessage(response.transcript);
    } catch (error) {
      setDiscussionError(error instanceof Error ? error.message : "Unable to process recorded audio.");
    } finally {
      setIsTranscribingDiscussion(false);
    }
  }

  async function startDiscussionRecording() {
    if (isRecordingDiscussion || isSendingDiscussion || isTranscribingDiscussion) {
      return;
    }
    if (typeof window === "undefined" || typeof MediaRecorder === "undefined") {
      setDiscussionError("Voice input is not supported in this browser.");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setDiscussionError("Microphone access is not available in this browser.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const preferredMimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : "audio/webm";
      const recorder = MediaRecorder.isTypeSupported(preferredMimeType)
        ? new MediaRecorder(stream, { mimeType: preferredMimeType })
        : new MediaRecorder(stream);

      discussionStreamRef.current = stream;
      discussionRecorderRef.current = recorder;
      discussionChunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          discussionChunksRef.current.push(event.data);
        }
      };

      recorder.onerror = () => {
        setDiscussionError("Recording failed.");
        setIsRecordingDiscussion(false);
        releaseDiscussionMediaResources();
      };

      recorder.onstop = () => {
        const chunks = [...discussionChunksRef.current];
        const blobType = recorder.mimeType || "audio/webm";
        setIsRecordingDiscussion(false);
        releaseDiscussionMediaResources();

        if (chunks.length === 0) {
          setDiscussionError("No audio captured.");
          return;
        }

        const recording = new Blob(chunks, { type: blobType });
        void transcribeDiscussionRecording(recording);
      };

      setDiscussionError("");
      recorder.start();
      setIsRecordingDiscussion(true);
    } catch (error) {
      setDiscussionError(error instanceof Error ? error.message : "Unable to access microphone.");
      setIsRecordingDiscussion(false);
      releaseDiscussionMediaResources();
    }
  }

  function stopDiscussionRecording() {
    if (!discussionRecorderRef.current || discussionRecorderRef.current.state === "inactive") {
      return;
    }
    discussionRecorderRef.current.stop();
  }

  async function startRealtimeVoiceSession() {
    if (isRealtimeVoiceActive || isRealtimeVoiceConnecting || isSendingDiscussion || isRecordingDiscussion || isTranscribingDiscussion) {
      return;
    }
    if (typeof window === "undefined" || typeof RTCPeerConnection === "undefined") {
      setDiscussionError("Live voice requires WebRTC support in this browser.");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setDiscussionError("Microphone access is not available in this browser.");
      return;
    }

    setDiscussionError("");
    setIsRealtimeVoiceConnecting(true);
    setRealtimeVoiceStatus("Connecting live voice...");

    try {
      const session = await requestJson<VoiceRealtimeSessionResponse>("/api/voice/realtime/session", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          voice: "cedar"
        })
      });

      const localStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      const peerConnection = new RTCPeerConnection();
      const audio = getRealtimeAudioElement();

      realtimeLocalStreamRef.current = localStream;
      realtimePeerConnectionRef.current = peerConnection;

      peerConnection.ontrack = (event) => {
        audio.srcObject = event.streams[0] ?? null;
        void audio.play().catch(() => {
          setRealtimeVoiceStatus("Live voice connected. Tap the page if audio playback is blocked.");
        });
      };

      peerConnection.onconnectionstatechange = () => {
        const state = peerConnection.connectionState;
        if (state === "connected") {
          setIsRealtimeVoiceConnecting(false);
          setIsRealtimeVoiceActive(true);
          setDiscussionModel(session.model);
          setRealtimeVoiceStatus(`Live voice connected on ${session.model} (${session.voice}).`);
          return;
        }

        if (state === "failed" || state === "closed" || state === "disconnected") {
          stopRealtimeVoiceSession();
          if (state === "failed") {
            setDiscussionError("Live voice connection dropped.");
          }
        }
      };

      const dataChannel = peerConnection.createDataChannel("oai-events");
      realtimeDataChannelRef.current = dataChannel;
      dataChannel.onopen = () => {
        setRealtimeVoiceStatus("Live voice ready. Start speaking.");
      };
      dataChannel.onclose = () => {
        if (realtimePeerConnectionRef.current) {
          stopRealtimeVoiceSession();
        }
      };
      dataChannel.onerror = () => {
        setDiscussionError("Live voice event channel failed.");
      };
      dataChannel.onmessage = (messageEvent) => {
        try {
          const event = JSON.parse(messageEvent.data) as unknown;
          handleRealtimeServerEvent(event, session.model, session.voice);
        } catch {
          // Ignore events we do not parse.
        }
      };

      localStream.getTracks().forEach((track) => {
        peerConnection.addTrack(track, localStream);
      });

      const offer = await peerConnection.createOffer();
      await peerConnection.setLocalDescription(offer);

      if (!offer.sdp) {
        throw new Error("Unable to create a WebRTC offer.");
      }

      const answerSdp = await requestRealtimeAnswer(
        session.webrtc_url,
        session.client_secret,
        offer.sdp
      );

      await peerConnection.setRemoteDescription({
        type: "answer",
        sdp: answerSdp
      });
    } catch (error) {
      stopRealtimeVoiceSession();
      setDiscussionError(error instanceof Error ? error.message : "Unable to start live voice.");
    } finally {
      setIsRealtimeVoiceConnecting(false);
    }
  }

  async function handleRealtimeVoiceToggle() {
    if (isRealtimeVoiceActive || isRealtimeVoiceConnecting) {
      stopRealtimeVoiceSession();
      return;
    }

    await startRealtimeVoiceSession();
  }

  function handleDiscussionReset() {
    if (discussionRecorderRef.current && discussionRecorderRef.current.state !== "inactive") {
      discussionRecorderRef.current.stop();
    }
    stopRealtimeVoiceSession();
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    releaseDiscussionMediaResources();
    setDiscussionMessages([]);
    setDiscussionModel(null);
    setDiscussionError("");
    setIsSendingDiscussion(false);
    setIsRecordingDiscussion(false);
    setIsTranscribingDiscussion(false);
  }

  function handleTextChatReset() {
    setChatMessages([]);
    setChatModel(null);
    setChatError("");
    setChatInput("");
    setIsSendingChat(false);
  }

  async function handleMusicGeneration() {
    const trimmedPrompt = musicPrompt.trim();
    if (!trimmedPrompt) {
      setMusicError("Enter a prompt.");
      return;
    }

    setIsGeneratingMusic(true);
    setMusicError("");

    try {
      const response = await requestJson<MusicGenerateResponse>("/api/music/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          prompt: trimmedPrompt,
          style: musicStyle.trim(),
          mood: musicMood.trim() || undefined
        })
      });

      setMusicResult(response);
      setMusicJob({
        job_id: response.job_id,
        status: response.status,
        provider: response.provider,
        audio_url: null,
        error: null
      });
    } catch (error) {
      setMusicResult(null);
      setMusicJob(null);
      setMusicError(error instanceof Error ? error.message : "Unable to generate music.");
    } finally {
      setIsGeneratingMusic(false);
    }
  }

  const hasUsage = studyPlan?.usage?.total_tokens != null;
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
  const activeViewCopy = useMemo(() => {
    if (activeView === "chat") {
      return "Use a simpler ChatGPT-style text thread for quick questions without carrying state from the other workspaces.";
    }
    if (activeView === "study") {
      return "Shape a sharper small-group guide with grounded context, discussion flow, and a companion passage image in one place.";
    }
    if (activeView === "discussion") {
      return "Keep the mentor conversation voice-first, with recording, live voice, and an isolated transcript for this workspace only.";
    }
    return "Turn a passage direction into a more coherent music brief, then follow the job through to generated audio.";
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
            <span className="app-divider" />
            <div>
              <p className="app-topbar-title">{activeViewLabel}</p>
              <p className="app-topbar-subtitle">{activeViewCopy}</p>
            </div>
          </div>

        </header>

        <main
          className={`workspace ${activeView === "discussion" || activeView === "chat" ? "workspace-single" : ""}`}
        >
          {activeView === "chat" ? (
            <TextChatWorkspace
              personaModel={chatModel}
              personaError={chatError}
              personaMessages={chatMessages}
              personaInput={chatInput}
              isSendingPersona={isSendingChat}
              onPersonaInputChange={setChatInput}
              onPersonaSend={handleTextChatSend}
            />
          ) : null}

          {activeView === "study" ? (
            <StudyWorkspace
              reference={reference}
              translation={translation}
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
              hasUsage={hasUsage}
              onReferenceChange={setReference}
              onTranslationChange={setTranslation}
              onGoalsChange={setGoals}
              onFetchPassage={handlePassageLookup}
              onGeneratePlan={handleStudyPlanGeneration}
              onGenerateImage={handlePassageImageGeneration}
            />
          ) : null}

          {activeView === "discussion" ? (
            <DiscussionWorkspace
              personaModel={discussionModel}
              personaError={discussionError}
              personaMessages={discussionMessages}
              enableVoiceReply={enableDiscussionVoiceReply}
              isSendingPersona={isSendingDiscussion}
              isRecordingPersona={isRecordingDiscussion}
              isTranscribingPersona={isTranscribingDiscussion}
              isRealtimeVoiceConnecting={isRealtimeVoiceConnecting}
              isRealtimeVoiceActive={isRealtimeVoiceActive}
              realtimeVoiceStatus={realtimeVoiceStatus}
              onPersonaVoiceToggle={isRecordingDiscussion ? stopDiscussionRecording : () => void startDiscussionRecording()}
              onRealtimeVoiceToggle={handleRealtimeVoiceToggle}
              onPersonaReset={handleDiscussionReset}
              onEnableVoiceReplyChange={setEnableDiscussionVoiceReply}
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
