import { useEffect, useMemo, useRef, useState } from "react";

import { DiscussionWorkspace } from "./components/DiscussionWorkspace";
import { MusicWorkspace } from "./components/MusicWorkspace";
import { StudyWorkspace } from "./components/StudyWorkspace";
import { ViewSwitcher } from "./components/ViewSwitcher";
import {
  blobToDataUrl,
  buildMusicPrompt,
  normalizeReference,
  requestJson,
  requestRealtimeAnswer,
  requestSse
} from "./lib/api";
import type {
  BiblePassageResponse,
  HealthStatus,
  MusicGenerateResponse,
  MusicJobResponse,
  PassageImageResponse,
  PersonaChatMessage,
  RealtimeVoice,
  TranslationCode,
  ViewMode,
  VoiceRealtimeSessionResponse,
  VoiceTranscriptionResponse,
  StudyPlanResponse
} from "./types";

export default function App() {
  const [activeView, setActiveView] = useState<ViewMode>("study");

  const [reference, setReference] = useState("Luke 21:5-28");
  const [translation, setTranslation] = useState<TranslationCode>("WEB");
  const [goals, setGoals] = useState("");
  const [userNotes, setUserNotes] = useState("");
  const [includeQuestionNotes, setIncludeQuestionNotes] = useState(false);

  const [healthStatus, setHealthStatus] = useState<HealthStatus>("checking");

  const [passage, setPassage] = useState<BiblePassageResponse | null>(null);
  const [studyPlan, setStudyPlan] = useState<StudyPlanResponse | null>(null);
  const [passageError, setPassageError] = useState("");
  const [studyPlanError, setStudyPlanError] = useState("");
  const [isLoadingPassage, setIsLoadingPassage] = useState(false);
  const [isLoadingStudyPlan, setIsLoadingStudyPlan] = useState(false);

  const [passageImage, setPassageImage] = useState<PassageImageResponse | null>(null);
  const [passageImageError, setPassageImageError] = useState("");
  const [isLoadingPassageImage, setIsLoadingPassageImage] = useState(false);

  const [personaInput, setPersonaInput] = useState("");
  const [personaMessages, setPersonaMessages] = useState<PersonaChatMessage[]>([]);
  const [personaModel, setPersonaModel] = useState<string | null>(null);
  const [personaError, setPersonaError] = useState("");
  const [isSendingPersona, setIsSendingPersona] = useState(false);
  const [isRecordingPersona, setIsRecordingPersona] = useState(false);
  const [isTranscribingPersona, setIsTranscribingPersona] = useState(false);
  const [enableVoiceReply, setEnableVoiceReply] = useState(true);
  const [isRealtimeVoiceConnecting, setIsRealtimeVoiceConnecting] = useState(false);
  const [isRealtimeVoiceActive, setIsRealtimeVoiceActive] = useState(false);
  const [realtimeVoiceStatus, setRealtimeVoiceStatus] = useState("");

  const personaRecorderRef = useRef<MediaRecorder | null>(null);
  const personaStreamRef = useRef<MediaStream | null>(null);
  const personaChunksRef = useRef<Blob[]>([]);
  const realtimePeerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const realtimeDataChannelRef = useRef<RTCDataChannel | null>(null);
  const realtimeLocalStreamRef = useRef<MediaStream | null>(null);
  const realtimeAudioRef = useRef<HTMLAudioElement | null>(null);
  const realtimeAssistantItemIdRef = useRef<string | null>(null);

  const [musicTitle, setMusicTitle] = useState("");
  const [musicPrompt, setMusicPrompt] = useState("");
  const [musicStyle, setMusicStyle] = useState("modern worship, acoustic");
  const [musicMood, setMusicMood] = useState("hopeful");
  const [musicResult, setMusicResult] = useState<MusicGenerateResponse | null>(null);
  const [musicJob, setMusicJob] = useState<MusicJobResponse | null>(null);
  const [musicError, setMusicError] = useState("");
  const [isGeneratingMusic, setIsGeneratingMusic] = useState(false);

  useEffect(() => {
    let cancelled = false;

    async function checkHealth() {
      try {
        await requestJson<{ status: string }>("/health");
        if (!cancelled) {
          setHealthStatus("online");
        }
      } catch {
        if (!cancelled) {
          setHealthStatus("offline");
        }
      }
    }

    void checkHealth();

    return () => {
      cancelled = true;
    };
  }, []);

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

  function releasePersonaMediaResources() {
    if (personaStreamRef.current) {
      personaStreamRef.current.getTracks().forEach((track) => track.stop());
    }
    personaStreamRef.current = null;
    personaRecorderRef.current = null;
    personaChunksRef.current = [];
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
    setIsRealtimeVoiceConnecting(false);
    setIsRealtimeVoiceActive(false);
    setRealtimeVoiceStatus("");
  }

  function appendRealtimeUserTranscript(transcript: string) {
    const cleaned = transcript.trim();
    if (!cleaned) {
      return;
    }

    setPersonaMessages((current) => [...current, { role: "user", content: cleaned }]);
  }

  function appendRealtimeAssistantDelta(itemId: string, delta: string) {
    if (!delta) {
      return;
    }

    setPersonaMessages((current) => {
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

    setPersonaMessages((current) => {
      if (current.length === 0) {
        return [{ role: "assistant", content: cleaned }];
      }

      const next = [...current];
      const last = next[next.length - 1];
      if (realtimeAssistantItemIdRef.current === itemId && last.role === "assistant") {
        next[next.length - 1] = { role: "assistant", content: cleaned };
        return next;
      }

      next.push({ role: "assistant", content: cleaned });
      return next;
    });

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
      setPersonaModel(model);
      setRealtimeVoiceStatus(`Live voice connected on ${model} (${voice}).`);
      return;
    }

    if (eventType === "conversation.item.input_audio_transcription.completed") {
      const transcript = "transcript" in event && typeof event.transcript === "string" ? event.transcript : "";
      appendRealtimeUserTranscript(transcript);
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
      setPersonaError(detail);
      stopRealtimeVoiceSession();
    }
  }

  useEffect(() => {
    return () => {
      if (personaRecorderRef.current && personaRecorderRef.current.state !== "inactive") {
        personaRecorderRef.current.stop();
      }
      stopRealtimeVoiceSession();
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
      releasePersonaMediaResources();
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

  function speakPersonaReply(replyText: string) {
    const cleaned = replyText.trim();
    if (!cleaned || !enableVoiceReply) {
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

  async function sendPersonaMessage(rawMessage: string) {
    const userMessage = rawMessage.trim();
    if (!userMessage) {
      setPersonaError("Type a message.");
      return;
    }

    const nextMessages = [...personaMessages, { role: "user", content: userMessage } as PersonaChatMessage];
    setPersonaMessages([...nextMessages, { role: "assistant", content: "" }]);
    setPersonaError("");
    setIsSendingPersona(true);

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
            messages: nextMessages,
            reference_context: reference.trim() || undefined,
            translation
          })
        },
        ({ event, data }) => {
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
              setPersonaMessages((current) => {
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
        setPersonaModel(streamedModel);
      }
      if (!receivedChunk) {
        setPersonaMessages((current) => {
          if (current.length === 0) {
            return current;
          }
          const last = current[current.length - 1];
          if (last.role === "assistant" && !last.content.trim()) {
            return current.slice(0, -1);
          }
          return current;
        });
      } else {
        speakPersonaReply(assistantReply);
      }
    } catch (error) {
      setPersonaMessages((current) => {
        if (current.length === 0) {
          return current;
        }
        const last = current[current.length - 1];
        if (last.role === "assistant" && !last.content.trim()) {
          return current.slice(0, -1);
        }
        return current;
      });
      setPersonaError(error instanceof Error ? error.message : "Unable to send message.");
    } finally {
      setIsSendingPersona(false);
    }
  }

  async function handlePersonaSend() {
    const userMessage = personaInput.trim();
    if (!userMessage) {
      setPersonaError("Type a message.");
      return;
    }
    setPersonaInput("");
    await sendPersonaMessage(userMessage);
  }

  async function transcribePersonaRecording(blob: Blob) {
    setIsTranscribingPersona(true);
    setPersonaError("");

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

      await sendPersonaMessage(response.transcript);
    } catch (error) {
      setPersonaError(error instanceof Error ? error.message : "Unable to process recorded audio.");
    } finally {
      setIsTranscribingPersona(false);
    }
  }

  async function startPersonaRecording() {
    if (isRecordingPersona || isSendingPersona || isTranscribingPersona) {
      return;
    }
    if (typeof window === "undefined" || typeof MediaRecorder === "undefined") {
      setPersonaError("Voice input is not supported in this browser.");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setPersonaError("Microphone access is not available in this browser.");
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

      personaStreamRef.current = stream;
      personaRecorderRef.current = recorder;
      personaChunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          personaChunksRef.current.push(event.data);
        }
      };

      recorder.onerror = () => {
        setPersonaError("Recording failed.");
        setIsRecordingPersona(false);
        releasePersonaMediaResources();
      };

      recorder.onstop = () => {
        const chunks = [...personaChunksRef.current];
        const blobType = recorder.mimeType || "audio/webm";
        setIsRecordingPersona(false);
        releasePersonaMediaResources();

        if (chunks.length === 0) {
          setPersonaError("No audio captured.");
          return;
        }

        const recording = new Blob(chunks, { type: blobType });
        void transcribePersonaRecording(recording);
      };

      setPersonaError("");
      recorder.start();
      setIsRecordingPersona(true);
    } catch (error) {
      setPersonaError(error instanceof Error ? error.message : "Unable to access microphone.");
      setIsRecordingPersona(false);
      releasePersonaMediaResources();
    }
  }

  function stopPersonaRecording() {
    if (!personaRecorderRef.current || personaRecorderRef.current.state === "inactive") {
      return;
    }
    personaRecorderRef.current.stop();
  }

  async function startRealtimeVoiceSession() {
    if (isRealtimeVoiceActive || isRealtimeVoiceConnecting || isSendingPersona || isRecordingPersona || isTranscribingPersona) {
      return;
    }
    if (typeof window === "undefined" || typeof RTCPeerConnection === "undefined") {
      setPersonaError("Live voice requires WebRTC support in this browser.");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setPersonaError("Microphone access is not available in this browser.");
      return;
    }

    setPersonaError("");
    setIsRealtimeVoiceConnecting(true);
    setRealtimeVoiceStatus("Connecting live voice...");

    try {
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
          setPersonaModel(session.model);
          setRealtimeVoiceStatus(`Live voice connected on ${session.model} (${session.voice}).`);
          return;
        }

        if (state === "failed" || state === "closed" || state === "disconnected") {
          stopRealtimeVoiceSession();
          if (state === "failed") {
            setPersonaError("Live voice connection dropped.");
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
        setPersonaError("Live voice event channel failed.");
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
        session.model,
        session.client_secret,
        offer.sdp
      );

      await peerConnection.setRemoteDescription({
        type: "answer",
        sdp: answerSdp
      });
    } catch (error) {
      stopRealtimeVoiceSession();
      setPersonaError(error instanceof Error ? error.message : "Unable to start live voice.");
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

  function handlePersonaReset() {
    if (personaRecorderRef.current && personaRecorderRef.current.state !== "inactive") {
      personaRecorderRef.current.stop();
    }
    stopRealtimeVoiceSession();
    if (typeof window !== "undefined" && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    releasePersonaMediaResources();
    setPersonaMessages([]);
    setPersonaModel(null);
    setPersonaError("");
    setPersonaInput("");
    setIsRecordingPersona(false);
    setIsTranscribingPersona(false);
  }

  async function handleMusicGeneration() {
    const trimmedReference = reference.trim();
    const trimmedPrompt = musicPrompt.trim();
    if (!trimmedReference && !trimmedPrompt) {
      setMusicError("Enter a reference or prompt.");
      return;
    }

    setIsGeneratingMusic(true);
    setMusicError("");

    try {
      const canReusePassage =
        passage !== null &&
        normalizeReference(passage.reference) === normalizeReference(trimmedReference) &&
        passage.translation === translation;
      const prompt = buildMusicPrompt(trimmedReference, canReusePassage ? passage.text : null, trimmedPrompt);

      const response = await requestJson<MusicGenerateResponse>("/api/music/generate", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          title: musicTitle.trim() || undefined,
          prompt,
          style_hint: musicStyle.trim(),
          mood_hint: musicMood.trim() || undefined
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

  return (
    <div className="shell">
      <div className="background-orb background-orb-left" />
      <div className="background-orb background-orb-right" />

      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">yaf-gpt</p>
          <h1>Study Workspace</h1>
        </div>

        <div className={`status-pill ${healthStatus}`}>
          <span className="status-dot" />
          <span>{healthStatus.toUpperCase()}</span>
        </div>
      </header>

      <ViewSwitcher activeView={activeView} onChange={setActiveView} />

      <main className={`workspace ${activeView === "discussion" ? "workspace-single" : ""}`}>
        {activeView === "study" ? (
          <StudyWorkspace
            reference={reference}
            translation={translation}
            goals={goals}
            userNotes={userNotes}
            includeQuestionNotes={includeQuestionNotes}
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
            onUserNotesChange={setUserNotes}
            onIncludeQuestionNotesChange={setIncludeQuestionNotes}
            onFetchPassage={handlePassageLookup}
            onGeneratePlan={handleStudyPlanGeneration}
            onGenerateImage={handlePassageImageGeneration}
          />
        ) : null}

        {activeView === "discussion" ? (
          <DiscussionWorkspace
            personaModel={personaModel}
            personaError={personaError}
            personaMessages={personaMessages}
            personaInput={personaInput}
            enableVoiceReply={enableVoiceReply}
            isSendingPersona={isSendingPersona}
            isRecordingPersona={isRecordingPersona}
            isTranscribingPersona={isTranscribingPersona}
            isRealtimeVoiceConnecting={isRealtimeVoiceConnecting}
            isRealtimeVoiceActive={isRealtimeVoiceActive}
            realtimeVoiceStatus={realtimeVoiceStatus}
            onPersonaInputChange={setPersonaInput}
            onPersonaSend={handlePersonaSend}
            onPersonaVoiceToggle={isRecordingPersona ? stopPersonaRecording : () => void startPersonaRecording()}
            onRealtimeVoiceToggle={handleRealtimeVoiceToggle}
            onPersonaReset={handlePersonaReset}
            onEnableVoiceReplyChange={setEnableVoiceReply}
          />
        ) : null}

        {activeView === "music" ? (
          <MusicWorkspace
            reference={reference}
            translation={translation}
            musicTitle={musicTitle}
            musicPrompt={musicPrompt}
            musicStyle={musicStyle}
            musicMood={musicMood}
            musicResult={musicResult}
            musicJob={musicJob}
            musicError={musicError}
            musicStatusLabel={musicStatusLabel}
            isGeneratingMusic={isGeneratingMusic}
            onReferenceChange={setReference}
            onTranslationChange={setTranslation}
            onMusicTitleChange={setMusicTitle}
            onMusicPromptChange={setMusicPrompt}
            onMusicStyleChange={setMusicStyle}
            onMusicMoodChange={setMusicMood}
            onGenerateMusic={handleMusicGeneration}
          />
        ) : null}
      </main>
    </div>
  );
}
