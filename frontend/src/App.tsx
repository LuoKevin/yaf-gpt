import { useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from "react";

import { DiscussionWorkspace } from "./components/DiscussionWorkspace";
import { MusicWorkspace } from "./components/MusicWorkspace";
import { StudyWorkspace } from "./components/StudyWorkspace";
import { TextChatWorkspace } from "./components/TextChatWorkspace";
import { ViewSwitcher } from "./components/ViewSwitcher";
import {
  blobToDataUrl,
  requestJson,
  requestSse
} from "./lib/api";
import type {
  BiblePassageResponse,
  MusicGenerateResponse,
  MusicJobResponse,
  PassageImageResponse,
  PersonaChatMessage,
  SseEventPayload,
  TranslationCode,
  ViewMode,
  VoiceChatTurnResponse,
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
  const [discussionError, setDiscussionError] = useState("");
  const [isRecordingDiscussion, setIsRecordingDiscussion] = useState(false);
  const [isProcessingDiscussion, setIsProcessingDiscussion] = useState(false);
  const [isPlayingDiscussionAudio, setIsPlayingDiscussionAudio] = useState(false);
  const [discussionVoiceStatus, setDiscussionVoiceStatus] = useState("");
  const [discussionVoiceVisualLevel, setDiscussionVoiceVisualLevel] = useState(0);

  const discussionRecorderRef = useRef<MediaRecorder | null>(null);
  const discussionStreamRef = useRef<MediaStream | null>(null);
  const discussionChunksRef = useRef<Blob[]>([]);
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

  function getDiscussionAudioElement() {
    if (!discussionAudioRef.current) {
      const audio = new Audio();
      audio.autoplay = true;
      audio.onplay = () => {
        setIsPlayingDiscussionAudio(true);
        setDiscussionVoiceStatus("Playing spoken reply...");
        setDiscussionVoiceVisualLevel(0.9);
      };
      audio.onended = () => {
        setIsPlayingDiscussionAudio(false);
        setDiscussionVoiceStatus("Voice reply ready.");
        setDiscussionVoiceVisualLevel(0);
      };
      audio.onpause = () => {
        setIsPlayingDiscussionAudio(false);
        setDiscussionVoiceVisualLevel(0);
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

  useEffect(() => {
    return () => {
      if (discussionRecorderRef.current && discussionRecorderRef.current.state !== "inactive") {
        discussionRecorderRef.current.stop();
      }
      stopDiscussionAudioPlayback();
      releaseDiscussionMediaResources();
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

  async function playDiscussionAudio(audioBase64: string, mimeType: string | null) {
    stopDiscussionAudioPlayback();

    const byteCharacters = window.atob(audioBase64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let index = 0; index < byteCharacters.length; index += 1) {
      byteNumbers[index] = byteCharacters.charCodeAt(index);
    }
    const audioBlob = new Blob([new Uint8Array(byteNumbers)], { type: mimeType || "audio/wav" });
    const audioUrl = URL.createObjectURL(audioBlob);
    discussionAudioUrlRef.current = audioUrl;

    const audio = getDiscussionAudioElement();
    audio.src = audioUrl;
    await audio.play();
  }

  async function createDiscussionVoiceTurn(blob: Blob) {
    setIsProcessingDiscussion(true);
    setDiscussionError("");
    setDiscussionVoiceStatus("Transcribing your message...");
    setDiscussionVoiceVisualLevel(0.65);

    try {
      const audioBase64 = await blobToDataUrl(blob);
      const response = await requestJson<VoiceChatTurnResponse>("/api/voice/chat-turn", {
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

      setDiscussionMessages((current) => [
        ...current,
        { role: "user", content: response.transcript },
        { role: "assistant", content: response.reply }
      ]);
      setDiscussionVoiceStatus(response.audio_base64 ? "Rendering spoken reply..." : "Voice reply unavailable.");

      if (response.audio_base64) {
        await playDiscussionAudio(response.audio_base64, response.audio_mime_type);
      } else {
        setDiscussionVoiceVisualLevel(0);
      }
    } catch (error) {
      setDiscussionVoiceStatus("");
      setDiscussionVoiceVisualLevel(0);
      setDiscussionError(error instanceof Error ? error.message : "Unable to process recorded audio.");
    } finally {
      setIsProcessingDiscussion(false);
    }
  }

  async function startDiscussionRecording() {
    if (isRecordingDiscussion || isProcessingDiscussion) {
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
      stopDiscussionAudioPlayback();
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
        setDiscussionVoiceStatus("");
        setDiscussionVoiceVisualLevel(0);
        setIsRecordingDiscussion(false);
        releaseDiscussionMediaResources();
      };

      recorder.onstop = () => {
        const chunks = [...discussionChunksRef.current];
        const blobType = recorder.mimeType || "audio/webm";
        setIsRecordingDiscussion(false);
        setDiscussionVoiceVisualLevel(0);
        releaseDiscussionMediaResources();

        if (chunks.length === 0) {
          setDiscussionError("No audio captured.");
          return;
        }

        const recording = new Blob(chunks, { type: blobType });
        void createDiscussionVoiceTurn(recording);
      };

      setDiscussionError("");
      setDiscussionVoiceStatus("Listening...");
      setDiscussionVoiceVisualLevel(0.95);
      recorder.start();
      setIsRecordingDiscussion(true);
    } catch (error) {
      setDiscussionError(error instanceof Error ? error.message : "Unable to access microphone.");
      setDiscussionVoiceStatus("");
      setDiscussionVoiceVisualLevel(0);
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

  async function handleDiscussionRecordToggle() {
    if (isRecordingDiscussion) {
      stopDiscussionRecording();
      return;
    }

    await startDiscussionRecording();
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
              personaError={discussionError}
              personaMessages={discussionMessages}
              isProcessingPersona={isProcessingDiscussion}
              isRecordingPersona={isRecordingDiscussion}
              isPlayingPersonaAudio={isPlayingDiscussionAudio}
              voiceVisualLevel={discussionVoiceVisualLevel}
              voiceStatus={discussionVoiceStatus}
              onRecordToggle={handleDiscussionRecordToggle}
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
