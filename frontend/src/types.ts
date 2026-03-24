export type TranslationCode = "WEB" | "KJV";
export type ImageStyle = "modern_editorial_illustration";
export type PersonaChatRole = "user" | "assistant";
export type MusicJobStatus = "queued" | "in_progress" | "completed" | "failed";
export type ViewMode = "study" | "chat" | "music" | "discussion";
export type RealtimeVoice =
  | "alloy"
  | "ash"
  | "ballad"
  | "cedar"
  | "coral"
  | "echo"
  | "marin"
  | "sage"
  | "shimmer"
  | "verse";

export type BibleVerse = {
  book: string;
  chapter: number;
  verse: number;
  text: string;
};

export type BiblePassageResponse = {
  reference: string;
  translation: TranslationCode;
  normalized_reference: string;
  text: string;
  verses: BibleVerse[];
};

export type UsageMetrics = {
  prompt_tokens: number | null;
  completion_tokens: number | null;
  total_tokens: number | null;
};

export type StudyPlanResponse = {
  reference: string;
  normalized_reference: string;
  translation: TranslationCode;
  passage_text: string;
  passage_title: string;
  context_points: string[];
  discussion_questions: string[];
  reflection_questions: string[];
  include_question_notes: boolean;
  discussion_question_notes: string[] | null;
  reflection_question_notes: string[] | null;
  model: string;
  usage: UsageMetrics | null;
};

export type PassageImageResponse = {
  reference: string;
  translation: TranslationCode;
  style: ImageStyle;
  prompt_used: string;
  image_b64_or_url: string;
  alt_text: string;
};

export type PersonaChatMessage = {
  role: PersonaChatRole;
  content: string;
};

export type PersonaChatResponse = {
  reply: string;
  model: string;
  usage: UsageMetrics | null;
};

export type VoiceTranscriptionResponse = {
  transcript: string;
  model: string;
};

export type VoiceRealtimeSessionResponse = {
  client_secret: string;
  expires_at: number;
  model: string;
  voice: RealtimeVoice;
  webrtc_url: string;
};

export type MusicGenerateResponse = {
  job_id: string;
  status: MusicJobStatus;
  provider: string;
  title: string;
  prompt: string;
};

export type MusicJobResponse = {
  job_id: string;
  status: MusicJobStatus;
  provider: string;
  audio_url: string | null;
  error: string | null;
};

export type SseEventPayload = {
  event: string;
  data: unknown;
};
