import { pgTable, text, serial, integer, boolean, json, timestamp } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const analyses = pgTable("analyses", {
  id: serial("id").primaryKey(),
  sessionId: text("session_id").notNull(),
  // Title for display in history panel
  title: text("title").notNull().default("Untitled Analysis"),
  // Store the URL of the media (image, video, document, text)
  mediaUrl: text("media_url").notNull(),
  // Indicate the type of content being analyzed
  mediaType: text("media_type", { enum: ["image", "video", "document", "text"] }).notNull(),
  // Store the face analysis data from AWS Rekognition and other services
  faceAnalysis: json("face_analysis"),
  // For videos, store additional analysis data
  videoAnalysis: json("video_analysis"),
  // For videos, store audio transcription
  audioTranscription: json("audio_transcription"),
  // For documents or text, store content analysis
  documentAnalysis: json("document_analysis"),
  // For text uploads, store the original text
  textContent: text("text_content"),
  // Store the comprehensive insights - for all analysis types
  personalityInsights: json("personality_insights").notNull(),
  // For image/video: number of people detected, for documents: relevant entities
  peopleCount: integer("people_count").default(1),
  // LLM model used for analysis
  modelUsed: text("model_used", { enum: ["openai", "anthropic", "perplexity"] }).notNull().default("openai"),
  // For document type tracking
  documentType: text("document_type", { enum: ["pdf", "docx", "other"] }),
  // Feature: Save download status for easy access
  hasDownloaded: boolean("has_downloaded").default(false),
  // Creation timestamp
  createdAt: timestamp("created_at").defaultNow(),
});

export const messages = pgTable("messages", {
  id: serial("id").primaryKey(),
  sessionId: text("session_id").notNull(),
  analysisId: integer("analysis_id").references(() => analyses.id),
  content: text("content").notNull(),
  role: text("role", { enum: ["user", "assistant"] }).notNull(),
  createdAt: timestamp("created_at").defaultNow(),
});

export const shares = pgTable("shares", {
  id: serial("id").primaryKey(),
  analysisId: integer("analysis_id").references(() => analyses.id).notNull(),
  senderEmail: text("sender_email").notNull(),
  recipientEmail: text("recipient_email").notNull(),
  status: text("status", { enum: ["pending", "sent", "error"] }).notNull().default("pending"),
  createdAt: timestamp("created_at").defaultNow(),
});

// For tracking user sessions
export const sessions = pgTable("sessions", {
  id: serial("id").primaryKey(),
  sessionId: text("session_id").notNull().unique(),
  name: text("name").notNull().default("Session"),
  isActive: boolean("is_active").notNull().default(true),
  createdAt: timestamp("created_at").defaultNow(),
  lastActiveAt: timestamp("last_active_at").defaultNow(),
});

export const insertAnalysisSchema = createInsertSchema(analyses).omit({
  id: true,
  createdAt: true,
  hasDownloaded: true,
});

export const insertMessageSchema = createInsertSchema(messages).omit({
  id: true,
  createdAt: true,
});

export const insertShareSchema = createInsertSchema(shares).omit({
  id: true,
  createdAt: true,
  status: true,
});

export const insertSessionSchema = createInsertSchema(sessions).omit({
  id: true,
  createdAt: true,
  lastActiveAt: true,
  isActive: true,
});

export type Analysis = typeof analyses.$inferSelect;
export type InsertAnalysis = z.infer<typeof insertAnalysisSchema>;
export type Message = typeof messages.$inferSelect;
export type InsertMessage = z.infer<typeof insertMessageSchema>;
export type Share = typeof shares.$inferSelect;
export type InsertShare = z.infer<typeof insertShareSchema>;
export type Session = typeof sessions.$inferSelect;
export type InsertSession = z.infer<typeof insertSessionSchema>;

// Schema for validating media uploads
export const uploadMediaSchema = z.object({
  mediaData: z.string(),
  mediaType: z.enum(["image", "video", "document", "text"]),
  sessionId: z.string(),
  maxPeople: z.number().min(1).max(5).optional().default(5), // Optional parameter to limit people count for image/video
  selectedModel: z.enum(["openai", "anthropic", "perplexity", "deepseek"]).optional().default("openai"), // Model selection
  documentType: z.enum(["pdf", "docx", "other"]).optional(), // For document uploads
  title: z.string().optional(), // For naming the analysis in history
});

// Schema for getting shared analysis
export const getSharedAnalysisSchema = z.object({
  shareId: z.string(),
});

// Schema for validating text input
export const textInputSchema = z.object({
  content: z.string().min(1).max(500000), // Up to 500,000 characters
  sessionId: z.string(),
  selectedModel: z.enum(["openai", "anthropic", "perplexity", "deepseek"]).optional().default("openai"),
  title: z.string().optional(),
});

// Schema for validating document analysis
export const documentAnalysisSchema = z.object({
  fileData: z.string(), // base64 encoded document
  fileName: z.string(),
  fileType: z.enum(["pdf", "docx"]),
  sessionId: z.string(),
  selectedModel: z.enum(["openai", "anthropic", "perplexity", "deepseek"]).optional().default("openai"),
  title: z.string().optional(),
});

// Schema for session-related operations
export const sessionSchema = z.object({
  sessionId: z.string(),
  name: z.string().optional(),
});

// Schema for downloading analysis
export const downloadAnalysisSchema = z.object({
  analysisId: z.number(),
  format: z.enum(["json", "pdf", "text"]).default("pdf"),
  includeCharts: z.boolean().default(true),
});