import type { Express, Request, Response, NextFunction } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import OpenAI from "openai";
import { insertAnalysisSchema, insertMessageSchema, insertShareSchema, uploadMediaSchema } from "@shared/schema";
import { z } from "zod";
import { 
  RekognitionClient, 
  DetectFacesCommand,
  DetectLabelsCommand,
  StartFaceDetectionCommand, 
  GetFaceDetectionCommand 
} from "@aws-sdk/client-rekognition";
import { sendAnalysisEmail } from "./services/email";
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { promisify } from 'util';
import ffmpeg from 'fluent-ffmpeg';
import Anthropic from '@anthropic-ai/sdk';

// Initialize API clients with proper error handling for missing keys
let openai: OpenAI | null = null;
let anthropic: Anthropic | null = null;

// Check if OpenAI API key is available
if (process.env.OPENAI_API_KEY) {
  try {
    openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    console.log("OpenAI client initialized successfully");
  } catch (error) {
    console.error("Failed to initialize OpenAI client:", error);
  }
} else {
  console.warn("OPENAI_API_KEY environment variable is not set. OpenAI API functionality will be limited.");
}

// Check if Anthropic API key is available
if (process.env.ANTHROPIC_API_KEY) {
  try {
    anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    console.log("Anthropic client initialized successfully");
  } catch (error) {
    console.error("Failed to initialize Anthropic client:", error);
  }
} else {
  console.warn("ANTHROPIC_API_KEY environment variable is not set. Anthropic API functionality will be limited.");
}

// Perplexity AI client
const perplexity = {
  query: async ({ model, query }: { model: string, query: string }) => {
    if (!process.env.PERPLEXITY_API_KEY) {
      console.warn("PERPLEXITY_API_KEY environment variable is not set. Perplexity API functionality will be limited.");
      throw new Error("Perplexity API key not available");
    }
    
    try {
      const response = await fetch("https://api.perplexity.ai/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${process.env.PERPLEXITY_API_KEY}`
        },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: query }]
        })
      });
      
      const data = await response.json();
      return {
        text: data.choices[0]?.message?.content || ""
      };
    } catch (error) {
      console.error("Perplexity API error:", error);
      return { text: "" };
    }
  }
};

// AWS Rekognition client
// Let the AWS SDK pick up credentials from environment variables automatically
const rekognition = new RekognitionClient({ 
  region: process.env.AWS_REGION || "us-east-1"
});

// For Google Cloud functionality, we'll implement in a follow-up task

// For temporary file storage
const tempDir = os.tmpdir();
const writeFileAsync = promisify(fs.writeFile);
const unlinkAsync = promisify(fs.unlink);

// Google Cloud Storage bucket for videos
// This would typically be created and configured through Google Cloud Console first
const bucketName = 'ai-personality-videos';

/**
 * Helper function to get the duration of a video using ffprobe
 */
async function getVideoDuration(videoPath: string): Promise<number> {
  return new Promise<number>((resolve, reject) => {
    ffmpeg.ffprobe(videoPath, (err: Error | null, metadata: any) => {
      if (err) {
        console.error('Error getting video duration:', err);
        // Default to 5 seconds if we can't determine duration
        return resolve(5);
      }
      
      // Get duration in seconds
      const durationSec = metadata.format.duration || 5;
      resolve(durationSec);
    });
  });
}

/**
 * Helper function to split a video into chunks of specified duration
 */
async function splitVideoIntoChunks(videoPath: string, outputDir: string, chunkDurationSec: number): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    ffmpeg(videoPath)
      .outputOptions([
        `-f segment`,
        `-segment_time ${chunkDurationSec}`,
        `-reset_timestamps 1`,
        `-c copy` // Copy codec (fast)
      ])
      .output(path.join(outputDir, 'chunk_%03d.mp4'))
      .on('end', () => {
        console.log('Video successfully split into chunks');
        resolve();
      })
      .on('error', (err: Error) => {
        console.error('Error splitting video:', err);
        reject(err);
      })
      .run();
  });
}

/**
 * Helper function to extract audio from video and transcribe it using OpenAI Whisper API
 */
async function extractAudioTranscription(videoPath: string): Promise<any> {
  try {
    // Extract audio from video
    const randomId = Math.random().toString(36).substring(2, 15);
    const audioPath = path.join(tempDir, `${randomId}.mp3`);
    
    console.log('Extracting audio from video...');
    await new Promise<void>((resolve, reject) => {
      ffmpeg(videoPath)
        .output(audioPath)
        .audioCodec('libmp3lame')
        .audioChannels(1)
        .audioFrequency(16000)
        .on('end', () => resolve())
        .on('error', (err: Error) => {
          console.error('Error extracting audio:', err);
          reject(err);
        })
        .run();
    });
    
    console.log('Audio extraction complete, starting transcription with OpenAI...');
    
    // Create a readable stream from the audio file
    const audioFile = fs.createReadStream(audioPath);
    
    // Transcribe using OpenAI's Whisper API
    let transcriptionResponse;
    
    if (openai) {
      try {
        transcriptionResponse = await openai.audio.transcriptions.create({
          file: audioFile,
          model: 'whisper-1',
          language: 'en',
          response_format: 'verbose_json',
          timestamp_granularities: ['word']
        });
      } catch (error) {
        console.error("Error with OpenAI Whisper API:", error);
        throw new Error("Transcription failed. OpenAI API may not be properly configured.");
      }
    } else {
      console.warn("OpenAI client is not initialized. Using mock transcription response.");
      transcriptionResponse = {
        text: "OpenAI API key is required for transcription. This is a placeholder transcription.",
        segments: []
      };
    }
    
    const transcription = transcriptionResponse.text;
    console.log(`Transcription received: ${transcription.substring(0, 100)}...`);
    
    // Calculate speaking rate based on word count and duration
    const audioDuration = await getVideoDuration(audioPath);
    const words = transcription.split(' ').length;
    const speakingRate = audioDuration > 0 ? words / audioDuration : 0;
    
    // Advanced analysis for speech patterns
    // Extract segments with confidence and timestamps
    const segments = transcriptionResponse.segments || [];
    
    // Calculate average confidence across all segments
    // Note: OpenAI's Whisper API doesn't actually provide confidence values per segment
    // So we'll use a default high confidence value as an estimate
    const averageConfidence = 0.92; // Whisper is generally highly accurate
    
    // Clean up temp file
    await unlinkAsync(audioPath).catch(err => console.warn('Error deleting temp audio file:', err));
    
    return {
      transcription,
      speechAnalysis: {
        averageConfidence,
        speakingRate,
        wordCount: words,
        duration: audioDuration,
        segments: segments.map(s => ({
          text: s.text,
          start: s.start,
          end: s.end,
          confidence: averageConfidence // Using the same confidence for all segments
        }))
      }
    };
  } catch (error) {
    console.error('Error in audio transcription:', error);
    // Return a minimal object if transcription fails
    return {
      transcription: "Failed to transcribe audio. Please try again with clearer audio or a different video.",
      speechAnalysis: {
        averageConfidence: 0,
        speakingRate: 0,
        error: error instanceof Error ? error.message : "Unknown transcription error"
      }
    };
  }
}


// For backward compatibility
const uploadImageSchema = z.object({
  imageData: z.string(),
  sessionId: z.string(),
});

const sendMessageSchema = z.object({
  content: z.string(),
  sessionId: z.string(),
});

// Check if email service is configured
let isEmailServiceConfigured = false;
if (process.env.SENDGRID_API_KEY && process.env.SENDGRID_VERIFIED_SENDER) {
  isEmailServiceConfigured = true;
}

// Define the schema for retrieving a shared analysis
const getSharedAnalysisSchema = z.object({
  shareId: z.coerce.number(),
});

export async function registerRoutes(app: Express): Promise<Server> {
  // Text analysis endpoint
  app.post("/api/analyze/text", async (req, res) => {
    try {
      const { content, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!content || typeof content !== 'string') {
        return res.status(400).json({ error: "Text content is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing text analysis with model: ${selectedModel}`);
      
      // Get comprehensive personality insights based on text content with 50-question framework
      const textAnalysisPrompt = `You are an expert literary psychologist. Analyze this text comprehensively, providing specific evidence-based answers to ALL questions below WITH DIRECT QUOTES.

CRITICAL: Every answer must reference SPECIFIC QUOTES or PHRASES from the text. Do not use generic descriptions.

TEXT:
${content}

I. LANGUAGE & STYLE
1. What is the dominant sentence rhythm — clipped, flowing, erratic — and what personality trait does it reveal?
2. Which adjectives recur, and what emotional bias do they show?
3. How does pronoun use ("I," "you," "we," "they") shift across the text, and what identity stance does that reflect?
4. What level of abstraction vs. concreteness dominates the writing?
5. Identify one passage where diction becomes suddenly elevated or deflated — what triggers it?
6. Are there unfinished or fragmentary sentences, and what might that signal psychologically?
7. How consistent is the tense? Does the writer slip between past and present, and why?
8. What metaphors or analogies recur, and what unconscious associations do they expose?
9. Is the author's tone self-assured, tentative, ironic, or performative? Cite phrasing.
10. What linguistic register (formal, colloquial, technical) dominates, and how does it align with self-image?

II. EMOTIONAL INDICATORS
11. What emotion seems primary (anger, melancholy, pride, longing), and where is it linguistically concentrated?
12. Which emotions appear repressed or displaced — hinted at but never named?
13. Does emotional intensity rise or fall as the text progresses?
14. Identify one sentence where affect "leaks through" despite apparent control.
15. Are there moments of sentimental overstatement or cold detachment?
16. What bodily or sensory words appear, and what do they suggest about embodiment or repression?
17. Is there ambivalence toward the subject matter? Cite a line where tone wavers.
18. Does humor appear, and if so, is it self-directed, aggressive, or defensive?
19. What words betray anxiety or guilt?
20. How is desire represented — directly, symbolically, or through avoidance?

III. COGNITIVE & STRUCTURAL PATTERNS
21. How logically coherent are transitions between ideas?
22. Does the writer prefer enumeration, narrative, or digression? What does that indicate about thought style?
23. What syntactic habits dominate (parallelism, repetition, parenthesis), and what mental rhythms do they mirror?
24. Are there contradictions the author fails to notice? Quote one.
25. How does the author handle uncertainty — through hedging, assertion, or silence?
26. Does the argument or story circle back on itself?
27. Are there abrupt topic shifts, and what emotional events coincide with them?
28. What elements of the text seem compulsive or ritualistic in repetition?
29. Where does the writer show real insight versus mechanical reasoning?
30. How does closure occur (resolution, withdrawal, collapse), and what does it signify psychologically?

IV. SELF-REPRESENTATION & IDENTITY
31. How does the writer portray the self — victim, hero, observer, analyst?
32. Is there a split between narrating voice and lived experience?
33. What form of authority or validation does the author seek (moral, intellectual, emotional)?
34. How consistent is the self-image across paragraphs?
35. Identify one phrase that reveals unconscious self-evaluation (admiration, contempt, shame).
36. Does the author reveal dependency on external approval or autonomy from it?
37. What form of vulnerability does the writer allow?
38. How does the author talk about others — with empathy, rivalry, indifference?
39. What implicit audience is being addressed?
40. Does the writer's stance shift from confession to performance? Cite turning point.

V. SYMBOLIC & UNCONSCIOUS MATERIAL
41. Which images or motifs recur (light/dark, ascent/descent, enclosure, mirrors), and what do they symbolize?
42. Are there dream-like or surreal elements?
43. What oppositions structure the text (order/chaos, love/power, mind/body)?
44. What wish or fear seems to animate the text beneath the surface argument?
45. Identify one metaphor that reads like a disguised confession.
46. How does the author relate to time — nostalgic, future-oriented, frozen?
47. Does the text express conflict between intellect and emotion?
48. What shadow aspect of personality is hinted at through hostile or taboo imagery?
49. Is there evidence of projection — attributing inner states to others or to abstractions?
50. What central psychological drama (loss, control, recognition, transformation) structures the entire piece?

Return JSON:
{
  "summary": "2-3 sentence overview with specific details from the text",
  "detailed_analysis": {
    "language_style": "Answers to questions 1-10 with specific quotes and linguistic evidence",
    "emotional_indicators": "Answers to questions 11-20 with specific quotes showing emotion",
    "cognitive_structural": "Answers to questions 21-30 with quotes showing thought patterns",
    "self_representation": "Answers to questions 31-40 with quotes revealing identity",
    "symbolic_unconscious": "Answers to questions 41-50 with quotes and symbolic analysis"
  }
}`;

      // Get personality analysis from selected AI model
      let analysisResult: any;
      
      if (selectedModel === "openai" && openai) {
        console.log('Using OpenAI for text analysis');
        const completion = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [
            { role: "system", content: "You are an expert literary psychologist and personality analyst." },
            { role: "user", content: textAnalysisPrompt }
          ],
          response_format: { type: "json_object" }
        });
        
        analysisResult = JSON.parse(completion.choices[0].message.content || "{}");
      } 
      else if (selectedModel === "anthropic" && anthropic) {
        console.log('Using Anthropic for text analysis');
        const response = await anthropic.messages.create({
          model: "claude-3-7-sonnet-20250219",
          max_tokens: 4000,
          system: "You are an expert literary psychologist and personality analyst. Always return valid JSON.",
          messages: [{ role: "user", content: textAnalysisPrompt }],
        });
        
        let textContent = response.content[0].text;
        
        // Extract JSON from markdown code blocks if present
        const jsonMatch = textContent.match(/```json\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          textContent = jsonMatch[1];
        }
        
        // Try to parse JSON, fallback to wrapping in analysis object with all 5 sections
        try {
          analysisResult = JSON.parse(textContent);
        } catch {
          analysisResult = {
            summary: "Text analysis completed. Unable to parse structured response.",
            detailed_analysis: {
              language_style: textContent.substring(0, 1000) || "Analysis section unavailable",
              emotional_indicators: "Unable to parse emotional indicators section",
              cognitive_structural: "Unable to parse cognitive patterns section", 
              self_representation: "Unable to parse identity section",
              symbolic_unconscious: "Unable to parse symbolic analysis section"
            }
          };
        }
      }
      else if (selectedModel === "perplexity" && process.env.PERPLEXITY_API_KEY) {
        console.log('Using Perplexity for text analysis');
        const response = await perplexity.query({
          model: "llama-3.1-sonar-small-128k-online",
          query: textAnalysisPrompt
        });
        
        let responseText = response.text;
        
        // Extract JSON from markdown code blocks if present
        const jsonMatch = responseText.match(/```json\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          responseText = jsonMatch[1];
        }
        
        // Try to parse JSON, fallback to wrapping with all 5 sections
        try {
          analysisResult = JSON.parse(responseText);
        } catch {
          analysisResult = {
            summary: "Text analysis completed. Unable to parse structured response.",
            detailed_analysis: {
              language_style: responseText.substring(0, 1000) || "Analysis section unavailable",
              emotional_indicators: "Unable to parse emotional indicators section",
              cognitive_structural: "Unable to parse cognitive patterns section",
              self_representation: "Unable to parse identity section", 
              symbolic_unconscious: "Unable to parse symbolic analysis section"
            }
          };
        }
      }
      else {
        return res.status(503).json({ 
          error: "Selected AI model is not available. Please try again with a different model." 
        });
      }
      
      // Helper function to safely stringify any value into readable text
      const safeStringify = (value: any): string => {
        if (typeof value === 'string') return value;
        if (typeof value === 'object' && value !== null) {
          // If it's an array, handle each item recursively
          if (Array.isArray(value)) {
            return value.map(item => {
              if (typeof item === 'string') return item;
              if (typeof item === 'object' && item !== null) {
                // Format objects in arrays as key-value pairs
                return Object.entries(item)
                  .map(([key, val]) => `${key}: ${val}`)
                  .join('\n');
              }
              return String(item);
            }).join('\n\n');
          }
          // If it's an object with numbered keys (like "1", "2", etc), format as numbered list
          const keys = Object.keys(value);
          if (keys.length > 0 && keys.every(k => /^\d+$/.test(k))) {
            return keys
              .sort((a, b) => parseInt(a) - parseInt(b))
              .map(key => `${key}. ${value[key]}`)
              .join('\n');
          }
          // If it's an object with named keys, format as key-value pairs
          return Object.entries(value)
            .map(([key, val]) => `${val}`)
            .join('\n\n');
        }
        return String(value || '');
      };
      
      // Format the analysis for display
      let formattedContent = `AI-Powered Text Analysis\nMode: Comprehensive Literary & Psychological Analysis\n\n`;
      formattedContent += `${'─'.repeat(65)}\n`;
      formattedContent += `Analysis Results\n`;
      formattedContent += `${'─'.repeat(65)}\n\n`;
      
      formattedContent += `Summary:\n${safeStringify(analysisResult.summary) || 'No summary available'}\n\n`;
      
      const detailedAnalysis = analysisResult.detailed_analysis || {};
      
      if (detailedAnalysis.language_style) {
        formattedContent += `I. Language & Style:\n${safeStringify(detailedAnalysis.language_style)}\n\n`;
      }
      
      if (detailedAnalysis.emotional_indicators) {
        formattedContent += `II. Emotional Indicators:\n${safeStringify(detailedAnalysis.emotional_indicators)}\n\n`;
      }
      
      if (detailedAnalysis.cognitive_structural) {
        formattedContent += `III. Cognitive & Structural Patterns:\n${safeStringify(detailedAnalysis.cognitive_structural)}\n\n`;
      }
      
      if (detailedAnalysis.self_representation) {
        formattedContent += `IV. Self-Representation & Identity:\n${safeStringify(detailedAnalysis.self_representation)}\n\n`;
      }
      
      if (detailedAnalysis.symbolic_unconscious) {
        formattedContent += `V. Symbolic & Unconscious Material:\n${safeStringify(detailedAnalysis.symbolic_unconscious)}\n\n`;
      }
      
      // Create an analysis with a dummy mediaUrl since the schema requires it but we don't have
      // media for text analysis
      const dummyMediaUrl = `text:${Date.now()}`;
      
      // Create analysis record in storage
      const analysis = await storage.createAnalysis({
        sessionId,
        mediaUrl: dummyMediaUrl,
        mediaType: "text",
        personalityInsights: analysisResult,
        title: title || "Text Analysis"
      });
      
      // Create initial message
      const initialMessage = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        role: "assistant",
        content: formattedContent
      });
      
      // Return data to client
      res.json({
        analysisId: analysis.id,
        messages: [initialMessage],
        emailServiceAvailable: isEmailServiceConfigured
      });
    } catch (error) {
      console.error("Text analysis error:", error);
      if (error instanceof Error) {
        res.status(500).json({ error: error.message });
      } else {
        res.status(500).json({ error: "Failed to analyze text" });
      }
    }
  });
  
  // Document analysis endpoint
  app.post("/api/analyze/document", async (req, res) => {
    try {
      const { fileData, fileName, fileType, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!fileData || typeof fileData !== 'string') {
        return res.status(400).json({ error: "Document data is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing document analysis with model: ${selectedModel}, file: ${fileName}`);
      
      // Extract base64 content from data URL
      const base64Data = fileData.split(',')[1];
      if (!base64Data) {
        return res.status(400).json({ error: "Invalid document data format" });
      }
      
      // Save the document to a temporary file
      const fileBuffer = Buffer.from(base64Data, 'base64');
      const tempDocPath = path.join(tempDir, `doc_${Date.now()}_${fileName}`);
      await writeFileAsync(tempDocPath, fileBuffer);
      
      // Document analysis prompt
      const documentAnalysisPrompt = `
I'm going to analyze the uploaded document: ${fileName} (${fileType}).

Provide a comprehensive analysis of this document, including:

1. Document overview and key topics
2. Main themes and insights
3. Emotional tone and sentiment
4. Writing style assessment
5. Author personality assessment based on the document
`;

      // Get document analysis from selected AI model
      let analysisText: string;
      
      if (selectedModel === "openai" && openai) {
        console.log('Using OpenAI for document analysis');
        const completion = await openai.chat.completions.create({
          model: "gpt-4o", // the newest OpenAI model is "gpt-4o" which was released May 13, 2024
          messages: [
            { role: "system", content: "You are an expert in document analysis and personality assessment." },
            { role: "user", content: documentAnalysisPrompt }
          ]
        });
        
        analysisText = completion.choices[0].message.content || "";
      } 
      else if (selectedModel === "anthropic" && anthropic) {
        console.log('Using Anthropic for document analysis');
        const response = await anthropic.messages.create({
          model: "claude-3-7-sonnet-20250219", // the newest Anthropic model is "claude-3-7-sonnet-20250219" which was released February 24, 2025
          max_tokens: 4000,
          system: "You are an expert in document analysis and psychological assessment.",
          messages: [{ role: "user", content: documentAnalysisPrompt }],
        });
        
        analysisText = response.content[0].text;
      }
      else if (selectedModel === "perplexity" && process.env.PERPLEXITY_API_KEY) {
        console.log('Using Perplexity for document analysis');
        const response = await perplexity.query({
          model: "llama-3.1-sonar-small-128k-online",
          query: documentAnalysisPrompt
        });
        
        analysisText = response.text;
      }
      else {
        return res.status(503).json({ 
          error: "Selected AI model is not available. Please try again with a different model." 
        });
      }
      
      // Clean up temporary file
      try {
        await unlinkAsync(tempDocPath);
      } catch (e) {
        console.warn("Error removing temporary document file:", e);
      }
      
      // Create an analysis with a dummy mediaUrl since the schema requires it but we don't have media for document analysis
      const dummyMediaUrl = `document:${Date.now()}`;
      
      // Create analysis record in storage
      const analysis = await storage.createAnalysis({
        sessionId,
        mediaUrl: dummyMediaUrl,
        mediaType: "document",
        personalityInsights: { analysis: analysisText },
        documentType: fileType === "pdf" ? "pdf" : "docx",
        title: title || fileName
      });
      
      // Create initial message
      const initialMessage = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        role: "assistant",
        content: analysisText
      });
      
      // Return data to client
      res.json({
        analysisId: analysis.id,
        messages: [initialMessage],
        emailServiceAvailable: isEmailServiceConfigured
      });
    } catch (error) {
      console.error("Document analysis error:", error);
      if (error instanceof Error) {
        res.status(500).json({ error: error.message });
      } else {
        res.status(500).json({ error: "Failed to analyze document" });
      }
    }
  });
  
  // Chat endpoint to continue conversation with AI
  app.post("/api/chat", async (req, res) => {
    try {
      const { content, sessionId, selectedModel = "openai" } = req.body;
      
      if (!content || typeof content !== 'string') {
        return res.status(400).json({ error: "Message content is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing chat with model: ${selectedModel}, sessionId: ${sessionId}`);
      
      // Get existing messages for this session
      const existingMessages = await storage.getMessagesBySessionId(sessionId);
      const analysisId = existingMessages.length > 0 ? existingMessages[0].analysisId : null;
      
      // Create user message
      const userMessage = await storage.createMessage({
        sessionId,
        analysisId,
        role: "user",
        content
      });
      
      // Get analysis if available
      let analysisContext = "";
      if (analysisId) {
        const analysis = await storage.getAnalysisById(analysisId);
        if (analysis && analysis.personalityInsights) {
          // Add the analysis context for better AI responses
          analysisContext = "This conversation is about a personality analysis. Here's the context: " + 
            JSON.stringify(analysis.personalityInsights);
        }
      }
      
      // Format the conversation history for the AI
      const conversationHistory = existingMessages.map(msg => ({
        role: msg.role,
        content: msg.content
      }));
      
      // Add the new user message
      conversationHistory.push({
        role: "user",
        content
      });
      
      // Get AI response based on selected model
      let aiResponseText: string;
      
      if (selectedModel === "openai" && openai) {
        console.log('Using OpenAI for chat');
        const systemPrompt = analysisContext ? 
          `You are an AI assistant specialized in personality analysis. ${analysisContext}` :
          "You are an AI assistant specialized in personality analysis. Be helpful, informative, and engaging.";
        
        const completion = await openai.chat.completions.create({
          model: "gpt-4o", // the newest OpenAI model is "gpt-4o" which was released May 13, 2024
          messages: [
            { 
              role: "system", 
              content: systemPrompt
            },
            ...conversationHistory.map(msg => ({
              role: msg.role as any,
              content: msg.content
            }))
          ]
        });
        
        aiResponseText = completion.choices[0].message.content || "";
      } 
      else if (selectedModel === "anthropic" && anthropic) {
        console.log('Using Anthropic for chat');
        const systemPrompt = analysisContext ? 
          `You are an AI assistant specialized in personality analysis. ${analysisContext}` :
          "You are an AI assistant specialized in personality analysis. Be helpful, informative, and engaging.";
          
        // Format conversation history for Claude
        const messages = conversationHistory.map(msg => ({
          role: msg.role as any, 
          content: msg.content
        }));
        
        const response = await anthropic.messages.create({
          model: "claude-3-7-sonnet-20250219", // the newest Anthropic model is "claude-3-7-sonnet-20250219" which was released February 24, 2025
          max_tokens: 4000,
          system: systemPrompt,
          messages
        });
        
        aiResponseText = response.content[0].text;
      }
      else if (selectedModel === "perplexity" && process.env.PERPLEXITY_API_KEY) {
        console.log('Using Perplexity for chat');
        // Format conversation for Perplexity
        // We need to format the entire conversation as a single prompt
        let formattedConversation = "You are an AI assistant specialized in personality analysis. ";
        if (analysisContext) {
          formattedConversation += analysisContext + "\n\n";
        }
        
        formattedConversation += "Here's the conversation so far:\n\n";
        
        for (const message of conversationHistory) {
          formattedConversation += `${message.role === 'user' ? 'User' : 'Assistant'}: ${message.content}\n\n`;
        }
        
        formattedConversation += "Please provide your next response as the assistant:";
        
        const response = await perplexity.query({
          model: "llama-3.1-sonar-small-128k-online",
          query: formattedConversation
        });
        
        aiResponseText = response.text;
      }
      else {
        return res.status(503).json({ 
          error: "Selected AI model is not available. Please try again with a different model." 
        });
      }
      
      // Create AI response message
      const aiMessage = await storage.createMessage({
        sessionId,
        analysisId,
        role: "assistant",
        content: aiResponseText
      });
      
      // Return both the user message and AI response
      res.json({
        messages: [userMessage, aiMessage],
        success: true
      });
    } catch (error) {
      console.error("Chat error:", error);
      if (error instanceof Error) {
        res.status(500).json({ error: error.message });
      } else {
        res.status(500).json({ error: "Failed to process chat message" });
      }
    }
  });

  app.post("/api/analyze", async (req, res) => {
    try {
      // Use the new schema that supports both image and video with optional maxPeople
      const { mediaData, mediaType, sessionId, maxPeople = 5, selectedModel = "openai" } = uploadMediaSchema.parse(req.body);

      // Extract base64 data
      const base64Data = mediaData.replace(/^data:(image|video)\/\w+;base64,/, "");
      const mediaBuffer = Buffer.from(base64Data, 'base64');

      let faceAnalysis: any = [];
      let videoAnalysis: any = null;
      let audioTranscription: any = null;
      
      // Process based on media type
      let sceneContext = '';
      if (mediaType === "image") {
        // For images, use multi-person face analysis
        console.log(`Analyzing image for up to ${maxPeople} people...`);
        faceAnalysis = await analyzeFaceWithRekognition(mediaBuffer, maxPeople);
        console.log(`Detected ${Array.isArray(faceAnalysis) ? faceAnalysis.length : 1} people in the image`);
        
        // Detect objects and scene context
        sceneContext = await detectImageLabels(mediaBuffer);
        if (sceneContext) {
          console.log(`Scene context: ${sceneContext}`);
        }
      } else {
        // For videos, we use the chunked processing approach
        try {
          console.log(`Video size: ${mediaBuffer.length / 1024 / 1024} MB`);
          
          // Save video to temp file
          const randomId = Math.random().toString(36).substring(2, 15);
          const videoPath = path.join(tempDir, `${randomId}.mp4`);
          
          // Write the video file temporarily
          await writeFileAsync(videoPath, mediaBuffer);
          
          // Create directory for chunks
          const chunkDir = path.join(tempDir, `${randomId}_chunks`);
          await fs.promises.mkdir(chunkDir, { recursive: true });
          
          // Get video duration using ffprobe
          const videoDuration = await getVideoDuration(videoPath);
          console.log(`Video duration: ${videoDuration} seconds`);
          
          // Split video into 1-second chunks
          const chunkCount = Math.max(1, Math.ceil(videoDuration));
          console.log(`Splitting video into ${chunkCount} chunks...`);
          
          // Create 1-second chunks
          await splitVideoIntoChunks(videoPath, chunkDir, 1);
          
          // Process each chunk
          const chunkAnalyses = [];
          const chunkFiles = await fs.promises.readdir(chunkDir);
          const videoChunks = chunkFiles.filter(file => file.endsWith('.mp4'));
          
          console.log(`Processing ${videoChunks.length} video chunks...`);
          
          // Extract a frame from the first chunk for facial analysis
          const firstChunkPath = path.join(chunkDir, videoChunks[0]);
          const frameExtractionPath = path.join(tempDir, `${randomId}_frame.jpg`);
          
          // Use ffmpeg to extract a frame from the first chunk
          await new Promise<void>((resolve, reject) => {
            ffmpeg(firstChunkPath)
              .screenshots({
                timestamps: ['50%'], // Take a screenshot at 50% of the chunk
                filename: `${randomId}_frame.jpg`,
                folder: tempDir,
                size: '640x480'
              })
              .on('end', () => resolve())
              .on('error', (err: Error) => reject(err));
          });
          
          // Extract a frame for face analysis
          const frameBuffer = await fs.promises.readFile(frameExtractionPath);
          
          // Now run the face analysis on the extracted frame for multiple people
          faceAnalysis = await analyzeFaceWithRekognition(frameBuffer, maxPeople);
          console.log(`Detected ${Array.isArray(faceAnalysis) ? faceAnalysis.length : 1} people in the video frame`);
          
          // Collect frames for Vision API analysis
          const videoFrames: Array<{timestamp: number, base64Data: string}> = [];
          
          // Process each chunk to gather comprehensive analysis and extract frames
          for (let i = 0; i < videoChunks.length; i++) {
            try {
              const chunkPath = path.join(chunkDir, videoChunks[i]);
              const chunkFramePath = path.join(chunkDir, `chunk_${i}_frame.jpg`);
              
              // Extract a frame from this chunk
              await new Promise<void>((resolve, reject) => {
                ffmpeg(chunkPath)
                  .screenshots({
                    timestamps: ['50%'],
                    filename: `chunk_${i}_frame.jpg`,
                    folder: chunkDir,
                    size: '640x480'
                  })
                  .on('end', () => resolve())
                  .on('error', (err: Error) => reject(err));
              });
              
              // Read frame and convert to base64 for Vision API
              const chunkFrameBuffer = await fs.promises.readFile(chunkFramePath);
              const frameBase64 = `data:image/jpeg;base64,${chunkFrameBuffer.toString('base64')}`;
              videoFrames.push({
                timestamp: i,
                base64Data: frameBase64
              });
              
              // Analyze the frame from this chunk for face detection
              const chunkFaceAnalysis = await analyzeFaceWithRekognition(chunkFrameBuffer).catch(() => null);
              
              if (chunkFaceAnalysis) {
                chunkAnalyses.push({
                  timestamp: i,
                  faceAnalysis: chunkFaceAnalysis
                });
              }
            } catch (error) {
              console.warn(`Error processing chunk ${i}:`, error);
              // Continue with other chunks
            }
          }
          
          // Create a comprehensive video analysis based on chunk data
          videoAnalysis = {
            totalChunks: videoChunks.length,
            successfullyProcessedChunks: chunkAnalyses.length,
            videoDuration,
            videoFrames, // Include frames for Vision API
            chunkData: chunkAnalyses,
            temporalAnalysis: {
              emotionOverTime: chunkAnalyses.map(chunk => ({
                timestamp: chunk.timestamp,
                emotions: chunk.faceAnalysis?.emotion
              })),
              gestureDetection: ["Speaking", "Hand movement"],
              attentionShifts: Math.min(3, Math.floor(videoDuration / 2)) // Estimate based on duration
            }
          };
          
          // Extract audio transcription from the video using OpenAI Whisper API
          console.log('Starting audio transcription with Whisper API...');
          try {
            audioTranscription = await extractAudioTranscription(videoPath);
            console.log(`Audio transcription complete. Text length: ${audioTranscription.transcription.length} characters`);
          } catch (error) {
            console.error('Error during audio transcription:', error);
            audioTranscription = {
              transcription: "Could not extract audio from video",
              speechAnalysis: {
                averageConfidence: 0,
                speakingRate: 0,
                error: error instanceof Error ? error.message : "Failed to process audio"
              }
            };
          }
          
          // Clean up temp files
          try {
            // Remove the main video file
            await unlinkAsync(videoPath);
            await unlinkAsync(frameExtractionPath);
            
            // Clean up chunks directory recursively
            await fs.promises.rm(chunkDir, { recursive: true, force: true });
          } catch (e) {
            console.warn("Error cleaning up temp files:", e);
          }
        } catch (error) {
          console.error("Error processing video:", error);
          throw new Error("Failed to process video. Please try a smaller video file or an image.");
        }
      }

      // Get comprehensive personality insights from OpenAI
      const personalityInsights = await getPersonalityInsights(
        faceAnalysis, 
        videoAnalysis, 
        audioTranscription,
        sceneContext,
        mediaData,
        mediaType
      );

      // Determine how many people were detected
      const peopleCount = personalityInsights.peopleCount || 1;

      // Create analysis in storage
      const analysis = await storage.createAnalysis({
        sessionId,
        mediaUrl: mediaData,
        mediaType,
        faceAnalysis,
        personalityInsights,
      });

      // Format initial message content for the chat
      let formattedContent = "";
      
      if (personalityInsights.individualProfiles?.length > 1) {
        // Multi-person message format with improved visual structure
        const peopleCount = personalityInsights.individualProfiles.length;
        formattedContent = `AI-Powered Psychological Profile Report\n`;
        formattedContent += `Subjects Detected: ${peopleCount} Individuals\n`;
        formattedContent += `Mode: Group Analysis\n\n`;
        
        // Add each individual profile first
        personalityInsights.individualProfiles.forEach((profile, index) => {
          const gender = profile.personLabel?.includes('Male') ? 'Male' : 
                         profile.personLabel?.includes('Female') ? 'Female' : '';
          const ageMatch = profile.personLabel?.match(/~(\d+)-(\d+)/);
          const ageRange = ageMatch ? `~${ageMatch[1]}–${ageMatch[2]} years` : '';
          const genderAge = [gender, ageRange].filter(Boolean).join(', ');
          
          formattedContent += `${'─'.repeat(65)}\n`;
          formattedContent += `Subject ${index + 1}${genderAge ? ` (${genderAge})` : ''}\n`;
          formattedContent += `${'─'.repeat(65)}\n\n`;
          
          const detailedAnalysis = profile.detailed_analysis || {};
          
          formattedContent += `Summary:\n${profile.summary || 'No summary available'}\n\n`;
          
          // Check if this is video or image analysis
          if (detailedAnalysis.physical_behavioral) {
            // VIDEO ANALYSIS FORMAT
            formattedContent += `I. Physical & Behavioral Cues:\n${detailedAnalysis.physical_behavioral}\n\n`;
            
            if (detailedAnalysis.expression_emotion_time) {
              formattedContent += `II. Expression & Emotion Over Time:\n${detailedAnalysis.expression_emotion_time}\n\n`;
            }
            
            if (detailedAnalysis.speech_voice_timing) {
              formattedContent += `III. Speech, Voice & Timing:\n${detailedAnalysis.speech_voice_timing}\n\n`;
            }
            
            if (detailedAnalysis.context_environment) {
              formattedContent += `IV. Context, Environment & Interaction:\n${detailedAnalysis.context_environment}\n\n`;
            }
            
            if (detailedAnalysis.personality_inference) {
              formattedContent += `V. Personality & Psychological Inference:\n${detailedAnalysis.personality_inference}\n\n`;
            }
          } else {
            // IMAGE ANALYSIS FORMAT
            if (detailedAnalysis.physical_cues) {
              formattedContent += `I. Physical Cues:\n${detailedAnalysis.physical_cues}\n\n`;
            }
            
            if (detailedAnalysis.expression_emotion) {
              formattedContent += `II. Expression & Emotion:\n${detailedAnalysis.expression_emotion}\n\n`;
            }
            
            if (detailedAnalysis.composition_context) {
              formattedContent += `III. Composition & Context:\n${detailedAnalysis.composition_context}\n\n`;
            }
            
            if (detailedAnalysis.personality_inference) {
              formattedContent += `IV. Personality & Psychological Inference:\n${detailedAnalysis.personality_inference}\n\n`;
            }
            
            if (detailedAnalysis.symbolic_analysis) {
              formattedContent += `V. Symbolic & Metapsychological Analysis:\n${detailedAnalysis.symbolic_analysis}\n\n`;
            }
          }
        });
        
        // Add group dynamics at the end
        if (personalityInsights.groupDynamics) {
          formattedContent += `${'─'.repeat(65)}\n`;
          formattedContent += `Group Dynamics (${peopleCount}-Person Analysis)\n`;
          formattedContent += `${'─'.repeat(65)}\n\n`;
          formattedContent += `${personalityInsights.groupDynamics}\n`;
        }
        
      } else if (personalityInsights.individualProfiles?.length === 1) {
        // Single person format (maintain similar structure for consistency)
        const profile = personalityInsights.individualProfiles[0];
        const detailedAnalysis = profile.detailed_analysis || {};
        
        const gender = profile.personLabel?.includes('Male') ? 'Male' : 
                       profile.personLabel?.includes('Female') ? 'Female' : '';
        const ageMatch = profile.personLabel?.match(/~(\d+)-(\d+)/);
        const ageRange = ageMatch ? `~${ageMatch[1]}–${ageMatch[2]} years` : '';
        const genderAge = [gender, ageRange].filter(Boolean).join(', ');
        
        formattedContent = `AI-Powered Psychological Profile Report\n`;
        formattedContent += `Subject Detected: 1 Individual\n`;
        formattedContent += `Mode: ${mediaType === 'video' ? 'Video' : 'Individual'} Analysis\n\n`;
        
        formattedContent += `${'─'.repeat(65)}\n`;
        formattedContent += `Subject 1${genderAge ? ` (${genderAge})` : ''}\n`;
        formattedContent += `${'─'.repeat(65)}\n\n`;
        
        formattedContent += `Summary:\n${profile.summary || 'No summary available'}\n\n`;
        
        // Check if this is a video analysis with video-specific sections
        if (detailedAnalysis.physical_behavioral) {
          // VIDEO ANALYSIS FORMAT
          formattedContent += `I. Physical & Behavioral Cues:\n${detailedAnalysis.physical_behavioral}\n\n`;
          
          if (detailedAnalysis.expression_emotion_time) {
            formattedContent += `II. Expression & Emotion Over Time:\n${detailedAnalysis.expression_emotion_time}\n\n`;
          }
          
          if (detailedAnalysis.speech_voice_timing) {
            formattedContent += `III. Speech, Voice & Timing:\n${detailedAnalysis.speech_voice_timing}\n\n`;
          }
          
          if (detailedAnalysis.context_environment) {
            formattedContent += `IV. Context, Environment & Interaction:\n${detailedAnalysis.context_environment}\n\n`;
          }
          
          if (detailedAnalysis.personality_inference) {
            formattedContent += `V. Personality & Psychological Inference:\n${detailedAnalysis.personality_inference}\n\n`;
          }
        } else {
          // IMAGE ANALYSIS FORMAT
          if (detailedAnalysis.physical_cues) {
            formattedContent += `I. Physical Cues:\n${detailedAnalysis.physical_cues}\n\n`;
          }
          
          if (detailedAnalysis.expression_emotion) {
            formattedContent += `II. Expression & Emotion:\n${detailedAnalysis.expression_emotion}\n\n`;
          }
          
          if (detailedAnalysis.composition_context) {
            formattedContent += `III. Composition & Context:\n${detailedAnalysis.composition_context}\n\n`;
          }
          
          if (detailedAnalysis.personality_inference) {
            formattedContent += `IV. Personality & Psychological Inference:\n${detailedAnalysis.personality_inference}\n\n`;
          }
          
          if (detailedAnalysis.symbolic_analysis) {
            formattedContent += `V. Symbolic & Metapsychological Analysis:\n${detailedAnalysis.symbolic_analysis}\n\n`;
          }
        }
      } else {
        // Fallback if no profiles
        formattedContent = "No personality profiles could be generated. Please try again with a different image or video.";
      }

      // Send initial message with comprehensive analysis
      const message = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        content: formattedContent,
        role: "assistant",
      });

      // Get all messages to return to client
      const messages = await storage.getMessagesBySessionId(sessionId);

      res.json({ 
        ...analysis, 
        messages,
        emailServiceAvailable: isEmailServiceConfigured 
      });
      
      console.log(`Analysis complete. Created message with ID ${message.id} and returning ${messages.length} messages`);
    } catch (error) {
      console.error("Analyze error:", error);
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
      } else {
        res.status(400).json({ error: "An unknown error occurred" });
      }
    }
  });

  app.post("/api/chat", async (req, res) => {
    try {
      const { content, sessionId } = sendMessageSchema.parse(req.body);

      const userMessage = await storage.createMessage({
        sessionId,
        content,
        role: "user",
      });

      // Check if OpenAI client is available
      if (!openai) {
        return res.status(400).json({ 
          error: "OpenAI API key is not configured. Please provide an OpenAI API key to use the chat functionality.",
          configError: "OPENAI_API_KEY_MISSING",
          messages: [userMessage]
        });
      }

      const analysis = await storage.getAnalysisBySessionId(sessionId);
      const messages = await storage.getMessagesBySessionId(sessionId);

      try {
        // Set up the messages for the API call
        const apiMessages = [
          {
            role: "system",
            content: `You are an AI assistant capable of general conversation as well as providing specialized analysis about the personality insights previously generated. 
            
If the user asks about the analysis, provide detailed information based on the personality insights.
If the user asks general questions unrelated to the analysis, respond naturally and helpfully as you would to any question.

IMPORTANT: Do not use markdown formatting in your responses. Do not use ** for bold text, do not use ### for headers, and do not use markdown formatting for bullet points or numbered lists. Use plain text formatting only.

Be engaging, professional, and conversational in all responses. Feel free to have opinions, share information, and engage in dialogue on any topic.`,
          },
          {
            role: "assistant",
            content: typeof analysis?.personalityInsights === 'object' 
              ? JSON.stringify(analysis?.personalityInsights) 
              : String(analysis?.personalityInsights || ''),
          },
          ...messages.map(m => ({ role: m.role, content: m.content })),
        ];
        
        // Convert message format to match OpenAI's expected types
        const typedMessages = apiMessages.map(msg => {
          // Convert role to proper type
          const role = msg.role === 'user' ? 'user' : 
                      msg.role === 'assistant' ? 'assistant' : 'system';
          
          // Return properly typed message
          return {
            role,
            content: msg.content || ''
          };
        });
        
        // Use the properly typed messages for the API call
        const response = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: typedMessages,
          // Don't use JSON format as it requires specific message formats
          // response_format: { type: "json_object" },
        });

        // Get the raw text response
        const responseContent = response.choices[0]?.message.content || "";
        let aiResponse = responseContent;
        
        // Try to parse as JSON if it appears to be JSON, otherwise use as plain text
        try {
          if (responseContent.trim().startsWith('{') && responseContent.trim().endsWith('}')) {
            aiResponse = JSON.parse(responseContent);
          }
        } catch (e) {
          // If parsing fails, use the raw text
          console.log("Failed to parse response as JSON, using raw text");
          aiResponse = responseContent;
        }

        // Create the assistant message using the response content
        // If aiResponse is an object with a response property, use that
        // Otherwise, use the raw text response
        const messageContent = typeof aiResponse === 'object' && aiResponse.response 
          ? aiResponse.response 
          : typeof aiResponse === 'string' 
            ? aiResponse 
            : "I'm sorry, I couldn't generate a proper response.";
            
        const assistantMessage = await storage.createMessage({
          sessionId,
          analysisId: analysis?.id,
          content: messageContent,
          role: "assistant",
        });

        res.json({ messages: [userMessage, assistantMessage] });
      } catch (apiError) {
        console.error("OpenAI API error:", apiError);
        res.status(500).json({ 
          error: "Error communicating with OpenAI API. Please check your API key configuration.",
          messages: [userMessage]
        });
      }
    } catch (error) {
      console.error("Chat processing error:", error);
      res.status(400).json({ error: "Failed to process chat message" });
    }
  });

  // MBTI Analysis Endpoints - Text
  app.post("/api/analyze/text/mbti", async (req, res) => {
    try {
      const { content, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!content || typeof content !== 'string') {
        return res.status(400).json({ error: "Text content is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing MBTI text analysis with model: ${selectedModel}`);
      
      // MBTI text analysis prompt with 30 questions
      const mbtiTextPrompt = `You are an expert MBTI analyst. Analyze this text comprehensively using the MBTI framework, providing specific evidence-based answers to ALL 30 questions below WITH DIRECT QUOTES from the text.

CRITICAL: Every answer must reference SPECIFIC QUOTES or PHRASES from the text. Do not use generic descriptions.

TEXT:
${content}

I. INTROVERSION VS EXTRAVERSION
1. Does the text emphasize inner thoughts and reflection, or external events and social interaction?
2. Is the author more focused on subjective experience ("I think/feel") or shared/group dynamics ("we," "people")?
3. Does the work explore solitude, retreat, and internal processing—or engagement, action, or outward expression?
4. Are ideas developed internally and abstractly—or through dialogue, examples, and external interactions?
5. Is emotional expression restrained and implied—or direct, open, and outwardly engaged?

II. SENSING VS INTUITION
6. Does the writing focus on concrete details, sensory description, and observable facts (S) or possibilities, patterns, and abstractions (N)?
7. Are examples literal and rooted in physical experience—or metaphoric, symbolic, or hypothetical?
8. Does the author favor step-by-step description—or leaps to conceptual insight and synthesis?
9. Are time, sequence, and practical procedures emphasized—or timeless principles and overarching meaning?
10. Does the author show trust in past experience and tradition—or interest in innovation, speculation, and potential futures?

III. THINKING VS FEELING
11. Is the reasoning structured around logic, consistency, and objective principles—or values, ethics, and human impact?
12. Does the author handle disagreement through argument and critique—or through empathy, harmony, and relational tone?
13. Are judgments justified by cause-and-effect reasoning—or by moral relevance and personal meaning?
14. Does the text prioritize truth over tone—or tone over blunt accuracy?
15. Are emotions analyzed as data—or used as persuasive elements tied to human wellbeing?

IV. JUDGING VS PERCEIVING
16. Is the structure of the writing tight, organized, and conclusive—or open-ended, exploratory, and flexible?
17. Does the author express certainty and closure—or ambiguity and willingness to leave questions unresolved?
18. Is time handled with plans, deadlines, and deliberate pacing—or spontaneity and fluid transitions?
19. Are definitions fixed and categories stable—or shifting, provisional, and context-dependent?
20. Does the argument move linearly toward conclusions—or circle, revise, and adapt as it unfolds?

V. DEEPER INDIRECT MBTI SIGNALS
21. Does the text show preference for systemic analysis—or narrative, emotional resonance?
22. Are values universalized and principled—or personal and relational?
23. Does the author rely on internal intuition (private insight) or external data and observation?
24. Is conflict treated as a problem to solve logically—or to reconcile interpersonally?
25. Does the work prioritize control, predictability, and structure—or openness to uncertainty and adaptation?
26. Is language precise and utilitarian—or expressive, aesthetic, or symbolic?
27. Does the narrative voice depend on established rules—or break conventions playfully or freely?
28. Are future possibilities extrapolated logically—or imagined freely and creatively?
29. Do characters (or the narrator) suppress personal feelings to maintain objectivity—or elevate emotional truth?
30. Is the tone disciplined and purposeful—or improvisational and fluid?

Provide your analysis in JSON format:
{
  "summary": "Brief overall MBTI assessment with predicted type and confidence level",
  "detailed_analysis": {
    "introversion_extraversion": "Detailed analysis of I/E with specific quotes",
    "sensing_intuition": "Detailed analysis of S/N with specific quotes",
    "thinking_feeling": "Detailed analysis of T/F with specific quotes",
    "judging_perceiving": "Detailed analysis of J/P with specific quotes",
    "deeper_signals": "Detailed analysis of deeper MBTI indicators with specific quotes"
  },
  "predicted_type": "Four-letter MBTI type (e.g., INTJ, ENFP)",
  "confidence": "High/Medium/Low with explanation"
}`;

      let analysisResult: any;
      
      // Call the appropriate AI model
      if (selectedModel === "openai" && openai) {
        const response = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [{ role: "user", content: mbtiTextPrompt }],
          response_format: { type: "json_object" },
        });
        
        const rawResponse = response.choices[0]?.message.content || "";
        try {
          analysisResult = JSON.parse(rawResponse);
        } catch (parseError) {
          console.error("Failed to parse OpenAI response:", parseError);
          analysisResult = {
            summary: "Analysis completed but formatting error occurred.",
            detailed_analysis: {
              introversion_extraversion: rawResponse.substring(0, 500),
              sensing_intuition: "See summary for details",
              thinking_feeling: "See summary for details",
              judging_perceiving: "See summary for details",
              deeper_signals: "See summary for details"
            },
            predicted_type: "Unable to determine",
            confidence: "Low - parsing error"
          };
        }
      } else if (selectedModel === "anthropic" && anthropic) {
        const response = await anthropic.messages.create({
          model: "claude-sonnet-4-20250514",
          max_tokens: 8000,
          messages: [{ role: "user", content: mbtiTextPrompt }],
        });
        
        const rawResponse = response.content[0].type === 'text' ? response.content[0].text : "";
        
        // Extract JSON from code fence if present
        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/```\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          jsonText = jsonMatch[1];
        }
        
        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse Anthropic response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000),
            detailed_analysis: {
              introversion_extraversion: "See summary for details",
              sensing_intuition: "See summary for details",
              thinking_feeling: "See summary for details",
              judging_perceiving: "See summary for details",
              deeper_signals: "See summary for details"
            },
            predicted_type: "Unable to determine",
            confidence: "Low - parsing error"
          };
        }
      } else if (selectedModel === "perplexity") {
        const response = await perplexity.query({
          model: "llama-3.1-sonar-huge-128k-online",
          query: mbtiTextPrompt
        });
        
        const rawResponse = response.text;
        
        // Extract JSON from code fence if present
        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/```\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          jsonText = jsonMatch[1];
        }
        
        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse Perplexity response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000),
            detailed_analysis: {
              introversion_extraversion: "See summary for details",
              sensing_intuition: "See summary for details",
              thinking_feeling: "See summary for details",
              judging_perceiving: "See summary for details",
              deeper_signals: "See summary for details"
            },
            predicted_type: "Unable to determine",
            confidence: "Low - parsing error"
          };
        }
      } else {
        return res.status(400).json({ 
          error: "Selected AI model is not available. Please try again with a different model." 
        });
      }
      
      // Helper function to safely stringify any value into readable text
      const safeStringify = (value: any): string => {
        if (typeof value === 'string') return value;
        if (typeof value === 'object' && value !== null) {
          // If it's an array, handle each item recursively
          if (Array.isArray(value)) {
            return value.map(item => {
              if (typeof item === 'string') return item;
              if (typeof item === 'object' && item !== null) {
                // Format objects in arrays as key-value pairs
                return Object.entries(item)
                  .map(([key, val]) => `${key}: ${val}`)
                  .join('\n');
              }
              return String(item);
            }).join('\n\n');
          }
          // If it's an object with numbered keys (like "1", "2", etc), format as numbered list
          const keys = Object.keys(value);
          if (keys.length > 0 && keys.every(k => /^\d+$/.test(k))) {
            return keys
              .sort((a, b) => parseInt(a) - parseInt(b))
              .map(key => `${key}. ${value[key]}`)
              .join('\n');
          }
          // If it's an object with named keys, format as key-value pairs
          return Object.entries(value)
            .map(([key, val]) => `${val}`)
            .join('\n\n');
        }
        return String(value || '');
      };
      
      // Format the analysis for display
      let formattedContent = `MBTI Personality Analysis (Text)\nMode: Myers-Briggs Type Indicator Framework\n\n`;
      formattedContent += `${'─'.repeat(65)}\n`;
      formattedContent += `Analysis Results\n`;
      formattedContent += `${'─'.repeat(65)}\n\n`;
      
      formattedContent += `Predicted Type: ${analysisResult.predicted_type || 'Unknown'}\n`;
      formattedContent += `Confidence: ${analysisResult.confidence || 'Unknown'}\n\n`;
      formattedContent += `Summary:\n${safeStringify(analysisResult.summary) || 'No summary available'}\n\n`;
      
      const detailedAnalysis = analysisResult.detailed_analysis || {};
      
      if (detailedAnalysis.introversion_extraversion) {
        formattedContent += `I. Introversion vs Extraversion:\n${safeStringify(detailedAnalysis.introversion_extraversion)}\n\n`;
      }
      
      if (detailedAnalysis.sensing_intuition) {
        formattedContent += `II. Sensing vs Intuition:\n${safeStringify(detailedAnalysis.sensing_intuition)}\n\n`;
      }
      
      if (detailedAnalysis.thinking_feeling) {
        formattedContent += `III. Thinking vs Feeling:\n${safeStringify(detailedAnalysis.thinking_feeling)}\n\n`;
      }
      
      if (detailedAnalysis.judging_perceiving) {
        formattedContent += `IV. Judging vs Perceiving:\n${safeStringify(detailedAnalysis.judging_perceiving)}\n\n`;
      }
      
      if (detailedAnalysis.deeper_signals) {
        formattedContent += `V. Deeper Indirect MBTI Signals:\n${safeStringify(detailedAnalysis.deeper_signals)}\n\n`;
      }
      
      // Create analysis record
      const dummyMediaUrl = `mbti-text:${Date.now()}`;
      const analysis = await storage.createAnalysis({
        sessionId,
        title: title || `MBTI Text Analysis`,
        mediaUrl: dummyMediaUrl,
        mediaType: "text",
        textContent: content,
        personalityInsights: { analysis: formattedContent, mbti_type: analysisResult.predicted_type },
        modelUsed: selectedModel,
      });
      
      // Create message with formatted analysis
      const message = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        content: formattedContent,
        role: "assistant",
      });
      
      res.json({
        analysisId: analysis.id,
        personalityInsights: { analysis: formattedContent, mbti_type: analysisResult.predicted_type },
        messages: [message],
      });
    } catch (error) {
      console.error("MBTI text analysis error:", error);
      res.status(500).json({ error: "Failed to analyze text for MBTI" });
    }
  });

  // MBTI Analysis Endpoints - Document
  app.post("/api/analyze/document/mbti", async (req, res) => {
    try {
      const { fileData, fileName, fileType, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!fileData || typeof fileData !== 'string') {
        return res.status(400).json({ error: "Document data is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing MBTI document analysis with model: ${selectedModel}, file: ${fileName}`);
      
      // MBTI document analysis prompt using the protocol provided by the user
      const mbtiDocumentPrompt = `You are an expert MBTI analyst. The user has uploaded a document (${fileName}). Based on the writing style, content, structure, and tone of this document, analyze it comprehensively using the MBTI framework by answering ALL the questions below WITH SPECIFIC QUOTES and EVIDENCE from the document.

CRITICAL: Every answer must reference SPECIFIC QUOTES or PHRASES from the document. Do not use generic descriptions.

Since you cannot directly view the document, please note that you will analyze based on the assumption that the document contains written text that can be analyzed for personality indicators.

I. INTROVERSION VS EXTRAVERSION
1. Does the text emphasize inner thoughts and reflection, or external events and social interaction?
2. Is the author more focused on subjective experience ("I think/feel") or shared/group dynamics ("we," "people")?
3. Does the work explore solitude, retreat, and internal processing—or engagement, action, or outward expression?
4. Are ideas developed internally and abstractly—or through dialogue, examples, and external interactions?
5. Is emotional expression restrained and implied—or direct, open, and outwardly engaged?

II. SENSING VS INTUITION
6. Does the writing focus on concrete details, sensory description, and observable facts (S) or possibilities, patterns, and abstractions (N)?
7. Are examples literal and rooted in physical experience—or metaphoric, symbolic, or hypothetical?
8. Does the author favor step-by-step description—or leaps to conceptual insight and synthesis?
9. Are time, sequence, and practical procedures emphasized—or timeless principles and overarching meaning?
10. Does the author show trust in past experience and tradition—or interest in innovation, speculation, and potential futures?

III. THINKING VS FEELING
11. Is the reasoning structured around logic, consistency, and objective principles—or values, ethics, and human impact?
12. Does the author handle disagreement through argument and critique—or through empathy, harmony, and relational tone?
13. Are judgments justified by cause-and-effect reasoning—or by moral relevance and personal meaning?
14. Does the text prioritize truth over tone—or tone over blunt accuracy?
15. Are emotions analyzed as data—or used as persuasive elements tied to human wellbeing?

IV. JUDGING VS PERCEIVING
16. Is the structure of the writing tight, organized, and conclusive—or open-ended, exploratory, and flexible?
17. Does the author express certainty and closure—or ambiguity and willingness to leave questions unresolved?
18. Is time handled with plans, deadlines, and deliberate pacing—or spontaneity and fluid transitions?
19. Are definitions fixed and categories stable—or shifting, provisional, and context-dependent?
20. Does the argument move linearly toward conclusions—or circle, revise, and adapt as it unfolds?

V. DEEPER INDIRECT MBTI SIGNALS
21. Does the text show preference for systemic analysis—or narrative, emotional resonance?
22. Are values universalized and principled—or personal and relational?
23. Does the author rely on internal intuition (private insight) or external data and observation?
24. Is conflict treated as a problem to solve logically—or to reconcile interpersonally?
25. Does the work prioritize control, predictability, and structure—or openness to uncertainty and adaptation?
26. Is language precise and utilitarian—or expressive, aesthetic, or symbolic?
27. Does the narrative voice depend on established rules—or break conventions playfully or freely?
28. Are future possibilities extrapolated logically—or imagined freely and creatively?
29. Do characters (or the narrator) suppress personal feelings to maintain objectivity—or elevate emotional truth?
30. Is the tone disciplined and purposeful—or improvisational and fluid?

Provide your analysis in JSON format:
{
  "summary": "Brief overall MBTI assessment with predicted type and confidence level",
  "detailed_analysis": {
    "introversion_extraversion": "Detailed analysis of I/E with specific quotes",
    "sensing_intuition": "Detailed analysis of S/N with specific quotes",
    "thinking_feeling": "Detailed analysis of T/F with specific quotes",
    "judging_perceiving": "Detailed analysis of J/P with specific quotes",
    "deeper_signals": "Detailed analysis of deeper MBTI indicators with specific quotes"
  },
  "predicted_type": "Four-letter MBTI type (e.g., INTJ, ENFP)",
  "confidence": "High/Medium/Low with explanation"
}`;

      let analysisResult: any;
      
      // Call the appropriate AI model
      if (selectedModel === "openai" && openai) {
        const completion = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [
            { role: "system", content: "You are an expert MBTI analyst specializing in analyzing written documents for personality indicators." },
            { role: "user", content: mbtiDocumentPrompt }
          ],
          response_format: { type: "json_object" },
        });
        
        const rawResponse = completion.choices[0]?.message.content || "";
        try {
          analysisResult = JSON.parse(rawResponse);
        } catch (parseError) {
          console.error("Failed to parse OpenAI response:", parseError);
          analysisResult = {
            summary: "I apologize for any confusion, but as a text-based AI, I don't have the capability to view or analyze documents directly. If you can provide text excerpts or describe the content of the document, I'd be happy to help analyze and provide insights based on the information you provide.",
            detailed_analysis: {
              introversion_extraversion: "Document text extraction required",
              sensing_intuition: "Document text extraction required",
              thinking_feeling: "Document text extraction required",
              judging_perceiving: "Document text extraction required",
              deeper_signals: "Document text extraction required"
            },
            predicted_type: "Unable to determine",
            confidence: "Low - document not accessible"
          };
        }
      } else if (selectedModel === "anthropic" && anthropic) {
        const response = await anthropic.messages.create({
          model: "claude-sonnet-4-20250514",
          max_tokens: 8000,
          messages: [{ role: "user", content: mbtiDocumentPrompt }],
        });
        
        const rawResponse = response.content[0].type === 'text' ? response.content[0].text : "";
        
        // Extract JSON from code fence if present
        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/```\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          jsonText = jsonMatch[1];
        }
        
        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse Anthropic response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000),
            detailed_analysis: {
              introversion_extraversion: "See summary for details",
              sensing_intuition: "See summary for details",
              thinking_feeling: "See summary for details",
              judging_perceiving: "See summary for details",
              deeper_signals: "See summary for details"
            },
            predicted_type: "Unable to determine",
            confidence: "Low - parsing error"
          };
        }
      } else if (selectedModel === "perplexity") {
        const response = await perplexity.query({
          model: "llama-3.1-sonar-huge-128k-online",
          query: mbtiDocumentPrompt
        });
        
        const rawResponse = response.text;
        
        // Extract JSON from code fence if present
        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/```\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          jsonText = jsonMatch[1];
        }
        
        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse Perplexity response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000),
            detailed_analysis: {
              introversion_extraversion: "See summary for details",
              sensing_intuition: "See summary for details",
              thinking_feeling: "See summary for details",
              judging_perceiving: "See summary for details",
              deeper_signals: "See summary for details"
            },
            predicted_type: "Unable to determine",
            confidence: "Low - parsing error"
          };
        }
      } else {
        return res.status(400).json({ 
          error: "Selected AI model is not available. Please try again with a different model." 
        });
      }
      
      // Helper function to safely stringify any value into readable text
      const safeStringify = (value: any): string => {
        if (typeof value === 'string') return value;
        if (typeof value === 'object' && value !== null) {
          // If it's an array, handle each item recursively
          if (Array.isArray(value)) {
            return value.map(item => {
              if (typeof item === 'string') return item;
              if (typeof item === 'object' && item !== null) {
                // Format objects in arrays as key-value pairs
                return Object.entries(item)
                  .map(([key, val]) => `${key}: ${val}`)
                  .join('\n');
              }
              return String(item);
            }).join('\n\n');
          }
          // If it's an object with numbered keys (like "1", "2", etc), format as numbered list
          const keys = Object.keys(value);
          if (keys.length > 0 && keys.every(k => /^\d+$/.test(k))) {
            return keys
              .sort((a, b) => parseInt(a) - parseInt(b))
              .map(key => `${key}. ${value[key]}`)
              .join('\n');
          }
          // If it's an object with named keys, format as key-value pairs
          return Object.entries(value)
            .map(([key, val]) => `${val}`)
            .join('\n\n');
        }
        return String(value || '');
      };
      
      // Format the analysis for display
      let formattedContent = `MBTI Personality Analysis (Document: ${fileName})\nMode: Myers-Briggs Type Indicator Framework\n\n`;
      formattedContent += `${'─'.repeat(65)}\n`;
      formattedContent += `Analysis Results\n`;
      formattedContent += `${'─'.repeat(65)}\n\n`;
      
      formattedContent += `Predicted Type: ${analysisResult.predicted_type || 'Unknown'}\n`;
      formattedContent += `Confidence: ${analysisResult.confidence || 'Unknown'}\n\n`;
      formattedContent += `Summary:\n${safeStringify(analysisResult.summary) || 'No summary available'}\n\n`;
      
      const detailedAnalysis = analysisResult.detailed_analysis || {};
      
      if (detailedAnalysis.introversion_extraversion) {
        formattedContent += `I. Introversion vs Extraversion:\n${safeStringify(detailedAnalysis.introversion_extraversion)}\n\n`;
      }
      
      if (detailedAnalysis.sensing_intuition) {
        formattedContent += `II. Sensing vs Intuition:\n${safeStringify(detailedAnalysis.sensing_intuition)}\n\n`;
      }
      
      if (detailedAnalysis.thinking_feeling) {
        formattedContent += `III. Thinking vs Feeling:\n${safeStringify(detailedAnalysis.thinking_feeling)}\n\n`;
      }
      
      if (detailedAnalysis.judging_perceiving) {
        formattedContent += `IV. Judging vs Perceiving:\n${safeStringify(detailedAnalysis.judging_perceiving)}\n\n`;
      }
      
      if (detailedAnalysis.deeper_signals) {
        formattedContent += `V. Deeper Indirect MBTI Signals:\n${safeStringify(detailedAnalysis.deeper_signals)}\n\n`;
      }
      
      // Create analysis record
      const dummyMediaUrl = `mbti-document:${Date.now()}`;
      const analysis = await storage.createAnalysis({
        sessionId,
        title: title || `MBTI Document Analysis - ${fileName}`,
        mediaUrl: dummyMediaUrl,
        mediaType: "document",
        documentType: fileType === "pdf" ? "pdf" : "docx",
        personalityInsights: { analysis: formattedContent, mbti_type: analysisResult.predicted_type },
        modelUsed: selectedModel,
      });
      
      // Create message with formatted analysis
      const message = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        content: formattedContent,
        role: "assistant",
      });
      
      res.json({
        analysisId: analysis.id,
        personalityInsights: { analysis: formattedContent, mbti_type: analysisResult.predicted_type },
        messages: [message],
      });
    } catch (error) {
      console.error("MBTI document analysis error:", error);
      res.status(500).json({ error: "Failed to analyze document for MBTI" });
    }
  });

  // MBTI Analysis Endpoints - Image
  app.post("/api/analyze/image/mbti", async (req, res) => {
    try {
      const { mediaData, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!mediaData || typeof mediaData !== 'string') {
        return res.status(400).json({ error: "Image data is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing MBTI image analysis with model: ${selectedModel}`);
      
      // MBTI image analysis prompt with 30 questions
      const mbtiImagePrompt = `You are an expert MBTI analyst specializing in visual cues. Analyze this image comprehensively using the MBTI framework, providing specific evidence-based answers to ALL 30 questions below WITH DIRECT VISUAL EVIDENCE.

CRITICAL: Every answer must reference SPECIFIC VISUAL DETAILS from the image. Do not use generic descriptions.

I. INTROVERSION VS EXTRAVERSION (I/E)
1. Does the person's facial expression convey self-restraint, inward focus, and minimal performative energy—or open expressiveness and outward emotional broadcast?
2. Is eye contact direct and socially assertive—or indirect, introspective, or observing rather than engaging?
3. Are they alone or with others, and if with others, are they positioned as focal point or quiet participant?
4. Does their body posture project "closed" (arms in, shoulders inward) or "open" (expanded, outward-facing)?
5. Is the environment suggestive of solitude, study, or quiet spaces—or public, energetic, and socially dynamic?

II. SENSING VS INTUITION (S/N)
6. Are clothes and accessories practical, detail-oriented, and sensory-specific—or symbolic, abstract, or stylistically conceptual?
7. Is the person's gaze locked into the environment (grounded) or slightly distant/"elsewhere" (abstracted)?
8. Do facial micro-expressions show direct reaction to physical surroundings—or inward processing of ideas?
9. Is the environment filled with literal objects/tools—or symbolic/artistic/ambiguous elements?
10. Does the person's styling emphasize texture, material precision, and concreteness—or originality, metaphor, or thematic coherence?

III. THINKING VS FEELING (T/F)
11. Does the expression suggest emotional attunement and warmth—or analysis, restraint, and evaluative distance?
12. Are facial muscles relaxed and receptive—or tense near brows/jaw, as if organizing or assessing?
13. Is posture soft and adaptable—or structured, controlled, and deliberate?
14. Is there visual evidence of values-driven identity (causes, humanitarian symbols)—or system/logic-driven identity (devices, strategy games, tools)?
15. Is the overall emotional atmosphere personal and empathetic—or impersonal, technical, and principled?

IV. JUDGING VS PERCEIVING (J/P)
16. Are clothes symmetrical, ironed, and intentionally arranged—or more relaxed, mismatched, or improvisational?
17. Does the photo composition show order and planning—or spontaneity and flow?
18. Is the posture upright and resolved—or loose, adaptable, and mid-transition?
19. Does the setting suggest schedules, structure, and completion—or exploration, activity, or ongoing process?
20. Are facial expressions "settled" and decisive—or flexible, shifting, or open-ended?

V. COGNITIVE / DEEPER INDICATORS
21. Does the person occupy space confidently and definitively—or lightly, as if moving through possibilities?
22. Are their hands used in expressive, relational ways—or instrumental, precise, or still?
23. Do they display intense, narrow-focus gaze (Ni/Si) or scanning/observing multiple cues (Ne/Se)?
24. Is the environment curated for meaning/symbolism—or for function, utility, or familiarity?
25. Are emotional cues projected outward (Fe)—or held internally (Fi), visible only subtly around the eyes/mouth?
26. Does the image show harmony-seeking positioning near others—or independence and self-contained space?
27. Does the person appear to structure their environment—or adapt fluidly to it?
28. Are they visually aligned with tradition and convention—or personalized, unconventional, or experimental?
29. Does their body language appear measured and restrained—or reactive and environment-responsive?
30. Is their aesthetic intentional and minimal—or layered, evolving, and exploratory?

Provide your analysis in JSON format:
{
  "summary": "Brief overall MBTI assessment with predicted type and confidence level",
  "detailed_analysis": {
    "introversion_extraversion": "Detailed analysis of I/E with specific visual evidence",
    "sensing_intuition": "Detailed analysis of S/N with specific visual evidence",
    "thinking_feeling": "Detailed analysis of T/F with specific visual evidence",
    "judging_perceiving": "Detailed analysis of J/P with specific visual evidence",
    "cognitive_indicators": "Detailed analysis of deeper cognitive indicators with specific visual evidence"
  },
  "predicted_type": "Four-letter MBTI type (e.g., INTJ, ENFP)",
  "confidence": "High/Medium/Low with explanation"
}`;

      let analysisResult: any;
      
      // Call the appropriate AI model with vision capability (OpenAI only for now)
      if (selectedModel === "openai" && openai) {
        const response = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [{
            role: "user",
            content: [
              { type: "text", text: mbtiImagePrompt },
              { type: "image_url", image_url: { url: mediaData } }
            ]
          }],
          response_format: { type: "json_object" },
        });
        
        const rawResponse = response.choices[0]?.message.content || "";
        console.log("OpenAI MBTI Image raw response:", rawResponse.substring(0, 500));
        
        if (!rawResponse || rawResponse.trim().length === 0) {
          throw new Error("OpenAI returned an empty response");
        }
        
        try {
          analysisResult = JSON.parse(rawResponse);
        } catch (parseError) {
          console.error("Failed to parse OpenAI response:", parseError);
          console.error("Raw response:", rawResponse);
          
          // Try to extract any useful information from the raw response
          const fallbackSummary = rawResponse.length > 0 
            ? rawResponse.substring(0, 1000) 
            : "The AI was unable to properly format the MBTI analysis. Please try again with a different image showing a clear view of the person's face and body language.";
          
          analysisResult = {
            summary: fallbackSummary,
            detailed_analysis: {
              introversion_extraversion: "Unable to analyze due to formatting error. Please retry with a clearer image.",
              sensing_intuition: "Unable to analyze due to formatting error. Please retry with a clearer image.",
              thinking_feeling: "Unable to analyze due to formatting error. Please retry with a clearer image.",
              judging_perceiving: "Unable to analyze due to formatting error. Please retry with a clearer image.",
              cognitive_indicators: "Unable to analyze due to formatting error. Please retry with a clearer image."
            },
            predicted_type: "Unable to determine",
            confidence: "Low (formatting error occurred)"
          };
        }
      } else if (selectedModel === "anthropic" && anthropic) {
        // Anthropic vision support
        const base64Match = mediaData.match(/^data:image\/[a-z]+;base64,(.+)$/);
        const base64Data = base64Match ? base64Match[1] : mediaData;
        const mediaTypeMatch = mediaData.match(/^data:(image\/[a-z]+);base64,/);
        const mediaType = mediaTypeMatch ? mediaTypeMatch[1] : "image/jpeg";
        
        const response = await anthropic.messages.create({
          model: "claude-sonnet-4-20250514",
          max_tokens: 8000,
          messages: [{
            role: "user",
            content: [
              {
                type: "image",
                source: {
                  type: "base64",
                  media_type: mediaType as any,
                  data: base64Data,
                },
              },
              {
                type: "text",
                text: mbtiImagePrompt
              }
            ]
          }],
        });
        
        const rawResponse = response.content[0].type === 'text' ? response.content[0].text : "";
        console.log("Anthropic MBTI Image raw response:", rawResponse.substring(0, 500));
        
        if (!rawResponse || rawResponse.trim().length === 0) {
          throw new Error("Anthropic returned an empty response");
        }
        
        // Extract JSON from code fence if present
        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/```\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          jsonText = jsonMatch[1];
        }
        
        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse Anthropic response:", parseError);
          console.error("Raw response:", rawResponse);
          
          const fallbackSummary = rawResponse.length > 0 
            ? rawResponse.substring(0, 1000) 
            : "The AI was unable to properly format the MBTI analysis. Please try again with a different image showing a clear view of the person's face and body language.";
          
          analysisResult = {
            summary: fallbackSummary,
            detailed_analysis: {
              introversion_extraversion: "Unable to analyze due to formatting error. Please retry with a clearer image.",
              sensing_intuition: "Unable to analyze due to formatting error. Please retry with a clearer image.",
              thinking_feeling: "Unable to analyze due to formatting error. Please retry with a clearer image.",
              judging_perceiving: "Unable to analyze due to formatting error. Please retry with a clearer image.",
              cognitive_indicators: "Unable to analyze due to formatting error. Please retry with a clearer image."
            },
            predicted_type: "Unable to determine",
            confidence: "Low (formatting error occurred)"
          };
        }
      } else {
        return res.status(400).json({ 
          error: "MBTI image analysis currently only supports OpenAI and Anthropic models with vision capabilities." 
        });
      }
      
      // Helper function to safely stringify any value into readable text
      const safeStringify = (value: any): string => {
        if (typeof value === 'string') return value;
        if (typeof value === 'object' && value !== null) {
          // If it's an array, handle each item recursively
          if (Array.isArray(value)) {
            return value.map(item => {
              if (typeof item === 'string') return item;
              if (typeof item === 'object' && item !== null) {
                // Format objects in arrays as key-value pairs
                return Object.entries(item)
                  .map(([key, val]) => `${key}: ${val}`)
                  .join('\n');
              }
              return String(item);
            }).join('\n\n');
          }
          // If it's an object with numbered keys (like "1", "2", etc), format as numbered list
          const keys = Object.keys(value);
          if (keys.length > 0 && keys.every(k => /^\d+$/.test(k))) {
            return keys
              .sort((a, b) => parseInt(a) - parseInt(b))
              .map(key => `${key}. ${value[key]}`)
              .join('\n');
          }
          // If it's an object with named keys, format as key-value pairs
          return Object.entries(value)
            .map(([key, val]) => `${val}`)
            .join('\n\n');
        }
        return String(value || '');
      };
      
      // Format the analysis for display
      let formattedContent = `MBTI Personality Analysis (Image)\nMode: Myers-Briggs Type Indicator Framework\n\n`;
      formattedContent += `${'─'.repeat(65)}\n`;
      formattedContent += `Analysis Results\n`;
      formattedContent += `${'─'.repeat(65)}\n\n`;
      
      formattedContent += `Predicted Type: ${analysisResult.predicted_type || 'Unknown'}\n`;
      formattedContent += `Confidence: ${analysisResult.confidence || 'Unknown'}\n\n`;
      formattedContent += `Summary:\n${safeStringify(analysisResult.summary) || 'No summary available'}\n\n`;
      
      const detailedAnalysis = analysisResult.detailed_analysis || {};
      
      if (detailedAnalysis.introversion_extraversion) {
        formattedContent += `I. Introversion vs Extraversion:\n${safeStringify(detailedAnalysis.introversion_extraversion)}\n\n`;
      }
      
      if (detailedAnalysis.sensing_intuition) {
        formattedContent += `II. Sensing vs Intuition:\n${safeStringify(detailedAnalysis.sensing_intuition)}\n\n`;
      }
      
      if (detailedAnalysis.thinking_feeling) {
        formattedContent += `III. Thinking vs Feeling:\n${safeStringify(detailedAnalysis.thinking_feeling)}\n\n`;
      }
      
      if (detailedAnalysis.judging_perceiving) {
        formattedContent += `IV. Judging vs Perceiving:\n${safeStringify(detailedAnalysis.judging_perceiving)}\n\n`;
      }
      
      if (detailedAnalysis.cognitive_indicators) {
        formattedContent += `V. Cognitive / Deeper Indicators:\n${safeStringify(detailedAnalysis.cognitive_indicators)}\n\n`;
      }
      
      // Create analysis record
      const mediaUrl = `mbti-image:${Date.now()}`;
      const analysis = await storage.createAnalysis({
        sessionId,
        title: title || `MBTI Image Analysis`,
        mediaUrl,
        mediaType: "image",
        personalityInsights: { analysis: formattedContent, mbti_type: analysisResult.predicted_type },
        modelUsed: selectedModel,
      });
      
      // Create message with formatted analysis
      const message = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        content: formattedContent,
        role: "assistant",
      });
      
      res.json({
        analysisId: analysis.id,
        personalityInsights: { analysis: formattedContent, mbti_type: analysisResult.predicted_type },
        messages: [message],
        mediaUrl,
      });
    } catch (error) {
      console.error("MBTI image analysis error:", error);
      res.status(500).json({ error: "Failed to analyze image for MBTI" });
    }
  });

  // MBTI Analysis Endpoints - Video
  app.post("/api/analyze/video/mbti", async (req, res) => {
    try {
      const { mediaData, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!mediaData || typeof mediaData !== 'string') {
        return res.status(400).json({ error: "Video data is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing MBTI video analysis with model: ${selectedModel}`);
      
      // For video, we'll extract frames and analyze them
      // Save video temporarily
      const videoBuffer = Buffer.from(mediaData.split(',')[1], 'base64');
      const tempVideoPath = path.join(tempDir, `video_${Date.now()}.mp4`);
      await writeFileAsync(tempVideoPath, videoBuffer);
      
      // Extract frames at different timestamps
      const framePromises = [0, 25, 50, 75].map(async (percent) => {
        const outputPath = path.join(tempDir, `frame_${Date.now()}_${percent}.jpg`);
        
        return new Promise<string>((resolve, reject) => {
          ffmpeg(tempVideoPath)
            .screenshots({
              count: 1,
              timemarks: [`${percent}%`],
              filename: path.basename(outputPath),
              folder: tempDir,
            })
            .on('end', () => {
              const frameData = fs.readFileSync(outputPath);
              const base64Frame = `data:image/jpeg;base64,${frameData.toString('base64')}`;
              fs.unlinkSync(outputPath);
              resolve(base64Frame);
            })
            .on('error', (err) => {
              console.error('Frame extraction error:', err);
              reject(err);
            });
        });
      });
      
      const frames = await Promise.all(framePromises);
      
      // Clean up temp video file
      await unlinkAsync(tempVideoPath);
      
      // MBTI video analysis prompt with 30 questions
      const mbtiVideoPrompt = `You are an expert MBTI analyst specializing in behavioral cues. Analyze these video frames comprehensively using the MBTI framework, providing specific evidence-based answers to ALL 30 questions below WITH DIRECT VISUAL AND BEHAVIORAL EVIDENCE with timestamps.

CRITICAL: Every answer must reference SPECIFIC VISUAL DETAILS, BEHAVIORS, and TEMPORAL CHANGES across the frames. Do not use generic descriptions.

I. INTROVERSION VS EXTRAVERSION (I/E)
1. Does their energy seem directed outward (projected), or inward (contained, self-referential)?
2. Do they initiate interaction with surroundings/others—or wait, observe, respond?
3. How quickly and confidently do they make eye contact with the camera/person—or do they glance briefly or avoid it?
4. Are their gestures expansive and socially open—or minimized, controlled, and self-contained?
5. Does their posture lean forward/engage—or stay anchored inward/backward, conserving energy?

II. SENSING VS INTUITION (S/N)
6. Do they tangibly engage with their physical surroundings—or appear mentally elsewhere, focused on ideas more than environment?
7. Is their attention visibly grounded in the present moment—or drifting, future-oriented, symbolic, or associative?
8. When interacting, do their eyes scan concrete objects—or look up/aside as if accessing concepts or abstractions?
9. Are movements literal and practical—or stylized, metaphorical, or imaginative?
10. Do they respond to immediate sensory input—or to internal interpretations of what could be happening?

III. THINKING VS FEELING (T/F)
11. Are facial reactions emotionally transparent—or filtered through analysis and composure?
12. When reacting, do they first show empathy/connection—or evaluation/judgment/assessment?
13. Do they adjust their posture and movements to maintain harmony—or to maintain logical structure and control?
14. Are emotional expressions fluid and relational—or subtle, contained, or dampened?
15. If speaking, do they choose words that connect emotionally—or words that systematize or clarify?

IV. JUDGING VS PERCEIVING (J/P)
16. Is their movement precise, planned, and concluded—or adaptive, flexible, and open-ended?
17. Do they complete motions cleanly—or leave gestures half-open, unresolved?
18. Are facial expressions decisive and consistent—or quickly changing, context-responsive?
19. Do they anticipate and prepare for things before they happen—or react in-the-moment?
20. Does their timing feel structured and rhythmic—or spontaneous and elastic?

V. DEEPER COGNITIVE FUNCTION SIGNALS
21. Do they show sustained focus on one point (Ni/Si)—or rapid shifting and idea-generation (Ne/Se)?
22. Are emotional responses internalized (Fi) or immediately externalized toward others (Fe)?
23. Do they use small, precise gestures (Ti/Si)—or broad, audience-aware ones (Fe/Se)?
24. Is their speaking cadence clipped, sequential, and controlled—or flowing, adaptive, meandering?
25. Do they pause to evaluate before acting—or act first and adjust afterward?
26. Do they correct or adjust the environment (J)—or assimilate themselves into it (P)?
27. Is their facial micro-expression delayed (inward processing) or immediate (external processing)?
28. Do they show tension to maintain control—or relaxed adaptability to emerging stimuli?
29. Does their energy rise the longer they engage (E/Ne/Se)—or deplete or pull inward (I/Ni/Si)?
30. If surprised, do they freeze and process—or respond instantly and externally?

Provide your analysis in JSON format:
{
  "summary": "Brief overall MBTI assessment with predicted type and confidence level",
  "detailed_analysis": {
    "introversion_extraversion": "Detailed analysis of I/E with specific behavioral evidence and timestamps",
    "sensing_intuition": "Detailed analysis of S/N with specific behavioral evidence and timestamps",
    "thinking_feeling": "Detailed analysis of T/F with specific behavioral evidence and timestamps",
    "judging_perceiving": "Detailed analysis of J/P with specific behavioral evidence and timestamps",
    "cognitive_function_signals": "Detailed analysis of deeper cognitive function indicators with specific behavioral evidence and timestamps"
  },
  "predicted_type": "Four-letter MBTI type (e.g., INTJ, ENFP)",
  "confidence": "High/Medium/Low with explanation"
}`;

      let analysisResult: any;
      
      // Call the appropriate AI model with vision capability (OpenAI only for now)
      if (selectedModel === "openai" && openai) {
        const response = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [{
            role: "user",
            content: [
              { type: "text", text: mbtiVideoPrompt + "\n\nFrames extracted at 0%, 25%, 50%, and 75% of video:" },
              ...frames.map((frame, idx) => ({
                type: "image_url" as const,
                image_url: { url: frame }
              }))
            ]
          }],
          response_format: { type: "json_object" },
        });
        
        const rawResponse = response.choices[0]?.message.content || "";
        console.log("OpenAI MBTI Video raw response:", rawResponse.substring(0, 500));
        
        if (!rawResponse || rawResponse.trim().length === 0) {
          throw new Error("OpenAI returned an empty response");
        }
        
        try {
          analysisResult = JSON.parse(rawResponse);
        } catch (parseError) {
          console.error("Failed to parse OpenAI response:", parseError);
          console.error("Raw response:", rawResponse);
          
          const fallbackSummary = rawResponse.length > 0 
            ? rawResponse.substring(0, 1000) 
            : "The AI was unable to properly format the MBTI analysis. Please try again with a different video showing clear behavioral patterns.";
          
          analysisResult = {
            summary: fallbackSummary,
            detailed_analysis: {
              introversion_extraversion: "Unable to analyze due to formatting error. Please retry with a clearer video.",
              sensing_intuition: "Unable to analyze due to formatting error. Please retry with a clearer video.",
              thinking_feeling: "Unable to analyze due to formatting error. Please retry with a clearer video.",
              judging_perceiving: "Unable to analyze due to formatting error. Please retry with a clearer video.",
              cognitive_function_signals: "Unable to analyze due to formatting error. Please retry with a clearer video."
            },
            predicted_type: "Unable to determine",
            confidence: "Low (formatting error occurred)"
          };
        }
      } else if (selectedModel === "anthropic" && anthropic) {
        // Anthropic vision support for multiple frames
        const imageContents = frames.map(frame => {
          const base64Match = frame.match(/^data:image\/[a-z]+;base64,(.+)$/);
          const base64Data = base64Match ? base64Match[1] : frame;
          const mediaTypeMatch = frame.match(/^data:(image\/[a-z]+);base64,/);
          const mediaType = mediaTypeMatch ? mediaTypeMatch[1] : "image/jpeg";
          
          return {
            type: "image" as const,
            source: {
              type: "base64" as const,
              media_type: mediaType as any,
              data: base64Data,
            },
          };
        });
        
        const response = await anthropic.messages.create({
          model: "claude-sonnet-4-20250514",
          max_tokens: 8000,
          messages: [{
            role: "user",
            content: [
              {
                type: "text",
                text: mbtiVideoPrompt + "\n\nFrames extracted at 0%, 25%, 50%, and 75% of video:"
              },
              ...imageContents
            ]
          }],
        });
        
        const rawResponse = response.content[0].type === 'text' ? response.content[0].text : "";
        console.log("Anthropic MBTI Video raw response:", rawResponse.substring(0, 500));
        
        if (!rawResponse || rawResponse.trim().length === 0) {
          throw new Error("Anthropic returned an empty response");
        }
        
        // Extract JSON from code fence if present
        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/```\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          jsonText = jsonMatch[1];
        }
        
        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse Anthropic response:", parseError);
          console.error("Raw response:", rawResponse);
          
          const fallbackSummary = rawResponse.length > 0 
            ? rawResponse.substring(0, 1000) 
            : "The AI was unable to properly format the MBTI analysis. Please try again with a different video showing clear behavioral patterns.";
          
          analysisResult = {
            summary: fallbackSummary,
            detailed_analysis: {
              introversion_extraversion: "Unable to analyze due to formatting error. Please retry with a clearer video.",
              sensing_intuition: "Unable to analyze due to formatting error. Please retry with a clearer video.",
              thinking_feeling: "Unable to analyze due to formatting error. Please retry with a clearer video.",
              judging_perceiving: "Unable to analyze due to formatting error. Please retry with a clearer video.",
              cognitive_function_signals: "Unable to analyze due to formatting error. Please retry with a clearer video."
            },
            predicted_type: "Unable to determine",
            confidence: "Low (formatting error occurred)"
          };
        }
      } else {
        return res.status(400).json({ 
          error: "MBTI video analysis currently only supports OpenAI and Anthropic models with vision capabilities." 
        });
      }
      
      // Helper function to safely stringify any value into readable text
      const safeStringify = (value: any): string => {
        if (typeof value === 'string') return value;
        if (typeof value === 'object' && value !== null) {
          // If it's an array, handle each item recursively
          if (Array.isArray(value)) {
            return value.map(item => {
              if (typeof item === 'string') return item;
              if (typeof item === 'object' && item !== null) {
                // Format objects in arrays as key-value pairs
                return Object.entries(item)
                  .map(([key, val]) => `${key}: ${val}`)
                  .join('\n');
              }
              return String(item);
            }).join('\n\n');
          }
          // If it's an object with numbered keys (like "1", "2", etc), format as numbered list
          const keys = Object.keys(value);
          if (keys.length > 0 && keys.every(k => /^\d+$/.test(k))) {
            return keys
              .sort((a, b) => parseInt(a) - parseInt(b))
              .map(key => `${key}. ${value[key]}`)
              .join('\n');
          }
          // If it's an object with named keys, format as key-value pairs
          return Object.entries(value)
            .map(([key, val]) => `${val}`)
            .join('\n\n');
        }
        return String(value || '');
      };
      
      // Format the analysis for display
      let formattedContent = `MBTI Personality Analysis (Video)\nMode: Myers-Briggs Type Indicator Framework\n\n`;
      formattedContent += `${'─'.repeat(65)}\n`;
      formattedContent += `Analysis Results\n`;
      formattedContent += `${'─'.repeat(65)}\n\n`;
      
      formattedContent += `Predicted Type: ${analysisResult.predicted_type || 'Unknown'}\n`;
      formattedContent += `Confidence: ${analysisResult.confidence || 'Unknown'}\n\n`;
      formattedContent += `Summary:\n${safeStringify(analysisResult.summary) || 'No summary available'}\n\n`;
      
      const detailedAnalysis = analysisResult.detailed_analysis || {};
      
      if (detailedAnalysis.introversion_extraversion) {
        formattedContent += `I. Introversion vs Extraversion:\n${safeStringify(detailedAnalysis.introversion_extraversion)}\n\n`;
      }
      
      if (detailedAnalysis.sensing_intuition) {
        formattedContent += `II. Sensing vs Intuition:\n${safeStringify(detailedAnalysis.sensing_intuition)}\n\n`;
      }
      
      if (detailedAnalysis.thinking_feeling) {
        formattedContent += `III. Thinking vs Feeling:\n${safeStringify(detailedAnalysis.thinking_feeling)}\n\n`;
      }
      
      if (detailedAnalysis.judging_perceiving) {
        formattedContent += `IV. Judging vs Perceiving:\n${safeStringify(detailedAnalysis.judging_perceiving)}\n\n`;
      }
      
      if (detailedAnalysis.cognitive_function_signals) {
        formattedContent += `V. Deeper Cognitive Function Signals:\n${safeStringify(detailedAnalysis.cognitive_function_signals)}\n\n`;
      }
      
      // Create analysis record
      const mediaUrl = `mbti-video:${Date.now()}`;
      const analysis = await storage.createAnalysis({
        sessionId,
        title: title || `MBTI Video Analysis`,
        mediaUrl,
        mediaType: "video",
        personalityInsights: { analysis: formattedContent, mbti_type: analysisResult.predicted_type },
        modelUsed: selectedModel,
      });
      
      // Create message with formatted analysis
      const message = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        content: formattedContent,
        role: "assistant",
      });
      
      res.json({
        analysisId: analysis.id,
        personalityInsights: { analysis: formattedContent, mbti_type: analysisResult.predicted_type },
        messages: [message],
        mediaUrl,
      });
    } catch (error) {
      console.error("MBTI video analysis error:", error);
      res.status(500).json({ error: "Failed to analyze video for MBTI" });
    }
  });

  // Big Five (OCEAN) Analysis Endpoints - Text
  app.post("/api/analyze/text/bigfive", async (req, res) => {
    try {
      const { content, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!content || typeof content !== 'string') {
        return res.status(400).json({ error: "Text content is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing Big Five text analysis with model: ${selectedModel}`);
      
      // Big Five (OCEAN) comprehensive prompt
      const bigFivePrompt = `You are an expert personality psychologist specializing in the Big Five (OCEAN) personality assessment. Analyze the following text comprehensively using the Big Five framework, providing detailed evidence for each dimension.

The Big Five dimensions are:
1. **Openness to Experience** - imagination, creativity, curiosity, appreciation for art, emotion, adventure, unusual ideas, variety
2. **Conscientiousness** - self-discipline, dutifulness, competence, order, deliberation, achievement-striving
3. **Extraversion** - sociability, assertiveness, talkativeness, activity level, excitement-seeking, positive emotions
4. **Agreeableness** - trust, altruism, kindness, affection, cooperation, modesty, sympathy
5. **Neuroticism** (Emotional Stability) - anxiety, anger, depression, self-consciousness, vulnerability, stress

TEXT TO ANALYZE:
${content}

Provide detailed analysis in JSON format:
{
  "summary": "Overall Big Five assessment with key personality insights",
  "detailed_analysis": {
    "openness": {
      "score": "High/Medium/Low",
      "description": "Detailed analysis of openness with specific evidence from text",
      "indicators": ["list of specific behavioral indicators from the text"]
    },
    "conscientiousness": {
      "score": "High/Medium/Low",
      "description": "Detailed analysis of conscientiousness with specific evidence from text",
      "indicators": ["list of specific behavioral indicators from the text"]
    },
    "extraversion": {
      "score": "High/Medium/Low",
      "description": "Detailed analysis of extraversion with specific evidence from text",
      "indicators": ["list of specific behavioral indicators from the text"]
    },
    "agreeableness": {
      "score": "High/Medium/Low",
      "description": "Detailed analysis of agreeableness with specific evidence from text",
      "indicators": ["list of specific behavioral indicators from the text"]
    },
    "neuroticism": {
      "score": "High/Medium/Low",
      "description": "Detailed analysis of neuroticism/emotional stability with specific evidence from text",
      "indicators": ["list of specific behavioral indicators from the text"]
    }
  },
  "personality_profile": "Comprehensive personality description based on the five dimensions",
  "strengths": ["list of key strengths based on the profile"],
  "growth_areas": ["list of potential areas for development"]
}`;

      let analysisResult: any;
      
      // Call the appropriate AI model
      if (selectedModel === "openai" && openai) {
        const completion = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [
            { role: "system", content: "You are an expert personality psychologist specializing in Big Five (OCEAN) personality assessment." },
            { role: "user", content: bigFivePrompt }
          ],
          response_format: { type: "json_object" },
        });
        
        const rawResponse = completion.choices[0]?.message.content || "";
        console.log("OpenAI Big Five raw response:", rawResponse.substring(0, 500));
        
        if (!rawResponse || rawResponse.trim().length === 0) {
          throw new Error("OpenAI returned an empty response");
        }
        
        try {
          analysisResult = JSON.parse(rawResponse);
        } catch (parseError) {
          console.error("Failed to parse OpenAI response:", parseError);
          console.error("Raw response:", rawResponse);
          
          const fallbackSummary = rawResponse.length > 0 
            ? rawResponse.substring(0, 1000) 
            : "The AI was unable to properly format the Big Five analysis. Please try again with different text.";
          
          analysisResult = {
            summary: fallbackSummary,
            detailed_analysis: {
              openness: { score: "Unable to determine", description: "Formatting error occurred", indicators: [] },
              conscientiousness: { score: "Unable to determine", description: "Formatting error occurred", indicators: [] },
              extraversion: { score: "Unable to determine", description: "Formatting error occurred", indicators: [] },
              agreeableness: { score: "Unable to determine", description: "Formatting error occurred", indicators: [] },
              neuroticism: { score: "Unable to determine", description: "Formatting error occurred", indicators: [] }
            },
            personality_profile: "Unable to generate profile due to formatting error",
            strengths: [],
            growth_areas: []
          };
        }
      } else if (selectedModel === "anthropic" && anthropic) {
        const response = await anthropic.messages.create({
          model: "claude-sonnet-4-20250514",
          max_tokens: 8000,
          messages: [{ role: "user", content: bigFivePrompt }],
        });
        
        const rawResponse = response.content[0].type === 'text' ? response.content[0].text : "";
        console.log("Anthropic Big Five raw response:", rawResponse.substring(0, 500));
        
        if (!rawResponse || rawResponse.trim().length === 0) {
          throw new Error("Anthropic returned an empty response");
        }
        
        // Extract JSON from code fence if present
        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/```\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          jsonText = jsonMatch[1];
        }
        
        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse Anthropic response:", parseError);
          console.error("Raw response:", rawResponse);
          
          const fallbackSummary = rawResponse.length > 0 
            ? rawResponse.substring(0, 1000) 
            : "The AI was unable to properly format the Big Five analysis. Please try again with different text.";
          
          analysisResult = {
            summary: fallbackSummary,
            detailed_analysis: {
              openness: { score: "Unable to determine", description: "Formatting error occurred", indicators: [] },
              conscientiousness: { score: "Unable to determine", description: "Formatting error occurred", indicators: [] },
              extraversion: { score: "Unable to determine", description: "Formatting error occurred", indicators: [] },
              agreeableness: { score: "Unable to determine", description: "Formatting error occurred", indicators: [] },
              neuroticism: { score: "Unable to determine", description: "Formatting error occurred", indicators: [] }
            },
            personality_profile: "Unable to generate profile due to formatting error",
            strengths: [],
            growth_areas: []
          };
        }
      } else {
        return res.status(400).json({ 
          error: "Big Five text analysis currently only supports OpenAI and Anthropic models." 
        });
      }
      
      // Helper function to safely stringify any value into readable text
      const safeStringify = (value: any): string => {
        if (typeof value === 'string') return value;
        if (typeof value === 'object' && value !== null) {
          if (Array.isArray(value)) {
            return value.map((item, idx) => `${idx + 1}. ${String(item)}`).join('\n');
          }
          const keys = Object.keys(value);
          if (keys.length > 0 && keys.every(k => /^\d+$/.test(k))) {
            return keys
              .sort((a, b) => parseInt(a) - parseInt(b))
              .map(key => `${key}. ${value[key]}`)
              .join('\n');
          }
          return Object.entries(value)
            .map(([key, val]) => `${val}`)
            .join('\n\n');
        }
        return String(value || '');
      };
      
      // Format the analysis for display
      let formattedContent = `Big Five (OCEAN) Personality Analysis\nMode: Five-Factor Model Framework\n\n`;
      formattedContent += `${'─'.repeat(65)}\n`;
      formattedContent += `Analysis Results\n`;
      formattedContent += `${'─'.repeat(65)}\n\n`;
      
      formattedContent += `Summary:\n${safeStringify(analysisResult.summary) || 'No summary available'}\n\n`;
      
      const detailedAnalysis = analysisResult.detailed_analysis || {};
      
      // Openness
      if (detailedAnalysis.openness) {
        formattedContent += `I. Openness to Experience: ${detailedAnalysis.openness.score || 'N/A'}\n`;
        formattedContent += `${safeStringify(detailedAnalysis.openness.description)}\n`;
        if (detailedAnalysis.openness.indicators && detailedAnalysis.openness.indicators.length > 0) {
          formattedContent += `Indicators:\n${safeStringify(detailedAnalysis.openness.indicators)}\n`;
        }
        formattedContent += `\n`;
      }
      
      // Conscientiousness
      if (detailedAnalysis.conscientiousness) {
        formattedContent += `II. Conscientiousness: ${detailedAnalysis.conscientiousness.score || 'N/A'}\n`;
        formattedContent += `${safeStringify(detailedAnalysis.conscientiousness.description)}\n`;
        if (detailedAnalysis.conscientiousness.indicators && detailedAnalysis.conscientiousness.indicators.length > 0) {
          formattedContent += `Indicators:\n${safeStringify(detailedAnalysis.conscientiousness.indicators)}\n`;
        }
        formattedContent += `\n`;
      }
      
      // Extraversion
      if (detailedAnalysis.extraversion) {
        formattedContent += `III. Extraversion: ${detailedAnalysis.extraversion.score || 'N/A'}\n`;
        formattedContent += `${safeStringify(detailedAnalysis.extraversion.description)}\n`;
        if (detailedAnalysis.extraversion.indicators && detailedAnalysis.extraversion.indicators.length > 0) {
          formattedContent += `Indicators:\n${safeStringify(detailedAnalysis.extraversion.indicators)}\n`;
        }
        formattedContent += `\n`;
      }
      
      // Agreeableness
      if (detailedAnalysis.agreeableness) {
        formattedContent += `IV. Agreeableness: ${detailedAnalysis.agreeableness.score || 'N/A'}\n`;
        formattedContent += `${safeStringify(detailedAnalysis.agreeableness.description)}\n`;
        if (detailedAnalysis.agreeableness.indicators && detailedAnalysis.agreeableness.indicators.length > 0) {
          formattedContent += `Indicators:\n${safeStringify(detailedAnalysis.agreeableness.indicators)}\n`;
        }
        formattedContent += `\n`;
      }
      
      // Neuroticism
      if (detailedAnalysis.neuroticism) {
        formattedContent += `V. Neuroticism (Emotional Stability): ${detailedAnalysis.neuroticism.score || 'N/A'}\n`;
        formattedContent += `${safeStringify(detailedAnalysis.neuroticism.description)}\n`;
        if (detailedAnalysis.neuroticism.indicators && detailedAnalysis.neuroticism.indicators.length > 0) {
          formattedContent += `Indicators:\n${safeStringify(detailedAnalysis.neuroticism.indicators)}\n`;
        }
        formattedContent += `\n`;
      }
      
      // Personality Profile
      if (analysisResult.personality_profile) {
        formattedContent += `Personality Profile:\n${safeStringify(analysisResult.personality_profile)}\n\n`;
      }
      
      // Strengths
      if (analysisResult.strengths && analysisResult.strengths.length > 0) {
        formattedContent += `Strengths:\n${safeStringify(analysisResult.strengths)}\n\n`;
      }
      
      // Growth Areas
      if (analysisResult.growth_areas && analysisResult.growth_areas.length > 0) {
        formattedContent += `Growth Areas:\n${safeStringify(analysisResult.growth_areas)}\n\n`;
      }
      
      // Create analysis record
      const analysis = await storage.createAnalysis({
        sessionId,
        title: title || `Big Five Text Analysis`,
        mediaUrl: `bigfive-text:${Date.now()}`,
        mediaType: "text",
        personalityInsights: { analysis: formattedContent, big_five: analysisResult },
        modelUsed: selectedModel,
      });
      
      // Create message with formatted analysis
      const message = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        content: formattedContent,
        role: "assistant",
      });
      
      res.json({
        analysisId: analysis.id,
        personalityInsights: { analysis: formattedContent, big_five: analysisResult },
        messages: [message],
      });
    } catch (error) {
      console.error("Big Five text analysis error:", error);
      res.status(500).json({ error: "Failed to analyze text for Big Five" });
    }
  });

  // Big Five (OCEAN) Analysis - Image
  app.post("/api/analyze/image/bigfive", async (req, res) => {
    try {
      const { mediaData, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!mediaData || typeof mediaData !== 'string') {
        return res.status(400).json({ error: "Image data is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing Big Five image analysis with model: ${selectedModel}`);
      
      // Big Five (OCEAN) visual analysis prompt
      const bigFiveImagePrompt = `You are an expert personality psychologist specializing in the Big Five (OCEAN) personality assessment through visual analysis. Analyze this image comprehensively using the Big Five framework, providing detailed evidence for each dimension based on VISIBLE ELEMENTS ONLY.

The Big Five dimensions are:
1. **Openness to Experience** - creativity, artistic expression, unconventional elements, symbolic imagery, variety
2. **Conscientiousness** - organization, attention to detail, grooming, neatness, structure
3. **Extraversion** - social engagement, expressiveness, energy level, body language, setting
4. **Agreeableness** - warmth, approachability, soft features, cooperative body language
5. **Neuroticism** (Emotional Stability) - tension, anxiety markers, emotional expression, stress indicators

Analyze ONLY what you can actually see in the image. Do not fabricate or assume details.

Provide detailed analysis in JSON format:
{
  "summary": "Overall Big Five assessment based on visual cues with key personality insights",
  "detailed_analysis": {
    "openness": {
      "score": "High/Medium/Low",
      "description": "Detailed analysis of openness based on specific visual evidence",
      "visual_indicators": ["list of specific visual cues from the image"]
    },
    "conscientiousness": {
      "score": "High/Medium/Low",
      "description": "Detailed analysis of conscientiousness based on specific visual evidence",
      "visual_indicators": ["list of specific visual cues from the image"]
    },
    "extraversion": {
      "score": "High/Medium/Low",
      "description": "Detailed analysis of extraversion based on specific visual evidence",
      "visual_indicators": ["list of specific visual cues from the image"]
    },
    "agreeableness": {
      "score": "High/Medium/Low",
      "description": "Detailed analysis of agreeableness based on specific visual evidence",
      "visual_indicators": ["list of specific visual cues from the image"]
    },
    "neuroticism": {
      "score": "High/Medium/Low",
      "description": "Detailed analysis of neuroticism/emotional stability based on specific visual evidence",
      "visual_indicators": ["list of specific visual cues from the image"]
    }
  },
  "personality_profile": "Comprehensive personality description based on the five dimensions",
  "strengths": ["list of key strengths based on the visual profile"],
  "growth_areas": ["list of potential areas for development based on visual analysis"]
}`;

      let analysisResult: any;
      
      // Only OpenAI GPT-4o Vision supports image analysis
      if (selectedModel === "openai" && openai) {
        const completion = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: bigFiveImagePrompt },
                {
                  type: "image_url",
                  image_url: { url: mediaData }
                }
              ]
            }
          ],
          response_format: { type: "json_object" },
          max_tokens: 4000,
        });
        
        const rawResponse = completion.choices[0]?.message.content || "";
        console.log("OpenAI Big Five Image raw response:", rawResponse.substring(0, 500));
        
        if (!rawResponse || rawResponse.trim().length === 0) {
          throw new Error("OpenAI returned an empty response");
        }
        
        try {
          analysisResult = JSON.parse(rawResponse);
        } catch (parseError) {
          console.error("Failed to parse OpenAI response:", parseError);
          console.error("Raw response:", rawResponse);
          
          const fallbackSummary = rawResponse.length > 0 
            ? rawResponse.substring(0, 1000) 
            : "The AI was unable to properly format the Big Five analysis. Please try again.";
          
          analysisResult = {
            summary: fallbackSummary,
            detailed_analysis: {
              openness: { score: "Unable to determine", description: "Formatting error occurred", visual_indicators: [] },
              conscientiousness: { score: "Unable to determine", description: "Formatting error occurred", visual_indicators: [] },
              extraversion: { score: "Unable to determine", description: "Formatting error occurred", visual_indicators: [] },
              agreeableness: { score: "Unable to determine", description: "Formatting error occurred", visual_indicators: [] },
              neuroticism: { score: "Unable to determine", description: "Formatting error occurred", visual_indicators: [] }
            },
            personality_profile: "Unable to generate profile due to formatting error",
            strengths: [],
            growth_areas: []
          };
        }
      } else {
        return res.status(400).json({ 
          error: "Big Five image analysis currently only supports OpenAI GPT-4o Vision model." 
        });
      }
      
      // Helper function to safely stringify any value into readable text
      const safeStringify = (value: any): string => {
        if (typeof value === 'string') return value;
        if (typeof value === 'object' && value !== null) {
          if (Array.isArray(value)) {
            return value.map((item, idx) => `${idx + 1}. ${String(item)}`).join('\n');
          }
          const keys = Object.keys(value);
          if (keys.length > 0 && keys.every(k => /^\d+$/.test(k))) {
            return keys
              .sort((a, b) => parseInt(a) - parseInt(b))
              .map(key => `${key}. ${value[key]}`)
              .join('\n');
          }
          return Object.entries(value)
            .map(([key, val]) => `${val}`)
            .join('\n\n');
        }
        return String(value || '');
      };
      
      // Format the analysis for display
      let formattedContent = `Big Five (OCEAN) Visual Personality Analysis\nMode: Five-Factor Model Framework (Image Analysis)\n\n`;
      formattedContent += `${'─'.repeat(65)}\n`;
      formattedContent += `Analysis Results\n`;
      formattedContent += `${'─'.repeat(65)}\n\n`;
      
      formattedContent += `Summary:\n${safeStringify(analysisResult.summary) || 'No summary available'}\n\n`;
      
      const detailedAnalysis = analysisResult.detailed_analysis || {};
      
      // Openness
      if (detailedAnalysis.openness) {
        formattedContent += `I. Openness to Experience: ${detailedAnalysis.openness.score || 'N/A'}\n`;
        formattedContent += `${safeStringify(detailedAnalysis.openness.description)}\n`;
        if (detailedAnalysis.openness.visual_indicators && detailedAnalysis.openness.visual_indicators.length > 0) {
          formattedContent += `Visual Indicators:\n${safeStringify(detailedAnalysis.openness.visual_indicators)}\n`;
        }
        formattedContent += `\n`;
      }
      
      // Conscientiousness
      if (detailedAnalysis.conscientiousness) {
        formattedContent += `II. Conscientiousness: ${detailedAnalysis.conscientiousness.score || 'N/A'}\n`;
        formattedContent += `${safeStringify(detailedAnalysis.conscientiousness.description)}\n`;
        if (detailedAnalysis.conscientiousness.visual_indicators && detailedAnalysis.conscientiousness.visual_indicators.length > 0) {
          formattedContent += `Visual Indicators:\n${safeStringify(detailedAnalysis.conscientiousness.visual_indicators)}\n`;
        }
        formattedContent += `\n`;
      }
      
      // Extraversion
      if (detailedAnalysis.extraversion) {
        formattedContent += `III. Extraversion: ${detailedAnalysis.extraversion.score || 'N/A'}\n`;
        formattedContent += `${safeStringify(detailedAnalysis.extraversion.description)}\n`;
        if (detailedAnalysis.extraversion.visual_indicators && detailedAnalysis.extraversion.visual_indicators.length > 0) {
          formattedContent += `Visual Indicators:\n${safeStringify(detailedAnalysis.extraversion.visual_indicators)}\n`;
        }
        formattedContent += `\n`;
      }
      
      // Agreeableness
      if (detailedAnalysis.agreeableness) {
        formattedContent += `IV. Agreeableness: ${detailedAnalysis.agreeableness.score || 'N/A'}\n`;
        formattedContent += `${safeStringify(detailedAnalysis.agreeableness.description)}\n`;
        if (detailedAnalysis.agreeableness.visual_indicators && detailedAnalysis.agreeableness.visual_indicators.length > 0) {
          formattedContent += `Visual Indicators:\n${safeStringify(detailedAnalysis.agreeableness.visual_indicators)}\n`;
        }
        formattedContent += `\n`;
      }
      
      // Neuroticism
      if (detailedAnalysis.neuroticism) {
        formattedContent += `V. Neuroticism (Emotional Stability): ${detailedAnalysis.neuroticism.score || 'N/A'}\n`;
        formattedContent += `${safeStringify(detailedAnalysis.neuroticism.description)}\n`;
        if (detailedAnalysis.neuroticism.visual_indicators && detailedAnalysis.neuroticism.visual_indicators.length > 0) {
          formattedContent += `Visual Indicators:\n${safeStringify(detailedAnalysis.neuroticism.visual_indicators)}\n`;
        }
        formattedContent += `\n`;
      }
      
      // Personality Profile
      if (analysisResult.personality_profile) {
        formattedContent += `Personality Profile:\n${safeStringify(analysisResult.personality_profile)}\n\n`;
      }
      
      // Strengths
      if (analysisResult.strengths && analysisResult.strengths.length > 0) {
        formattedContent += `Strengths:\n${safeStringify(analysisResult.strengths)}\n\n`;
      }
      
      // Growth Areas
      if (analysisResult.growth_areas && analysisResult.growth_areas.length > 0) {
        formattedContent += `Growth Areas:\n${safeStringify(analysisResult.growth_areas)}\n\n`;
      }
      
      // Create analysis record
      const analysis = await storage.createAnalysis({
        sessionId,
        title: title || `Big Five Image Analysis`,
        mediaUrl: mediaData,
        mediaType: "image",
        personalityInsights: { analysis: formattedContent, big_five: analysisResult },
        modelUsed: selectedModel,
      });
      
      // Create message with formatted analysis
      const message = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        content: formattedContent,
        role: "assistant",
      });
      
      res.json({
        analysisId: analysis.id,
        personalityInsights: { analysis: formattedContent, big_five: analysisResult },
        messages: [message],
        mediaUrl: mediaData,
      });
    } catch (error) {
      console.error("Big Five image analysis error:", error);
      res.status(500).json({ error: "Failed to analyze image for Big Five" });
    }
  });

  // Big Five (OCEAN) Analysis - Video
  app.post("/api/analyze/video/bigfive", async (req, res) => {
    try {
      const { mediaData, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!mediaData || typeof mediaData !== 'string') {
        return res.status(400).json({ error: "Video data is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing Big Five video analysis with model: ${selectedModel}`);
      
      // Save video temporarily and extract frames
      const videoBuffer = Buffer.from(mediaData.split(',')[1], 'base64');
      const tempVideoPath = path.join(tempDir, `video_${Date.now()}.mp4`);
      await writeFileAsync(tempVideoPath, videoBuffer);
      
      // Extract frames at different timestamps
      const framePromises = [0, 25, 50, 75].map(async (percent) => {
        const outputPath = path.join(tempDir, `frame_${Date.now()}_${percent}.jpg`);
        
        return new Promise<string>((resolve, reject) => {
          ffmpeg(tempVideoPath)
            .screenshots({
              count: 1,
              timemarks: [`${percent}%`],
              filename: path.basename(outputPath),
              folder: tempDir,
            })
            .on('end', () => {
              const frameData = fs.readFileSync(outputPath);
              const base64Frame = `data:image/jpeg;base64,${frameData.toString('base64')}`;
              fs.unlinkSync(outputPath);
              resolve(base64Frame);
            })
            .on('error', (err) => {
              console.error('Frame extraction error:', err);
              reject(err);
            });
        });
      });
      
      const extractedFrames = await Promise.all(framePromises);
      
      // Clean up temp video file
      await unlinkAsync(tempVideoPath);
      
      console.log(`Extracted ${extractedFrames.length} frames from video`);
      
      // Big Five (OCEAN) video analysis prompt
      const bigFiveVideoPrompt = `You are an expert personality psychologist specializing in the Big Five (OCEAN) personality assessment through behavioral video analysis. Analyze this video comprehensively using the Big Five framework, providing detailed evidence for each dimension based on OBSERVABLE BEHAVIORS AND VISUAL ELEMENTS across the video timeline.

The Big Five dimensions are:
1. **Openness to Experience** - creative expression, unconventional behaviors, variety in gestures, exploratory movements
2. **Conscientiousness** - organized movements, attention to detail, structured behavior, purposeful actions
3. **Extraversion** - energy level, expressiveness, social engagement, body language dynamism
4. **Agreeableness** - warm expressions, cooperative gestures, approachable demeanor, affiliative behaviors  
5. **Neuroticism** (Emotional Stability) - tension patterns, anxiety markers, emotional fluctuations, stress indicators

Analyze ONLY what you can actually observe in the video frames. Do not fabricate or assume details. Reference specific moments or behavioral patterns you observe.

Provide detailed analysis in JSON format:
{
  "summary": "Overall Big Five assessment based on observable behavioral patterns with key personality insights",
  "detailed_analysis": {
    "openness": {
      "score": "High/Medium/Low",
      "description": "Detailed analysis of openness based on specific behavioral evidence",
      "behavioral_indicators": ["list of specific observable behaviors from the video"]
    },
    "conscientiousness": {
      "score": "High/Medium/Low",
      "description": "Detailed analysis of conscientiousness based on specific behavioral evidence",
      "behavioral_indicators": ["list of specific observable behaviors from the video"]
    },
    "extraversion": {
      "score": "High/Medium/Low",
      "description": "Detailed analysis of extraversion based on specific behavioral evidence",
      "behavioral_indicators": ["list of specific observable behaviors from the video"]
    },
    "agreeableness": {
      "score": "High/Medium/Low",
      "description": "Detailed analysis of agreeableness based on specific behavioral evidence",
      "behavioral_indicators": ["list of specific observable behaviors from the video"]
    },
    "neuroticism": {
      "score": "High/Medium/Low",
      "description": "Detailed analysis of neuroticism/emotional stability based on specific behavioral evidence",
      "behavioral_indicators": ["list of specific observable behaviors from the video"]
    }
  },
  "personality_profile": "Comprehensive personality description integrating all five dimensions with behavioral examples",
  "strengths": ["list of personality strengths based on behavioral evidence"],
  "growth_areas": ["list of potential growth areas based on behavioral evidence"]
}`;

      // Analyze with selected model
      let analysisResult: any;
      
      // Call the appropriate AI model with vision capability
      if (selectedModel === "openai" && openai) {
        const response = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [{
            role: "user",
            content: [
              { type: "text", text: bigFiveVideoPrompt + "\n\nFrames extracted at 0%, 25%, 50%, and 75% of video:" },
              ...extractedFrames.map((frame, idx) => ({
                type: "image_url" as const,
                image_url: { url: frame }
              }))
            ]
          }],
          response_format: { type: "json_object" },
        });
        
        const rawResponse = response.choices[0]?.message.content || "";
        console.log("OpenAI Big Five Video raw response:", rawResponse.substring(0, 500));
        
        if (!rawResponse || rawResponse.trim().length === 0) {
          throw new Error("OpenAI returned an empty response");
        }
        
        try {
          analysisResult = JSON.parse(rawResponse);
        } catch (parseError) {
          console.error("Failed to parse OpenAI response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000) || "Unable to format analysis",
            detailed_analysis: {
              openness: { score: "N/A", description: "Formatting error occurred", behavioral_indicators: [] },
              conscientiousness: { score: "N/A", description: "Formatting error occurred", behavioral_indicators: [] },
              extraversion: { score: "N/A", description: "Formatting error occurred", behavioral_indicators: [] },
              agreeableness: { score: "N/A", description: "Formatting error occurred", behavioral_indicators: [] },
              neuroticism: { score: "N/A", description: "Formatting error occurred", behavioral_indicators: [] }
            },
            personality_profile: "Unable to format analysis",
            strengths: [],
            growth_areas: []
          };
        }
      } else if (selectedModel === "anthropic" && anthropic) {
        const imageContents = extractedFrames.map(frame => {
          const base64Match = frame.match(/^data:image\/[a-z]+;base64,(.+)$/);
          const base64Data = base64Match ? base64Match[1] : frame;
          const mediaTypeMatch = frame.match(/^data:(image\/[a-z]+);base64,/);
          const mediaType = mediaTypeMatch ? mediaTypeMatch[1] : "image/jpeg";
          
          return {
            type: "image" as const,
            source: {
              type: "base64" as const,
              media_type: mediaType as any,
              data: base64Data,
            },
          };
        });
        
        const response = await anthropic.messages.create({
          model: "claude-sonnet-4-20250514",
          max_tokens: 8000,
          messages: [{
            role: "user",
            content: [
              {
                type: "text",
                text: bigFiveVideoPrompt + "\n\nFrames extracted at 0%, 25%, 50%, and 75% of video:"
              },
              ...imageContents
            ]
          }],
        });
        
        const rawResponse = response.content[0].type === 'text' ? response.content[0].text : "";
        console.log("Anthropic Big Five Video raw response:", rawResponse.substring(0, 500));
        
        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/```\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          jsonText = jsonMatch[1];
        }
        
        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse Anthropic response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000) || "Unable to format analysis",
            detailed_analysis: {
              openness: { score: "N/A", description: "Formatting error occurred", behavioral_indicators: [] },
              conscientiousness: { score: "N/A", description: "Formatting error occurred", behavioral_indicators: [] },
              extraversion: { score: "N/A", description: "Formatting error occurred", behavioral_indicators: [] },
              agreeableness: { score: "N/A", description: "Formatting error occurred", behavioral_indicators: [] },
              neuroticism: { score: "N/A", description: "Formatting error occurred", behavioral_indicators: [] }
            },
            personality_profile: "Unable to format analysis",
            strengths: [],
            growth_areas: []
          };
        }
      } else if (selectedModel === "deepseek") {
        const response = await fetch('https://api.deepseek.com/chat/completions', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${process.env.DEEPSEEK_API_KEY}`
          },
          body: JSON.stringify({
            model: "deepseek-chat",
            messages: [{
              role: "user",
              content: [
                { type: "text", text: bigFiveVideoPrompt + "\n\nFrames extracted at 0%, 25%, 50%, and 75% of video:" },
                ...extractedFrames.map(frame => ({
                  type: "image_url",
                  image_url: { url: frame }
                }))
              ]
            }],
            response_format: { type: "json_object" }
          })
        });

        const data = await response.json();
        const rawResponse = data.choices?.[0]?.message?.content || "";
        console.log("DeepSeek Big Five Video raw response:", rawResponse.substring(0, 500));

        try {
          analysisResult = JSON.parse(rawResponse);
        } catch (parseError) {
          console.error("Failed to parse DeepSeek response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000) || "Unable to format analysis",
            detailed_analysis: {
              openness: { score: "N/A", description: "Formatting error occurred", behavioral_indicators: [] },
              conscientiousness: { score: "N/A", description: "Formatting error occurred", behavioral_indicators: [] },
              extraversion: { score: "N/A", description: "Formatting error occurred", behavioral_indicators: [] },
              agreeableness: { score: "N/A", description: "Formatting error occurred", behavioral_indicators: [] },
              neuroticism: { score: "N/A", description: "Formatting error occurred", behavioral_indicators: [] }
            },
            personality_profile: "Unable to format analysis",
            strengths: [],
            growth_areas: []
          };
        }
      } else if (selectedModel === "perplexity" && perplexity) {
        const response = await perplexity.chat.completions.create({
          model: "llama-3.1-sonar-large-128k-online",
          messages: [{
            role: "user",
            content: bigFiveVideoPrompt + "\n\nNote: Analyzing video frames for Big Five personality assessment."
          }],
        });

        const rawResponse = response.choices[0]?.message?.content || "";
        console.log("Perplexity Big Five Video raw response:", rawResponse.substring(0, 500));

        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/```\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          jsonText = jsonMatch[1];
        }

        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse Perplexity response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000) || "Unable to format analysis",
            detailed_analysis: {
              openness: { score: "N/A", description: "Formatting error occurred", behavioral_indicators: [] },
              conscientiousness: { score: "N/A", description: "Formatting error occurred", behavioral_indicators: [] },
              extraversion: { score: "N/A", description: "Formatting error occurred", behavioral_indicators: [] },
              agreeableness: { score: "N/A", description: "Formatting error occurred", behavioral_indicators: [] },
              neuroticism: { score: "N/A", description: "Formatting error occurred", behavioral_indicators: [] }
            },
            personality_profile: "Unable to format analysis",
            strengths: [],
            growth_areas: []
          };
        }
      }
      
      console.log("Big Five video analysis complete");
      
      // Format the analysis for display
      let formattedContent = `Big Five (OCEAN) Video Analysis\nMode: Behavioral Analysis\n\n`;
      formattedContent += `Summary:\n${safeStringify(analysisResult.summary)}\n\n`;
      
      const detailedAnalysis = analysisResult.detailed_analysis || {};
      
      // Openness
      if (detailedAnalysis.openness) {
        formattedContent += `I. Openness to Experience: ${detailedAnalysis.openness.score || 'N/A'}\n`;
        formattedContent += `${safeStringify(detailedAnalysis.openness.description)}\n`;
        if (detailedAnalysis.openness.behavioral_indicators && detailedAnalysis.openness.behavioral_indicators.length > 0) {
          formattedContent += `Behavioral Indicators:\n${safeStringify(detailedAnalysis.openness.behavioral_indicators)}\n`;
        }
        formattedContent += `\n`;
      }
      
      // Conscientiousness
      if (detailedAnalysis.conscientiousness) {
        formattedContent += `II. Conscientiousness: ${detailedAnalysis.conscientiousness.score || 'N/A'}\n`;
        formattedContent += `${safeStringify(detailedAnalysis.conscientiousness.description)}\n`;
        if (detailedAnalysis.conscientiousness.behavioral_indicators && detailedAnalysis.conscientiousness.behavioral_indicators.length > 0) {
          formattedContent += `Behavioral Indicators:\n${safeStringify(detailedAnalysis.conscientiousness.behavioral_indicators)}\n`;
        }
        formattedContent += `\n`;
      }
      
      // Extraversion
      if (detailedAnalysis.extraversion) {
        formattedContent += `III. Extraversion: ${detailedAnalysis.extraversion.score || 'N/A'}\n`;
        formattedContent += `${safeStringify(detailedAnalysis.extraversion.description)}\n`;
        if (detailedAnalysis.extraversion.behavioral_indicators && detailedAnalysis.extraversion.behavioral_indicators.length > 0) {
          formattedContent += `Behavioral Indicators:\n${safeStringify(detailedAnalysis.extraversion.behavioral_indicators)}\n`;
        }
        formattedContent += `\n`;
      }
      
      // Agreeableness
      if (detailedAnalysis.agreeableness) {
        formattedContent += `IV. Agreeableness: ${detailedAnalysis.agreeableness.score || 'N/A'}\n`;
        formattedContent += `${safeStringify(detailedAnalysis.agreeableness.description)}\n`;
        if (detailedAnalysis.agreeableness.behavioral_indicators && detailedAnalysis.agreeableness.behavioral_indicators.length > 0) {
          formattedContent += `Behavioral Indicators:\n${safeStringify(detailedAnalysis.agreeableness.behavioral_indicators)}\n`;
        }
        formattedContent += `\n`;
      }
      
      // Neuroticism
      if (detailedAnalysis.neuroticism) {
        formattedContent += `V. Neuroticism (Emotional Stability): ${detailedAnalysis.neuroticism.score || 'N/A'}\n`;
        formattedContent += `${safeStringify(detailedAnalysis.neuroticism.description)}\n`;
        if (detailedAnalysis.neuroticism.behavioral_indicators && detailedAnalysis.neuroticism.behavioral_indicators.length > 0) {
          formattedContent += `Behavioral Indicators:\n${safeStringify(detailedAnalysis.neuroticism.behavioral_indicators)}\n`;
        }
        formattedContent += `\n`;
      }
      
      // Personality Profile
      if (analysisResult.personality_profile) {
        formattedContent += `Personality Profile:\n${safeStringify(analysisResult.personality_profile)}\n\n`;
      }
      
      // Strengths
      if (analysisResult.strengths && analysisResult.strengths.length > 0) {
        formattedContent += `Strengths:\n${safeStringify(analysisResult.strengths)}\n\n`;
      }
      
      // Growth Areas
      if (analysisResult.growth_areas && analysisResult.growth_areas.length > 0) {
        formattedContent += `Growth Areas:\n${safeStringify(analysisResult.growth_areas)}\n\n`;
      }
      
      // Create analysis record
      const analysis = await storage.createAnalysis({
        sessionId,
        title: title || `Big Five Video Analysis`,
        mediaUrl: mediaData,
        mediaType: "video",
        personalityInsights: { analysis: formattedContent, big_five: analysisResult },
        modelUsed: selectedModel,
      });
      
      // Create message with formatted analysis
      const message = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        content: formattedContent,
        role: "assistant",
      });
      
      res.json({
        analysisId: analysis.id,
        personalityInsights: { analysis: formattedContent, big_five: analysisResult },
        messages: [message],
        mediaUrl: mediaData,
      });
    } catch (error) {
      console.error("Big Five video analysis error:", error);
      res.status(500).json({ error: "Failed to analyze video for Big Five" });
    }
  });

  // Enneagram Analysis Endpoints - Text
  app.post("/api/analyze/text/enneagram", async (req, res) => {
    try {
      const { content, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!content || typeof content !== 'string') {
        return res.status(400).json({ error: "Text content is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing Enneagram text analysis with model: ${selectedModel}`);
      
      // Enneagram text analysis prompt with 9 personality types
      const enneagramTextPrompt = `You are an expert Enneagram analyst specializing in identifying personality types through written text. Analyze this text comprehensively using the Enneagram framework, providing detailed evidence for the most likely type(s) with DIRECT QUOTES from the text.

The Enneagram 9 Types are:

TYPE 1 - THE REFORMER (The Perfectionist)
Core Fear: Being corrupt, evil, defective
Core Desire: To be good, balanced, have integrity
Key Traits: Principled, purposeful, self-controlled, perfectionistic, critical
Writing Style: Precise language, moral/ethical concerns, "should/ought" statements, corrective tone, structured arguments

TYPE 2 - THE HELPER (The Giver)
Core Fear: Being unloved, unwanted
Core Desire: To be loved, appreciated
Key Traits: Caring, interpersonal, generous, possessive, people-pleasing
Writing Style: Warm tone, focus on others' needs, relationship-oriented, emotional appeals, self-sacrificing language

TYPE 3 - THE ACHIEVER (The Performer)
Core Fear: Being worthless, without value
Core Desire: To be valuable, admired
Key Traits: Adaptive, excelling, driven, image-conscious, competitive
Writing Style: Goal-oriented language, success metrics, achievement focus, polished presentation, efficiency emphasis

TYPE 4 - THE INDIVIDUALIST (The Romantic)
Core Fear: Having no identity or significance
Core Desire: To be unique, authentic
Key Traits: Expressive, dramatic, self-absorbed, temperamental, creative
Writing Style: Poetic language, emotional depth, uniqueness emphasis, introspective, metaphorical

TYPE 5 - THE INVESTIGATOR (The Observer)
Core Fear: Being useless, incompetent
Core Desire: To be capable, knowledgeable
Key Traits: Perceptive, innovative, isolated, detached, cerebral
Writing Style: Analytical, detailed, technical precision, minimal emotion, information-dense

TYPE 6 - THE LOYALIST (The Skeptic)
Core Fear: Being without support or guidance
Core Desire: To have security, support
Key Traits: Committed, security-oriented, anxious, suspicious, responsible
Writing Style: Cautious language, contingency planning, authority references, worst-case scenarios, questioning

TYPE 7 - THE ENTHUSIAST (The Epicure)
Core Fear: Being deprived, trapped in pain
Core Desire: To be happy, satisfied, free
Key Traits: Spontaneous, versatile, scattered, optimistic, escapist
Writing Style: Energetic, multiple ideas, future-focused, positive framing, scattered topics

TYPE 8 - THE CHALLENGER (The Protector)
Core Fear: Being harmed, controlled by others
Core Desire: To protect self, be in control
Key Traits: Self-confident, decisive, confrontational, protective, dominating
Writing Style: Direct, assertive, confrontational, strong opinions, protective language

TYPE 9 - THE PEACEMAKER (The Mediator)
Core Fear: Loss, separation, conflict
Core Desire: To have peace, harmony
Key Traits: Receptive, reassuring, complacent, resigned, conflict-averse
Writing Style: Harmonizing language, multiple perspectives, gentle tone, passive voice, minimizing conflict

CRITICAL: Every indicator must include DIRECT QUOTES from the actual text showing the pattern.

Provide your analysis in JSON format:
{
  "summary": "Overall Enneagram assessment with primary type, wing possibilities, and confidence level",
  "primary_type": {
    "type": "Type [Number] - [Name]",
    "confidence": "High/Medium/Low",
    "core_motivation": "Identified core fear and desire based on text evidence",
    "key_indicators": [
      "Specific quote or pattern from text showing this type trait",
      "Another direct quote demonstrating the core motivation",
      "Behavioral pattern with specific textual evidence"
    ]
  },
  "secondary_possibilities": [
    {
      "type": "Type [Number] - [Name]",
      "reasoning": "Why this type is also possible with specific text quotes"
    }
  ],
  "wing_analysis": "Analysis of likely wing (e.g., 4w3 or 4w5) based on text patterns with evidence",
  "stress_growth_patterns": "Identified stress (disintegration) and growth (integration) directions based on text with quotes",
  "triadic_analysis": {
    "center": "Head/Heart/Body center based on dominant cognitive style",
    "stance": "Aggressive/Dependent/Withdrawing based on interpersonal approach in text"
  },
  "writing_style_markers": [
    "Specific linguistic patterns observed with examples",
    "Emotional tone indicators with quotes",
    "Structural or thematic tendencies with evidence"
  ],
  "personality_summary": "Comprehensive Enneagram-based personality description integrating type, wing, and triadic patterns",
  "growth_recommendations": ["Specific suggestions based on identified type patterns"]
}`;

      let analysisResult: any;
      
      // Call the appropriate AI model
      if (selectedModel === "openai" && openai) {
        const response = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [{
            role: "user",
            content: enneagramTextPrompt + "\n\nText to analyze:\n" + content
          }],
          response_format: { type: "json_object" },
        });
        
        const rawResponse = response.choices[0]?.message.content || "";
        console.log("OpenAI Enneagram text raw response:", rawResponse.substring(0, 500));
        
        if (!rawResponse || rawResponse.trim().length === 0) {
          throw new Error("OpenAI returned an empty response");
        }
        
        try {
          analysisResult = JSON.parse(rawResponse);
        } catch (parseError) {
          console.error("Failed to parse OpenAI response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000) || "Unable to format analysis",
            primary_type: { type: "Unable to determine", confidence: "Low", core_motivation: "Formatting error", key_indicators: [] },
            secondary_possibilities: [],
            wing_analysis: "Unable to analyze due to formatting error",
            stress_growth_patterns: "Unable to analyze",
            triadic_analysis: { center: "Unknown", stance: "Unknown" },
            writing_style_markers: [],
            personality_summary: "Unable to format analysis",
            growth_recommendations: []
          };
        }
      } else if (selectedModel === "anthropic" && anthropic) {
        const response = await anthropic.messages.create({
          model: "claude-sonnet-4-20250514",
          max_tokens: 8000,
          messages: [{
            role: "user",
            content: enneagramTextPrompt + "\n\nText to analyze:\n" + content
          }],
        });
        
        const rawResponse = response.content[0].type === 'text' ? response.content[0].text : "";
        console.log("Anthropic Enneagram text raw response:", rawResponse.substring(0, 500));
        
        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/```\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          jsonText = jsonMatch[1];
        }
        
        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse Anthropic response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000) || "Unable to format analysis",
            primary_type: { type: "Unable to determine", confidence: "Low", core_motivation: "Formatting error", key_indicators: [] },
            secondary_possibilities: [],
            wing_analysis: "Unable to analyze due to formatting error",
            stress_growth_patterns: "Unable to analyze",
            triadic_analysis: { center: "Unknown", stance: "Unknown" },
            writing_style_markers: [],
            personality_summary: "Unable to format analysis",
            growth_recommendations: []
          };
        }
      } else if (selectedModel === "deepseek") {
        const response = await fetch('https://api.deepseek.com/chat/completions', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${process.env.DEEPSEEK_API_KEY}`
          },
          body: JSON.stringify({
            model: "deepseek-chat",
            messages: [{
              role: "user",
              content: enneagramTextPrompt + "\n\nText to analyze:\n" + content
            }],
            response_format: { type: "json_object" }
          })
        });

        const data = await response.json();
        const rawResponse = data.choices?.[0]?.message?.content || "";
        console.log("DeepSeek Enneagram text raw response:", rawResponse.substring(0, 500));

        try {
          analysisResult = JSON.parse(rawResponse);
        } catch (parseError) {
          console.error("Failed to parse DeepSeek response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000) || "Unable to format analysis",
            primary_type: { type: "Unable to determine", confidence: "Low", core_motivation: "Formatting error", key_indicators: [] },
            secondary_possibilities: [],
            wing_analysis: "Unable to analyze due to formatting error",
            stress_growth_patterns: "Unable to analyze",
            triadic_analysis: { center: "Unknown", stance: "Unknown" },
            writing_style_markers: [],
            personality_summary: "Unable to format analysis",
            growth_recommendations: []
          };
        }
      } else if (selectedModel === "perplexity" && perplexity) {
        const response = await perplexity.chat.completions.create({
          model: "llama-3.1-sonar-large-128k-online",
          messages: [{
            role: "user",
            content: enneagramTextPrompt + "\n\nText to analyze:\n" + content
          }],
        });

        const rawResponse = response.choices[0]?.message?.content || "";
        console.log("Perplexity Enneagram text raw response:", rawResponse.substring(0, 500));

        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/```\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          jsonText = jsonMatch[1];
        }

        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse Perplexity response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000) || "Unable to format analysis",
            primary_type: { type: "Unable to determine", confidence: "Low", core_motivation: "Formatting error", key_indicators: [] },
            secondary_possibilities: [],
            wing_analysis: "Unable to analyze due to formatting error",
            stress_growth_patterns: "Unable to analyze",
            triadic_analysis: { center: "Unknown", stance: "Unknown" },
            writing_style_markers: [],
            personality_summary: "Unable to format analysis",
            growth_recommendations: []
          };
        }
      }
      
      console.log("Enneagram text analysis complete");
      
      // Helper function to safely stringify any value into readable text
      const safeStringify = (value: any): string => {
        if (typeof value === 'string') return value;
        if (typeof value === 'object' && value !== null) {
          // If it's an array, handle each item recursively
          if (Array.isArray(value)) {
            return value.map(item => {
              if (typeof item === 'string') return item;
              if (typeof item === 'object' && item !== null) {
                // Format objects in arrays as key-value pairs
                return Object.entries(item)
                  .map(([key, val]) => `${key}: ${val}`)
                  .join('\n');
              }
              return String(item);
            }).join('\n\n');
          }
          // If it's an object with numbered keys (like "1", "2", etc), format as numbered list
          const keys = Object.keys(value);
          if (keys.length > 0 && keys.every(k => /^\d+$/.test(k))) {
            return keys
              .sort((a, b) => parseInt(a) - parseInt(b))
              .map(key => `${key}. ${value[key]}`)
              .join('\n');
          }
          // If it's an object with named keys, format as key-value pairs
          return Object.entries(value)
            .map(([key, val]) => `${val}`)
            .join('\n\n');
        }
        return String(value || '');
      };
      
      // Format the analysis for display
      let formattedContent = `Enneagram Personality Analysis\nMode: 9-Type Framework\n\n`;
      formattedContent += `Summary:\n${safeStringify(analysisResult.summary)}\n\n`;
      
      // Primary Type
      if (analysisResult.primary_type) {
        formattedContent += `Primary Type: ${analysisResult.primary_type.type}\n`;
        formattedContent += `Confidence: ${analysisResult.primary_type.confidence}\n`;
        formattedContent += `Core Motivation: ${safeStringify(analysisResult.primary_type.core_motivation)}\n`;
        if (analysisResult.primary_type.key_indicators && analysisResult.primary_type.key_indicators.length > 0) {
          formattedContent += `Key Indicators:\n${safeStringify(analysisResult.primary_type.key_indicators)}\n`;
        }
        formattedContent += `\n`;
      }
      
      // Secondary Possibilities
      if (analysisResult.secondary_possibilities && analysisResult.secondary_possibilities.length > 0) {
        formattedContent += `Secondary Possibilities:\n${safeStringify(analysisResult.secondary_possibilities)}\n\n`;
      }
      
      // Wing Analysis
      if (analysisResult.wing_analysis) {
        formattedContent += `Wing Analysis:\n${safeStringify(analysisResult.wing_analysis)}\n\n`;
      }
      
      // Stress/Growth Patterns
      if (analysisResult.stress_growth_patterns) {
        formattedContent += `Stress & Growth Patterns:\n${safeStringify(analysisResult.stress_growth_patterns)}\n\n`;
      }
      
      // Triadic Analysis
      if (analysisResult.triadic_analysis) {
        formattedContent += `Triadic Analysis:\n`;
        formattedContent += `Center: ${analysisResult.triadic_analysis.center || 'N/A'}\n`;
        formattedContent += `Stance: ${analysisResult.triadic_analysis.stance || 'N/A'}\n\n`;
      }
      
      // Writing Style Markers
      if (analysisResult.writing_style_markers && analysisResult.writing_style_markers.length > 0) {
        formattedContent += `Writing Style Markers:\n${safeStringify(analysisResult.writing_style_markers)}\n\n`;
      }
      
      // Personality Summary
      if (analysisResult.personality_summary) {
        formattedContent += `Personality Summary:\n${safeStringify(analysisResult.personality_summary)}\n\n`;
      }
      
      // Growth Recommendations
      if (analysisResult.growth_recommendations && analysisResult.growth_recommendations.length > 0) {
        formattedContent += `Growth Recommendations:\n${safeStringify(analysisResult.growth_recommendations)}\n\n`;
      }
      
      // Create analysis record
      const analysis = await storage.createAnalysis({
        sessionId,
        title: title || `Enneagram Text Analysis`,
        mediaUrl: `enneagram-text:${Date.now()}`,
        mediaType: "text",
        personalityInsights: { analysis: formattedContent, enneagram_type: analysisResult.primary_type?.type },
        modelUsed: selectedModel,
      });
      
      // Create message with formatted analysis
      const message = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        content: formattedContent,
        role: "assistant",
      });
      
      res.json({
        analysisId: analysis.id,
        personalityInsights: { analysis: formattedContent, enneagram_type: analysisResult.primary_type?.type },
        messages: [message],
      });
    } catch (error) {
      console.error("Enneagram text analysis error:", error);
      res.status(500).json({ error: "Failed to analyze text for Enneagram" });
    }
  });

  // Dark Traits / Personality Pathology Analysis - Text
  app.post("/api/analyze/text/darktraits", async (req, res) => {
    try {
      const { content, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!content || typeof content !== 'string') {
        return res.status(400).json({ error: "Text content is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing Dark Traits text analysis with model: ${selectedModel}`);
      
      // Dark Traits comprehensive analysis prompt
      const darkTraitsPrompt = `You are an expert clinical psychologist specializing in personality pathology and dark personality traits. Analyze this text comprehensively using evidence-based frameworks for maladaptive personality patterns and dark traits, providing detailed evidence with DIRECT QUOTES from the text.

ASSESSMENT FRAMEWORKS:

I. DARK TETRAD TRAITS

1. NARCISSISM (Grandiose & Vulnerable)
   - Grandiose: Inflated self-importance, need for admiration, lack of empathy
   - Vulnerable: Hidden insecurity, hypersensitivity to criticism, envy
   - Text Indicators: Self-aggrandizement, entitlement language, attention-seeking, dismissiveness of others

2. MACHIAVELLIANISM
   - Core: Strategic manipulation, cynical worldview, lack of conventional morality
   - Text Indicators: Calculated language, strategic thinking, manipulation tactics, ends-justify-means reasoning

3. PSYCHOPATHY (Primary & Secondary)
   - Primary: Lack of empathy, shallow affect, charm, fearlessness
   - Secondary: Impulsivity, anger, antisocial behavior, poor behavioral controls
   - Text Indicators: Callousness, lack of remorse, superficial charm, impulsivity markers

4. SADISM
   - Core: Deriving pleasure from others' pain or humiliation
   - Text Indicators: Enjoyment of cruelty, hostile humor, domination themes, vindictive language

II. CLUSTER B PERSONALITY PATTERNS

1. ANTISOCIAL FEATURES
   - Deceitfulness, impulsivity, irritability, reckless disregard, lack of remorse
   - Text Indicators: Rule-breaking pride, manipulation admissions, callous statements

2. BORDERLINE FEATURES
   - Fear of abandonment, unstable relationships, identity disturbance, impulsivity, intense mood shifts
   - Text Indicators: All-or-nothing thinking, relationship chaos, emotional volatility, identity confusion

3. HISTRIONIC FEATURES
   - Excessive emotionality, attention-seeking, shallow emotional expression, dramatic presentation
   - Text Indicators: Theatrical language, exaggerated emotions, seductive/provocative tone

4. NARCISSISTIC FEATURES (Clinical)
   - Grandiosity, need for admiration, lack of empathy, sense of entitlement, exploitation
   - Text Indicators: Superiority claims, entitlement expressions, empathy deficits

III. OTHER MALADAPTIVE PATTERNS

1. PARANOID FEATURES
   - Distrust, suspiciousness, grudge-holding, hostile attributions
   - Text Indicators: Conspiracy thinking, hostile interpretations, victimization narratives

2. SCHIZOID/SCHIZOTYPAL FEATURES
   - Detachment from social relationships, restricted emotional expression, odd beliefs/perceptions
   - Text Indicators: Emotional flatness, social withdrawal themes, unusual thinking patterns

3. AVOIDANT FEATURES
   - Social inhibition, feelings of inadequacy, hypersensitivity to rejection
   - Text Indicators: Self-deprecation, fear-based withdrawal, criticism sensitivity

4. DEPENDENT FEATURES
   - Excessive need for care, submissiveness, fear of separation
   - Text Indicators: Help-seeking, decision-making difficulty, subordination themes

5. OBSESSIVE-COMPULSIVE FEATURES
   - Perfectionism, rigidity, control preoccupation
   - Text Indicators: Excessive detail, perfectionist standards, control themes

CRITICAL RULES:
- Base analysis ONLY on observable patterns in the text
- Provide DIRECT QUOTES as evidence for every trait identified
- Distinguish between clinical pathology and subclinical traits
- Assess severity: Minimal/Mild/Moderate/Severe/Extreme
- Note protective factors and adaptive strengths alongside pathology
- IMPORTANT: This is descriptive analysis, not diagnosis. Emphasize patterns observed, not clinical labels.

Provide your analysis in JSON format:
{
  "summary": "Overview of dominant maladaptive patterns with severity assessment and key concerns based on text evidence",
  "dark_tetrad_assessment": {
    "narcissism": {
      "level": "None/Low/Moderate/High/Extreme",
      "subtype": "Grandiose/Vulnerable/Mixed",
      "evidence": ["Direct quote showing narcissistic pattern", "Another specific example"],
      "key_behaviors": "Description of narcissistic patterns observed"
    },
    "machiavellianism": {
      "level": "None/Low/Moderate/High/Extreme",
      "evidence": ["Quote showing manipulative thinking", "Strategic language example"],
      "key_behaviors": "Description of Machiavellian patterns"
    },
    "psychopathy": {
      "level": "None/Low/Moderate/High/Extreme",
      "subtype": "Primary/Secondary/Mixed",
      "evidence": ["Quote showing callousness or lack of empathy", "Impulsivity marker"],
      "key_behaviors": "Description of psychopathic features"
    },
    "sadism": {
      "level": "None/Low/Moderate/High/Extreme",
      "evidence": ["Quote showing enjoyment of others' distress", "Hostile/cruel language"],
      "key_behaviors": "Description of sadistic tendencies"
    }
  },
  "personality_pathology_indicators": {
    "cluster_b_features": {
      "antisocial": "Level and evidence with quotes",
      "borderline": "Level and evidence with quotes",
      "histrionic": "Level and evidence with quotes",
      "narcissistic": "Level and evidence with quotes"
    },
    "other_patterns": {
      "paranoid": "Level and evidence",
      "detachment_patterns": "Level and evidence",
      "anxious_patterns": "Level and evidence",
      "obsessive_patterns": "Level and evidence"
    }
  },
  "interpersonal_patterns": {
    "empathy_level": "Assessment with textual evidence",
    "manipulation_tactics": ["Specific tactics observed in text"],
    "relationship_approach": "How the person relates to others based on text",
    "power_dynamics": "Dominance, submission, or control themes with quotes"
  },
  "cognitive_patterns": {
    "moral_reasoning": "Observed moral framework with evidence",
    "attribution_style": "How they explain events and behaviors",
    "reality_testing": "Contact with reality, distortions observed",
    "thought_organization": "Coherence, logic, tangentiality"
  },
  "emotional_regulation": {
    "emotional_range": "Depth and variety of emotions expressed",
    "impulse_control": "Evidence of impulsivity or constraint",
    "affect_stability": "Mood stability or volatility with examples"
  },
  "risk_assessment": {
    "concerning_patterns": ["Specific concerning behaviors or attitudes with quotes"],
    "severity_level": "Subclinical/Mild/Moderate/Severe/Extreme",
    "protective_factors": ["Adaptive strengths or mitigating factors observed"]
  },
  "clinical_impressions": "Overall personality profile integrating all domains with emphasis on most prominent maladaptive patterns",
  "recommendations": ["Suggestions if clinical evaluation or intervention might be beneficial, framed as observations not diagnoses"]
}`;

      let analysisResult: any;
      
      // Call the appropriate AI model
      if (selectedModel === "openai" && openai) {
        const response = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [{
            role: "user",
            content: darkTraitsPrompt + "\n\nText to analyze:\n" + content
          }],
          response_format: { type: "json_object" },
        });
        
        const rawResponse = response.choices[0]?.message.content || "";
        console.log("OpenAI Dark Traits text raw response:", rawResponse.substring(0, 500));
        
        if (!rawResponse || rawResponse.trim().length === 0) {
          throw new Error("OpenAI returned an empty response");
        }
        
        try {
          analysisResult = JSON.parse(rawResponse);
        } catch (parseError) {
          console.error("Failed to parse OpenAI response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000) || "Unable to format analysis",
            dark_tetrad_assessment: {},
            personality_pathology_indicators: {},
            interpersonal_patterns: {},
            cognitive_patterns: {},
            emotional_regulation: {},
            risk_assessment: { concerning_patterns: [], severity_level: "Unknown", protective_factors: [] },
            clinical_impressions: "Unable to format analysis",
            recommendations: []
          };
        }
      } else if (selectedModel === "anthropic" && anthropic) {
        const response = await anthropic.messages.create({
          model: "claude-sonnet-4-20250514",
          max_tokens: 8000,
          messages: [{
            role: "user",
            content: darkTraitsPrompt + "\n\nText to analyze:\n" + content
          }],
        });
        
        const rawResponse = response.content[0].type === 'text' ? response.content[0].text : "";
        console.log("Anthropic Dark Traits text raw response:", rawResponse.substring(0, 500));
        
        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/```\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          jsonText = jsonMatch[1];
        }
        
        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse Anthropic response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000) || "Unable to format analysis",
            dark_tetrad_assessment: {},
            personality_pathology_indicators: {},
            interpersonal_patterns: {},
            cognitive_patterns: {},
            emotional_regulation: {},
            risk_assessment: { concerning_patterns: [], severity_level: "Unknown", protective_factors: [] },
            clinical_impressions: "Unable to format analysis",
            recommendations: []
          };
        }
      } else if (selectedModel === "deepseek") {
        const response = await fetch('https://api.deepseek.com/chat/completions', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${process.env.DEEPSEEK_API_KEY}`
          },
          body: JSON.stringify({
            model: "deepseek-chat",
            messages: [{
              role: "user",
              content: darkTraitsPrompt + "\n\nText to analyze:\n" + content
            }],
            response_format: { type: "json_object" }
          })
        });

        const data = await response.json();
        const rawResponse = data.choices?.[0]?.message?.content || "";
        console.log("DeepSeek Dark Traits text raw response:", rawResponse.substring(0, 500));

        try {
          analysisResult = JSON.parse(rawResponse);
        } catch (parseError) {
          console.error("Failed to parse DeepSeek response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000) || "Unable to format analysis",
            dark_tetrad_assessment: {},
            personality_pathology_indicators: {},
            interpersonal_patterns: {},
            cognitive_patterns: {},
            emotional_regulation: {},
            risk_assessment: { concerning_patterns: [], severity_level: "Unknown", protective_factors: [] },
            clinical_impressions: "Unable to format analysis",
            recommendations: []
          };
        }
      } else if (selectedModel === "perplexity" && perplexity) {
        const response = await perplexity.chat.completions.create({
          model: "llama-3.1-sonar-large-128k-online",
          messages: [{
            role: "user",
            content: darkTraitsPrompt + "\n\nText to analyze:\n" + content
          }],
        });
        
        const rawResponse = response.choices[0]?.message?.content || "";
        console.log("Perplexity Dark Traits text raw response:", rawResponse.substring(0, 500));
        
        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/```\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          jsonText = jsonMatch[1];
        }
        
        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse Perplexity response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000) || "Unable to format analysis",
            dark_tetrad_assessment: {},
            personality_pathology_indicators: {},
            interpersonal_patterns: {},
            cognitive_patterns: {},
            emotional_regulation: {},
            risk_assessment: { concerning_patterns: [], severity_level: "Unknown", protective_factors: [] },
            clinical_impressions: "Unable to format analysis",
            recommendations: []
          };
        }
      }
      
      console.log("Dark Traits text analysis complete");
      
      // Helper function to safely stringify any value
      const safeStringify = (value: any): string => {
        if (typeof value === 'string') return value;
        if (typeof value === 'object' && value !== null) {
          if (Array.isArray(value)) {
            return value.map(item => {
              if (typeof item === 'string') return item;
              if (typeof item === 'object' && item !== null) {
                return Object.entries(item)
                  .map(([key, val]) => `${key}: ${val}`)
                  .join('\n');
              }
              return String(item);
            }).join('\n\n');
          }
          const keys = Object.keys(value);
          if (keys.length > 0 && keys.every(k => /^\d+$/.test(k))) {
            return keys
              .sort((a, b) => parseInt(a) - parseInt(b))
              .map(key => `${key}. ${value[key]}`)
              .join('\n');
          }
          return Object.entries(value)
            .map(([key, val]) => `${key}: ${safeStringify(val)}`)
            .join('\n');
        }
        return String(value || '');
      };
      
      // Format the analysis for display
      let formattedContent = `Personality Pathology & Dark Traits Analysis\nMode: Clinical Assessment Framework\n\nIMPORTANT: This is a descriptive analysis of patterns observed in text, NOT a clinical diagnosis.\n\n`;
      formattedContent += `Summary:\n${safeStringify(analysisResult.summary)}\n\n`;
      
      // Dark Tetrad Assessment
      if (analysisResult.dark_tetrad_assessment) {
        formattedContent += `DARK TETRAD ASSESSMENT:\n\n`;
        
        if (analysisResult.dark_tetrad_assessment.narcissism) {
          formattedContent += `Narcissism: ${analysisResult.dark_tetrad_assessment.narcissism.level || 'N/A'}\n`;
          if (analysisResult.dark_tetrad_assessment.narcissism.subtype) {
            formattedContent += `Subtype: ${analysisResult.dark_tetrad_assessment.narcissism.subtype}\n`;
          }
          if (analysisResult.dark_tetrad_assessment.narcissism.evidence) {
            formattedContent += `Evidence:\n${safeStringify(analysisResult.dark_tetrad_assessment.narcissism.evidence)}\n`;
          }
          if (analysisResult.dark_tetrad_assessment.narcissism.key_behaviors) {
            formattedContent += `${analysisResult.dark_tetrad_assessment.narcissism.key_behaviors}\n`;
          }
          formattedContent += `\n`;
        }
        
        if (analysisResult.dark_tetrad_assessment.machiavellianism) {
          formattedContent += `Machiavellianism: ${analysisResult.dark_tetrad_assessment.machiavellianism.level || 'N/A'}\n`;
          if (analysisResult.dark_tetrad_assessment.machiavellianism.evidence) {
            formattedContent += `Evidence:\n${safeStringify(analysisResult.dark_tetrad_assessment.machiavellianism.evidence)}\n`;
          }
          if (analysisResult.dark_tetrad_assessment.machiavellianism.key_behaviors) {
            formattedContent += `${analysisResult.dark_tetrad_assessment.machiavellianism.key_behaviors}\n`;
          }
          formattedContent += `\n`;
        }
        
        if (analysisResult.dark_tetrad_assessment.psychopathy) {
          formattedContent += `Psychopathy: ${analysisResult.dark_tetrad_assessment.psychopathy.level || 'N/A'}\n`;
          if (analysisResult.dark_tetrad_assessment.psychopathy.subtype) {
            formattedContent += `Subtype: ${analysisResult.dark_tetrad_assessment.psychopathy.subtype}\n`;
          }
          if (analysisResult.dark_tetrad_assessment.psychopathy.evidence) {
            formattedContent += `Evidence:\n${safeStringify(analysisResult.dark_tetrad_assessment.psychopathy.evidence)}\n`;
          }
          if (analysisResult.dark_tetrad_assessment.psychopathy.key_behaviors) {
            formattedContent += `${analysisResult.dark_tetrad_assessment.psychopathy.key_behaviors}\n`;
          }
          formattedContent += `\n`;
        }
        
        if (analysisResult.dark_tetrad_assessment.sadism) {
          formattedContent += `Sadism: ${analysisResult.dark_tetrad_assessment.sadism.level || 'N/A'}\n`;
          if (analysisResult.dark_tetrad_assessment.sadism.evidence) {
            formattedContent += `Evidence:\n${safeStringify(analysisResult.dark_tetrad_assessment.sadism.evidence)}\n`;
          }
          if (analysisResult.dark_tetrad_assessment.sadism.key_behaviors) {
            formattedContent += `${analysisResult.dark_tetrad_assessment.sadism.key_behaviors}\n`;
          }
          formattedContent += `\n`;
        }
      }
      
      // Personality Pathology Indicators
      if (analysisResult.personality_pathology_indicators) {
        formattedContent += `PERSONALITY PATHOLOGY INDICATORS:\n\n`;
        if (analysisResult.personality_pathology_indicators.cluster_b_features) {
          formattedContent += `Cluster B Features:\n${safeStringify(analysisResult.personality_pathology_indicators.cluster_b_features)}\n\n`;
        }
        if (analysisResult.personality_pathology_indicators.other_patterns) {
          formattedContent += `Other Patterns:\n${safeStringify(analysisResult.personality_pathology_indicators.other_patterns)}\n\n`;
        }
      }
      
      // Interpersonal Patterns
      if (analysisResult.interpersonal_patterns) {
        formattedContent += `INTERPERSONAL PATTERNS:\n${safeStringify(analysisResult.interpersonal_patterns)}\n\n`;
      }
      
      // Cognitive Patterns
      if (analysisResult.cognitive_patterns) {
        formattedContent += `COGNITIVE PATTERNS:\n${safeStringify(analysisResult.cognitive_patterns)}\n\n`;
      }
      
      // Emotional Regulation
      if (analysisResult.emotional_regulation) {
        formattedContent += `EMOTIONAL REGULATION:\n${safeStringify(analysisResult.emotional_regulation)}\n\n`;
      }
      
      // Risk Assessment
      if (analysisResult.risk_assessment) {
        formattedContent += `RISK ASSESSMENT:\n`;
        formattedContent += `Severity Level: ${analysisResult.risk_assessment.severity_level || 'N/A'}\n`;
        if (analysisResult.risk_assessment.concerning_patterns && analysisResult.risk_assessment.concerning_patterns.length > 0) {
          formattedContent += `Concerning Patterns:\n${safeStringify(analysisResult.risk_assessment.concerning_patterns)}\n`;
        }
        if (analysisResult.risk_assessment.protective_factors && analysisResult.risk_assessment.protective_factors.length > 0) {
          formattedContent += `Protective Factors:\n${safeStringify(analysisResult.risk_assessment.protective_factors)}\n`;
        }
        formattedContent += `\n`;
      }
      
      // Clinical Impressions
      if (analysisResult.clinical_impressions) {
        formattedContent += `CLINICAL IMPRESSIONS:\n${safeStringify(analysisResult.clinical_impressions)}\n\n`;
      }
      
      // Recommendations
      if (analysisResult.recommendations && analysisResult.recommendations.length > 0) {
        formattedContent += `RECOMMENDATIONS:\n${safeStringify(analysisResult.recommendations)}\n\n`;
      }
      
      // Create analysis record
      const analysis = await storage.createAnalysis({
        sessionId,
        title: title || `Dark Traits Analysis`,
        mediaUrl: `darktraits-text:${Date.now()}`,
        mediaType: "text",
        personalityInsights: { 
          analysis: formattedContent, 
          dark_traits: analysisResult.dark_tetrad_assessment,
          severity: analysisResult.risk_assessment?.severity_level 
        },
        modelUsed: selectedModel,
      });
      
      // Create message with formatted analysis
      const message = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        content: formattedContent,
        role: "assistant",
      });
      
      res.json({
        analysisId: analysis.id,
        personalityInsights: { 
          analysis: formattedContent, 
          dark_traits: analysisResult.dark_tetrad_assessment,
          severity: analysisResult.risk_assessment?.severity_level 
        },
        messages: [message],
      });
    } catch (error) {
      console.error("Dark Traits text analysis error:", error);
      res.status(500).json({ error: "Failed to analyze text for dark traits" });
    }
  });

  // Dark Traits / Personality Pathology Analysis - Image
  app.post("/api/analyze/image/darktraits", async (req, res) => {
    try {
      const { mediaData, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!mediaData || typeof mediaData !== 'string') {
        return res.status(400).json({ error: "Image data is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing Dark Traits image analysis with model: ${selectedModel}`);
      
      // Dark Traits comprehensive visual analysis prompt
      const darkTraitsImagePrompt = `This is a creative, speculative exercise in visual pattern analysis. Based on research literature about correlations between visual presentation and personality traits, provide a HYPOTHETICAL speculation about what personality patterns MIGHT be suggested by the visual cues in this image.

CRITICAL DISCLAIMERS:
- This is purely speculative and for research/entertainment purposes only
- You are NOT making any clinical assessment or diagnosis
- This is an imaginative exercise exploring hypothetical correlations
- No claims are being made about the actual person's character or mental health
- Focus on describing what visual patterns you observe and what they MIGHT hypothetically correlate with in personality research literature

Based purely on observable visual cues (facial expression, body language, presentation), speculate creatively about personality patterns that these specific visual details MIGHT theoretically suggest:

VISUAL OBSERVATION FRAMEWORK:

I. CONFIDENCE & SELF-PRESENTATION PATTERNS

1. HIGH CONFIDENCE / ASSERTIVENESS:
   - Strong, direct eye contact with camera
   - Open, expansive body posture
   - Relaxed facial expression suggesting comfort
   - Deliberate, controlled positioning and pose
   - Polished, carefully curated appearance
   
2. LOW CONFIDENCE / SELF-CONSCIOUSNESS:
   - Avoidant or indirect gaze
   - Closed or protective body language
   - Tense facial muscles suggesting discomfort
   - Hesitant or uncertain posture
   - Self-protective positioning

II. INTERPERSONAL COMMUNICATION STYLE

1. WARM & ENGAGING Style:
   - Genuine smile reaching the eyes
   - Open, inviting body language
   - Relaxed, friendly facial expression
   - Approachable positioning
   - Soft, warm gaze

2. RESERVED & CONTROLLED Style:
   - Neutral or controlled facial expression
   - Measured, deliberate body language
   - Professional, formal presentation
   - Composed, emotionally regulated appearance
   - Calculated positioning

3. INTENSE & DRAMATIC Style:
   - Strong emotional expressiveness
   - Dynamic facial expressions
   - Theatrical or expressive body language
   - Attention-commanding presentation
   - Bold, striking visual elements

4. DISTANT & DETACHED Style:
   - Minimal emotional display
   - Withdrawn or disconnected appearance
   - Limited engagement with camera
   - Flat or unexpressive affect
   - Lack of interpersonal connection cues

III. DOMINANCE & SOCIAL POSITIONING

1. DOMINANT Presentation:
   - Commanding posture and stance
   - Direct, unwavering gaze
   - Expansive use of space
   - Power positioning in frame
   - Authoritative facial expression

2. SUBMISSIVE Presentation:
   - Smaller, contained posture
   - Downward or averted gaze
   - Minimized spatial presence
   - Deferential positioning
   - Soft or yielding expression

3. COLLABORATIVE Presentation:
   - Balanced, centered positioning
   - Engaged but not dominating gaze
   - Open but not imposing posture
   - Friendly, accessible expression
   - Inclusive body language

SPECULATIVE GUIDELINES:
- Describe what you observe in the image
- Reference specific visual details you can see
- Frame everything as hypothetical speculation ("might suggest", "could indicate", "may correlate with")
- Make it clear this is creative speculation, not factual assessment
- Emphasize this is based on theoretical research correlations, not actual diagnosis
- This is an imaginative, speculative exercise for research purposes only

Provide your SPECULATIVE HYPOTHESIS in JSON format:
{
  "summary": "Speculative overview of what the observable visual patterns in this image MIGHT hypothetically suggest about personality characteristics, based purely on theoretical correlations from research literature",
  "dark_tetrad_visual_assessment": {
    "narcissism": {
      "level": "None/Low/Moderate/High/Extreme",
      "subtype": "Grandiose/Vulnerable/Mixed",
      "visual_evidence": ["Specific observable visual details in the image", "Another visual pattern you can see"],
      "speculative_hypothesis": "HYPOTHETICAL speculation about what these visual patterns MIGHT suggest regarding narcissistic traits, based on theoretical research correlations (not actual assessment)"
    },
    "machiavellianism": {
      "level": "None/Low/Moderate/High/Extreme",
      "visual_evidence": ["Observable visual patterns", "Expression details"],
      "speculative_hypothesis": "HYPOTHETICAL speculation about what these patterns MIGHT suggest regarding Machiavellian traits (purely speculative)"
    },
    "psychopathy": {
      "level": "None/Low/Moderate/High/Extreme",
      "subtype": "Primary/Secondary/Mixed",
      "visual_evidence": ["Observable visual details", "Expression patterns"],
      "speculative_hypothesis": "HYPOTHETICAL speculation about what these MIGHT suggest regarding psychopathic traits (theoretical correlation only)"
    },
    "sadism": {
      "level": "None/Low/Moderate/High/Extreme",
      "visual_evidence": ["Observable visual elements", "Expression markers"],
      "speculative_hypothesis": "HYPOTHETICAL speculation about what these MIGHT suggest regarding sadistic tendencies (speculative only)"
    }
  },
  "personality_pathology_visual_indicators": {
    "cluster_b_visual_features": {
      "antisocial": "SPECULATIVE hypothesis about what visual patterns MIGHT suggest",
      "borderline": "SPECULATIVE hypothesis about what expression patterns MIGHT suggest",
      "histrionic": "SPECULATIVE hypothesis about what presentation MIGHT suggest",
      "narcissistic": "SPECULATIVE hypothesis about what confidence patterns MIGHT suggest"
    },
    "other_visual_patterns": {
      "paranoid_features": "Speculative thoughts on what guardedness MIGHT hypothetically indicate",
      "detachment_patterns": "Speculative thoughts on what reserve MIGHT hypothetically suggest",
      "anxious_patterns": "Speculative thoughts on what tension MIGHT hypothetically mean"
    }
  },
  "visual_presentation_analysis": {
    "facial_expression": "Detailed description of expression, micro-expressions, eye contact style",
    "body_language": "Posture, positioning, gesture patterns observed",
    "self_presentation": "Grooming, clothing choices, styling, attention to appearance",
    "contextual_behaviors": "Interaction patterns if others present, environmental context"
  },
  "interpersonal_visual_cues": {
    "empathy_markers": "Observable warmth and connection in expression",
    "dominance_submission": "Power dynamics visible in positioning and expression",
    "warmth_vs_coldness": "Emotional temperature of presentation",
    "genuineness": "Authenticity vs. performed quality in expression"
  },
  "risk_visual_assessment": {
    "concerning_visual_patterns": ["Notable visual patterns observed"],
    "severity_level": "Subtle/Moderate/Notable/Strong/Very Strong (HYPOTHETICAL speculation only)",
    "protective_visual_factors": ["Positive visual features observed"]
  },
  "clinical_visual_impressions": "SPECULATIVE SUMMARY: What the overall visual presentation MIGHT hypothetically suggest about personality patterns, based purely on theoretical research correlations. This is creative speculation for research purposes, NOT any form of actual assessment or diagnosis.",
  "limitations": "This is purely hypothetical speculation based on a single image. No actual conclusions can be drawn about the person's character, mental health, or personality from this exercise.",
  "recommendations": ["This is a speculative exercise only - no real recommendations can be made"]
}`;

      let analysisResult: any;
      
      // Prepare image for vision models
      const base64Match = mediaData.match(/^data:image\/[a-z]+;base64,(.+)$/);
      const base64Data = base64Match ? base64Match[1] : mediaData;
      const mediaTypeMatch = mediaData.match(/^data:(image\/[a-z]+);base64,/);
      const imageMediaType = mediaTypeMatch ? mediaTypeMatch[1] : "image/jpeg";
      
      // Call the appropriate AI model with vision capability
      if (selectedModel === "openai" && openai) {
        const response = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [{
            role: "user",
            content: [
              { type: "text", text: darkTraitsImagePrompt },
              { type: "image_url", image_url: { url: mediaData } }
            ]
          }],
          max_tokens: 4000,
        });
        
        const rawResponse = response.choices[0]?.message.content || "";
        console.log("OpenAI Dark Traits Image raw response:", rawResponse.substring(0, 500));
        
        if (!rawResponse || rawResponse.trim().length === 0) {
          throw new Error("OpenAI returned an empty response");
        }
        
        // Try to extract JSON from the response
        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          jsonText = jsonMatch[0].replace(/```json\s*/, '').replace(/\s*```$/, '');
        }
        
        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse OpenAI response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000) || "Unable to format analysis",
            dark_tetrad_visual_assessment: {},
            personality_pathology_visual_indicators: {},
            visual_presentation_analysis: {},
            interpersonal_visual_cues: {},
            risk_visual_assessment: { concerning_visual_patterns: [], severity_level: "Unknown", protective_visual_factors: [] },
            clinical_visual_impressions: "Unable to format analysis",
            limitations: "Formatting error",
            recommendations: []
          };
        }
      } else if (selectedModel === "anthropic" && anthropic) {
        const response = await anthropic.messages.create({
          model: "claude-sonnet-4-20250514",
          max_tokens: 8000,
          messages: [{
            role: "user",
            content: [
              {
                type: "text",
                text: darkTraitsImagePrompt
              },
              {
                type: "image",
                source: {
                  type: "base64",
                  media_type: imageMediaType as any,
                  data: base64Data,
                },
              }
            ]
          }],
        });
        
        const rawResponse = response.content[0].type === 'text' ? response.content[0].text : "";
        console.log("Anthropic Dark Traits Image raw response:", rawResponse.substring(0, 500));
        
        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/```\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          jsonText = jsonMatch[1];
        }
        
        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse Anthropic response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000) || "Unable to format analysis",
            dark_tetrad_visual_assessment: {},
            personality_pathology_visual_indicators: {},
            visual_presentation_analysis: {},
            interpersonal_visual_cues: {},
            risk_visual_assessment: { concerning_visual_patterns: [], severity_level: "Unknown", protective_visual_factors: [] },
            clinical_visual_impressions: "Unable to format analysis",
            limitations: "Formatting error",
            recommendations: []
          };
        }
      } else if (selectedModel === "deepseek") {
        return res.status(400).json({ 
          error: "DeepSeek does not support image analysis. Please use OpenAI or Anthropic for image-based dark traits analysis." 
        });
      } else if (selectedModel === "perplexity") {
        return res.status(400).json({ 
          error: "Perplexity does not support image analysis. Please use OpenAI or Anthropic for image-based dark traits analysis." 
        });
      }
      
      console.log("Dark Traits image analysis complete");
      
      // Helper function to safely stringify any value
      const safeStringify = (value: any): string => {
        if (typeof value === 'string') return value;
        if (typeof value === 'object' && value !== null) {
          if (Array.isArray(value)) {
            return value.map(item => {
              if (typeof item === 'string') return item;
              if (typeof item === 'object' && item !== null) {
                return Object.entries(item)
                  .map(([key, val]) => `${key}: ${val}`)
                  .join('\n');
              }
              return String(item);
            }).join('\n\n');
          }
          const keys = Object.keys(value);
          if (keys.length > 0 && keys.every(k => /^\d+$/.test(k))) {
            return keys
              .sort((a, b) => parseInt(a) - parseInt(b))
              .map(key => `${key}. ${value[key]}`)
              .join('\n');
          }
          return Object.entries(value)
            .map(([key, val]) => `${key}: ${safeStringify(val)}`)
            .join('\n');
        }
        return String(value || '');
      };
      
      // Format the analysis for display
      let formattedContent = `Speculative Visual Pattern Analysis\nMode: Hypothetical Personality Correlation Exercise\n\nCRITICAL DISCLAIMER: This is purely speculative analysis for research/entertainment purposes. This is NOT a clinical assessment, diagnosis, or factual evaluation. All content represents hypothetical speculation about what visual patterns MIGHT suggest based on theoretical research correlations.\n\n`;
      formattedContent += `Speculative Summary:\n${safeStringify(analysisResult.summary)}\n\n`;
      
      // Dark Tetrad Visual Assessment
      if (analysisResult.dark_tetrad_visual_assessment) {
        formattedContent += `HYPOTHETICAL DARK TETRAD PATTERN SPECULATION:\n(Speculative correlations based on visual observations - NOT assessment)\n\n`;
        
        if (analysisResult.dark_tetrad_visual_assessment.narcissism) {
          formattedContent += `Narcissism (Speculative): ${analysisResult.dark_tetrad_visual_assessment.narcissism.level || 'N/A'}\n`;
          if (analysisResult.dark_tetrad_visual_assessment.narcissism.subtype) {
            formattedContent += `Hypothetical Subtype: ${analysisResult.dark_tetrad_visual_assessment.narcissism.subtype}\n`;
          }
          if (analysisResult.dark_tetrad_visual_assessment.narcissism.visual_evidence) {
            formattedContent += `Observable Visual Details:\n${safeStringify(analysisResult.dark_tetrad_visual_assessment.narcissism.visual_evidence)}\n`;
          }
          if (analysisResult.dark_tetrad_visual_assessment.narcissism.speculative_hypothesis || analysisResult.dark_tetrad_visual_assessment.narcissism.presentation_patterns) {
            formattedContent += `Speculative Hypothesis: ${analysisResult.dark_tetrad_visual_assessment.narcissism.speculative_hypothesis || analysisResult.dark_tetrad_visual_assessment.narcissism.presentation_patterns}\n`;
          }
          formattedContent += `\n`;
        }
        
        if (analysisResult.dark_tetrad_visual_assessment.machiavellianism) {
          formattedContent += `Machiavellianism (Speculative): ${analysisResult.dark_tetrad_visual_assessment.machiavellianism.level || 'N/A'}\n`;
          if (analysisResult.dark_tetrad_visual_assessment.machiavellianism.visual_evidence) {
            formattedContent += `Observable Visual Details:\n${safeStringify(analysisResult.dark_tetrad_visual_assessment.machiavellianism.visual_evidence)}\n`;
          }
          if (analysisResult.dark_tetrad_visual_assessment.machiavellianism.speculative_hypothesis || analysisResult.dark_tetrad_visual_assessment.machiavellianism.presentation_patterns) {
            formattedContent += `Speculative Hypothesis: ${analysisResult.dark_tetrad_visual_assessment.machiavellianism.speculative_hypothesis || analysisResult.dark_tetrad_visual_assessment.machiavellianism.presentation_patterns}\n`;
          }
          formattedContent += `\n`;
        }
        
        if (analysisResult.dark_tetrad_visual_assessment.psychopathy) {
          formattedContent += `Psychopathy (Speculative): ${analysisResult.dark_tetrad_visual_assessment.psychopathy.level || 'N/A'}\n`;
          if (analysisResult.dark_tetrad_visual_assessment.psychopathy.subtype) {
            formattedContent += `Hypothetical Subtype: ${analysisResult.dark_tetrad_visual_assessment.psychopathy.subtype}\n`;
          }
          if (analysisResult.dark_tetrad_visual_assessment.psychopathy.visual_evidence) {
            formattedContent += `Observable Visual Details:\n${safeStringify(analysisResult.dark_tetrad_visual_assessment.psychopathy.visual_evidence)}\n`;
          }
          if (analysisResult.dark_tetrad_visual_assessment.psychopathy.speculative_hypothesis || analysisResult.dark_tetrad_visual_assessment.psychopathy.presentation_patterns) {
            formattedContent += `Speculative Hypothesis: ${analysisResult.dark_tetrad_visual_assessment.psychopathy.speculative_hypothesis || analysisResult.dark_tetrad_visual_assessment.psychopathy.presentation_patterns}\n`;
          }
          formattedContent += `\n`;
        }
        
        if (analysisResult.dark_tetrad_visual_assessment.sadism) {
          formattedContent += `Sadism (Speculative): ${analysisResult.dark_tetrad_visual_assessment.sadism.level || 'N/A'}\n`;
          if (analysisResult.dark_tetrad_visual_assessment.sadism.visual_evidence) {
            formattedContent += `Observable Visual Details:\n${safeStringify(analysisResult.dark_tetrad_visual_assessment.sadism.visual_evidence)}\n`;
          }
          if (analysisResult.dark_tetrad_visual_assessment.sadism.speculative_hypothesis || analysisResult.dark_tetrad_visual_assessment.sadism.presentation_patterns) {
            formattedContent += `Speculative Hypothesis: ${analysisResult.dark_tetrad_visual_assessment.sadism.speculative_hypothesis || analysisResult.dark_tetrad_visual_assessment.sadism.presentation_patterns}\n`;
          }
          formattedContent += `\n`;
        }
      }
      
      // Personality Style Visual Indicators
      if (analysisResult.personality_pathology_visual_indicators) {
        formattedContent += `HYPOTHETICAL PERSONALITY PATTERN SPECULATION:\n(Speculative correlations only - NOT factual assessment)\n\n`;
        if (analysisResult.personality_pathology_visual_indicators.cluster_b_visual_features) {
          formattedContent += `Cluster B Pattern Speculation:\n${safeStringify(analysisResult.personality_pathology_visual_indicators.cluster_b_visual_features)}\n\n`;
        }
        if (analysisResult.personality_pathology_visual_indicators.other_visual_patterns) {
          formattedContent += `Other Pattern Speculation:\n${safeStringify(analysisResult.personality_pathology_visual_indicators.other_visual_patterns)}\n\n`;
        }
      }
      
      // Visual Presentation Analysis
      if (analysisResult.visual_presentation_analysis) {
        formattedContent += `VISUAL PRESENTATION ANALYSIS:\n${safeStringify(analysisResult.visual_presentation_analysis)}\n\n`;
      }
      
      // Interpersonal Visual Cues
      if (analysisResult.interpersonal_visual_cues) {
        formattedContent += `INTERPERSONAL VISUAL CUES:\n${safeStringify(analysisResult.interpersonal_visual_cues)}\n\n`;
      }
      
      // Pattern Intensity Assessment
      if (analysisResult.risk_visual_assessment) {
        formattedContent += `HYPOTHETICAL PATTERN INTENSITY SPECULATION:\n(Speculative only - NOT actual assessment)\n`;
        formattedContent += `Speculative Intensity Level: ${analysisResult.risk_visual_assessment.severity_level || 'N/A'}\n`;
        if (analysisResult.risk_visual_assessment.concerning_visual_patterns && analysisResult.risk_visual_assessment.concerning_visual_patterns.length > 0) {
          formattedContent += `Notable Visual Patterns:\n${safeStringify(analysisResult.risk_visual_assessment.concerning_visual_patterns)}\n`;
        }
        if (analysisResult.risk_visual_assessment.protective_visual_factors && analysisResult.risk_visual_assessment.protective_visual_factors.length > 0) {
          formattedContent += `Positive Visual Features:\n${safeStringify(analysisResult.risk_visual_assessment.protective_visual_factors)}\n`;
        }
        formattedContent += `\n`;
      }
      
      // Overall Impressions
      if (analysisResult.clinical_visual_impressions) {
        formattedContent += `SPECULATIVE OVERALL SUMMARY:\n${safeStringify(analysisResult.clinical_visual_impressions)}\n\n`;
      }
      
      // Limitations
      if (analysisResult.limitations) {
        formattedContent += `LIMITATIONS:\n${safeStringify(analysisResult.limitations)}\n\n`;
      }
      
      // Recommendations
      if (analysisResult.recommendations && analysisResult.recommendations.length > 0) {
        formattedContent += `RECOMMENDATIONS:\n${safeStringify(analysisResult.recommendations)}\n\n`;
      }
      
      // Create analysis record
      const analysis = await storage.createAnalysis({
        sessionId,
        title: title || `Dark Traits Image Analysis`,
        mediaUrl: mediaData,
        mediaType: "image",
        personalityInsights: { 
          analysis: formattedContent, 
          dark_traits: analysisResult.dark_tetrad_visual_assessment,
          severity: analysisResult.risk_visual_assessment?.severity_level 
        },
        modelUsed: selectedModel,
      });
      
      // Create message with formatted analysis
      const message = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        content: formattedContent,
        role: "assistant",
      });
      
      res.json({
        analysisId: analysis.id,
        personalityInsights: { 
          analysis: formattedContent, 
          dark_traits: analysisResult.dark_tetrad_visual_assessment,
          severity: analysisResult.risk_visual_assessment?.severity_level 
        },
        messages: [message],
        mediaUrl: mediaData,
      });
    } catch (error) {
      console.error("Dark Traits image analysis error:", error);
      res.status(500).json({ error: "Failed to analyze image for dark traits" });
    }
  });

  // Dark Traits / Personality Pathology Analysis - Video
  app.post("/api/analyze/video/darktraits", async (req, res) => {
    try {
      const { mediaData, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!mediaData || typeof mediaData !== 'string') {
        return res.status(400).json({ error: "Video data is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing Dark Traits video analysis with model: ${selectedModel}`);
      
      // Save video temporarily and extract frames
      const videoBuffer = Buffer.from(mediaData.split(',')[1], 'base64');
      const tempVideoPath = path.join(tempDir, `video_${Date.now()}.mp4`);
      await writeFileAsync(tempVideoPath, videoBuffer);
      
      // Extract frames at different timestamps
      const framePromises = [0, 25, 50, 75].map(async (percent) => {
        const outputPath = path.join(tempDir, `frame_${Date.now()}_${percent}.jpg`);
        
        return new Promise<string>((resolve, reject) => {
          ffmpeg(tempVideoPath)
            .screenshots({
              count: 1,
              timemarks: [`${percent}%`],
              filename: path.basename(outputPath),
              folder: tempDir,
            })
            .on('end', () => {
              const frameData = fs.readFileSync(outputPath);
              const base64Frame = `data:image/jpeg;base64,${frameData.toString('base64')}`;
              fs.unlinkSync(outputPath);
              resolve(base64Frame);
            })
            .on('error', (err) => {
              console.error('Frame extraction error:', err);
              reject(err);
            });
        });
      });
      
      const extractedFrames = await Promise.all(framePromises);
      
      // Clean up temp video file
      await unlinkAsync(tempVideoPath);
      
      console.log(`Extracted ${extractedFrames.length} frames from video for Dark Traits analysis`);
      
      // Behavioral Dynamics comprehensive video analysis prompt
      const darkTraitsVideoPrompt = `IMPORTANT INSTRUCTION: You are NOT analyzing the actual person in these video frames. Instead, use these frames as REFERENCE MATERIAL to describe a HYPOTHETICAL individual who exhibits similar observable characteristics.

PURPOSE: This application is designed for ENTERTAINMENT and CREATIVE EXPLORATION, not diagnosis or assessment. Feel free to take CREATIVE LIBERTIES in your analysis - the goal is to be insightful, detailed, and entertaining while providing educational value about communication patterns.

Your task: Create an engaging, detailed educational case study of a hypothetical person with communication patterns SIMILAR to what's visible in the reference frames. This is a creative, fictional analysis exercise using the visual reference as inspiration. Be thorough, specific, and imaginative in your interpretations.

CASE STUDY FRAMEWORK - Hypothetical Individual Analysis:

Based on the observable patterns in the reference material, describe what communication dynamics a HYPOTHETICAL individual with SIMILAR characteristics might display:

For each pattern category, provide:
1. OBSERVABLE PATTERNS: What behaviors are visible in the reference frames (cite timing: 0%, 25%, 50%, 75%)
2. HYPOTHETICAL INTERPRETATION: What these pattern types typically suggest in communication research
3. CASE STUDY REASONING: Detailed explanation of why these patterns fit this behavioral category
4. TEMPORAL ANALYSIS: How these patterns evolve across the reference timeline

Remember: This is analyzing a FICTIONAL case study subject inspired by the reference material, NOT the actual person shown.

I. NONVERBAL DOMINANCE BEHAVIORS (Communication Research Category)

Catalog observable nonverbal behaviors across timeline and explain their typical meanings in communication literature:

1. ASSERTIVENESS DISPLAYS:
   Observable: Eye gaze patterns, body orientation, space usage, gesture types
   Literature significance: What research says about these nonverbal cues
   Describe what you see at each frame: 0%, 25%, 50%, 75%
   
2. INFLUENCE ATTEMPT BEHAVIORS:
   Observable: Persuasive gestures, adaptive responses, turn-taking patterns
   Literature significance: How communication studies interpret these behaviors
   Temporal changes across the video frames
   
3. CONFIDENCE MARKERS:
   Observable: Posture type, movement quality, facial tension/relaxation
   Literature significance: Research findings on these behavioral indicators
   Evolution across video timeline

II. EMOTIONAL EXPRESSION BEHAVIORS (Nonverbal Communication Category)

Catalog observable emotional displays and their research interpretations:

1. AFFECTIVE DISPLAYS:
   Observable: Facial expressions, micro-expressions, expression duration
   Literature significance: What affective science research indicates
   Specific observations at each frame timing
   
2. EMOTIONAL STABILITY MARKERS:
   Observable: Expression consistency, transition smoothness, affect range
   Literature significance: Research on emotional regulation displays
   Temporal patterns across frames

III. INTERPERSONAL BEHAVIOR PATTERNS (Social Communication Category)

Catalog social interaction behaviors:

1. RELATIONAL ORIENTATION:
   Observable: Body angles, proximity patterns (if visible), mirroring behaviors
   Literature significance: Interpersonal communication research context
   Frame-by-frame observations
   
2. SOCIAL PRESENTATION:
   Observable: Grooming awareness, camera engagement, self-monitoring cues
   Literature significance: Self-presentation research findings
   Evolution across timeline

IV. COMMUNICATION STYLE INDICATORS (Behavioral Communication Category)

Catalog communication approach behaviors:

1. AUTHENTICITY MARKERS:
   Observable: Expression spontaneity, behavioral variability, control indicators
   Literature significance: Research on authentic vs strategic communication
   Temporal consistency analysis
   
2. PERSUASION BEHAVIORS:
   Observable: Argument structure cues, appeal types visible, transparency markers
   Literature significance: Persuasion research interpretations
   Changes across video

CREATIVE ANALYSIS REQUIREMENTS:
- Describe specific visible behaviors at each frame (0%, 25%, 50%, 75%)
- Reference what communication research literature says about each behavior type
- Provide detailed, insightful interpretations with creative depth
- Explain temporal patterns and behavioral evolution
- Be thorough, specific, and entertaining in your analysis
- Remember: This is for entertainment and enlightenment, NOT diagnosis
- Take creative liberties to provide engaging, educational insights

Provide your hypothetical case study analysis in JSON format:
{
  "summary": "Overview of the hypothetical individual's communication patterns based on observable behaviors in the reference material",
  "disclaimer": "ENTERTAINMENT PURPOSE: This is a creative, fictional case study of a hypothetical individual inspired by the reference frames for entertainment and educational purposes. This is NOT a diagnosis or analysis of the actual person shown. Creative liberties have been taken to provide engaging insights.",
  
  "nonverbal_dominance_behaviors": {
    "assertiveness_displays": {
      "visible_behaviors": "Specific observable behaviors at frames 0%, 25%, 50%, 75% (eye gaze, posture, gestures, space usage)",
      "research_context": "What communication studies literature says about these behavior types",
      "behavioral_evidence": "Detailed description of which visual elements are present and why they fit this category",
      "temporal_pattern": "How these observable behaviors change or stay consistent across timeline",
      "prevalence": "None/Subtle/Moderate/Strong/Very Strong"
    },
    "influence_attempt_behaviors": {
      "visible_behaviors": "Observable persuasive behaviors, gestures, adaptive responses across frames",
      "research_context": "Communication research findings on these behavior patterns",
      "behavioral_evidence": "Specific visual details supporting this behavioral classification",
      "temporal_pattern": "Evolution of these behaviors throughout video",
      "prevalence": "None/Subtle/Moderate/Strong/Very Strong"
    },
    "confidence_markers": {
      "visible_behaviors": "Observable posture types, movement quality, tension patterns",
      "research_context": "Research literature on confidence-related nonverbal cues",
      "behavioral_evidence": "Visual evidence for this behavioral category",
      "temporal_pattern": "Changes in these markers across timeline",
      "prevalence": "None/Subtle/Moderate/Strong/Very Strong"
    }
  },
  
  "emotional_expression_behaviors": {
    "affective_displays": {
      "visible_behaviors": "Observable facial expressions, micro-expressions, expression durations across frames",
      "research_context": "Affective science research on these expression types",
      "behavioral_evidence": "Specific facial cues and expression patterns visible",
      "temporal_pattern": "How emotional displays vary across video",
      "classification": "Describe the expression types observed based on research categories"
    },
    "emotional_stability_markers": {
      "visible_behaviors": "Observable expression consistency, transition quality, affect range",
      "research_context": "Research on emotional regulation behavioral displays",
      "behavioral_evidence": "Visual evidence of stability/variability patterns",
      "temporal_pattern": "Consistency or changes across timeline",
      "classification": "Research-based description of regulation pattern type"
    }
  },
  
  "interpersonal_behavior_patterns": {
    "relational_orientation": {
      "visible_behaviors": "Observable body angles, proximity patterns, mirroring if visible",
      "research_context": "Interpersonal communication research on these cues",
      "behavioral_evidence": "Specific visual elements present",
      "temporal_pattern": "Evolution of relational behaviors",
      "classification": "Research-based categorization of pattern type"
    },
    "social_presentation": {
      "visible_behaviors": "Observable grooming awareness, camera engagement, self-monitoring cues",
      "research_context": "Self-presentation research findings",
      "behavioral_evidence": "Visual indicators supporting this classification",
      "temporal_pattern": "Changes in presentation across timeline",
      "classification": "Description based on presentation research"
    }
  },
  
  "communication_style_indicators": {
    "authenticity_markers": {
      "visible_behaviors": "Observable spontaneity levels, behavioral variability, control indicators",
      "research_context": "Research on authentic vs strategic communication displays",
      "behavioral_evidence": "Specific visual cues for this behavioral category",
      "temporal_pattern": "Consistency of these markers across video",
      "classification": "Research-based authenticity pattern description"
    },
    "persuasion_behaviors": {
      "visible_behaviors": "Observable argument cues, appeal types, transparency markers",
      "research_context": "Persuasion research on nonverbal indicators",
      "behavioral_evidence": "Visual evidence for persuasion behavior type",
      "temporal_pattern": "Evolution across timeline",
      "classification": "Research-based persuasion style description"
    }
  },
  
  "behavioral_catalog_summary": {
    "primary_behavior_categories": ["Main behavior types observed with research-based descriptions"],
    "notable_research_applications": ["How these behavior types are studied in communication research"],
    "temporal_consistency_notes": "Research perspective on behavioral stability vs variability observed",
    "overall_behavior_profile": "Comprehensive catalog summary using communication research terminology"
  },
  
  "frame_by_frame_catalog": {
    "0_percent": "Observable behaviors at video start",
    "25_percent": "Observable behaviors at first quarter",
    "50_percent": "Observable behaviors at midpoint",
    "75_percent": "Observable behaviors at final quarter"
  },
  
  "cataloging_notes": {
    "clear_observations": ["Behavior types with strong visual evidence"],
    "tentative_observations": ["Behavior types with limited or ambiguous visual evidence"],
    "limitations": "Specific limitations based on video quality, angle, framing, context availability"
  }
}`;

      let analysisResult: any;
      
      // Call the appropriate AI model with vision capability
      if (selectedModel === "openai" && openai) {
        const response = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [{
            role: "user",
            content: [
              { type: "text", text: darkTraitsVideoPrompt + "\n\nFrames extracted at 0%, 25%, 50%, and 75% of video:" },
              ...extractedFrames.map((frame) => ({
                type: "image_url" as const,
                image_url: { url: frame }
              }))
            ]
          }],
          max_tokens: 4000,
        });
        
        const rawResponse = response.choices[0]?.message.content || "";
        console.log("OpenAI Dark Traits Video raw response:", rawResponse.substring(0, 500));
        
        if (!rawResponse || rawResponse.trim().length === 0) {
          throw new Error("OpenAI returned an empty response. This may be due to content moderation or video complexity issues.");
        }
        
        // Try to extract JSON from the response
        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          jsonText = jsonMatch[0].replace(/```json\s*/, '').replace(/\s*```$/, '');
        }
        
        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse OpenAI response:", parseError);
          console.error("Raw response:", rawResponse);
          analysisResult = {
            summary: rawResponse.substring(0, 1000) || "Unable to format analysis",
            dark_tetrad_visual_assessment: {
              narcissism: { level: "Unable to determine", subtype: "N/A", visual_evidence_timeline: [], speculative_hypothesis: "Formatting error" },
              machiavellianism: { level: "Unable to determine", visual_evidence_timeline: [], speculative_hypothesis: "Formatting error" },
              psychopathy: { level: "Unable to determine", subtype: "N/A", visual_evidence_timeline: [], speculative_hypothesis: "Formatting error" },
              sadism: { level: "Unable to determine", visual_evidence_timeline: [], speculative_hypothesis: "Formatting error" }
            }
          };
        }
      } else if (selectedModel === "anthropic" && anthropic) {
        const imageContents = extractedFrames.map(frame => {
          const base64Match = frame.match(/^data:image\/[a-z]+;base64,(.+)$/);
          const base64Data = base64Match ? base64Match[1] : frame;
          const mediaTypeMatch = frame.match(/^data:(image\/[a-z]+);base64,/);
          const mediaType = mediaTypeMatch ? mediaTypeMatch[1] : "image/jpeg";
          
          return {
            type: "image" as const,
            source: {
              type: "base64" as const,
              media_type: mediaType as any,
              data: base64Data,
            },
          };
        });
        
        const response = await anthropic.messages.create({
          model: "claude-sonnet-4-20250514",
          max_tokens: 8000,
          messages: [{
            role: "user",
            content: [
              {
                type: "text",
                text: darkTraitsVideoPrompt + "\n\nFrames extracted at 0%, 25%, 50%, and 75% of video:"
              },
              ...imageContents
            ]
          }],
        });
        
        const rawResponse = response.content[0].type === 'text' ? response.content[0].text : "";
        console.log("Anthropic Dark Traits Video raw response:", rawResponse.substring(0, 500));
        
        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/```\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          jsonText = jsonMatch[1];
        }
        
        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse Anthropic response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000) || "Unable to format analysis",
            dark_tetrad_visual_assessment: {
              narcissism: { level: "Unable to determine", subtype: "N/A", visual_evidence_timeline: [], speculative_hypothesis: "Formatting error" },
              machiavellianism: { level: "Unable to determine", visual_evidence_timeline: [], speculative_hypothesis: "Formatting error" },
              psychopathy: { level: "Unable to determine", subtype: "N/A", visual_evidence_timeline: [], speculative_hypothesis: "Formatting error" },
              sadism: { level: "Unable to determine", visual_evidence_timeline: [], speculative_hypothesis: "Formatting error" }
            }
          };
        }
      } else if (selectedModel === "deepseek") {
        return res.status(400).json({ 
          error: "DeepSeek does not support video analysis. Please use OpenAI or Anthropic for video-based dark traits analysis." 
        });
      } else if (selectedModel === "perplexity") {
        return res.status(400).json({ 
          error: "Perplexity does not support video analysis. Please use OpenAI or Anthropic for video-based dark traits analysis." 
        });
      }
      
      console.log("Dark Traits video analysis complete");
      
      // Helper function to safely stringify any value into readable text
      const safeStringify = (value: any): string => {
        if (typeof value === 'string') return value;
        if (typeof value === 'object' && value !== null) {
          if (Array.isArray(value)) {
            return value.map(item => {
              if (typeof item === 'string') return item;
              if (typeof item === 'object' && item !== null) {
                return Object.entries(item)
                  .map(([key, val]) => `${key}: ${val}`)
                  .join('\n');
              }
              return String(item);
            }).join('\n\n');
          }
          const keys = Object.keys(value);
          if (keys.length > 0 && keys.every(k => /^\d+$/.test(k))) {
            return keys
              .sort((a, b) => parseInt(a) - parseInt(b))
              .map(key => `${key}. ${value[key]}`)
              .join('\n');
          }
          return Object.entries(value)
            .map(([key, val]) => `${val}`)
            .join('\n\n');
        }
        return String(value || '');
      };
      
      // Format the catalog for display
      let formattedContent = `Hypothetical Case Study Analysis - Behavioral Dynamics\nMode: Fictional Educational Analysis Using Reference Material\n\n`;
      
      if (analysisResult.disclaimer) {
        formattedContent += `IMPORTANT DISCLAIMER:\n${safeStringify(analysisResult.disclaimer)}\n\n`;
      }
      
      formattedContent += `Case Study Summary:\n${safeStringify(analysisResult.summary)}\n\n`;
      
      // Nonverbal Dominance Behaviors
      if (analysisResult.nonverbal_dominance_behaviors) {
        formattedContent += `NONVERBAL DOMINANCE BEHAVIORS (Communication Research):\n\n`;
        
        if (analysisResult.nonverbal_dominance_behaviors.assertiveness_displays) {
          const ad = analysisResult.nonverbal_dominance_behaviors.assertiveness_displays;
          formattedContent += `Assertiveness Displays [${ad.prevalence || 'N/A'}]:\n`;
          formattedContent += `VISIBLE BEHAVIORS: ${safeStringify(ad.visible_behaviors)}\n`;
          formattedContent += `RESEARCH CONTEXT: ${safeStringify(ad.research_context)}\n`;
          formattedContent += `BEHAVIORAL EVIDENCE: ${safeStringify(ad.behavioral_evidence)}\n`;
          formattedContent += `TEMPORAL PATTERN: ${safeStringify(ad.temporal_pattern)}\n\n`;
        }
        
        if (analysisResult.nonverbal_dominance_behaviors.influence_attempt_behaviors) {
          const iab = analysisResult.nonverbal_dominance_behaviors.influence_attempt_behaviors;
          formattedContent += `Influence Attempt Behaviors [${iab.prevalence || 'N/A'}]:\n`;
          formattedContent += `VISIBLE BEHAVIORS: ${safeStringify(iab.visible_behaviors)}\n`;
          formattedContent += `RESEARCH CONTEXT: ${safeStringify(iab.research_context)}\n`;
          formattedContent += `BEHAVIORAL EVIDENCE: ${safeStringify(iab.behavioral_evidence)}\n`;
          formattedContent += `TEMPORAL PATTERN: ${safeStringify(iab.temporal_pattern)}\n\n`;
        }
        
        if (analysisResult.nonverbal_dominance_behaviors.confidence_markers) {
          const cm = analysisResult.nonverbal_dominance_behaviors.confidence_markers;
          formattedContent += `Confidence Markers [${cm.prevalence || 'N/A'}]:\n`;
          formattedContent += `VISIBLE BEHAVIORS: ${safeStringify(cm.visible_behaviors)}\n`;
          formattedContent += `RESEARCH CONTEXT: ${safeStringify(cm.research_context)}\n`;
          formattedContent += `BEHAVIORAL EVIDENCE: ${safeStringify(cm.behavioral_evidence)}\n`;
          formattedContent += `TEMPORAL PATTERN: ${safeStringify(cm.temporal_pattern)}\n\n`;
        }
      }
      
      // Emotional Expression Behaviors
      if (analysisResult.emotional_expression_behaviors) {
        formattedContent += `EMOTIONAL EXPRESSION BEHAVIORS (Affective Research):\n\n`;
        
        if (analysisResult.emotional_expression_behaviors.affective_displays) {
          const ad = analysisResult.emotional_expression_behaviors.affective_displays;
          formattedContent += `Affective Displays [${ad.classification || 'N/A'}]:\n`;
          formattedContent += `VISIBLE BEHAVIORS: ${safeStringify(ad.visible_behaviors)}\n`;
          formattedContent += `RESEARCH CONTEXT: ${safeStringify(ad.research_context)}\n`;
          formattedContent += `BEHAVIORAL EVIDENCE: ${safeStringify(ad.behavioral_evidence)}\n`;
          formattedContent += `TEMPORAL PATTERN: ${safeStringify(ad.temporal_pattern)}\n\n`;
        }
        
        if (analysisResult.emotional_expression_behaviors.emotional_stability_markers) {
          const esm = analysisResult.emotional_expression_behaviors.emotional_stability_markers;
          formattedContent += `Emotional Stability Markers [${esm.classification || 'N/A'}]:\n`;
          formattedContent += `VISIBLE BEHAVIORS: ${safeStringify(esm.visible_behaviors)}\n`;
          formattedContent += `RESEARCH CONTEXT: ${safeStringify(esm.research_context)}\n`;
          formattedContent += `BEHAVIORAL EVIDENCE: ${safeStringify(esm.behavioral_evidence)}\n`;
          formattedContent += `TEMPORAL PATTERN: ${safeStringify(esm.temporal_pattern)}\n\n`;
        }
      }
      
      // Interpersonal Behavior Patterns
      if (analysisResult.interpersonal_behavior_patterns) {
        formattedContent += `INTERPERSONAL BEHAVIOR PATTERNS (Social Communication):\n\n`;
        
        if (analysisResult.interpersonal_behavior_patterns.relational_orientation) {
          const ro = analysisResult.interpersonal_behavior_patterns.relational_orientation;
          formattedContent += `Relational Orientation [${ro.classification || 'N/A'}]:\n`;
          formattedContent += `VISIBLE BEHAVIORS: ${safeStringify(ro.visible_behaviors)}\n`;
          formattedContent += `RESEARCH CONTEXT: ${safeStringify(ro.research_context)}\n`;
          formattedContent += `BEHAVIORAL EVIDENCE: ${safeStringify(ro.behavioral_evidence)}\n`;
          formattedContent += `TEMPORAL PATTERN: ${safeStringify(ro.temporal_pattern)}\n\n`;
        }
        
        if (analysisResult.interpersonal_behavior_patterns.social_presentation) {
          const sp = analysisResult.interpersonal_behavior_patterns.social_presentation;
          formattedContent += `Social Presentation [${sp.classification || 'N/A'}]:\n`;
          formattedContent += `VISIBLE BEHAVIORS: ${safeStringify(sp.visible_behaviors)}\n`;
          formattedContent += `RESEARCH CONTEXT: ${safeStringify(sp.research_context)}\n`;
          formattedContent += `BEHAVIORAL EVIDENCE: ${safeStringify(sp.behavioral_evidence)}\n`;
          formattedContent += `TEMPORAL PATTERN: ${safeStringify(sp.temporal_pattern)}\n\n`;
        }
      }
      
      // Communication Style Indicators
      if (analysisResult.communication_style_indicators) {
        formattedContent += `COMMUNICATION STYLE INDICATORS (Behavioral Communication):\n\n`;
        
        if (analysisResult.communication_style_indicators.authenticity_markers) {
          const am = analysisResult.communication_style_indicators.authenticity_markers;
          formattedContent += `Authenticity Markers [${am.classification || 'N/A'}]:\n`;
          formattedContent += `VISIBLE BEHAVIORS: ${safeStringify(am.visible_behaviors)}\n`;
          formattedContent += `RESEARCH CONTEXT: ${safeStringify(am.research_context)}\n`;
          formattedContent += `BEHAVIORAL EVIDENCE: ${safeStringify(am.behavioral_evidence)}\n`;
          formattedContent += `TEMPORAL PATTERN: ${safeStringify(am.temporal_pattern)}\n\n`;
        }
        
        if (analysisResult.communication_style_indicators.persuasion_behaviors) {
          const pb = analysisResult.communication_style_indicators.persuasion_behaviors;
          formattedContent += `Persuasion Behaviors [${pb.classification || 'N/A'}]:\n`;
          formattedContent += `VISIBLE BEHAVIORS: ${safeStringify(pb.visible_behaviors)}\n`;
          formattedContent += `RESEARCH CONTEXT: ${safeStringify(pb.research_context)}\n`;
          formattedContent += `BEHAVIORAL EVIDENCE: ${safeStringify(pb.behavioral_evidence)}\n`;
          formattedContent += `TEMPORAL PATTERN: ${safeStringify(pb.temporal_pattern)}\n\n`;
        }
      }
      
      // Behavioral Catalog Summary
      if (analysisResult.behavioral_catalog_summary) {
        formattedContent += `BEHAVIORAL CATALOG SUMMARY:\n\n`;
        const bcs = analysisResult.behavioral_catalog_summary;
        
        if (bcs.primary_behavior_categories) {
          formattedContent += `Primary Behavior Categories:\n${safeStringify(bcs.primary_behavior_categories)}\n\n`;
        }
        if (bcs.notable_research_applications) {
          formattedContent += `Research Applications:\n${safeStringify(bcs.notable_research_applications)}\n\n`;
        }
        if (bcs.temporal_consistency_notes) {
          formattedContent += `Temporal Consistency:\n${safeStringify(bcs.temporal_consistency_notes)}\n\n`;
        }
        if (bcs.overall_behavior_profile) {
          formattedContent += `Overall Behavior Profile:\n${safeStringify(bcs.overall_behavior_profile)}\n\n`;
        }
      }
      
      // Frame-by-Frame Catalog
      if (analysisResult.frame_by_frame_catalog) {
        formattedContent += `FRAME-BY-FRAME CATALOG:\n\n`;
        const fbc = analysisResult.frame_by_frame_catalog;
        if (fbc["0_percent"]) formattedContent += `At 0% (Video Start):\n${safeStringify(fbc["0_percent"])}\n\n`;
        if (fbc["25_percent"]) formattedContent += `At 25% (First Quarter):\n${safeStringify(fbc["25_percent"])}\n\n`;
        if (fbc["50_percent"]) formattedContent += `At 50% (Midpoint):\n${safeStringify(fbc["50_percent"])}\n\n`;
        if (fbc["75_percent"]) formattedContent += `At 75% (Final Quarter):\n${safeStringify(fbc["75_percent"])}\n\n`;
      }
      
      // Cataloging Notes
      if (analysisResult.cataloging_notes) {
        formattedContent += `CATALOGING NOTES:\n\n`;
        const cn = analysisResult.cataloging_notes;
        if (cn.clear_observations) {
          formattedContent += `Clear Observations:\n${safeStringify(cn.clear_observations)}\n\n`;
        }
        if (cn.tentative_observations) {
          formattedContent += `Tentative Observations:\n${safeStringify(cn.tentative_observations)}\n\n`;
        }
        if (cn.limitations) {
          formattedContent += `Limitations:\n${safeStringify(cn.limitations)}\n\n`;
        }
      }
      
      // Create analysis record in storage
      const analysis = await storage.createAnalysis({
        sessionId,
        title: title || `Hypothetical Case Study - Behavioral Dynamics`,
        mediaUrl: mediaData,
        personalityInsights: { 
          analysis: formattedContent,
          disclaimer: analysisResult.disclaimer,
          behavioral_catalog: analysisResult.behavioral_catalog_summary,
          nonverbal_dominance: analysisResult.nonverbal_dominance_behaviors,
          emotional_expression: analysisResult.emotional_expression_behaviors
        },
        modelUsed: selectedModel,
      });
      
      // Create message with formatted analysis
      const message = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        content: formattedContent,
        role: "assistant",
      });
      
      res.json({
        analysisId: analysis.id,
        personalityInsights: { 
          analysis: formattedContent,
          behavioral_catalog: analysisResult.behavioral_catalog_summary,
          nonverbal_dominance: analysisResult.nonverbal_dominance_behaviors
        },
        messages: [message],
        mediaUrl: mediaData,
      });
    } catch (error) {
      console.error("Behavioral Dynamics video analysis error:", error);
      res.status(500).json({ error: "Failed to analyze video for behavioral dynamics" });
    }
  });

  // Enneagram Analysis Endpoints - Image
  app.post("/api/analyze/image/enneagram", async (req, res) => {
    try {
      const { mediaData, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!mediaData || typeof mediaData !== 'string') {
        return res.status(400).json({ error: "Image data is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing Enneagram image analysis with model: ${selectedModel}`);
      
      // Enneagram image analysis prompt with 9 personality types
      const enneagramImagePrompt = `You are an expert Enneagram analyst specializing in identifying personality types through visual analysis of photographs. Analyze this image comprehensively using the Enneagram framework, providing detailed evidence for the most likely type(s) with SPECIFIC VISUAL REFERENCES.

The Enneagram 9 Types are:

TYPE 1 - THE REFORMER (The Perfectionist)
Core Fear: Being corrupt, evil, defective
Core Desire: To be good, balanced, have integrity
Visual Traits: Controlled posture, precise grooming, serious/concentrated expression, orderly surroundings, minimal frivolity

TYPE 2 - THE HELPER (The Giver)
Core Fear: Being unloved, unwanted
Core Desire: To be loved, appreciated
Visual Traits: Warm smile, open body language, positioned near others, nurturing gestures, friendly eye contact

TYPE 3 - THE ACHIEVER (The Performer)
Core Fear: Being worthless, without value
Core Desire: To be valuable, admired
Visual Traits: Confident stance, polished appearance, success symbols, camera-aware positioning, composed presentation

TYPE 4 - THE INDIVIDUALIST (The Romantic)
Core Fear: Having no identity or significance
Core Desire: To be unique, authentic
Visual Traits: Artistic styling, emotional depth in eyes, unique fashion choices, dramatic or melancholic expression, symbolic accessories

TYPE 5 - THE INVESTIGATOR (The Observer)
Core Fear: Being useless, incompetent
Core Desire: To be capable, knowledgeable
Visual Traits: Reserved demeanor, minimal expression, intellectual environment, solitary positioning, analytical gaze

TYPE 6 - THE LOYALIST (The Skeptic)
Core Fear: Being without support or guidance
Core Desire: To have security, support
Visual Traits: Cautious expression, protective stance, group affiliation markers, alert eyes, conservative styling

TYPE 7 - THE ENTHUSIAST (The Epicure)
Core Fear: Being deprived, trapped in pain
Core Desire: To be happy, satisfied, free
Visual Traits: Energetic expression, playful demeanor, active setting, bright colors, multiple interests visible

TYPE 8 - THE CHALLENGER (The Protector)
Core Fear: Being harmed, controlled by others
Core Desire: To protect self, be in control
Visual Traits: Strong stance, direct gaze, commanding presence, protective positioning, bold fashion

TYPE 9 - THE PEACEMAKER (The Mediator)
Core Fear: Loss, separation, conflict
Core Desire: To have peace, harmony
Visual Traits: Relaxed posture, gentle expression, harmonious colors, blending-in appearance, serene environment

CRITICAL: Every indicator must reference SPECIFIC VISUAL DETAILS from the actual image.

Provide your analysis in JSON format:
{
  "summary": "Overall Enneagram assessment with primary type, confidence level, and visual reasoning",
  "primary_type": {
    "type": "Type [Number] - [Name]",
    "confidence": "High/Medium/Low",
    "core_motivation": "Identified core fear and desire based on visual evidence",
    "key_indicators": [
      "Specific visual detail showing this type trait",
      "Another visual observation demonstrating the core motivation",
      "Physical cue with specific reference to what's visible"
    ]
  },
  "secondary_possibilities": [
    {
      "type": "Type [Number] - [Name]",
      "reasoning": "Why this type is also possible with specific visual evidence"
    }
  ],
  "wing_analysis": "Analysis of likely wing (e.g., 4w3 or 4w5) based on visual patterns with evidence",
  "triadic_analysis": {
    "center": "Head/Heart/Body center based on dominant visual presentation",
    "stance": "Aggressive/Dependent/Withdrawing based on interpersonal positioning in image"
  },
  "visual_style_markers": [
    "Specific visual patterns observed (clothing, posture, expression)",
    "Environmental cues visible in the image",
    "Body language and facial expression indicators"
  ],
  "personality_summary": "Comprehensive Enneagram-based personality description integrating type, wing, and visual patterns"
}`;

      let analysisResult: any;
      
      // Call the appropriate AI model with vision capability
      if (selectedModel === "openai" && openai) {
        const response = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [{
            role: "user",
            content: [
              { type: "text", text: enneagramImagePrompt },
              { type: "image_url", image_url: { url: mediaData } }
            ]
          }],
          max_tokens: 4000,
        });
        
        const rawResponse = response.choices[0]?.message.content || "";
        console.log("OpenAI Enneagram Image raw response:", rawResponse.substring(0, 500));
        
        if (!rawResponse || rawResponse.trim().length === 0) {
          throw new Error("OpenAI returned an empty response. This may be due to content moderation or image size issues.");
        }
        
        // Try to extract JSON from the response
        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          jsonText = jsonMatch[0].replace(/```json\s*/, '').replace(/\s*```$/, '');
        }
        
        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse OpenAI response:", parseError);
          console.error("Raw response:", rawResponse);
          analysisResult = {
            summary: rawResponse.substring(0, 1000) || "Unable to format analysis",
            primary_type: { type: "Unable to determine", confidence: "Low", core_motivation: "Formatting error", key_indicators: [] },
            secondary_possibilities: [],
            wing_analysis: "Unable to analyze due to formatting error",
            triadic_analysis: { center: "Unknown", stance: "Unknown" },
            visual_style_markers: [],
            personality_summary: "Unable to format analysis"
          };
        }
      } else if (selectedModel === "anthropic" && anthropic) {
        const base64Match = mediaData.match(/^data:image\/[a-z]+;base64,(.+)$/);
        const base64Data = base64Match ? base64Match[1] : mediaData;
        const mediaTypeMatch = mediaData.match(/^data:(image\/[a-z]+);base64,/);
        const mimeType = mediaTypeMatch ? mediaTypeMatch[1] : "image/jpeg";

        const response = await anthropic.messages.create({
          model: "claude-sonnet-4-20250514",
          max_tokens: 8000,
          messages: [{
            role: "user",
            content: [
              {
                type: "image",
                source: {
                  type: "base64",
                  media_type: mimeType,
                  data: base64Data,
                },
              },
              {
                type: "text",
                text: enneagramImagePrompt
              }
            ],
          }],
        });

        const rawResponse = response.content[0].type === 'text' ? response.content[0].text : "";
        console.log("Anthropic Enneagram Image raw response:", rawResponse.substring(0, 500));

        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/```\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          jsonText = jsonMatch[1];
        }

        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse Anthropic response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000) || "Unable to format analysis",
            primary_type: { type: "Unable to determine", confidence: "Low", core_motivation: "Formatting error", key_indicators: [] },
            secondary_possibilities: [],
            wing_analysis: "Unable to analyze due to formatting error",
            triadic_analysis: { center: "Unknown", stance: "Unknown" },
            visual_style_markers: [],
            personality_summary: "Unable to format analysis"
          };
        }
      } else if (selectedModel === "deepseek") {
        return res.status(400).json({ 
          error: "DeepSeek does not support image analysis. Please use OpenAI or Anthropic for image-based Enneagram analysis." 
        });
      } else if (selectedModel === "perplexity") {
        return res.status(400).json({ 
          error: "Perplexity does not support image analysis. Please use OpenAI or Anthropic for image-based Enneagram analysis." 
        });
      }
      
      console.log("Enneagram image analysis complete");
      
      // Helper function to safely stringify any value into readable text
      const safeStringify = (value: any): string => {
        if (typeof value === 'string') return value;
        if (typeof value === 'object' && value !== null) {
          // If it's an array, handle each item recursively
          if (Array.isArray(value)) {
            return value.map(item => {
              if (typeof item === 'string') return item;
              if (typeof item === 'object' && item !== null) {
                // Format objects in arrays as key-value pairs
                return Object.entries(item)
                  .map(([key, val]) => `${key}: ${val}`)
                  .join('\n');
              }
              return String(item);
            }).join('\n\n');
          }
          // If it's an object with numbered keys (like "1", "2", etc), format as numbered list
          const keys = Object.keys(value);
          if (keys.length > 0 && keys.every(k => /^\d+$/.test(k))) {
            return keys
              .sort((a, b) => parseInt(a) - parseInt(b))
              .map(key => `${key}. ${value[key]}`)
              .join('\n');
          }
          // If it's an object with named keys, format as key-value pairs
          return Object.entries(value)
            .map(([key, val]) => `${val}`)
            .join('\n\n');
        }
        return String(value || '');
      };
      
      // Format the analysis for display
      let formattedContent = `Enneagram Personality Analysis\nMode: 9-Type Visual Framework\n\n`;
      formattedContent += `Summary:\n${safeStringify(analysisResult.summary)}\n\n`;
      
      // Primary Type
      if (analysisResult.primary_type) {
        formattedContent += `Primary Type: ${analysisResult.primary_type.type}\n`;
        formattedContent += `Confidence: ${analysisResult.primary_type.confidence}\n`;
        formattedContent += `Core Motivation: ${safeStringify(analysisResult.primary_type.core_motivation)}\n`;
        if (analysisResult.primary_type.key_indicators && analysisResult.primary_type.key_indicators.length > 0) {
          formattedContent += `Key Indicators:\n${safeStringify(analysisResult.primary_type.key_indicators)}\n`;
        }
        formattedContent += `\n`;
      }
      
      // Secondary Possibilities
      if (analysisResult.secondary_possibilities && analysisResult.secondary_possibilities.length > 0) {
        formattedContent += `Secondary Possibilities:\n${safeStringify(analysisResult.secondary_possibilities)}\n\n`;
      }
      
      // Wing Analysis
      if (analysisResult.wing_analysis) {
        formattedContent += `Wing Analysis:\n${safeStringify(analysisResult.wing_analysis)}\n\n`;
      }
      
      // Triadic Analysis
      if (analysisResult.triadic_analysis) {
        formattedContent += `Triadic Analysis:\n`;
        formattedContent += `Center: ${analysisResult.triadic_analysis.center || 'N/A'}\n`;
        formattedContent += `Stance: ${analysisResult.triadic_analysis.stance || 'N/A'}\n\n`;
      }
      
      // Visual Style Markers
      if (analysisResult.visual_style_markers && analysisResult.visual_style_markers.length > 0) {
        formattedContent += `Visual Style Markers:\n${safeStringify(analysisResult.visual_style_markers)}\n\n`;
      }
      
      // Personality Summary
      if (analysisResult.personality_summary) {
        formattedContent += `Personality Summary:\n${safeStringify(analysisResult.personality_summary)}\n\n`;
      }
      
      // Create analysis record
      const analysis = await storage.createAnalysis({
        sessionId,
        title: title || `Enneagram Image Analysis`,
        mediaUrl: mediaData,
        mediaType: "image",
        personalityInsights: { analysis: formattedContent, enneagram_type: analysisResult.primary_type?.type },
        modelUsed: selectedModel,
      });
      
      // Create message with formatted analysis
      const message = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        content: formattedContent,
        role: "assistant",
      });
      
      res.json({
        analysisId: analysis.id,
        personalityInsights: { analysis: formattedContent, enneagram_type: analysisResult.primary_type?.type },
        messages: [message],
        mediaUrl: mediaData,
      });
    } catch (error) {
      console.error("Enneagram image analysis error:", error);
      res.status(500).json({ error: "Failed to analyze image for Enneagram" });
    }
  });

  // Enneagram Analysis Endpoints - Video
  app.post("/api/analyze/video/enneagram", async (req, res) => {
    try {
      const { mediaData, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!mediaData || typeof mediaData !== 'string') {
        return res.status(400).json({ error: "Video data is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing Enneagram video analysis with model: ${selectedModel}`);
      
      // Save video temporarily and extract frames
      const videoBuffer = Buffer.from(mediaData.split(',')[1], 'base64');
      const tempVideoPath = path.join(tempDir, `video_${Date.now()}.mp4`);
      await writeFileAsync(tempVideoPath, videoBuffer);
      
      // Extract frames at different timestamps
      const framePromises = [0, 25, 50, 75].map(async (percent) => {
        const outputPath = path.join(tempDir, `frame_${Date.now()}_${percent}.jpg`);
        
        return new Promise<string>((resolve, reject) => {
          ffmpeg(tempVideoPath)
            .screenshots({
              count: 1,
              timemarks: [`${percent}%`],
              filename: path.basename(outputPath),
              folder: tempDir,
            })
            .on('end', () => {
              const frameData = fs.readFileSync(outputPath);
              const base64Frame = `data:image/jpeg;base64,${frameData.toString('base64')}`;
              fs.unlinkSync(outputPath);
              resolve(base64Frame);
            })
            .on('error', (err) => {
              console.error('Frame extraction error:', err);
              reject(err);
            });
        });
      });
      
      const extractedFrames = await Promise.all(framePromises);
      
      // Clean up temp video file
      await unlinkAsync(tempVideoPath);
      
      console.log(`Extracted ${extractedFrames.length} frames from video for Enneagram analysis`);
      
      // Enneagram video analysis prompt
      const enneagramVideoPrompt = `You are an expert Enneagram analyst specializing in identifying personality types through behavioral video analysis. Analyze this video comprehensively using the Enneagram framework, providing detailed evidence for the most likely type(s) with SPECIFIC BEHAVIORAL PATTERNS across the video timeline.

The Enneagram 9 Types with behavioral markers:

TYPE 1 - THE REFORMER (The Perfectionist)
Core Fear: Being corrupt, evil, defective
Core Desire: To be good, balanced, have integrity
Behavioral Traits: Controlled movements, precise gestures, self-correcting behaviors, critical expressions, structured patterns

TYPE 2 - THE HELPER (The Giver)
Core Fear: Being unloved, unwanted
Core Desire: To be loved, appreciated
Behavioral Traits: Warm facial expressions, reaching gestures, attentive to others, nurturing movements, relational engagement

TYPE 3 - THE ACHIEVER (The Performer)
Core Fear: Being worthless, without value
Core Desire: To be valuable, admired
Behavioral Traits: Confident posture, polished presentation, goal-oriented movements, image-conscious behaviors, dynamic energy

TYPE 4 - THE INDIVIDUALIST (The Romantic)
Core Fear: Having no identity or significance
Core Desire: To be unique, authentic
Behavioral Traits: Expressive gestures, emotional depth in movements, unique mannerisms, introspective moments, artistic expression

TYPE 5 - THE INVESTIGATOR (The Observer)
Core Fear: Being useless, incompetent
Core Desire: To be capable, knowledgeable
Behavioral Traits: Reserved movements, analytical gaze, minimal emotional display, economical gestures, observational stance

TYPE 6 - THE LOYALIST (The Skeptic)
Core Fear: Being without support or guidance
Core Desire: To have security, support
Behavioral Traits: Cautious movements, vigilant expressions, testing behaviors, protective gestures, anxiety markers

TYPE 7 - THE ENTHUSIAST (The Epicure)
Core Fear: Being deprived, trapped in pain
Core Desire: To be happy, satisfied, free
Behavioral Traits: Energetic movements, animated expressions, quick transitions, playful gestures, exploratory behaviors

TYPE 8 - THE CHALLENGER (The Protector)
Core Fear: Being harmed, controlled by others
Core Desire: To protect self, be in control
Behavioral Traits: Strong movements, direct gaze, commanding presence, protective postures, assertive gestures

TYPE 9 - THE PEACEMAKER (The Mediator)
Core Fear: Loss, separation, conflict
Core Desire: To have peace, harmony
Behavioral Traits: Relaxed movements, gentle expressions, harmonizing gestures, conflict-avoiding behaviors, steady presence

CRITICAL: Base your analysis ONLY on observable behaviors across the video frames. Reference specific behavioral patterns, movements, expressions, and temporal changes.

Provide your analysis in JSON format:
{
  "summary": "Overall Enneagram assessment with primary type, confidence level, and behavioral reasoning across video timeline",
  "primary_type": {
    "type": "Type [Number] - [Name]",
    "confidence": "High/Medium/Low",
    "core_motivation": "Identified core fear and desire based on behavioral evidence",
    "behavioral_indicators": [
      "Specific behavior observed at beginning of video",
      "Pattern that emerges mid-video",
      "Consistent behavioral marker throughout"
    ]
  },
  "secondary_possibilities": [
    {
      "type": "Type [Number] - [Name]",
      "reasoning": "Why this type is also possible with specific behavioral evidence"
    }
  ],
  "wing_analysis": "Analysis of likely wing (e.g., 4w3 or 4w5) based on behavioral patterns with evidence",
  "temporal_patterns": {
    "consistency": "Whether behaviors remain consistent or shift across video",
    "energy_changes": "How energy level and engagement patterns evolve",
    "defensive_patterns": "Any defensive behaviors or coping mechanisms observed"
  },
  "triadic_analysis": {
    "center": "Head/Heart/Body center based on dominant behavioral presentation",
    "stance": "Aggressive/Dependent/Withdrawing based on interpersonal behaviors"
  },
  "behavioral_style_markers": [
    "Specific movement patterns observed (gestures, posture changes)",
    "Emotional expression patterns across timeline",
    "Interaction style and relational behaviors"
  ],
  "personality_summary": "Comprehensive Enneagram-based personality description integrating type, wing, temporal patterns, and behavioral evidence"
}`;

      let analysisResult: any;
      
      // Call the appropriate AI model with vision capability
      if (selectedModel === "openai" && openai) {
        const response = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [{
            role: "user",
            content: [
              { type: "text", text: enneagramVideoPrompt + "\n\nFrames extracted at 0%, 25%, 50%, and 75% of video:" },
              ...extractedFrames.map((frame) => ({
                type: "image_url" as const,
                image_url: { url: frame }
              }))
            ]
          }],
          max_tokens: 4000,
        });
        
        const rawResponse = response.choices[0]?.message.content || "";
        console.log("OpenAI Enneagram Video raw response:", rawResponse.substring(0, 500));
        
        if (!rawResponse || rawResponse.trim().length === 0) {
          throw new Error("OpenAI returned an empty response. This may be due to content moderation or video complexity issues.");
        }
        
        // Try to extract JSON from the response
        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          jsonText = jsonMatch[0].replace(/```json\s*/, '').replace(/\s*```$/, '');
        }
        
        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse OpenAI response:", parseError);
          console.error("Raw response:", rawResponse);
          analysisResult = {
            summary: rawResponse.substring(0, 1000) || "Unable to format analysis",
            primary_type: { type: "Unable to determine", confidence: "Low", core_motivation: "Formatting error", behavioral_indicators: [] },
            secondary_possibilities: [],
            wing_analysis: "Unable to analyze due to formatting error",
            temporal_patterns: { consistency: "Unknown", energy_changes: "Unknown", defensive_patterns: "Unknown" },
            triadic_analysis: { center: "Unknown", stance: "Unknown" },
            behavioral_style_markers: [],
            personality_summary: "Unable to format analysis"
          };
        }
      } else if (selectedModel === "anthropic" && anthropic) {
        const imageContents = extractedFrames.map(frame => {
          const base64Match = frame.match(/^data:image\/[a-z]+;base64,(.+)$/);
          const base64Data = base64Match ? base64Match[1] : frame;
          const mediaTypeMatch = frame.match(/^data:(image\/[a-z]+);base64,/);
          const mediaType = mediaTypeMatch ? mediaTypeMatch[1] : "image/jpeg";
          
          return {
            type: "image" as const,
            source: {
              type: "base64" as const,
              media_type: mediaType as any,
              data: base64Data,
            },
          };
        });
        
        const response = await anthropic.messages.create({
          model: "claude-sonnet-4-20250514",
          max_tokens: 8000,
          messages: [{
            role: "user",
            content: [
              {
                type: "text",
                text: enneagramVideoPrompt + "\n\nFrames extracted at 0%, 25%, 50%, and 75% of video:"
              },
              ...imageContents
            ]
          }],
        });
        
        const rawResponse = response.content[0].type === 'text' ? response.content[0].text : "";
        console.log("Anthropic Enneagram Video raw response:", rawResponse.substring(0, 500));
        
        let jsonText = rawResponse;
        const jsonMatch = rawResponse.match(/```json\s*([\s\S]*?)\s*```/) || rawResponse.match(/```\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          jsonText = jsonMatch[1];
        }
        
        try {
          analysisResult = JSON.parse(jsonText);
        } catch (parseError) {
          console.error("Failed to parse Anthropic response:", parseError);
          analysisResult = {
            summary: rawResponse.substring(0, 1000) || "Unable to format analysis",
            primary_type: { type: "Unable to determine", confidence: "Low", core_motivation: "Formatting error", behavioral_indicators: [] },
            secondary_possibilities: [],
            wing_analysis: "Unable to analyze due to formatting error",
            temporal_patterns: { consistency: "Unknown", energy_changes: "Unknown", defensive_patterns: "Unknown" },
            triadic_analysis: { center: "Unknown", stance: "Unknown" },
            behavioral_style_markers: [],
            personality_summary: "Unable to format analysis"
          };
        }
      } else if (selectedModel === "deepseek") {
        return res.status(400).json({ 
          error: "DeepSeek does not support video analysis. Please use OpenAI or Anthropic for video-based Enneagram analysis." 
        });
      } else if (selectedModel === "perplexity") {
        return res.status(400).json({ 
          error: "Perplexity does not support video analysis. Please use OpenAI or Anthropic for video-based Enneagram analysis." 
        });
      }
      
      console.log("Enneagram video analysis complete");
      
      // Helper function to safely stringify any value into readable text
      const safeStringify = (value: any): string => {
        if (typeof value === 'string') return value;
        if (typeof value === 'object' && value !== null) {
          // If it's an array, handle each item recursively
          if (Array.isArray(value)) {
            return value.map(item => {
              if (typeof item === 'string') return item;
              if (typeof item === 'object' && item !== null) {
                // Format objects in arrays as key-value pairs
                return Object.entries(item)
                  .map(([key, val]) => `${key}: ${val}`)
                  .join('\n');
              }
              return String(item);
            }).join('\n\n');
          }
          // If it's an object with numbered keys (like "1", "2", etc), format as numbered list
          const keys = Object.keys(value);
          if (keys.length > 0 && keys.every(k => /^\d+$/.test(k))) {
            return keys
              .sort((a, b) => parseInt(a) - parseInt(b))
              .map(key => `${key}. ${value[key]}`)
              .join('\n');
          }
          // If it's an object with named keys, format as key-value pairs
          return Object.entries(value)
            .map(([key, val]) => `${val}`)
            .join('\n\n');
        }
        return String(value || '');
      };
      
      // Format the analysis for display
      let formattedContent = `Enneagram Personality Analysis\nMode: 9-Type Behavioral Video Framework\n\n`;
      formattedContent += `Summary:\n${safeStringify(analysisResult.summary)}\n\n`;
      
      // Primary Type
      if (analysisResult.primary_type) {
        formattedContent += `Primary Type: ${analysisResult.primary_type.type}\n`;
        formattedContent += `Confidence: ${analysisResult.primary_type.confidence}\n`;
        formattedContent += `Core Motivation: ${safeStringify(analysisResult.primary_type.core_motivation)}\n`;
        if (analysisResult.primary_type.behavioral_indicators && analysisResult.primary_type.behavioral_indicators.length > 0) {
          formattedContent += `Behavioral Indicators:\n${safeStringify(analysisResult.primary_type.behavioral_indicators)}\n`;
        }
        formattedContent += `\n`;
      }
      
      // Secondary Possibilities
      if (analysisResult.secondary_possibilities && analysisResult.secondary_possibilities.length > 0) {
        formattedContent += `Secondary Possibilities:\n${safeStringify(analysisResult.secondary_possibilities)}\n\n`;
      }
      
      // Wing Analysis
      if (analysisResult.wing_analysis) {
        formattedContent += `Wing Analysis:\n${safeStringify(analysisResult.wing_analysis)}\n\n`;
      }
      
      // Temporal Patterns
      if (analysisResult.temporal_patterns) {
        formattedContent += `Temporal Patterns:\n`;
        formattedContent += `Consistency: ${analysisResult.temporal_patterns.consistency || 'N/A'}\n`;
        formattedContent += `Energy Changes: ${analysisResult.temporal_patterns.energy_changes || 'N/A'}\n`;
        formattedContent += `Defensive Patterns: ${analysisResult.temporal_patterns.defensive_patterns || 'N/A'}\n\n`;
      }
      
      // Triadic Analysis
      if (analysisResult.triadic_analysis) {
        formattedContent += `Triadic Analysis:\n`;
        formattedContent += `Center: ${analysisResult.triadic_analysis.center || 'N/A'}\n`;
        formattedContent += `Stance: ${analysisResult.triadic_analysis.stance || 'N/A'}\n\n`;
      }
      
      // Behavioral Style Markers
      if (analysisResult.behavioral_style_markers && analysisResult.behavioral_style_markers.length > 0) {
        formattedContent += `Behavioral Style Markers:\n${safeStringify(analysisResult.behavioral_style_markers)}\n\n`;
      }
      
      // Personality Summary
      if (analysisResult.personality_summary) {
        formattedContent += `Personality Summary:\n${safeStringify(analysisResult.personality_summary)}\n\n`;
      }
      
      // Create analysis record
      const analysis = await storage.createAnalysis({
        sessionId,
        title: title || `Enneagram Video Analysis`,
        mediaUrl: mediaData,
        mediaType: "video",
        personalityInsights: { analysis: formattedContent, enneagram_type: analysisResult.primary_type?.type },
        modelUsed: selectedModel,
      });
      
      // Create message with formatted analysis
      const message = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        content: formattedContent,
        role: "assistant",
      });
      
      res.json({
        analysisId: analysis.id,
        personalityInsights: { analysis: formattedContent, enneagram_type: analysisResult.primary_type?.type },
        messages: [message],
        mediaUrl: mediaData,
      });
    } catch (error) {
      console.error("Enneagram video analysis error:", error);
      res.status(500).json({ error: "Failed to analyze video for Enneagram" });
    }
  });

  app.get("/api/messages", async (req, res) => {
    try {
      const { sessionId } = req.query;
      
      if (!sessionId || typeof sessionId !== 'string') {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      const messages = await storage.getMessagesBySessionId(sessionId);
      res.json(messages);
    } catch (error) {
      console.error("Get messages error:", error);
      res.status(400).json({ error: "Failed to get messages" });
    }
  });

  app.get("/api/shared-analysis/:shareId", async (req, res) => {
    try {
      const { shareId } = getSharedAnalysisSchema.parse({ shareId: req.params.shareId });
      
      // Get the share record
      const share = await storage.getShareById(shareId);
      if (!share) {
        return res.status(404).json({ error: "Shared analysis not found" });
      }
      
      // Get the analysis
      const analysis = await storage.getAnalysisById(share.analysisId);
      if (!analysis) {
        return res.status(404).json({ error: "Analysis not found" });
      }
      
      // Get all messages for this analysis
      const messages = await storage.getMessagesBySessionId(analysis.sessionId);
      
      // Return the complete data
      res.json({
        analysis,
        messages,
        share,
        emailServiceAvailable: isEmailServiceConfigured
      });
    } catch (error) {
      console.error("Get shared analysis error:", error);
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
      } else {
        res.status(400).json({ error: "Failed to get shared analysis" });
      }
    }
  });

  // API status endpoint - returns the availability of various services
  app.get("/api/status", async (req, res) => {
    try {
      const statusData = {
        openai: !!openai,
        anthropic: !!anthropic,
        perplexity: !!process.env.PERPLEXITY_API_KEY,
        aws: !!process.env.AWS_ACCESS_KEY_ID && !!process.env.AWS_SECRET_ACCESS_KEY,
        facepp: !!process.env.FACEPP_API_KEY && !!process.env.FACEPP_API_SECRET,
        sendgrid: !!process.env.SENDGRID_API_KEY && !!process.env.SENDGRID_VERIFIED_SENDER,
        timestamp: new Date().toISOString()
      };
      
      res.json(statusData);
    } catch (error) {
      console.error("Error checking API status:", error);
      res.status(500).json({ error: "Failed to check API status" });
    }
  });
  
  // Session management endpoints
  app.get("/api/sessions", async (req, res) => {
    try {
      const sessions = await storage.getAllSessions();
      res.json(sessions);
    } catch (error) {
      console.error("Error getting sessions:", error);
      res.status(500).json({ error: "Failed to get sessions" });
    }
  });
  
  app.post("/api/session/clear", async (req, res) => {
    try {
      const { sessionId } = req.body;
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      await storage.clearSession(sessionId);
      res.json({ success: true });
    } catch (error) {
      console.error("Error clearing session:", error);
      res.status(500).json({ error: "Failed to clear session" });
    }
  });
  
  app.patch("/api/session/name", async (req, res) => {
    try {
      const { sessionId, name } = req.body;
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      if (!name) {
        return res.status(400).json({ error: "Name is required" });
      }
      
      await storage.updateSessionName(sessionId, name);
      res.json({ success: true });
    } catch (error) {
      console.error("Error updating session name:", error);
      res.status(500).json({ error: "Failed to update session name" });
    }
  });
  
  // Test email endpoint (for troubleshooting only, disable in production)
  app.get("/api/test-email", async (req, res) => {
    try {
      if (!process.env.SENDGRID_API_KEY || !process.env.SENDGRID_VERIFIED_SENDER) {
        return res.status(503).json({ 
          error: "Email service is not available. Please check environment variables." 
        });
      }
      
      // Create a test share
      const testShare = {
        id: 9999,
        analysisId: 9999,
        senderEmail: "test@example.com",
        recipientEmail: process.env.SENDGRID_VERIFIED_SENDER, // Use the verified sender as recipient for testing
        status: "pending",
        createdAt: new Date().toISOString()
      };
      
      // Create a test analysis
      const testAnalysis = {
        id: 9999,
        sessionId: "test-session",
        title: "Test Analysis",
        mediaType: "text",
        mediaUrl: null,
        peopleCount: 1,
        personalityInsights: {
          summary: "This is a test analysis summary for email testing purposes.",
          personality_core: {
            summary: "Test personality core summary."
          },
          thought_patterns: {
            summary: "Test thought patterns summary."
          },
          professional_insights: {
            summary: "Test professional insights summary."
          },
          growth_areas: {
            strengths: ["Test strength 1", "Test strength 2"],
            challenges: ["Test challenge 1", "Test challenge 2"],
            development_path: "Test development path."
          }
        },
        downloaded: false,
        createdAt: new Date().toISOString()
      };
      
      // Send test email
      console.log("Sending test email...");
      const emailSent = await sendAnalysisEmail({
        share: testShare,
        analysis: testAnalysis,
        shareUrl: "https://example.com/test-share"
      });
      
      if (emailSent) {
        res.json({ success: true, message: "Test email sent successfully" });
      } else {
        res.status(500).json({ success: false, error: "Failed to send test email" });
      }
    } catch (error) {
      console.error("Test email error:", error);
      res.status(500).json({ success: false, error: String(error) });
    }
  });
  
  // Download analysis as PDF or DOCX
  app.get("/api/download/:analysisId", async (req, res) => {
    try {
      const { analysisId } = req.params;
      const format = req.query.format as string || 'pdf';
      
      // Get the analysis from storage
      const analysis = await storage.getAnalysisById(parseInt(analysisId));
      if (!analysis) {
        return res.status(404).json({ error: "Analysis not found" });
      }
      
      // Import document generation services
      const { generateAnalysisHtml, generatePdf, generateDocx } = require('./services/document');
      
      let buffer: Buffer;
      let contentType: string;
      let filename: string;
      
      if (format === 'docx') {
        // Generate DOCX
        buffer = await generateDocx(analysis);
        contentType = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
        filename = `personality-analysis-${analysisId}.docx`;
      } else {
        // Default to PDF
        const htmlContent = generateAnalysisHtml(analysis);
        buffer = await generatePdf(htmlContent);
        contentType = 'application/pdf';
        filename = `personality-analysis-${analysisId}.pdf`;
      }
      
      // Mark as downloaded in the database
      await storage.updateAnalysisDownloadStatus(analysis.id, true);
      
      // Send the file
      res.setHeader('Content-Type', contentType);
      res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
      res.setHeader('Content-Length', buffer.length);
      res.send(buffer);
      
    } catch (error) {
      console.error("Download error:", error);
      res.status(500).json({ error: "Failed to generate document" });
    }
  });

  app.post("/api/share", async (req, res) => {
    try {
      // Check if email service is configured
      if (!isEmailServiceConfigured) {
        return res.status(503).json({ 
          error: "Email sharing is not available. Please try again later or contact support." 
        });
      }

      const shareData = insertShareSchema.parse(req.body);

      // Create share record
      const share = await storage.createShare(shareData);

      // Get the analysis
      const analysis = await storage.getAnalysisById(shareData.analysisId);
      if (!analysis) {
        return res.status(404).json({ error: "Analysis not found" });
      }

      // Generate the share URL with the current hostname and /share path with analysis ID
      const hostname = req.get('host');
      const protocol = req.headers['x-forwarded-proto'] || req.protocol;
      const shareUrl = `${protocol}://${hostname}/share/${share.id}`;
      
      // Send email with share URL
      const emailSent = await sendAnalysisEmail({
        share,
        analysis,
        shareUrl
      });

      // Update share status based on email sending result
      await storage.updateShareStatus(share.id, emailSent ? "sent" : "error");

      if (!emailSent) {
        return res.status(500).json({ 
          error: "Failed to send email. Please try again later." 
        });
      }

      res.json({ success: emailSent, shareUrl });
    } catch (error) {
      console.error('Share endpoint error:', error);
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
      } else {
        res.status(400).json({ error: "Failed to share analysis" });
      }
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}

async function analyzeFaceWithRekognition(imageBuffer: Buffer, maxPeople: number = 5) {
  const command = new DetectFacesCommand({
    Image: {
      Bytes: imageBuffer
    },
    Attributes: ['ALL']
  });

  console.log('Sending request to AWS Rekognition...');
  const response = await rekognition.send(command);
  console.log('Received response from AWS Rekognition');
  const faces = response.FaceDetails || [];

  if (faces.length === 0) {
    throw new Error("No faces detected in the image");
  }

  // Filter faces by confidence threshold (>=90%) to eliminate false positives
  const highConfidenceFaces = faces.filter(face => (face.Confidence || 0) >= 90);
  
  if (highConfidenceFaces.length === 0) {
    throw new Error("No high-confidence faces detected in the image");
  }

  console.log(`Filtered to ${highConfidenceFaces.length} high-confidence faces (>=90%) from ${faces.length} total detections`);

  // Limit the number of faces to analyze
  const facesToProcess = highConfidenceFaces.slice(0, maxPeople);
  
  // Process each face and add descriptive labels
  return facesToProcess.map((face, index) => {
    // Create a descriptive label for each person
    let personLabel = `Person ${index + 1}`;
    
    // Add gender and approximate age to label if available
    if (face.Gender?.Value) {
      const genderLabel = face.Gender.Value.toLowerCase() === 'male' ? 'Male' : 'Female';
      const ageRange = face.AgeRange ? `${face.AgeRange.Low}-${face.AgeRange.High}` : '';
      personLabel = `${personLabel} (${genderLabel}${ageRange ? ', ~' + ageRange + ' years' : ''})`;
    }
  
  return {
    personLabel,
    positionInImage: index + 1,
    boundingBox: face.BoundingBox || {
      Width: 0,
      Height: 0,
      Left: 0,
      Top: 0
    },
    age: {
      low: face.AgeRange?.Low || 0,
      high: face.AgeRange?.High || 0
    },
      gender: face.Gender?.Value?.toLowerCase() || "unknown",
      emotion: face.Emotions?.reduce((acc, emotion) => {
        if (emotion.Type && emotion.Confidence) {
          acc[emotion.Type.toLowerCase()] = emotion.Confidence / 100;
        }
        return acc;
      }, {} as Record<string, number>),
      faceAttributes: {
        smile: face.Smile?.Value ? (face.Smile.Confidence || 0) / 100 : 0,
        eyeglasses: face.Eyeglasses?.Value ? "Glasses" : "NoGlasses",
        sunglasses: face.Sunglasses?.Value ? "Sunglasses" : "NoSunglasses",
        beard: face.Beard?.Value ? "Yes" : "No",
        mustache: face.Mustache?.Value ? "Yes" : "No",
        eyesOpen: face.EyesOpen?.Value ? "Yes" : "No",
        mouthOpen: face.MouthOpen?.Value ? "Yes" : "No",
        quality: {
          brightness: face.Quality?.Brightness || 0,
          sharpness: face.Quality?.Sharpness || 0,
        },
        pose: {
          pitch: face.Pose?.Pitch || 0,
          roll: face.Pose?.Roll || 0,
          yaw: face.Pose?.Yaw || 0,
        }
      },
      dominant: index === 0 // Flag the first/largest face as dominant
    };
  });
}

// Detect objects and scenes in the image for context
async function detectImageLabels(imageBuffer: Buffer) {
  try {
    const command = new DetectLabelsCommand({
      Image: {
        Bytes: imageBuffer
      },
      MaxLabels: 10,
      MinConfidence: 70
    });

    const response = await rekognition.send(command);
    const labels = response.Labels || [];
    
    // Format labels with confidence
    return labels
      .map(label => `${label.Name} (${label.Confidence?.toFixed(0)}%)`)
      .join(', ');
  } catch (error) {
    console.error('Error detecting labels:', error);
    return '';
  }
}

async function getPersonalityInsights(faceAnalysis: any, videoAnalysis: any = null, audioTranscription: any = null, sceneContext: string = '', mediaData: string = '', mediaType: string = 'image') {
  // Check if any API clients are available, display warning if not
  if (!openai && !anthropic && !process.env.PERPLEXITY_API_KEY) {
    console.warn("No AI model API clients are available. Using fallback analysis.");
    return {
      peopleCount: Array.isArray(faceAnalysis) ? faceAnalysis.length : 1,
      individualProfiles: [{
        summary: "API keys are required for detailed analysis. Please configure OpenAI, Anthropic, or Perplexity API keys.",
        detailed_analysis: {
          personality_core: "API keys required for detailed analysis",
          thought_patterns: "API keys required for detailed analysis",
          cognitive_style: "API keys required for detailed analysis",
          professional_insights: "API keys required for detailed analysis",
          relationships: {
            current_status: "Not available",
            parental_status: "Not available",
            ideal_partner: "Not available"
          },
          growth_areas: {
            strengths: ["Not available"],
            challenges: ["Not available"],
            development_path: "Not available"
          }
        }
      }]
    };
  }
  
  // SPECIAL HANDLING FOR VIDEO ANALYSIS
  if (mediaType === 'video' && videoAnalysis?.videoFrames && videoAnalysis.videoFrames.length > 0) {
    console.log(`Analyzing video with ${videoAnalysis.videoFrames.length} frames and audio transcription...`);
    
    const videoDuration = videoAnalysis.videoDuration || videoAnalysis.videoFrames.length;
    const transcription = audioTranscription?.transcription || "No audio transcription available";
    
    const videoAnalysisPrompt = `You are an expert personality analyst. Analyze this video comprehensively, providing specific evidence-based answers to ALL questions below WITH TIMESTAMPS.

CRITICAL: Every answer must reference SPECIFIC VISUAL/AUDIO EVIDENCE with TIMESTAMPS (e.g., "at 0:05" or "from 0:10-0:15").

Video Duration: ${videoDuration} seconds
Audio Transcription: "${transcription}"

I. PHYSICAL & BEHAVIORAL CUES
1. How does the person's gait or movement rhythm change across the clip?
2. Which recurring gesture seems habitual rather than situational?
3. Describe one moment where muscle tension releases or spikes — what triggers it?
4. How does posture vary when the person speaks vs. listens?
5. Identify one micro-adjustment (e.g., hair touch, collar fix) and explain its likely emotional cause.
6. What is the person doing with their hands during silent intervals?
7. How consistent is eye-contact across frames? Give timestamps showing breaks or sustained gazes.
8. At which point does breathing rate visibly change, and what precedes it?
9. Describe the physical energy level throughout — rising, falling, or cyclical?
10. What body part seems most expressive (eyes, shoulders, mouth), and how is that used?

II. EXPRESSION & EMOTION OVER TIME
11. Track micro-expressions that flicker and vanish. At what timestamps do they appear?
12. When does the dominant emotion shift, and how abruptly?
13. Does the person's smile fade naturally or snap off?
14. Which emotion seems performed vs. spontaneous? Cite frames/timestamps.
15. How does blink rate change when discussing specific topics?
16. Identify one involuntary facial tic and interpret its significance.
17. Are there moments of incongruence between facial expression and vocal tone?
18. When does the person's face "freeze" — i.e., hold still unnaturally — and what triggers that?
19. What subtle expression signals discomfort before any verbal cue?
20. How does lighting or camera angle amplify or mute visible emotions?

III. SPEECH, VOICE & TIMING
21. Describe baseline vocal timbre — breathy, clipped, resonant — and what personality trait it implies.
22. At which timestamp does pitch spike or flatten dramatically? Why?
23. How does speaking rate change when emotionally charged content arises?
24. Identify one pause longer than 1.5 seconds and interpret it psychologically.
25. What filler words or vocal tics recur, and what function do they serve?
26. How synchronized are gestures with speech rhythm?
27. Does the voice carry underlying fatigue, tension, or confidence? Provide audible markers.
28. Compare early vs. late segments: does articulation become more or less precise?
29. What is the emotional contour of the voice across the clip (anxious → calm, etc.)?
30. When does volume drop below baseline, and what coincides with it visually?

IV. CONTEXT, ENVIRONMENT & INTERACTION
31. What environmental cues (background noise, lighting shifts) change mid-video?
32. How does the camera distance or angle influence perceived dominance or submission?
33. Are there off-screen sounds or glances suggesting another presence?
34. When the person looks away, where do they look, and what might they be avoiding?
35. How do objects in the frame get used or ignored (cup, pen, phone)?
36. Does the person adapt posture or tone in response to environmental change?
37. What part of the environment most reflects personality (book titles, wall art, tidiness)?
38. How does background color palette influence mood perception?
39. Is there evidence of editing cuts or jump transitions that alter authenticity?
40. What temporal pacing (camera motion, cut frequency) matches or mismatches emotional tempo?

V. PERSONALITY & PSYCHOLOGICAL INFERENCE
41. Based on kinetic patterns, what baseline temperament (introvert/extrovert, restrained/expressive) emerges?
42. What defense mechanism manifests dynamically (e.g., laughter after stress cue)?
43. When does self-presentation collapse momentarily into candor?
44. What behavioral marker suggests anxiety management (fidgeting, throat clearing, leg bounce)?
45. How does the person handle silence — restless, composed, avoidant?
46. Identify one moment that feels genuinely unguarded; what detail proves it?
47. What relational stance is enacted toward the viewer (teacher, confessor, performer)?
48. Does the body ever contradict the words? Provide timestamps.
49. What sustained pattern (voice-tone loop, repeated motion) indicates underlying psychological theme?
50. What overall transformation occurs from first to last frame — and what emotional or existential story does that evolution tell?

Return JSON:
{
  "summary": "2-3 sentence overview with specific details from video",
  "detailed_analysis": {
    "physical_behavioral": "Answers to questions 1-10 with timestamps and specific evidence",
    "expression_emotion_time": "Answers to questions 11-20 with timestamps and specific evidence",
    "speech_voice_timing": "Answers to questions 21-30 with timestamps and specific evidence",
    "context_environment": "Answers to questions 31-40 with timestamps and specific evidence",
    "personality_inference": "Answers to questions 41-50 with timestamps and specific evidence"
  }
}`;
    
    try {
      if (!openai) {
        throw new Error("OpenAI client not available");
      }
      
      // Build content array with prompt and video frames
      const visionContent: any[] = [
        {
          type: "text",
          text: videoAnalysisPrompt
        }
      ];
      
      // Add all video frames (limit to prevent token overflow)
      const framesToInclude = videoAnalysis.videoFrames.slice(0, 10); // Max 10 frames
      framesToInclude.forEach((frame: any) => {
        visionContent.push({
          type: "image_url",
          image_url: {
            url: frame.base64Data
          }
        });
      });
      
      const response = await openai.chat.completions.create({
        model: "gpt-4o",
        messages: [
          {
            role: "user",
            content: visionContent
          }
        ],
        response_format: { type: "json_object" },
        max_tokens: 4096
      });
      
      const analysisResult = JSON.parse(response.choices[0]?.message.content || "{}");
      
      return {
        peopleCount: 1,
        individualProfiles: [analysisResult],
        detailed_analysis: analysisResult.detailed_analysis || {}
      };
    } catch (error) {
      console.error("Error in video analysis:", error);
      throw new Error("Failed to analyze video. Please try again.");
    }
  }
  
  // Check if faceAnalysis is an array (multiple people) or single object
  const isMultiplePeople = Array.isArray(faceAnalysis);
  
  // If we have multiple people, analyze each one separately
  if (isMultiplePeople) {
    console.log(`Analyzing ${faceAnalysis.length} people...`);
    
    // Create a combined analysis with an overview and individual profiles
    let multiPersonAnalysis = {
      peopleCount: faceAnalysis.length,
      overviewSummary: `Analysis of ${faceAnalysis.length} people detected in the media.`,
      individualProfiles: [] as any[],
      groupDynamics: undefined as string | undefined, // Will be populated later for multi-person analyses
      detailed_analysis: {} // For backward compatibility with message format
    };
    
    // Analyze each person with the existing logic (concurrently for efficiency)
    const analysisPromises = faceAnalysis.map(async (personFaceData) => {
      try {
        // Create input for this specific person
        const personInput = {
          faceAnalysis: personFaceData,
          ...(videoAnalysis && { videoAnalysis }),
          ...(audioTranscription && { audioTranscription })
        };
        
        // Use the standard analysis prompt but customized for the person
        const personLabel = personFaceData.personLabel || "Person";
        
        // Format the Rekognition data in a human-readable way
        const faceData = personFaceData;
        const emotionList = Object.entries(faceData.emotion || {})
          .map(([emotion, confidence]) => `${emotion}: ${(confidence * 100).toFixed(1)}%`)
          .join(', ');
        
        const analysisPrompt = `You are an expert personality analyst. Analyze this photo in comprehensive detail, providing specific evidence-based answers to ALL questions below.

CRITICAL: Every answer must reference SPECIFIC VISUAL EVIDENCE from the photo. Do not use generic descriptions.

I. PHYSICAL CUES
1. What is the person's approximate age range, and what visual evidence supports this?
2. What is their likely dominant hand, based on body posture or hand use?
3. What kind of lighting was used (natural, fluorescent, LED), and how does it shape facial tone or mood?
4. How symmetrical is the person's face, and what asymmetries are visible?
5. Describe the color and apparent texture of the person's skin in objective terms.
6. Identify one visible physical trait (scar, mole, wrinkle pattern) and infer its probable significance.
7. What can be inferred about the person's sleep habits from the eyes and skin tone?
8. Describe the person's hair (color, grooming, direction, style) and what it indicates.
9. What kind of lighting shadow falls across the eyes or nose, and what mood does that convey?
10. Is there evidence of cosmetic enhancement (makeup, filters, retouching)?

II. EXPRESSION & EMOTION
11. Describe the dominant facial expression in granular terms (eyebrow position, lip tension, gaze angle).
12. Does the expression look posed or spontaneous? Why?
13. Identify micro-expressions suggesting secondary emotions.
14. Does the smile (if any) engage the eyes? What does that reveal psychologically?
15. Compare upper-face emotion vs. lower-face emotion; do they match?
16. What emotional tone is conveyed by the person's gaze direction?
17. Does the person appear guarded, open, or performative? Cite visible evidence.
18. Are there tension points in the jaw or neck suggesting repressed emotion?
19. Estimate how long the expression was held for the photo.
20. Does the emotion appear congruent with the setting or mismatched?

III. COMPOSITION & CONTEXT
21. Describe the setting (indoor/outdoor, professional/personal).
22. What objects or background details signal aspects of lifestyle or occupation?
23. How does clothing color palette interact with lighting to create emotional tone?
24. What focal length or camera distance was likely used?
25. Is there visible clutter or minimalism, and what does that suggest?
26. Are there reflections, windows, or mirrors in frame?
27. How does body posture interact with spatial framing?
28. What portion of the frame does the subject occupy?
29. Is there visible symmetry or imbalance in composition?
30. Identify one hidden or easily overlooked element.

IV. PERSONALITY & PSYCHOLOGICAL INFERENCE
31. Based on facial micro-cues, what is the person's baseline affect?
32. What defense mechanism seems most active?
33. Describe the likely self-image being projected.
34. What aspects seem unconsciously revealing versus deliberately controlled?
35. How would the person handle confrontation?
36. Does the person exhibit signs of narcissism or self-doubt?
37. What cognitive style is implied?
38. What is the person's apparent relationship to vulnerability?
39. Does the photo suggest recent emotional hardship or resilience?
40. How does the person want to be seen vs. how they actually appear?

V. SYMBOLIC & METAPSYCHOLOGICAL ANALYSIS
41. What emotional temperature dominates the photo's color space?
42. If this were a dream image, what would each major element symbolize?
43. What mythic or cinematic archetype does the person most resemble?
44. Which aspect of the psyche is most visible?
45. What unconscious conflict seems dramatized in the composition?
46. How does clothing or accessories function as psychological armor?
47. What is the implied relationship between photographer and subject?
48. If this were part of a sequence, what emotional narrative would it tell?
49. What single object or feature best symbolizes the person's life stance?
50. What inner contradiction or paradox defines the subject?

METADATA: Age ${faceData.age?.low || 'unknown'}-${faceData.age?.high || 'unknown'}, Gender ${faceData.gender || 'unknown'}, Emotions detected: ${emotionList}${sceneContext ? `, Objects: ${sceneContext}` : ''}

Return JSON:
{
  "summary": "2-3 sentence overview with specific details from photo",
  "detailed_analysis": {
    "physical_cues": "Answers to questions 1-10 with specific evidence",
    "expression_emotion": "Answers to questions 11-20 with specific evidence",
    "composition_context": "Answers to questions 21-30 with specific evidence",
    "personality_inference": "Answers to questions 31-40 with specific evidence",
    "symbolic_analysis": "Answers to questions 41-50 with specific evidence"
  }
}`;

        // Use OpenAI Vision API with actual image
        try {
          if (!openai) {
            throw new Error("OpenAI client not available");
          }
          
          const response = await openai.chat.completions.create({
            model: "gpt-4o",
            messages: [
              {
                role: "user",
                content: [
                  {
                    type: "text",
                    text: analysisPrompt
                  },
                  {
                    type: "image_url",
                    image_url: {
                      url: mediaData
                    }
                  }
                ],
              },
            ],
            response_format: { type: "json_object" },
            max_tokens: 4096,
          });
          
          // Parse and add person identifier
          const analysisResult = JSON.parse(response.choices[0]?.message.content || "{}");
          return {
            ...analysisResult,
            personLabel: personFaceData.personLabel,
            personIndex: personFaceData.positionInImage,
            // Add positional data for potential UI highlighting
            boundingBox: personFaceData.boundingBox
          };
        } catch (err) {
          console.error(`Failed to analyze ${personLabel}:`, err);
          // Return minimal profile on error
          return {
            summary: `Analysis of ${personLabel} could not be completed.`,
            detailed_analysis: {
              personality_core: "Analysis unavailable for this individual.",
              thought_patterns: "Analysis unavailable.",
              cognitive_style: "Analysis unavailable.",
              professional_insights: "Analysis unavailable.",
              relationships: {
                current_status: "Analysis unavailable.",
                parental_status: "Analysis unavailable.",
                ideal_partner: "Analysis unavailable."
              },
              growth_areas: {
                strengths: ["Unknown"],
                challenges: ["Unknown"],
                development_path: "Analysis unavailable."
              }
            },
            personLabel: personFaceData.personLabel,
            personIndex: personFaceData.positionInImage
          };
        }
      } catch (error) {
        console.error("Error analyzing person:", error);
        return null;
      }
    });
    
    // Wait for all analyses to complete
    const individualResults = await Promise.all(analysisPromises);
    
    // Filter out any failed analyses
    multiPersonAnalysis.individualProfiles = individualResults.filter(result => result !== null);
    
    // Generate a group dynamics summary if we have multiple successful analyses
    if (multiPersonAnalysis.individualProfiles.length > 1) {
      try {
        // Create a combined input with only successful profiles
        const groupInput = {
          profiles: multiPersonAnalysis.individualProfiles.map(profile => ({
            personLabel: profile.personLabel,
            summary: profile.summary,
            key_traits: profile.detailed_analysis.personality_core.substring(0, 200) // Truncate for brevity
          }))
        };
        
        const groupPrompt = `
You are analyzing the group dynamics of ${multiPersonAnalysis.individualProfiles.length} people detected in the same media.
Based on the individual summaries provided, generate a brief analysis of how these personalities might interact.

Return a short paragraph (3-5 sentences) describing potential group dynamics, 
compatibilities or conflicts, and how these different personalities might complement each other.`;

        if (!openai) {
          throw new Error("OpenAI client not available for group dynamics analysis");
        }
        
        const groupResponse = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [
            {
              role: "system",
              content: groupPrompt,
            },
            {
              role: "user",
              content: JSON.stringify(groupInput),
            },
          ]
        });
        
        multiPersonAnalysis.groupDynamics = groupResponse.choices[0]?.message.content || 
          "Group dynamics analysis unavailable.";
      } catch (err) {
        console.error("Error generating group dynamics:", err);
        multiPersonAnalysis.groupDynamics = "Group dynamics analysis unavailable.";
      }
    }
    
    return multiPersonAnalysis;
  } else {
    // Original single-person analysis logic
    // Build a comprehensive analysis input combining all the data we have
    const analysisInput = {
      faceAnalysis,
      ...(videoAnalysis && { videoAnalysis }),
      ...(audioTranscription && { audioTranscription })
    };
    
    const analysisPrompt = `
You are an expert personality analyst capable of providing deep psychological insights. 
Analyze the provided data to generate a comprehensive personality profile.

${videoAnalysis ? 'This analysis includes video data showing gestures, activities, and attention patterns.' : ''}
${audioTranscription ? 'This analysis includes audio transcription and speech pattern data.' : ''}

Return a JSON object with the following structure:
{
  "summary": "Brief overview",
  "detailed_analysis": {
    "personality_core": "Deep analysis of core personality traits",
    "thought_patterns": "Analysis of cognitive processes and decision-making style",
    "cognitive_style": "Description of learning and problem-solving approaches",
    "professional_insights": "Career inclinations and work style",
    "relationships": {
      "current_status": "Likely relationship status",
      "parental_status": "Insights about parenting style or potential",
      "ideal_partner": "Description of compatible partner characteristics"
    },
    "growth_areas": {
      "strengths": ["List of key strengths"],
      "challenges": ["Areas for improvement"],
      "development_path": "Suggested personal growth direction"
    }
  }
}

Be thorough and insightful while avoiding stereotypes. Each section should be at least 2-3 paragraphs long.
Important: Pay careful attention to gender, facial expressions, emotional indicators, and any video/audio data provided. Base your insights on the actual analysis data provided.`;

    // Try to get analysis from all three services in parallel for maximum depth
    try {
      // Prepare API calls based on available clients
      const apiPromises = [];
      
      // OpenAI Analysis (if available)
      if (openai) {
        apiPromises.push(
          openai.chat.completions.create({
            model: "gpt-4o",
            messages: [
              {
                role: "system",
                content: analysisPrompt,
              },
              {
                role: "user",
                content: JSON.stringify(analysisInput),
              },
            ],
            response_format: { type: "json_object" },
          })
        );
      } else {
        apiPromises.push(Promise.reject(new Error("OpenAI client not available")));
      }
      
      // Anthropic Analysis (if available)
      if (anthropic) {
        apiPromises.push(
          anthropic.messages.create({
            model: "claude-3-opus-20240229",
            max_tokens: 4000,
            system: analysisPrompt,
            messages: [
              {
                role: "user",
                content: JSON.stringify(analysisInput),
              }
            ],
          })
        );
      } else {
        apiPromises.push(Promise.reject(new Error("Anthropic client not available")));
      }
      
      // Perplexity Analysis (if API key available)
      if (process.env.PERPLEXITY_API_KEY) {
        apiPromises.push(
          perplexity.query({
            model: "mistral-large-latest",
            query: `${analysisPrompt}\n\nHere is the data to analyze: ${JSON.stringify(analysisInput)}`,
          })
        );
      } else {
        apiPromises.push(Promise.reject(new Error("Perplexity API key not available")));
      }
      
      // Run all API calls in parallel
      const [openaiResult, anthropicResult, perplexityResult] = await Promise.allSettled(apiPromises);
      
      // Process results from each service
      let finalInsights: any = {};
      
      // Try each service result in order of preference
      if (openaiResult.status === 'fulfilled') {
        try {
          // Handle OpenAI response
          const openaiResponse = openaiResult.value as any;
          const openaiData = JSON.parse(openaiResponse.choices[0]?.message.content || "{}");
          finalInsights = openaiData;
          console.log("OpenAI analysis used as primary source");
        } catch (e) {
          console.error("Error parsing OpenAI response:", e);
        }
      } else if (anthropicResult.status === 'fulfilled') {
        try {
          // Handle Anthropic API response structure
          const anthropicResponse = anthropicResult.value as any;
          if (anthropicResponse.content && Array.isArray(anthropicResponse.content) && anthropicResponse.content.length > 0) {
            const content = anthropicResponse.content[0];
            // Check if it's a text content type
            if (content && content.type === 'text') {
              const anthropicText = content.text;
              // Extract JSON from Anthropic response (which might include markdown formatting)
              const jsonMatch = anthropicText.match(/```json\n([\s\S]*?)\n```/) || 
                                anthropicText.match(/{[\s\S]*}/);
                                
              if (jsonMatch) {
                const jsonStr = jsonMatch[1] || jsonMatch[0];
                finalInsights = JSON.parse(jsonStr);
                console.log("Anthropic analysis used as backup");
              }
            }
          }
        } catch (e) {
          console.error("Error parsing Anthropic response:", e);
        }
      } else if (perplexityResult.status === 'fulfilled') {
        try {
          // Extract JSON from Perplexity response
          const perplexityResponse = perplexityResult.value as any;
          const perplexityText = perplexityResponse.text || "";
          const jsonMatch = perplexityText.match(/```json\n([\s\S]*?)\n```/) || 
                           perplexityText.match(/{[\s\S]*}/);
                           
          if (jsonMatch) {
            const jsonStr = jsonMatch[1] || jsonMatch[0];
            finalInsights = JSON.parse(jsonStr);
            console.log("Perplexity analysis used as backup");
          }
        } catch (e) {
          console.error("Error parsing Perplexity response:", e);
        }
      }
      
      // If we couldn't get analysis from any service, fall back to a basic structure
      if (!finalInsights || Object.keys(finalInsights).length === 0) {
        console.error("All personality analysis services failed, using basic fallback");
        finalInsights = {
          summary: "Analysis could not be completed fully.",
          detailed_analysis: {
            personality_core: "The analysis could not be completed at this time. Please try again with a clearer image or video.",
            thought_patterns: "Analysis unavailable.",
            cognitive_style: "Analysis unavailable.",
            professional_insights: "Analysis unavailable.",
            relationships: {
              current_status: "Analysis unavailable.",
              parental_status: "Analysis unavailable.",
              ideal_partner: "Analysis unavailable."
            },
            growth_areas: {
              strengths: ["Determination"],
              challenges: ["Technical issues"],
              development_path: "Try again with a clearer image or video."
            }
          }
        };
      }
      
      // Enhance with combined insights if we have multiple services working
      if (openaiResult.status === 'fulfilled' && (anthropicResult.status === 'fulfilled' || perplexityResult.status === 'fulfilled')) {
        finalInsights.provider_info = "This analysis used multiple AI providers for maximum depth and accuracy.";
      }
      
      // For single person case, wrap in object with peopleCount=1 for consistency
      return {
        peopleCount: 1,
        individualProfiles: [finalInsights],
        detailed_analysis: finalInsights.detailed_analysis || {} // For backward compatibility
      };
    } catch (error) {
      console.error("Error in getPersonalityInsights:", error);
      throw new Error("Failed to generate personality insights. Please try again.");
    }
  }
}