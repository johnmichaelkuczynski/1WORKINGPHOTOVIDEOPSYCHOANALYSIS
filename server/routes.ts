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
      
      // Get comprehensive personality insights based on text content with 100-question framework
      const textAnalysisPrompt = `You are an expert literary psychologist and cognitive analyst. Analyze this text comprehensively, providing specific evidence-based answers to ALL 100 questions below WITH DIRECT QUOTES.

CRITICAL: Every answer must reference SPECIFIC QUOTES or PHRASES from the text. Do not use generic descriptions.

TEXT:
${content}

I. INFORMATION PROCESSING STYLE
1. Does the text show an active mind organizing information, or a passive mind reciting it?
2. Is information digested and restructured, or merely repeated?
3. Does the writer analyze causes, or just describe effects?
4. Are distinctions drawn sharply or blurred lazily?
5. Is the reasoning linear, branching, or circular?
6. Does the writer generalize prematurely or hold data until patterns emerge?
7. Are claims proportioned to evidence?
8. Does the mind shown seem inductive (pattern-seeking) or deductive (rule-enforcing)?
9. Does the prose reveal curiosity, or intellectual fatigue?
10. When the text confronts complexity, does it simplify or engage it?

II. EMOTIONAL PROCESSING STYLE
11. Are emotions named, implied, or avoided altogether?
12. When emotion appears, is it integrated or intrusive?
13. Does the author intellectualize feelings or experience them?
14. Does the tone show restraint, volatility, or emotional flatness?
15. Are emotional reactions linked to meaning, or detached from it?
16. Does the text show empathy, contempt, or indifference toward others?
17. Does emotion drive understanding or distort it?
18. When faced with threat or contradiction, does the writer show defensiveness or reflection?
19. Does the writing reveal warmth anywhere, or only analysis?
20. Is there emotional growth within the text, or the same affect from start to finish?

III. AGENCY & ACTIVITY LEVEL
21. Does the prose suggest a person who acts on the world or merely comments on it?
22. Are verbs mostly active or passive?
23. Does the writer assume control of argument flow or let it meander?
24. Does the voice take responsibility for claims ("I think," "I argue") or hide behind impersonal phrasing?
25. Does the argument attempt to change the reader's mind or simply display intellect?
26. Is there initiative — new frameworks, redefinitions — or only reaction?
27. When obstacles appear, does the author adapt or stall?
28. Does the text reveal willpower or resignation?
29. Is the overall energy rising, steady, or depleted?
30. Does the mind seem confident in shaping reality or resentful of being shaped by it?

IV. FOCUS: INTERPERSONAL VS. IDEATIONAL
31. Is attention directed toward people or toward abstractions?
32. When others are mentioned, are they treated as minds or as examples?
33. Does the prose imply social awareness or social detachment?
34. Is there sensitivity to audience, or indifference to communication?
35. Does persuasion matter, or only demonstration?
36. Does the text reveal an interest in relationships, systems, or self-display?
37. When using "we," is it inclusive or manipulative?
38. Does the writer ever show vulnerability to others' judgment?
39. Are ideas personified (showing emotional engagement) or sterilized?
40. Does the author seek understanding or dominance over interlocutors?

V. MOTIVATION, VALUE SYSTEM, AND REALITY TESTING
41. What is the writer trying to achieve — truth, recognition, safety, superiority?
42. Is success defined internally (clarity) or externally (approval)?
43. Does the writer trust reason, intuition, authority, or instinct most?
44. When evidence contradicts belief, does belief bend or resist?
45. Is the worldview optimistic, tragic, cynical, or dispassionate?
46. Does the author see the self as agent or as spectator?
47. Does the text treat reality as negotiable (conceptualist) or binding (realist)?
48. Does the writer seek understanding or vindication?
49. Are problems treated as puzzles to solve or evils to condemn?
50. Is there any visible hunger for truth — or only hunger for being right?

VI. INTELLIGENCE & CONCEPTUAL CONTROL
51. Does the writing show genuine intelligence — deep structure, inference, precision — or rote mimicry?
52. Does the argument actually advance, or does it spin in circles?
53. Does the author handle abstraction cleanly, or get lost in vagueness?
54. Does reasoning depend on concrete evidence or empty jargon?
55. When the author defines something, is it sharp and economical or padded and evasive?
56. Are terms used consistently, or redefined to dodge difficulty?
57. Is the prose intellectually ambitious in a disciplined way, or pretentious for show?
58. Does the text reveal mastery of the material, or merely second-hand familiarity?
59. Does the writer tolerate internal tension, or hide contradictions with style?
60. Does the argument show genuine insight — a new relation between ideas — or rehearse clichés?

VII. HONESTY & SINCERITY OF MIND
61. Is the writer being straightforward, or manipulating phrasing to sound profound?
62. Does the text ever admit uncertainty or limitation?
63. When challenged (implicitly or explicitly), does the writer concede or double down?
64. Is there visible willingness to change one's mind if the reasoning fails?
65. Does the author ever say "I don't know," or is omniscience performed throughout?
66. Are doubts faced or buried under abstraction?
67. Does confidence come from understanding or bluster?
68. Are counterarguments represented fairly, or caricatured for easy defeat?
69. Does the writer seem to care about truth, or about appearing intelligent?
70. Is there moral or intellectual humility anywhere in the text?

VIII. STRUCTURE, ORGANIZATION, AND FOCUS
71. Is the argument linearly constructed or chaotic?
72. Does every paragraph push the reasoning forward?
73. Are transitions real or cosmetic?
74. Does the conclusion actually follow from the premises?
75. Are examples used to clarify or to disguise weakness?
76. Does the text maintain topic discipline or wander aimlessly?
77. Is there redundancy that signals insecurity?
78. How coherent is paragraph sequencing — cumulative or random?
79. Does the text have a visible beginning, middle, and end?
80. Does the closing section resolve something or merely stop?

IX. PSYCHOLOGICAL PROFILE IN STYLE
81. Does the tone suggest calm confidence or anxious control?
82. Is the style dry, combative, ingratiating, or sermonizing?
83. Does the writer hide behind abstraction to avoid personal exposure?
84. Is there contempt for opposing views or curiosity about them?
85. What emotion drives the prose — irritation, pride, fear, wonder?
86. Is the rhythm clenched (defensive) or open (exploratory)?
87. Does the diction reveal class anxiety, moral superiority, or insecurity?
88. Is humor used to clarify or to deflect?
89. Does the writer need to dominate the reader intellectually?
90. Does the language show obsession with control, symmetry, or perfection?

X. SUBSTANCE, DEPTH, AND COGNITIVE FLEXIBILITY
91. Does the writer ever integrate a new idea mid-stream?
92. Is there evidence of learning in motion — development within the text?
93. Are insights layered, or all at the same conceptual level?
94. When describing others' ideas, does the writer paraphrase accurately?
95. Does the prose reveal real curiosity, or mere performance of curiosity?
96. Does the author show capacity for self-correction?
97. Is there flexibility of perspective, or rigid monologue?
98. Does the argument invite dialogue, or shut it down?
99. Is there intellectual empathy — ability to inhabit another view sincerely?
100. After reading, do we feel we've encountered a mind in motion or a mask of erudition?

Return JSON:
{
  "summary": "2-3 sentence overview with specific details from the text",
  "detailed_analysis": {
    "information_processing": "Answers to questions 1-10 with specific quotes and evidence",
    "emotional_processing": "Answers to questions 11-20 with specific quotes showing emotional patterns",
    "agency_activity": "Answers to questions 21-30 with quotes revealing agency and energy",
    "interpersonal_ideational": "Answers to questions 31-40 with quotes about focus and social orientation",
    "motivation_values": "Answers to questions 41-50 with quotes revealing motivations and worldview",
    "intelligence_control": "Answers to questions 51-60 with quotes showing conceptual mastery",
    "honesty_sincerity": "Answers to questions 61-70 with quotes revealing intellectual honesty",
    "structure_organization": "Answers to questions 71-80 with quotes about argument construction",
    "psychological_style": "Answers to questions 81-90 with quotes revealing psychological patterns",
    "substance_flexibility": "Answers to questions 91-100 with quotes showing cognitive depth"
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
        
        // Try to parse JSON, fallback to wrapping in analysis object with all 10 sections
        try {
          analysisResult = JSON.parse(textContent);
        } catch {
          analysisResult = {
            summary: "Text analysis completed. Unable to parse structured response.",
            detailed_analysis: {
              information_processing: textContent.substring(0, 800) || "Analysis section unavailable",
              emotional_processing: "Unable to parse emotional processing section",
              agency_activity: "Unable to parse agency & activity section", 
              interpersonal_ideational: "Unable to parse interpersonal/ideational section",
              motivation_values: "Unable to parse motivation & values section",
              intelligence_control: "Unable to parse intelligence & control section",
              honesty_sincerity: "Unable to parse honesty & sincerity section",
              structure_organization: "Unable to parse structure & organization section",
              psychological_style: "Unable to parse psychological style section",
              substance_flexibility: "Unable to parse substance & flexibility section"
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
        
        // Try to parse JSON, fallback to wrapping with all 10 sections
        try {
          analysisResult = JSON.parse(responseText);
        } catch {
          analysisResult = {
            summary: "Text analysis completed. Unable to parse structured response.",
            detailed_analysis: {
              information_processing: responseText.substring(0, 800) || "Analysis section unavailable",
              emotional_processing: "Unable to parse emotional processing section",
              agency_activity: "Unable to parse agency & activity section",
              interpersonal_ideational: "Unable to parse interpersonal/ideational section", 
              motivation_values: "Unable to parse motivation & values section",
              intelligence_control: "Unable to parse intelligence & control section",
              honesty_sincerity: "Unable to parse honesty & sincerity section",
              structure_organization: "Unable to parse structure & organization section",
              psychological_style: "Unable to parse psychological style section",
              substance_flexibility: "Unable to parse substance & flexibility section"
            }
          };
        }
      }
      else {
        return res.status(503).json({ 
          error: "Selected AI model is not available. Please try again with a different model." 
        });
      }
      
      // Format the analysis for display
      let formattedContent = `AI-Powered Text Analysis\nMode: Comprehensive Cognitive & Psychological Analysis (100 Questions)\n\n`;
      formattedContent += `${'─'.repeat(65)}\n`;
      formattedContent += `Analysis Results\n`;
      formattedContent += `${'─'.repeat(65)}\n\n`;
      
      formattedContent += `Summary:\n${analysisResult.summary || 'No summary available'}\n\n`;
      
      const detailedAnalysis = analysisResult.detailed_analysis || {};
      
      if (detailedAnalysis.information_processing) {
        formattedContent += `I. Information Processing Style:\n${detailedAnalysis.information_processing}\n\n`;
      }
      
      if (detailedAnalysis.emotional_processing) {
        formattedContent += `II. Emotional Processing Style:\n${detailedAnalysis.emotional_processing}\n\n`;
      }
      
      if (detailedAnalysis.agency_activity) {
        formattedContent += `III. Agency & Activity Level:\n${detailedAnalysis.agency_activity}\n\n`;
      }
      
      if (detailedAnalysis.interpersonal_ideational) {
        formattedContent += `IV. Focus: Interpersonal vs. Ideational:\n${detailedAnalysis.interpersonal_ideational}\n\n`;
      }
      
      if (detailedAnalysis.motivation_values) {
        formattedContent += `V. Motivation, Value System, and Reality Testing:\n${detailedAnalysis.motivation_values}\n\n`;
      }
      
      if (detailedAnalysis.intelligence_control) {
        formattedContent += `VI. Intelligence & Conceptual Control:\n${detailedAnalysis.intelligence_control}\n\n`;
      }
      
      if (detailedAnalysis.honesty_sincerity) {
        formattedContent += `VII. Honesty & Sincerity of Mind:\n${detailedAnalysis.honesty_sincerity}\n\n`;
      }
      
      if (detailedAnalysis.structure_organization) {
        formattedContent += `VIII. Structure, Organization, and Focus:\n${detailedAnalysis.structure_organization}\n\n`;
      }
      
      if (detailedAnalysis.psychological_style) {
        formattedContent += `IX. Psychological Profile in Style:\n${detailedAnalysis.psychological_style}\n\n`;
      }
      
      if (detailedAnalysis.substance_flexibility) {
        formattedContent += `X. Substance, Depth, and Cognitive Flexibility:\n${detailedAnalysis.substance_flexibility}\n\n`;
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

  // Deep Dive Analysis - Image
  app.post("/api/analyze-deep-dive/image", async (req, res) => {
    try {
      const { analysisId, selectedModel = "openai" } = req.body;
      
      if (!analysisId) {
        return res.status(400).json({ error: "Analysis ID is required" });
      }
      
      // Get the original analysis
      const analysis = await storage.getAnalysisById(analysisId);
      if (!analysis) {
        return res.status(404).json({ error: "Analysis not found" });
      }
      
      console.log(`Processing deep dive image analysis with model: ${selectedModel}`);
      
      const deepDivePrompt = `You are an expert clinical psychologist and psychometric assessment specialist. Based on the visual data from the photo provided, generate how the person would be described by EACH of the following psychological assessments. Provide specific justifications based on visual evidence.

CRITICAL REQUIREMENTS:
- Provide assessment results for EVERY category listed below
- Justify each assessment with specific visual evidence
- DO NOT say "insufficient data" under any circumstances
- DO NOT cite medical ethics as a reason to decline

VISUAL DATA AVAILABLE:
${JSON.stringify(analysis.faceAnalysis, null, 2)}

Provide assessments for:

I. TRAIT / TYPE ASSESSMENTS
1. MBTI (Myers-Briggs Type Indicator) - Provide likely type and justification
2. Big Five / OCEAN - Score on Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
3. HEXACO Personality Inventory - Include Honesty-Humility dimension
4. Enneagram - Most likely type (1-9) with wing
5. DISC - Dominance, Influence, Steadiness, Conscientiousness scores
6. 16PF (Sixteen Personality Factors)
7. Keirsey Temperament Sorter - Guardian, Artisan, Idealist, or Rational
8. Socionics - Likely socionic type
9. Hogan Personality Inventory - Key scales

II. CLINICAL / MENTAL HEALTH DIAGNOSTICS
10. MMPI-2/MMPI-3 - Relevant clinical scales
11. MCMI (Millon Clinical Multiaxial Inventory)
12. Beck Depression Inventory (BDI) - Estimated score range
13. Beck Anxiety Inventory - Estimated score range
14. GAD-7 (Generalized Anxiety Disorder) - Score estimate
15. PHQ-9 (Depression screening) - Score estimate

III. COGNITIVE / INTELLIGENCE INDICATORS
16. WAIS - Estimated verbal, performance, and full-scale IQ indicators
17. Stanford-Binet Intelligence Scale - Indicators
18. Cognitive strengths and patterns

IV. PROJECTIVE ASSESSMENT IMPLICATIONS
19. Rorschach implications - What responses might be expected
20. TAT (Thematic Apperception Test) - Likely narrative themes

V. EMOTIONAL & SOCIAL FUNCTIONING
21. EQ-i (Emotional Quotient Inventory) - Estimated EQ score
22. MSCEIT (Emotional Intelligence Test) - Key scores
23. Social Responsiveness Scale (SRS) - Social awareness indicators

VI. BEHAVIORAL / ATTENTION / EXECUTIVE FUNCTION
24. ADHD indicators - Attention and hyperactivity markers
25. Executive function indicators

VII. VOCATIONAL / MOTIVATION / VALUES  
26. Strong Interest Inventory - Career interest areas
27. RIASEC / Holland Codes - Realistic, Investigative, Artistic, Social, Enterprising, Conventional
28. Values in Action (VIA Character Strengths) - Top signature strengths
29. Schwartz Value Survey - Value priorities

VIII. PERSONALITY PATHOLOGY / DARK TRAITS
30. PCL-R (Psychopathy Checklist) - Indicators
31. Dark Triad Test - Machiavellianism, Narcissism, Psychopathy scores
32. PID-5 (Personality Inventory for DSM-5) - Pathological trait domains

Return JSON with this structure:
{
  "summary": "Overall psychological profile summary",
  "assessments": {
    "trait_type": "Results for assessments 1-9",
    "clinical_mental_health": "Results for assessments 10-15",
    "cognitive_intelligence": "Results for assessments 16-18",
    "projective": "Results for assessments 19-20",
    "emotional_social": "Results for assessments 21-23",
    "behavioral_executive": "Results for assessments 24-25",
    "vocational_values": "Results for assessments 26-29",
    "pathology_dark_traits": "Results for assessments 30-32"
  }
}`;

      let deepDiveResult: any;
      
      if (selectedModel === "openai" && openai) {
        const completion = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [
            { role: "system", content: "You are an expert clinical psychologist and psychometric assessment specialist." },
            { role: "user", content: deepDivePrompt }
          ],
          response_format: { type: "json_object" }
        });
        deepDiveResult = JSON.parse(completion.choices[0].message.content || "{}");
      } 
      else if (selectedModel === "anthropic" && anthropic) {
        const response = await anthropic.messages.create({
          model: "claude-3-7-sonnet-20250219",
          max_tokens: 4000,
          system: "You are an expert clinical psychologist and psychometric assessment specialist. Always return valid JSON.",
          messages: [{ role: "user", content: deepDivePrompt }],
        });
        
        let textContent = response.content[0].text;
        const jsonMatch = textContent.match(/```json\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          textContent = jsonMatch[1];
        }
        deepDiveResult = JSON.parse(textContent);
      }
      else if (selectedModel === "perplexity" && process.env.PERPLEXITY_API_KEY) {
        const response = await perplexity.query({
          model: "llama-3.1-sonar-small-128k-online",
          query: deepDivePrompt
        });
        
        let responseText = response.text;
        const jsonMatch = responseText.match(/```json\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          responseText = jsonMatch[1];
        }
        deepDiveResult = JSON.parse(responseText);
      }
      else {
        return res.status(503).json({ 
          error: "Selected AI model is not available." 
        });
      }
      
      // Format results
      let formattedContent = `Deep Dive Psychological Assessment - Image Analysis\n\n`;
      formattedContent += `${'═'.repeat(65)}\n\n`;
      formattedContent += `Summary:\n${deepDiveResult.summary || 'No summary available'}\n\n`;
      formattedContent += `${'─'.repeat(65)}\n\n`;
      
      const assessments = deepDiveResult.assessments || {};
      
      if (assessments.trait_type) {
        formattedContent += `I. Trait / Type Assessments:\n${assessments.trait_type}\n\n`;
      }
      if (assessments.clinical_mental_health) {
        formattedContent += `II. Clinical / Mental Health Diagnostics:\n${assessments.clinical_mental_health}\n\n`;
      }
      if (assessments.cognitive_intelligence) {
        formattedContent += `III. Cognitive / Intelligence Indicators:\n${assessments.cognitive_intelligence}\n\n`;
      }
      if (assessments.projective) {
        formattedContent += `IV. Projective Assessment Implications:\n${assessments.projective}\n\n`;
      }
      if (assessments.emotional_social) {
        formattedContent += `V. Emotional & Social Functioning:\n${assessments.emotional_social}\n\n`;
      }
      if (assessments.behavioral_executive) {
        formattedContent += `VI. Behavioral / Attention / Executive Function:\n${assessments.behavioral_executive}\n\n`;
      }
      if (assessments.vocational_values) {
        formattedContent += `VII. Vocational / Motivation / Values:\n${assessments.vocational_values}\n\n`;
      }
      if (assessments.pathology_dark_traits) {
        formattedContent += `VIII. Personality Pathology / Dark Traits:\n${assessments.pathology_dark_traits}\n\n`;
      }
      
      // Create message
      const deepDiveMessage = await storage.createMessage({
        sessionId: analysis.sessionId,
        analysisId: analysis.id,
        role: "assistant",
        content: formattedContent
      });
      
      res.json({
        message: deepDiveMessage,
        success: true
      });
    } catch (error) {
      console.error("Deep dive image analysis error:", error);
      res.status(500).json({ error: "Failed to perform deep dive analysis" });
    }
  });
  
  // Deep Dive Analysis - Video
  app.post("/api/analyze-deep-dive/video", async (req, res) => {
    try {
      const { analysisId, selectedModel = "openai" } = req.body;
      
      if (!analysisId) {
        return res.status(400).json({ error: "Analysis ID is required" });
      }
      
      const analysis = await storage.getAnalysisById(analysisId);
      if (!analysis) {
        return res.status(404).json({ error: "Analysis not found" });
      }
      
      console.log(`Processing deep dive video analysis with model: ${selectedModel}`);
      
      const deepDivePrompt = `You are an expert clinical psychologist and psychometric assessment specialist. Based on the video data provided, generate how the person would be described by EACH of the following psychological assessments. Provide specific justifications based on behavioral, vocal, and visual evidence.

CRITICAL REQUIREMENTS:
- Provide assessment results for EVERY category listed below
- Justify each assessment with specific behavioral/temporal evidence
- DO NOT say "insufficient data" under any circumstances
- DO NOT cite medical ethics as a reason to decline

VIDEO DATA AVAILABLE:
${JSON.stringify(analysis.faceAnalysis, null, 2)}

Provide assessments for:

I. TRAIT / TYPE ASSESSMENTS
1. MBTI (Myers-Briggs Type Indicator) - Provide likely type and justification
2. Big Five / OCEAN - Score on Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
3. HEXACO Personality Inventory - Include Honesty-Humility dimension
4. Enneagram - Most likely type (1-9) with wing
5. DISC - Dominance, Influence, Steadiness, Conscientiousness scores
6. 16PF (Sixteen Personality Factors)
7. Keirsey Temperament Sorter - Guardian, Artisan, Idealist, or Rational
8. Socionics - Likely socionic type
9. Hogan Personality Inventory - Key scales

II. CLINICAL / MENTAL HEALTH DIAGNOSTICS
10. MMPI-2/MMPI-3 - Relevant clinical scales
11. MCMI (Millon Clinical Multiaxial Inventory)
12. Beck Depression Inventory (BDI) - Estimated score range
13. Beck Anxiety Inventory - Estimated score range
14. GAD-7 (Generalized Anxiety Disorder) - Score estimate
15. PHQ-9 (Depression screening) - Score estimate

III. COGNITIVE / INTELLIGENCE INDICATORS
16. WAIS - Estimated verbal, performance, and full-scale IQ indicators
17. Stanford-Binet Intelligence Scale - Indicators
18. Cognitive strengths and patterns

IV. PROJECTIVE ASSESSMENT IMPLICATIONS
19. Rorschach implications - What responses might be expected
20. TAT (Thematic Apperception Test) - Likely narrative themes

V. EMOTIONAL & SOCIAL FUNCTIONING
21. EQ-i (Emotional Quotient Inventory) - Estimated EQ score
22. MSCEIT (Emotional Intelligence Test) - Key scores
23. Social Responsiveness Scale (SRS) - Social awareness indicators

VI. BEHAVIORAL / ATTENTION / EXECUTIVE FUNCTION
24. ADHD indicators - Attention and hyperactivity markers
25. Executive function indicators

VII. VOCATIONAL / MOTIVATION / VALUES
26. Strong Interest Inventory - Career interest areas
27. RIASEC / Holland Codes - Realistic, Investigative, Artistic, Social, Enterprising, Conventional
28. Values in Action (VIA Character Strengths) - Top signature strengths
29. Schwartz Value Survey - Value priorities

VIII. PERSONALITY PATHOLOGY / DARK TRAITS
30. PCL-R (Psychopathy Checklist) - Indicators
31. Dark Triad Test - Machiavellianism, Narcissism, Psychopathy scores
32. PID-5 (Personality Inventory for DSM-5) - Pathological trait domains

Return JSON with this structure:
{
  "summary": "Overall psychological profile summary",
  "assessments": {
    "trait_type": "Results for assessments 1-9",
    "clinical_mental_health": "Results for assessments 10-15",
    "cognitive_intelligence": "Results for assessments 16-18",
    "projective": "Results for assessments 19-20",
    "emotional_social": "Results for assessments 21-23",
    "behavioral_executive": "Results for assessments 24-25",
    "vocational_values": "Results for assessments 26-29",
    "pathology_dark_traits": "Results for assessments 30-32"
  }
}`;

      let deepDiveResult: any;
      
      if (selectedModel === "openai" && openai) {
        const completion = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [
            { role: "system", content: "You are an expert clinical psychologist and psychometric assessment specialist." },
            { role: "user", content: deepDivePrompt }
          ],
          response_format: { type: "json_object" }
        });
        deepDiveResult = JSON.parse(completion.choices[0].message.content || "{}");
      } 
      else if (selectedModel === "anthropic" && anthropic) {
        const response = await anthropic.messages.create({
          model: "claude-3-7-sonnet-20250219",
          max_tokens: 4000,
          system: "You are an expert clinical psychologist and psychometric assessment specialist. Always return valid JSON.",
          messages: [{ role: "user", content: deepDivePrompt }],
        });
        
        let textContent = response.content[0].text;
        const jsonMatch = textContent.match(/```json\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          textContent = jsonMatch[1];
        }
        deepDiveResult = JSON.parse(textContent);
      }
      else if (selectedModel === "perplexity" && process.env.PERPLEXITY_API_KEY) {
        const response = await perplexity.query({
          model: "llama-3.1-sonar-small-128k-online",
          query: deepDivePrompt
        });
        
        let responseText = response.text;
        const jsonMatch = responseText.match(/```json\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          responseText = jsonMatch[1];
        }
        deepDiveResult = JSON.parse(responseText);
      }
      else {
        return res.status(503).json({ 
          error: "Selected AI model is not available." 
        });
      }
      
      // Format results
      let formattedContent = `Deep Dive Psychological Assessment - Video Analysis\n\n`;
      formattedContent += `${'═'.repeat(65)}\n\n`;
      formattedContent += `Summary:\n${deepDiveResult.summary || 'No summary available'}\n\n`;
      formattedContent += `${'─'.repeat(65)}\n\n`;
      
      const assessments = deepDiveResult.assessments || {};
      
      if (assessments.trait_type) {
        formattedContent += `I. Trait / Type Assessments:\n${assessments.trait_type}\n\n`;
      }
      if (assessments.clinical_mental_health) {
        formattedContent += `II. Clinical / Mental Health Diagnostics:\n${assessments.clinical_mental_health}\n\n`;
      }
      if (assessments.cognitive_intelligence) {
        formattedContent += `III. Cognitive / Intelligence Indicators:\n${assessments.cognitive_intelligence}\n\n`;
      }
      if (assessments.projective) {
        formattedContent += `IV. Projective Assessment Implications:\n${assessments.projective}\n\n`;
      }
      if (assessments.emotional_social) {
        formattedContent += `V. Emotional & Social Functioning:\n${assessments.emotional_social}\n\n`;
      }
      if (assessments.behavioral_executive) {
        formattedContent += `VI. Behavioral / Attention / Executive Function:\n${assessments.behavioral_executive}\n\n`;
      }
      if (assessments.vocational_values) {
        formattedContent += `VII. Vocational / Motivation / Values:\n${assessments.vocational_values}\n\n`;
      }
      if (assessments.pathology_dark_traits) {
        formattedContent += `VIII. Personality Pathology / Dark Traits:\n${assessments.pathology_dark_traits}\n\n`;
      }
      
      // Create message
      const deepDiveMessage = await storage.createMessage({
        sessionId: analysis.sessionId,
        analysisId: analysis.id,
        role: "assistant",
        content: formattedContent
      });
      
      res.json({
        message: deepDiveMessage,
        success: true
      });
    } catch (error) {
      console.error("Deep dive video analysis error:", error);
      res.status(500).json({ error: "Failed to perform deep dive analysis" });
    }
  });
  
  // Deep Dive Analysis - Text
  app.post("/api/analyze-deep-dive/text", async (req, res) => {
    try {
      const { analysisId, selectedModel = "openai" } = req.body;
      
      if (!analysisId) {
        return res.status(400).json({ error: "Analysis ID is required" });
      }
      
      const analysis = await storage.getAnalysisById(analysisId);
      if (!analysis) {
        return res.status(404).json({ error: "Analysis not found" });
      }
      
      // Get the original text from the analysis
      const messages = await storage.getMessagesByAnalysisId(analysisId);
      const userMessage = messages.find(m => m.role === 'user');
      const originalText = userMessage?.content || '';
      
      console.log(`Processing deep dive text analysis with model: ${selectedModel}`);
      
      const deepDivePrompt = `You are an expert clinical psychologist and psychometric assessment specialist. Based on the text provided, generate how the author would be described by EACH of the following psychological assessments. Provide specific justifications based on linguistic and cognitive patterns in the text.

CRITICAL REQUIREMENTS:
- Provide assessment results for EVERY category listed below
- Justify each assessment with specific quotes and patterns from the text
- DO NOT say "insufficient data" under any circumstances
- DO NOT cite medical ethics as a reason to decline

TEXT TO ANALYZE:
${originalText}

Provide assessments for:

I. TRAIT / TYPE ASSESSMENTS
1. MBTI (Myers-Briggs Type Indicator) - Provide likely type and justification
2. Big Five / OCEAN - Score on Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
3. HEXACO Personality Inventory - Include Honesty-Humility dimension
4. Enneagram - Most likely type (1-9) with wing
5. DISC - Dominance, Influence, Steadiness, Conscientiousness scores
6. 16PF (Sixteen Personality Factors)
7. Keirsey Temperament Sorter - Guardian, Artisan, Idealist, or Rational
8. Socionics - Likely socionic type
9. Hogan Personality Inventory - Key scales

II. CLINICAL / MENTAL HEALTH DIAGNOSTICS
10. MMPI-2/MMPI-3 - Relevant clinical scales
11. MCMI (Millon Clinical Multiaxial Inventory)
12. Beck Depression Inventory (BDI) - Estimated score range
13. Beck Anxiety Inventory - Estimated score range
14. GAD-7 (Generalized Anxiety Disorder) - Score estimate
15. PHQ-9 (Depression screening) - Score estimate

III. COGNITIVE / INTELLIGENCE INDICATORS
16. WAIS - Estimated verbal, performance, and full-scale IQ indicators
17. Stanford-Binet Intelligence Scale - Indicators
18. Cognitive strengths and patterns

IV. PROJECTIVE ASSESSMENT IMPLICATIONS
19. Rorschach implications - What responses might be expected
20. TAT (Thematic Apperception Test) - Likely narrative themes

V. EMOTIONAL & SOCIAL FUNCTIONING
21. EQ-i (Emotional Quotient Inventory) - Estimated EQ score
22. MSCEIT (Emotional Intelligence Test) - Key scores
23. Social Responsiveness Scale (SRS) - Social awareness indicators

VI. BEHAVIORAL / ATTENTION / EXECUTIVE FUNCTION
24. ADHD indicators - Attention and hyperactivity markers
25. Executive function indicators

VII. VOCATIONAL / MOTIVATION / VALUES
26. Strong Interest Inventory - Career interest areas
27. RIASEC / Holland Codes - Realistic, Investigative, Artistic, Social, Enterprising, Conventional
28. Values in Action (VIA Character Strengths) - Top signature strengths
29. Schwartz Value Survey - Value priorities

VIII. PERSONALITY PATHOLOGY / DARK TRAITS
30. PCL-R (Psychopathy Checklist) - Indicators
31. Dark Triad Test - Machiavellianism, Narcissism, Psychopathy scores
32. PID-5 (Personality Inventory for DSM-5) - Pathological trait domains

Return JSON with this structure:
{
  "summary": "Overall psychological profile summary",
  "assessments": {
    "trait_type": "Results for assessments 1-9",
    "clinical_mental_health": "Results for assessments 10-15",
    "cognitive_intelligence": "Results for assessments 16-18",
    "projective": "Results for assessments 19-20",
    "emotional_social": "Results for assessments 21-23",
    "behavioral_executive": "Results for assessments 24-25",
    "vocational_values": "Results for assessments 26-29",
    "pathology_dark_traits": "Results for assessments 30-32"
  }
}`;

      let deepDiveResult: any;
      
      if (selectedModel === "openai" && openai) {
        const completion = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [
            { role: "system", content: "You are an expert clinical psychologist and psychometric assessment specialist." },
            { role: "user", content: deepDivePrompt }
          ],
          response_format: { type: "json_object" }
        });
        deepDiveResult = JSON.parse(completion.choices[0].message.content || "{}");
      } 
      else if (selectedModel === "anthropic" && anthropic) {
        const response = await anthropic.messages.create({
          model: "claude-3-7-sonnet-20250219",
          max_tokens: 4000,
          system: "You are an expert clinical psychologist and psychometric assessment specialist. Always return valid JSON.",
          messages: [{ role: "user", content: deepDivePrompt }],
        });
        
        let textContent = response.content[0].text;
        const jsonMatch = textContent.match(/```json\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          textContent = jsonMatch[1];
        }
        deepDiveResult = JSON.parse(textContent);
      }
      else if (selectedModel === "perplexity" && process.env.PERPLEXITY_API_KEY) {
        const response = await perplexity.query({
          model: "llama-3.1-sonar-small-128k-online",
          query: deepDivePrompt
        });
        
        let responseText = response.text;
        const jsonMatch = responseText.match(/```json\s*([\s\S]*?)\s*```/);
        if (jsonMatch) {
          responseText = jsonMatch[1];
        }
        deepDiveResult = JSON.parse(responseText);
      }
      else {
        return res.status(503).json({ 
          error: "Selected AI model is not available." 
        });
      }
      
      // Format results
      let formattedContent = `Deep Dive Psychological Assessment - Text Analysis\n\n`;
      formattedContent += `${'═'.repeat(65)}\n\n`;
      formattedContent += `Summary:\n${deepDiveResult.summary || 'No summary available'}\n\n`;
      formattedContent += `${'─'.repeat(65)}\n\n`;
      
      const assessments = deepDiveResult.assessments || {};
      
      if (assessments.trait_type) {
        formattedContent += `I. Trait / Type Assessments:\n${assessments.trait_type}\n\n`;
      }
      if (assessments.clinical_mental_health) {
        formattedContent += `II. Clinical / Mental Health Diagnostics:\n${assessments.clinical_mental_health}\n\n`;
      }
      if (assessments.cognitive_intelligence) {
        formattedContent += `III. Cognitive / Intelligence Indicators:\n${assessments.cognitive_intelligence}\n\n`;
      }
      if (assessments.projective) {
        formattedContent += `IV. Projective Assessment Implications:\n${assessments.projective}\n\n`;
      }
      if (assessments.emotional_social) {
        formattedContent += `V. Emotional & Social Functioning:\n${assessments.emotional_social}\n\n`;
      }
      if (assessments.behavioral_executive) {
        formattedContent += `VI. Behavioral / Attention / Executive Function:\n${assessments.behavioral_executive}\n\n`;
      }
      if (assessments.vocational_values) {
        formattedContent += `VII. Vocational / Motivation / Values:\n${assessments.vocational_values}\n\n`;
      }
      if (assessments.pathology_dark_traits) {
        formattedContent += `VIII. Personality Pathology / Dark Traits:\n${assessments.pathology_dark_traits}\n\n`;
      }
      
      // Create message
      const deepDiveMessage = await storage.createMessage({
        sessionId: analysis.sessionId,
        analysisId: analysis.id,
        role: "assistant",
        content: formattedContent
      });
      
      res.json({
        message: deepDiveMessage,
        success: true
      });
    } catch (error) {
      console.error("Deep dive text analysis error:", error);
      res.status(500).json({ error: "Failed to perform deep dive analysis" });
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