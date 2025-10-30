# AI Personality Analysis Platform

## Overview
An advanced AI-powered personality insights platform that provides sophisticated emotional and visual analysis through intelligent interaction technologies.

## Key Features
- Multi-modal analysis (images, videos, documents, text)
- Multiple AI model support (OpenAI GPT-4o, Anthropic Claude, Perplexity)
- MBTI personality analysis (text, images, videos)
- Big Five/OCEAN personality analysis (text, images, videos)
- Enneagram 9-Type personality analysis (text, images)
- Facial analysis using AWS Rekognition and Face++
- Audio transcription with OpenAI Whisper
- Email sharing with SendGrid integration
- Session management with analysis history
- Clean UI with re-analysis capabilities

## Technologies
- **Frontend**: React.js with TypeScript, Tailwind CSS, shadcn/ui components
- **Backend**: Express.js with TypeScript
- **AI Services**: OpenAI, Anthropic, Perplexity
- **Analysis Services**: AWS Rekognition, Face++
- **Email**: SendGrid
- **Storage**: In-memory with session management

## Project Architecture
- `client/` - React frontend application
- `server/` - Express.js backend with API routes
- `shared/` - Shared TypeScript schemas and types
- Session-based analysis storage with proper clearing functionality

## User Preferences
- Prefers simple, everyday language in communications
- Values working solutions over technical explanations
- Needs comprehensive testing of new features
- Wants clean state management (no stacking of old analyses)

## Recent Changes
### October 2025 (Latest)
- **NEW ENNEAGRAM (IMAGE) ANALYSIS FEATURE**: Added visual Enneagram personality type detection
  * Analyzes photographs to identify 9 Enneagram personality types with visual evidence
  * Comprehensive personality type framework covering all 9 types (Reformer, Helper, Achiever, Individualist, Investigator, Loyalist, Enthusiast, Challenger, Peacemaker)
  * Identifies primary type with confidence level (High/Medium/Low)
  * Provides secondary possibilities and wing analysis (e.g., 4w3 or 4w5)
  * Includes triadic analysis (Head/Heart/Body center, Aggressive/Dependent/Withdrawing stance)
  * Visual style markers based on specific image details
  * Works with OpenAI (GPT-4o) and Anthropic (Claude Sonnet) vision models
  * New sidebar button with dedicated file input ref to prevent routing conflicts
  * Backend endpoint: /api/analyze/image/enneagram
  * Frontend API function: analyzeEnneagramImage
  * safeStringify helper function for clean result formatting
  * Mutation handler: handleEnneagramImageAnalysis

- **NEW MBTI ANALYSIS FEATURE**: Added three dedicated MBTI analysis functions
  * MBTI Text Analysis - 30-question framework (5 I/E + 5 S/N + 5 T/F + 5 J/P + 10 deeper signals) with direct text quotes
  * MBTI Image Analysis - 30-question visual framework (5 I/E + 5 S/N + 5 T/F + 5 J/P + 10 cognitive indicators) analyzing visual cues
  * MBTI Video Analysis - 30-question behavioral framework (5 I/E + 5 S/N + 5 T/F + 5 J/P + 10 cognitive signals) with timestamps
  * All MBTI analyses provide predicted type (e.g., INTJ, ENFP) with confidence levels
  * Works with OpenAI, Anthropic, and Perplexity models (vision capabilities for image/video)
  * Three new buttons added to frontend (separate from comprehensive analysis)
  * Evidence-based results with specific visual/textual references
  * MBTI endpoints: /api/analyze/text/mbti, /api/analyze/image/mbti, /api/analyze/video/mbti
  * Fixed [object Object] display bug with safeStringify helper function in all three endpoints
  * Corrected prompt documentation from "50 questions" to accurate "30 questions"
  * Frontend buttons with variant="secondary" styling and proper test IDs
  * API integration complete with proper error handling and fallbacks

- **BIG FIVE/OCEAN ANALYSIS FEATURES**: Added three Big Five personality analysis functions
  * Big Five Text Analysis - Measures Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
  * Big Five Image Analysis - Visual assessment of Big Five traits
  * Big Five Video Analysis - Behavioral assessment of Big Five traits with temporal patterns
  * All analyses provide trait scores with evidence-based reasoning
  * Frontend buttons in main input area and sidebar for easy access
  
- **ENNEAGRAM TEXT ANALYSIS**: Fixed critical routing bug
  * Was calling generic analysis endpoint instead of Enneagram-specific endpoint
  * Now correctly predicts Enneagram Type 1-9 with confidence levels
  * Added missing safeStringify helper function (was causing ReferenceError crash)
  * Fixed "[object Object]" display bug with recursive object formatting
  * Improved UX by adding button to main input area alongside MBTI
  * Backend endpoint: /api/analyze/text/enneagram
  
### October 2025
- **COMPREHENSIVE TEXT ANALYSIS UPGRADE - 100 QUESTIONS**: Fully implemented 100-question framework for text analysis
  * All 100 questions across 10 categories:
    - I. Information Processing Style (10)
    - II. Emotional Processing Style (10)
    - III. Agency & Activity Level (10)
    - IV. Focus: Interpersonal vs. Ideational (10)
    - V. Motivation, Value System, and Reality Testing (10)
    - VI. Intelligence & Conceptual Control (10)
    - VII. Honesty & Sincerity of Mind (10)
    - VIII. Structure, Organization, and Focus (10)
    - IX. Psychological Profile in Style (10)
    - X. Substance, Depth, and Cognitive Flexibility (10)
  * Evidence-based analysis requiring direct quotes and specific phrases from text
  * Robust JSON parsing for OpenAI, Anthropic, and Perplexity with code fence extraction
  * Comprehensive fallback structure maintaining all 10 sections if parsing fails
  * Display formatting with Roman numeral section headers matching framework
  * Removed duplicate handlers for cleaner codebase

### January 2025
- **Fixed session clearing**: Added server-side session clearing to prevent analyses from stacking up when using "New Analysis" button
- **Enhanced "New Analysis" functionality**: Button now properly creates fresh sessions with clean state
- **Added session management endpoints**: `/api/session/clear` and related endpoints for proper session handling
- **Verified email sharing**: SendGrid integration working properly with all required API credentials
- **Confirmed re-analysis capability**: Users can analyze same content with different AI models
- **All AI services connected**: OpenAI, Anthropic, Perplexity, AWS Rekognition, Face++ all working
- **Fixed text analysis bug**: Resolved "[object Object]" display issue in text analysis results
- **Removed markdown formatting**: Cleaned up all analysis outputs to remove # and * symbols
- **Updated AI model display names**: Changed to 知 1 (Anthropic), 知 2 (OpenAI), 知 3 (DeepSeek), 知 4 (Perplexity) using Chinese character for "knowledge"
- **Fixed dropdown order**: Models now display in correct 知 1-4 sequence
- **Implemented Chinese character**: Used authentic "知" character for elegant model naming
- **Fixed text analysis formatting**: Removed raw JSON code display, now shows clean readable text output
- **Fixed DeepSeek validation**: Added "deepseek" to backend validation schemas to enable 知 3 for all analysis types
- **CRITICAL FIX - Face detection accuracy**: Added 90% confidence threshold filter to AWS Rekognition to eliminate false positives (shadows, reflections, background elements)
- **CRITICAL FIX - Grounded analysis**: AI prompts now receive structured visual data (age, gender, emotions, facial features) instead of raw JSON
- **CRITICAL FIX - Scene context**: Added AWS DetectLabels to identify objects/scenes in photos (e.g., piano, musical instrument) for contextual analysis
- **CRITICAL FIX - Fabrication prevention**: AI explicitly instructed to base analysis only on actual visual data, not fabricate details
- **REVOLUTIONARY UPGRADE - GPT-4o Vision Integration**: Now sends actual images to GPT-4o Vision API instead of just metadata
- **COMPREHENSIVE IMAGE ANALYSIS - 50-Question Framework**: Implemented detailed photo analysis covering:
  * I. Physical Cues (10 questions) - age, lighting, skin, hair, symmetry, cosmetics
  * II. Expression & Emotion (10 questions) - facial expression, micro-expressions, gaze, posed vs spontaneous
  * III. Composition & Context (10 questions) - setting, objects, clothing, camera distance, spatial framing
  * IV. Personality & Psychological Inference (10 questions) - baseline affect, defense mechanisms, self-image, vulnerability
  * V. Symbolic & Metapsychological Analysis (10 questions) - archetypes, dream symbolism, unconscious conflicts
- **COMPREHENSIVE VIDEO ANALYSIS - 50-Question Framework**: Implemented detailed video analysis with timestamps covering:
  * I. Physical & Behavioral Cues (10 questions) - gait, gestures, muscle tension, posture, hand movements, eye contact, breathing
  * II. Expression & Emotion Over Time (10 questions) - micro-expressions with timestamps, emotion shifts, blink rate, facial tics, incongruence
  * III. Speech, Voice & Timing (10 questions) - vocal timbre, pitch changes, speaking rate, pauses, filler words, gesture-speech sync
  * IV. Context, Environment & Interaction (10 questions) - environmental cues, camera angles, off-screen presence, object use, background
  * V. Personality & Psychological Inference (10 questions) - temperament, defense mechanisms, anxiety markers, unguarded moments, transformation
- **VIDEO VISION API**: Extracts multiple frames from videos and sends them to GPT-4o Vision API for temporal analysis
- **COMPREHENSIVE TEXT ANALYSIS - 50-Question Framework**: Implemented detailed text analysis with quotes covering:
  * I. Language & Style (10 questions) - sentence rhythm, adjectives, pronoun use, abstraction, diction shifts, fragments, tense, metaphors, tone, register
  * II. Emotional Indicators (10 questions) - primary emotion, repressed emotions, intensity progression, affect leakage, detachment, sensory words, ambivalence, humor
  * III. Cognitive & Structural Patterns (10 questions) - coherence, thought style, syntactic habits, contradictions, uncertainty handling, circularity, topic shifts, repetition, insight
  * IV. Self-Representation & Identity (10 questions) - self-portrayal, voice splits, authority seeking, self-image consistency, self-evaluation, vulnerability, relationship to others, audience
  * V. Symbolic & Unconscious Material (10 questions) - recurring motifs, surreal elements, oppositions, wishes/fears, metaphors, time relation, intellect vs emotion, shadow aspects, projection
- **EVIDENCE-BASED RESULTS**: All analysis answers anchored in specific visual/audio evidence with timestamps from actual media, or direct quotes from text

## API Keys Required
- `OPENAI_API_KEY` - For GPT-4o analysis and Whisper transcription
- `ANTHROPIC_API_KEY` - For Claude analysis
- `PERPLEXITY_API_KEY` - For Perplexity analysis
- `AWS_ACCESS_KEY_ID` & `AWS_SECRET_ACCESS_KEY` - For Rekognition facial analysis
- `FACEPP_API_KEY` & `FACEPP_API_SECRET` - For Face++ analysis
- `SENDGRID_API_KEY` & `SENDGRID_VERIFIED_SENDER` - For email sharing

## Status
✅ **WORKING PERFECTLY** - User confirmed all major functionality is working correctly:
- Session clearing prevents analysis stacking
- New Analysis button creates fresh sessions
- Re-analysis with different models works
- Email sharing functional
- All AI services connected and operational
- Text analysis displaying properly (no more [object Object] errors)
- Image analysis working flawlessly
- Video analysis processing correctly
- All analysis outputs clean without markdown formatting
- AI model names elegantly displayed as 知 1-4 using Chinese character for knowledge
- Text analysis output now clean and professional without raw code or JSON formatting
- DeepSeek (知 3) now working properly for video, image, text, and document analysis

## Next Steps
Ready for user's next request or feature enhancement.