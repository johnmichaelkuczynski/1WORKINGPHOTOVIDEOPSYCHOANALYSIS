# AI Personality Analysis Platform

## Overview
An advanced AI-powered personality insights platform that provides sophisticated emotional and visual analysis through intelligent interaction technologies.

## Key Features
- Multi-modal analysis (images, videos, documents, text)
- Multiple AI model support (OpenAI GPT-4o, Anthropic Claude, Perplexity)
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
- **EVIDENCE-BASED RESULTS**: All analysis answers anchored in specific visual/audio evidence with timestamps from actual media

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