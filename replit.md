# AI Personality Analysis Platform

## Overview
An advanced AI-powered personality insights platform that provides sophisticated emotional and visual analysis through intelligent interaction technologies. The platform offers multi-modal analysis (images, videos, documents, text) using various AI models to deliver detailed personality assessments. Its core purpose is to offer in-depth insights into personality traits, emotional states, and behavioral patterns, with applications spanning personal development, psychological research, and user understanding.

## User Preferences
- Prefers simple, everyday language in communications
- Values working solutions over technical explanations
- Needs comprehensive testing of new features
- Wants clean state management (no stacking of old analyses)

## System Architecture
The platform is built with a React.js frontend (TypeScript, Tailwind CSS, shadcn/ui) and an Express.js backend (TypeScript). It supports multiple AI models (OpenAI, Anthropic, Perplexity) for diverse analytical capabilities.

**Key Features:**
- **Multi-modal Analysis:** Supports analysis across text, images, videos, and documents.
- **Personality Frameworks:** Offers MBTI, Big Five/OCEAN, Enneagram (9-Type), and Dark Traits / Personality Pathology analysis across multiple modalities.
- **Personality Structure Analysis:** Comprehensive consolidated analysis synthesizing 8 major personality frameworks (Big Five/OCEAN, HEXACO, 16PF, MBTI, Keirsey, Socionics, Hogan, DISC) across text, image, and video modalities for maximum insight depth.
- **Clinical Analysis:** Comprehensive clinical assessment combining 4 major clinical frameworks (MMPI-2/MMPI-3, MCMI, DSM-5 SCID, PID-5) across text, image, and video modalities for hypothetical case study analysis with appropriate disclaimers.
- **Dark Traits Visual Assessment:** Analyzes personality pathology through visual cues including Dark Tetrad (narcissism, Machiavellianism, psychopathy, sadism), Cluster B patterns, and other maladaptive traits using facial expressions, body language, grooming patterns, and interpersonal markers.
- **Facial Analysis:** Integrates AWS Rekognition and Face++ for detailed facial feature detection.
- **Audio Transcription:** Utilizes OpenAI Whisper for transcribing audio content.
- **Session Management:** Provides session-based storage of analysis history with clear state management to prevent analysis stacking.
- **UI/UX:** Features a clean interface with capabilities for re-analysis and email sharing. Sidebar width optimized to 14rem (56) for full button text visibility.
- **Comprehensive Analysis Frameworks:** Implements detailed 50-question frameworks for image, video, and text analysis, and a 100-question framework for comprehensive text analysis, ensuring evidence-based results with specific references.
- **Robust Error Handling:** Consistent JSON.stringify usage for result formatting.
- **AI Model Naming:** Displays AI models elegantly as 知 1-4 using the Chinese character for "knowledge" for enhanced user experience.
- **Grounded Analysis:** AI prompts are designed to receive structured visual data and scene context, preventing fabrication and ensuring analysis is based on actual visual evidence.

## External Dependencies
- **AI Services**: OpenAI (GPT-4o, Whisper), Anthropic (Claude), Perplexity
- **Facial Analysis**: AWS Rekognition, Face++
- **Email Service**: SendGrid
- **Storage**: In-memory session management