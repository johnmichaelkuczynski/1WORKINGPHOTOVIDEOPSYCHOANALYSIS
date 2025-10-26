import { apiRequest } from "./queryClient";

// Type definitions for enhanced API functionality
export type ModelType = "openai" | "anthropic" | "perplexity";
export type MediaType = "image" | "video" | "document" | "text";

export async function uploadMedia(
  mediaData: string, 
  mediaType: MediaType, 
  sessionId: string,
  options: {
    maxPeople?: number;
    selectedModel?: ModelType;
    title?: string;
    documentType?: "pdf" | "docx" | "other";
  } = {}
) {
  const { 
    maxPeople = 5, 
    selectedModel = "openai", 
    title,
    documentType
  } = options;
  
  console.log(`Uploading ${mediaType} for analysis with model: ${selectedModel}, sessionId: ${sessionId}`);
  
  const res = await apiRequest("POST", "/api/analyze", { 
    mediaData, 
    mediaType, 
    sessionId,
    maxPeople,
    selectedModel,
    title,
    documentType
  });
  
  const data = await res.json();
  console.log("Media analysis response:", data);
  
  // Extract the analysis text into a proper message format if missing
  if (data.analysisId && (!data.messages || data.messages.length === 0)) {
    if (data.personalityInsights) {
      console.log("Creating message from personality insights");
      let analysisContent = '';
      
      // Try to extract analysis text from different possible formats
      if (typeof data.personalityInsights === 'string') {
        analysisContent = data.personalityInsights;
      } else if (data.personalityInsights.analysis) {
        analysisContent = data.personalityInsights.analysis;
      } else if (data.personalityInsights.individualProfiles && 
                data.personalityInsights.individualProfiles.length > 0) {
        // Format multiple profiles into a single analysis
        const profiles = data.personalityInsights.individualProfiles;
        analysisContent = `# Personality Analysis\n\n`;
        
        profiles.forEach((profile: any, index: number) => {
          const summary = profile.summary || '';
          analysisContent += `## Person ${index + 1}\n\n${summary}\n\n`;
          
          if (profile.detailed_analysis) {
            Object.entries(profile.detailed_analysis).forEach(([key, value]) => {
              analysisContent += `### ${key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}\n${value}\n\n`;
            });
          }
        });
      }
      
      if (analysisContent) {
        data.messages = [{
          id: Date.now(),
          analysisId: data.analysisId,
          sessionId,
          role: "assistant",
          content: analysisContent,
          createdAt: new Date().toISOString()
        }];
      }
    }
  }
  
  return data;
}

export async function sendMessage(content: string, sessionId: string, selectedModel: ModelType = "openai") {
  const res = await apiRequest("POST", "/api/chat", { 
    content, 
    sessionId,
    selectedModel 
  });
  return res.json();
}

export async function analyzeText(
  content: string, 
  sessionId: string, 
  selectedModel: ModelType = "openai",
  title?: string
) {
  console.log(`Analyzing text with model: ${selectedModel}, sessionId: ${sessionId}`);
  
  const res = await apiRequest("POST", "/api/analyze/text", { 
    content, 
    sessionId,
    selectedModel,
    title
  });
  
  const data = await res.json();
  console.log("Text analysis response:", data);
  
  // Extract the analysis text into a proper message format if missing
  if (data.analysisId && (!data.messages || data.messages.length === 0)) {
    if (data.personalityInsights) {
      console.log("Creating message from text analysis insights");
      let analysisContent = '';
      
      // Try to extract analysis text from different possible formats
      if (typeof data.personalityInsights === 'string') {
        analysisContent = data.personalityInsights;
      } else if (data.personalityInsights.analysis) {
        analysisContent = data.personalityInsights.analysis;
      }
      
      if (analysisContent) {
        data.messages = [{
          id: Date.now(),
          analysisId: data.analysisId,
          sessionId,
          role: "assistant",
          content: analysisContent,
          createdAt: new Date().toISOString()
        }];
      }
    }
  }
  
  return data;
}

export async function analyzeDocument(
  fileData: string,
  fileName: string,
  fileType: "pdf" | "docx",
  sessionId: string,
  selectedModel: ModelType = "openai",
  title?: string
) {
  console.log(`Analyzing document "${fileName}" with model: ${selectedModel}, sessionId: ${sessionId}`);
  
  const res = await apiRequest("POST", "/api/analyze/document", { 
    fileData, 
    fileName,
    fileType,
    sessionId,
    selectedModel,
    title
  });
  
  const data = await res.json();
  console.log("Document analysis response:", data);
  
  // Extract the analysis text into a proper message format if missing
  if (data.analysisId && (!data.messages || data.messages.length === 0)) {
    if (data.personalityInsights) {
      console.log("Creating message from document analysis insights");
      let analysisContent = '';
      
      // Try to extract analysis text from different possible formats
      if (typeof data.personalityInsights === 'string') {
        analysisContent = data.personalityInsights;
      } else if (data.personalityInsights.analysis) {
        analysisContent = data.personalityInsights.analysis;
      }
      
      if (analysisContent) {
        data.messages = [{
          id: Date.now(),
          analysisId: data.analysisId,
          sessionId,
          role: "assistant",
          content: analysisContent,
          createdAt: new Date().toISOString()
        }];
      }
    }
  }
  
  return data;
}

export async function shareAnalysis(analysisId: number, senderEmail: string, recipientEmail: string) {
  const res = await apiRequest("POST", "/api/share", { 
    analysisId,
    senderEmail,
    recipientEmail
  });
  return res.json();
}

export async function getSharedAnalysis(shareId: string) {
  const res = await apiRequest("GET", `/api/shared-analysis/${shareId}`, null);
  return res.json();
}

// Session management functions
export async function getAllSessions() {
  const res = await apiRequest("GET", "/api/sessions", null);
  return res.json();
}

export async function clearSession(sessionId: string) {
  const res = await apiRequest("POST", "/api/session/clear", { sessionId });
  return res.json();
}

export async function updateSessionName(sessionId: string, name: string) {
  const res = await apiRequest("PATCH", "/api/session/name", { sessionId, name });
  return res.json();
}

// Analysis functions
export async function getAllAnalysesBySession(sessionId: string) {
  const res = await apiRequest("GET", `/api/analyses?sessionId=${sessionId}`, null);
  return res.json();
}

export async function downloadAnalysis(analysisId: number, format: "pdf" | "docx" = "pdf") {
  // Direct download approach using window.open
  window.open(`/api/download/${analysisId}?format=${format}`, '_blank');
  return true;
}

export async function updateAnalysisTitle(analysisId: number, title: string) {
  const res = await apiRequest("PATCH", `/api/analysis/${analysisId}/title`, { title });
  return res.json();
}

// API status check
export async function checkAPIStatus() {
  const res = await apiRequest("GET", "/api/status", null);
  return res.json();
}