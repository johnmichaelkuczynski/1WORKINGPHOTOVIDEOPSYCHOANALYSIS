import { useState, useCallback, useRef, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useToast } from "@/hooks/use-toast";
import { nanoid } from "nanoid";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter, DialogTrigger } from "@/components/ui/dialog";
import { Form, FormField, FormItem, FormLabel, FormControl, FormMessage } from "@/components/ui/form";
import { Progress } from "@/components/ui/progress";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { uploadMedia, sendMessage, shareAnalysis, getSharedAnalysis, analyzeText, analyzeDocument, downloadAnalysis, clearSession, analyzeMBTIText, analyzeMBTIImage, analyzeMBTIVideo, analyzeMBTIDocument, analyzeBigFiveText, analyzeBigFiveImage, analyzeBigFiveVideo, analyzeEnneagramText, analyzeEnneagramImage, analyzeEnneagramVideo, analyzeDarkTraitsText, analyzeDarkTraitsImage, analyzeDarkTraitsVideo, analyzeStanfordBinetText, analyzeStanfordBinetImage, analyzeStanfordBinetVideo, analyzeVocationalText, analyzeVocationalImage, analyzeVocationalVideo, analyzePersonalityStructureText, analyzePersonalityStructureImage, analyzePersonalityStructureVideo, analyzeClinicalText, analyzeClinicalImage, analyzeClinicalVideo, analyzeAnxietyText, analyzeAnxietyImage, analyzeAnxietyVideo, analyzeEvoText, analyzeEvoImage, analyzeEvoVideo, analyzeVerticalHorizontalText, analyzeVerticalHorizontalImage, analyzeVerticalHorizontalVideo, ModelType, MediaType } from "@/lib/api";
import { Upload, Send, FileImage, Film, Share2, AlertCircle, FileText, File, Download, Copy, Check } from "lucide-react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";

// Define schemas for forms
const shareSchema = z.object({
  senderEmail: z.string().email("Please enter a valid email"),
  recipientEmail: z.string().email("Please enter a valid email"),
});

// Helper function to resize images
async function resizeImage(file: File, maxWidth: number): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (readerEvent) => {
      const img = new Image();
      img.onload = () => {
        // Calculate new dimensions while maintaining aspect ratio
        let width = img.width;
        let height = img.height;
        
        if (width > maxWidth) {
          height = Math.round((height * maxWidth) / width);
          width = maxWidth;
        }
        
        // Create canvas for resizing
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        
        // Draw and resize image on canvas
        const ctx = canvas.getContext('2d');
        if (!ctx) {
          reject(new Error('Could not get canvas context'));
          return;
        }
        
        ctx.drawImage(img, 0, 0, width, height);
        
        // Convert canvas to data URL
        try {
          const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
          resolve(dataUrl);
        } catch (e) {
          reject(e);
        }
      };
      
      img.onerror = () => {
        reject(new Error('Failed to load image'));
      };
      
      if (typeof readerEvent.target?.result === 'string') {
        img.src = readerEvent.target.result;
      } else {
        reject(new Error('Failed to read file'));
      }
    };
    
    reader.onerror = () => {
      reject(new Error('Failed to read file'));
    };
    
    reader.readAsDataURL(file);
  });
}

export default function Home({ isShareMode = false, shareId }: { isShareMode?: boolean, shareId?: string }) {
  const { toast } = useToast();
  // Persist sessionId to localStorage so it survives page reloads
  const [sessionId] = useState(() => {
    const stored = localStorage.getItem('personality-analysis-session-id');
    if (stored) {
      return stored;
    }
    const newId = nanoid();
    localStorage.setItem('personality-analysis-session-id', newId);
    return newId;
  });
  const [messages, setMessages] = useState<any[]>([]);
  const [input, setInput] = useState("");
  const [textInput, setTextInput] = useState("");
  const queryClient = useQueryClient();
  
  // Media states
  const [uploadedMedia, setUploadedMedia] = useState<string | null>(null);
  const [mediaType, setMediaType] = useState<MediaType>("image");
  const [mediaData, setMediaData] = useState<string | null>(null); // Store media data for re-analysis
  const [analysisId, setAnalysisId] = useState<number | null>(null);
  const [isShareDialogOpen, setIsShareDialogOpen] = useState(false);
  const [emailServiceAvailable, setEmailServiceAvailable] = useState(false);
  const [copied, setCopied] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedModel, setSelectedModel] = useState<ModelType>("grok");
  const [documentName, setDocumentName] = useState<string>("");
  const [selectedAnalysisType, setSelectedAnalysisType] = useState<string | null>(null);
  
  // References
  const videoRef = useRef<HTMLVideoElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const documentInputRef = useRef<HTMLInputElement>(null);
  const documentMBTIInputRef = useRef<HTMLInputElement>(null);
  const imageMBTIInputRef = useRef<HTMLInputElement>(null);
  const videoMBTIInputRef = useRef<HTMLInputElement>(null);
  const bigFiveImageInputRef = useRef<HTMLInputElement>(null);
  const bigFiveVideoInputRef = useRef<HTMLInputElement>(null);
  const enneagramTextInputRef = useRef<HTMLInputElement>(null);
  const enneagramImageInputRef = useRef<HTMLInputElement>(null);
  const enneagramVideoInputRef = useRef<HTMLInputElement>(null);
  const darkTraitsImageInputRef = useRef<HTMLInputElement>(null);
  const darkTraitsVideoInputRef = useRef<HTMLInputElement>(null);
  const stanfordBinetImageInputRef = useRef<HTMLInputElement>(null);
  const stanfordBinetVideoInputRef = useRef<HTMLInputElement>(null);
  const vocationalImageInputRef = useRef<HTMLInputElement>(null);
  const vocationalVideoInputRef = useRef<HTMLInputElement>(null);
  const personalityStructureImageInputRef = useRef<HTMLInputElement>(null);
  const personalityStructureVideoInputRef = useRef<HTMLInputElement>(null);
  const clinicalImageInputRef = useRef<HTMLInputElement>(null);
  const clinicalVideoInputRef = useRef<HTMLInputElement>(null);
  const anxietyImageInputRef = useRef<HTMLInputElement>(null);
  const anxietyVideoInputRef = useRef<HTMLInputElement>(null);
  const evoImageInputRef = useRef<HTMLInputElement>(null);
  const evoVideoInputRef = useRef<HTMLInputElement>(null);
  const vhImageInputRef = useRef<HTMLInputElement>(null);
  const vhVideoInputRef = useRef<HTMLInputElement>(null);

  // Check API status on component mount
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await fetch('/api/status');
        const status = await res.json();
        setEmailServiceAvailable(status.sendgrid || false);
      } catch (error) {
        console.error("Error checking API status:", error);
      }
    };
    
    checkStatus();
  }, []);

  // Load shared analysis when shareId is provided
  useEffect(() => {
    if (shareId) {
      // Fetch and display the shared analysis
      getSharedAnalysis(shareId)
        .then(data => {
          if (data.analysis && data.messages) {
            // Set the analysis data
            setAnalysisId(data.analysis.id);
            
            // Set uploaded media preview if available
            if (data.analysis.mediaUrl) {
              setUploadedMedia(data.analysis.mediaUrl);
              setMediaType(data.analysis.mediaType as MediaType);
            }
            
            // Set messages
            setMessages(prev => [...prev, ...data.messages]);
            
            // Set email service status
            setEmailServiceAvailable(data.emailServiceAvailable);
            
            toast({
              title: "Shared Analysis Loaded",
              description: "Viewing a shared personality analysis."
            });
          }
        })
        .catch(error => {
          console.error("Error loading shared analysis:", error);
          toast({
            variant: "destructive",
            title: "Error",
            description: "Failed to load shared analysis. It may have expired or been removed."
          });
        });
    }
  }, [shareId, toast]);
  
  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);
  
  // Text analysis
  const handleTextAnalysis = useMutation({
    mutationFn: async (text: string) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        const response = await analyzeText(text, sessionId, selectedModel);
        
        setAnalysisProgress(80);
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Text analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze text. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: (data) => {
      // Get all messages for the session to be sure we have the latest
      if (data?.analysisId) {
        // If we received an analysis ID, fetch any messages related to it
        fetch(`/api/messages?sessionId=${sessionId}`)
          .then(res => res.json())
          .then(data => {
            if (data && Array.isArray(data) && data.length > 0) {
              console.log("Fetched messages after text analysis:", data);
              setMessages(data);
            }
          })
          .catch(err => console.error("Error fetching messages after text analysis:", err));
      }
      
      toast({
        title: "Analysis Complete",
        description: "Your text has been successfully analyzed.",
      });
      setTextInput("");
    }
  });

  // Document analysis with file upload
  const handleDocumentAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        setDocumentName(file.name);
        setAnalysisProgress(30);
        
        // Read the file as data URL
        const reader = new FileReader();
        const fileData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setAnalysisProgress(50);
        
        // Determine file type
        const fileExt = file.name.split('.').pop()?.toLowerCase();
        const fileType = fileExt === 'pdf' ? 'pdf' : 'docx';
        
        const response = await analyzeDocument(
          fileData,
          file.name,
          fileType,
          sessionId,
          selectedModel
        );
        
        setAnalysisProgress(80);
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Document analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze document. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: (data) => {
      // Get all messages for the session to be sure we have the latest
      if (data?.analysisId) {
        // If we received an analysis ID, try to fetch any messages related to it
        fetch(`/api/messages?sessionId=${sessionId}`)
          .then(res => res.json())
          .then(data => {
            if (data && Array.isArray(data) && data.length > 0) {
              console.log("Fetched messages after document analysis:", data);
              setMessages(data);
            }
          })
          .catch(err => console.error("Error fetching messages after document analysis:", err));
      }
      
      toast({
        title: "Analysis Complete",
        description: "Your document has been successfully analyzed.",
      });
    }
  });

  // Document MBTI analysis with file upload
  const handleDocumentMBTIAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        setDocumentName(file.name);
        setAnalysisProgress(30);
        
        // Read the file as data URL
        const reader = new FileReader();
        const fileData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setAnalysisProgress(50);
        
        // Determine file type
        const fileExt = file.name.split('.').pop()?.toLowerCase();
        const fileType = fileExt === 'pdf' ? 'pdf' : 'docx';
        
        const response = await analyzeMBTIDocument(
          fileData,
          file.name,
          fileType,
          sessionId,
          selectedModel
        );
        
        setAnalysisProgress(80);
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Document MBTI analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze document for MBTI. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: (data) => {
      // Get all messages for the session to be sure we have the latest
      if (data?.analysisId) {
        // If we received an analysis ID, try to fetch any messages related to it
        fetch(`/api/messages?sessionId=${sessionId}`)
          .then(res => res.json())
          .then(data => {
            if (data && Array.isArray(data) && data.length > 0) {
              console.log("Fetched messages after document MBTI analysis:", data);
              setMessages(data);
            }
          })
          .catch(err => console.error("Error fetching messages after document MBTI analysis:", err));
      }
      
      toast({
        title: "MBTI Analysis Complete",
        description: "Your document has been successfully analyzed for MBTI personality type.",
      });
    }
  });

  // Image MBTI analysis with file upload
  const handleImageMBTIAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        setAnalysisProgress(30);
        
        // Resize image if needed
        let mediaData: string;
        if (file.size > 4 * 1024 * 1024) {
          mediaData = await resizeImage(file, 1600);
        } else {
          const reader = new FileReader();
          mediaData = await new Promise<string>((resolve) => {
            reader.onload = (e) => resolve(e.target?.result as string);
            reader.readAsDataURL(file);
          });
        }
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("image");
        setAnalysisProgress(50);
        
        const response = await analyzeMBTIImage(
          mediaData,
          sessionId,
          selectedModel
        );
        
        setAnalysisProgress(80);
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Image MBTI analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze image for MBTI. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "MBTI Analysis Complete",
        description: "Your image has been successfully analyzed for MBTI personality type.",
      });
    }
  });

  // Video MBTI analysis with file upload
  const handleVideoMBTIAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        setAnalysisProgress(30);
        
        // Read video file as data URL
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("video");
        setAnalysisProgress(50);
        
        const response = await analyzeMBTIVideo(
          mediaData,
          sessionId,
          selectedModel
        );
        
        setAnalysisProgress(80);
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Video MBTI analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze video for MBTI. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "MBTI Analysis Complete",
        description: "Your video has been successfully analyzed for MBTI personality type.",
      });
    }
  });

  // Big Five (OCEAN) text analysis
  const handleBigFiveTextAnalysis = useMutation({
    mutationFn: async (text: string) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        if (!text.trim()) {
          throw new Error("Please provide text to analyze");
        }
        
        setAnalysisProgress(30);
        
        const response = await analyzeBigFiveText(
          text,
          sessionId,
          selectedModel,
          `Big Five Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Big Five text analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze text for Big Five. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Big Five Analysis Complete",
        description: "Your text has been successfully analyzed using the Five-Factor Model.",
      });
      setTextInput("");
    }
  });

  // Big Five (OCEAN) image analysis
  const handleBigFiveImageAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        // Read the image file
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("image");
        setAnalysisProgress(30);
        
        const response = await analyzeBigFiveImage(
          mediaData,
          sessionId,
          selectedModel,
          `Big Five Image Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Big Five image analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze image for Big Five. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Big Five Analysis Complete",
        description: "Your image has been successfully analyzed using the Five-Factor Model.",
      });
    }
  });

  // Big Five (OCEAN) video analysis
  const handleBigFiveVideoAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        // Read the video file
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("video");
        setAnalysisProgress(30);
        
        const response = await analyzeBigFiveVideo(
          mediaData,
          sessionId,
          selectedModel,
          `Big Five Video Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Big Five video analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze video for Big Five. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Big Five Analysis Complete",
        description: "Your video has been successfully analyzed using the Five-Factor Model.",
      });
    }
  });

  // Enneagram text analysis
  const handleEnneagramTextAnalysis = useMutation({
    mutationFn: async (text: string) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        if (!text.trim()) {
          throw new Error("Please provide text to analyze");
        }
        
        setAnalysisProgress(30);
        
        const response = await analyzeEnneagramText(
          text,
          sessionId,
          selectedModel,
          `Enneagram Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Enneagram text analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze text for Enneagram. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Enneagram Analysis Complete",
        description: "Your text has been successfully analyzed for Enneagram personality type.",
      });
    }
  });

  // Enneagram image analysis
  const handleEnneagramImageAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        // Read the image file
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("image");
        setAnalysisProgress(30);
        
        const response = await analyzeEnneagramImage(
          mediaData,
          sessionId,
          selectedModel,
          `Enneagram Image Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Enneagram image analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze image for Enneagram. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Enneagram Analysis Complete",
        description: "Your image has been successfully analyzed for Enneagram personality type.",
      });
    }
  });

  // Enneagram video analysis
  const handleEnneagramVideoAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        // Read the video file
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("video");
        setAnalysisProgress(30);
        
        const response = await analyzeEnneagramVideo(
          mediaData,
          sessionId,
          selectedModel,
          `Enneagram Video Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Enneagram video analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze video for Enneagram. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Enneagram Analysis Complete",
        description: "Your video has been successfully analyzed for Enneagram personality type.",
      });
    }
  });

  // Dark Traits text analysis
  const handleDarkTraitsTextAnalysis = useMutation({
    mutationFn: async (text: string) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        if (!text.trim()) {
          throw new Error("Please provide text to analyze");
        }
        
        setAnalysisProgress(30);
        
        const response = await analyzeDarkTraitsText(
          text,
          sessionId,
          selectedModel,
          `Dark Traits Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Dark traits text analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze text for dark traits. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Dark Traits Analysis Complete",
        description: "Your text has been successfully analyzed for personality pathology and dark traits.",
      });
    }
  });

  // Dark Traits image analysis
  const handleDarkTraitsImageAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        // Read the image file
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("image");
        setAnalysisProgress(30);
        
        const response = await analyzeDarkTraitsImage(
          mediaData,
          sessionId,
          selectedModel,
          `Dark Traits Image Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Dark traits image analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze image for dark traits. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Dark Traits Analysis Complete",
        description: "Your image has been successfully analyzed for personality pathology and dark traits.",
      });
    }
  });

  // Dark Traits video analysis
  const handleDarkTraitsVideoAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        // Read the video file
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("video");
        setAnalysisProgress(30);
        
        const response = await analyzeDarkTraitsVideo(
          mediaData,
          sessionId,
          selectedModel,
          `Dark Traits Video Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Dark traits video analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze video for dark traits. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Dark Traits Analysis Complete",
        description: "Your video has been successfully analyzed for personality pathology and dark traits.",
      });
    }
  });

  // Stanford-Binet Intelligence Scale text analysis
  const handleStanfordBinetTextAnalysis = useMutation({
    mutationFn: async (text: string) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        if (!text.trim()) {
          throw new Error("Please provide text to analyze");
        }
        
        setAnalysisProgress(30);
        
        const response = await analyzeStanfordBinetText(
          text,
          sessionId,
          selectedModel,
          `Stanford-Binet Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Stanford-Binet text analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze text for Stanford-Binet. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Stanford-Binet Analysis Complete",
        description: "Your text has been successfully analyzed using the Stanford-Binet Intelligence Scale.",
      });
      setTextInput("");
    }
  });

  // Vocational / Motivation / Values text analysis
  const handleVocationalTextAnalysis = useMutation({
    mutationFn: async (text: string) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        if (!text.trim()) {
          throw new Error("Please provide text to analyze");
        }
        
        setAnalysisProgress(30);
        
        const response = await analyzeVocationalText(
          text,
          sessionId,
          selectedModel,
          `Vocational / Motivation / Values Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Vocational text analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze text for Vocational/Motivation/Values. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Vocational Analysis Complete",
        description: "Your text has been successfully analyzed for career interests, work values, and motivational drivers.",
      });
      setTextInput("");
    }
  });

  // Vocational / Motivation / Values image analysis
  const handleVocationalImageAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        // Read the image file
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("image");
        setAnalysisProgress(30);
        
        const response = await analyzeVocationalImage(
          mediaData,
          sessionId,
          selectedModel,
          `Vocational / Motivation / Values Image Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Vocational image analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze image for Vocational/Motivation/Values. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Vocational Analysis Complete",
        description: "Your image has been successfully analyzed for career interests, work values, and motivational drivers.",
      });
    }
  });

  // Vocational / Motivation / Values video analysis
  const handleVocationalVideoAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        // Read the video file
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("video");
        setAnalysisProgress(30);
        
        const response = await analyzeVocationalVideo(
          mediaData,
          sessionId,
          selectedModel,
          `Vocational / Motivation / Values Video Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Vocational video analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze video for Vocational/Motivation/Values. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Vocational Analysis Complete",
        description: "Your video has been successfully analyzed for career interests, work values, and motivational drivers.",
      });
    }
  });

  // Personality Structure image analysis
  const handlePersonalityStructureImageAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        // Read the image file
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("image");
        setAnalysisProgress(30);
        
        const response = await analyzePersonalityStructureImage(
          mediaData,
          sessionId,
          selectedModel,
          `Personality Structure Image Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Personality Structure image analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze image for consolidated personality structure. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Personality Structure Analysis Complete",
        description: "Your image has been analyzed across 8 major personality frameworks.",
      });
    }
  });

  // Personality Structure video analysis
  const handlePersonalityStructureVideoAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        // Read the video file
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("video");
        setAnalysisProgress(30);
        
        const response = await analyzePersonalityStructureVideo(
          mediaData,
          sessionId,
          selectedModel,
          `Personality Structure Video Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Personality Structure video analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze video for consolidated personality structure. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Personality Structure Analysis Complete",
        description: "Your video has been analyzed across 8 major personality frameworks.",
      });
    }
  });

  // Clinical image analysis
  const handleClinicalImageAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        // Read the image file
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("image");
        setAnalysisProgress(30);
        
        const response = await analyzeClinicalImage(
          mediaData,
          sessionId,
          selectedModel,
          `Clinical Psychopathology Image Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Clinical image analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze image for clinical/psychopathology assessment. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Clinical Analysis Complete",
        description: "Your image has been analyzed across 4 major clinical frameworks (MMPI, MCMI, DSM-5 SCID, PID-5).",
      });
    }
  });

  // Clinical video analysis
  const handleClinicalVideoAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        // Read the video file
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("video");
        setAnalysisProgress(30);
        
        const response = await analyzeClinicalVideo(
          mediaData,
          sessionId,
          selectedModel,
          `Clinical Psychopathology Video Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Clinical video analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze video for clinical/psychopathology assessment. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Clinical Analysis Complete",
        description: "Your video has been analyzed across 4 major clinical frameworks (MMPI, MCMI, DSM-5 SCID, PID-5).",
      });
    }
  });

  // Anxiety image analysis
  const handleAnxietyImageAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        
        // Read the image file
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("image");
        setAnalysisProgress(30);
        
        const response = await analyzeAnxietyImage(
          mediaData,
          sessionId,
          selectedModel,
          `Anxiety/Affective Image Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Anxiety image analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze image for anxiety/affective assessment. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Anxiety Analysis Complete",
        description: "Your image has been analyzed across 5 affective/anxiety scales (BDI, HDRS, BAI, GAD-7, PHQ-9).",
      });
    }
  });

  // Anxiety video analysis
  const handleAnxietyVideoAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        // Read the video file
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("video");
        setAnalysisProgress(30);
        
        const response = await analyzeAnxietyVideo(
          mediaData,
          sessionId,
          selectedModel,
          `Anxiety/Affective Video Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Anxiety video analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze video for anxiety/affective assessment. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Anxiety Analysis Complete",
        description: "Your video has been analyzed across 5 affective/anxiety scales (BDI, HDRS, BAI, GAD-7, PHQ-9).",
      });
    }
  });

  // EVO Psych (Evolutionary Psychology) image analysis
  const handleEvoImageAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        
        // Read the image file
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("image");
        setAnalysisProgress(30);
        
        const response = await analyzeEvoImage(
          mediaData,
          sessionId,
          selectedModel,
          `EVO Psych Image Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('EVO Psych image analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze image for EVO Psych assessment. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "EVO Psych Analysis Complete",
        description: "Your image has been analyzed across the 10-pole evolutionary psychology framework.",
      });
    }
  });

  // EVO Psych (Evolutionary Psychology) video analysis
  const handleEvoVideoAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        
        // Read the video file
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("video");
        setAnalysisProgress(30);
        
        const response = await analyzeEvoVideo(
          mediaData,
          sessionId,
          selectedModel,
          `EVO Psych Video Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('EVO Psych video analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze video for EVO Psych assessment. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "EVO Psych Analysis Complete",
        description: "Your video has been analyzed across the 10-pole evolutionary psychology framework with temporal dynamics.",
      });
    }
  });

  // V/H (Vertical/Horizontal) Orientation image analysis
  const handleVHImageAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        
        // Read the image file
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("image");
        setAnalysisProgress(30);
        
        const response = await analyzeVerticalHorizontalImage(
          mediaData,
          sessionId,
          selectedModel,
          `V/H Image Orientation - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('V/H Image orientation analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze image for V/H orientation. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "V/H Orientation Complete",
        description: "Your image has been analyzed for Vertical vs. Horizontal visual orientation.",
      });
    }
  });

  // V/H (Vertical/Horizontal) Orientation video analysis (Full Version - Speech Dominant)
  const handleVHVideoAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        
        // Read the video file
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("video");
        setAnalysisProgress(30);
        
        const response = await analyzeVerticalHorizontalVideo(
          mediaData,
          sessionId,
          selectedModel,
          `V/H Video Orientation - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('V/H Video orientation analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze video for V/H orientation. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "V/H Video Orientation Complete",
        description: "Your video has been analyzed for Vertical vs. Horizontal orientation (speech + visual + prosody).",
      });
    }
  });

  // Stanford-Binet Intelligence Scale image analysis
  const handleStanfordBinetImageAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        // Read the image file
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("image");
        setAnalysisProgress(30);
        
        const response = await analyzeStanfordBinetImage(
          mediaData,
          sessionId,
          selectedModel,
          `Stanford-Binet Image Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Stanford-Binet image analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze image for Stanford-Binet. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Stanford-Binet Analysis Complete",
        description: "Your image has been successfully analyzed using the Stanford-Binet Intelligence Scale.",
      });
    }
  });

  // Stanford-Binet Intelligence Scale video analysis
  const handleStanfordBinetVideoAnalysis = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(10);
        // Messages now append instead of clear
        
        // Read the video file
        const reader = new FileReader();
        const mediaData = await new Promise<string>((resolve) => {
          reader.onload = (e) => resolve(e.target?.result as string);
          reader.readAsDataURL(file);
        });
        
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setMediaType("video");
        setAnalysisProgress(30);
        
        const response = await analyzeStanfordBinetVideo(
          mediaData,
          sessionId,
          selectedModel,
          `Stanford-Binet Video Analysis - ${new Date().toLocaleDateString()}`
        );
        
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(prev => [...prev, ...response.messages]);
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Stanford-Binet video analysis error:', error);
        toast({
          title: "Analysis Failed",
          description: error.message || "Failed to analyze video for Stanford-Binet. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: () => {
      toast({
        title: "Stanford-Binet Analysis Complete",
        description: "Your video has been successfully analyzed using the Stanford-Binet Intelligence Scale.",
      });
    }
  });

  // Media upload and analysis
  const handleUploadMedia = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(0);
        // Messages now append instead of clear
        
        // Determine media type and set it
        const fileType = file.type.split('/')[0];
        const isVideo = fileType === 'video';
        const mediaFileType: MediaType = isVideo ? "video" : "image";
        setMediaType(mediaFileType);
        
        // Update progress
        setAnalysisProgress(20);
        
        // For images, resize if needed to meet AWS limits
        let mediaData: string;
        if (mediaFileType === "image" && file.size > 4 * 1024 * 1024) {
          mediaData = await resizeImage(file, 1600);
        } else {
          // For videos or smaller images, read as data URL
          const reader = new FileReader();
          mediaData = await new Promise<string>((resolve) => {
            reader.onload = (e) => resolve(e.target?.result as string);
            reader.readAsDataURL(file);
          });
        }
        
        // Set preview and store media data for re-analysis
        setUploadedMedia(mediaData);
        setMediaData(mediaData);
        setAnalysisProgress(50);
        
        // Maximum 5 people to analyze
        const maxPeople = 5;
        
        // Upload for analysis
        const response = await uploadMedia(
          mediaData, 
          mediaFileType, 
          sessionId, 
          { 
            selectedModel, 
            maxPeople 
          }
        );
        
        setAnalysisProgress(90);
        
        if (response && response.analysisId) {
          setAnalysisId(response.analysisId);
        }
        
        console.log("Response from uploadMedia:", response);
        
        // Make sure we update the messages state with the response
        if (response && response.messages && Array.isArray(response.messages) && response.messages.length > 0) {
          console.log("Setting messages from response:", response.messages);
          setMessages(prev => [...prev, ...response.messages]);
        } else {
          // If no messages were returned, let's add a default message
          console.warn("No messages returned from analysis");
          if (response?.analysisInsights) {
            setMessages([{
              role: "assistant",
              content: response.analysisInsights,
              id: Date.now(),
              createdAt: new Date().toISOString(),
              sessionId,
              analysisId: response.analysisId
            }]);
          }
        }
        
        setAnalysisProgress(100);
        return response;
      } catch (error: any) {
        console.error('Upload error:', error);
        toast({
          title: "Upload Failed",
          description: error.message || "Failed to upload media. Please try again.",
          variant: "destructive",
        });
        setAnalysisProgress(0);
        throw error;
      } finally {
        setIsAnalyzing(false);
      }
    },
    onSuccess: (data) => {
      // Get all messages for the session to be sure we have the latest
      if (data?.analysisId) {
        // If we received an analysis ID, try to fetch any messages related to it
        fetch(`/api/messages?sessionId=${sessionId}`)
          .then(res => res.json())
          .then(data => {
            if (data && Array.isArray(data) && data.length > 0) {
              console.log("Fetched messages after analysis:", data);
              setMessages(data);
            }
          })
          .catch(err => console.error("Error fetching messages:", err));
      }
      
      toast({
        title: "Analysis Complete",
        description: "Your media has been successfully analyzed.",
      });
    },
    onError: () => {
      setUploadedMedia(null);
    }
  });

  // Chat with AI
  const chatMutation = useMutation({
    mutationFn: async (content: string) => {
      return sendMessage(content, sessionId, selectedModel);
    },
    onSuccess: (data) => {
      if (data && data.messages && Array.isArray(data.messages)) {
        // Add the new messages
        setMessages((prev) => [...prev, ...data.messages]);
        queryClient.invalidateQueries({ queryKey: ["/api/chat"] });
      }
    },
    onError: (error: any) => {
      console.error("Chat error:", error);
      toast({
        variant: "destructive",
        title: "Error",
        description: error.message || "Failed to send message.",
      });
    },
  });

  // Email sharing
  const shareForm = useForm<z.infer<typeof shareSchema>>({
    resolver: zodResolver(shareSchema),
  });
  
  const shareMutation = useMutation({
    mutationFn: async (data: z.infer<typeof shareSchema>) => {
      if (!analysisId) throw new Error("No analysis to share");
      return shareAnalysis(analysisId, data.senderEmail, data.recipientEmail);
    },
    onSuccess: () => {
      toast({
        title: "Success",
        description: "Analysis shared successfully!",
      });
      setIsShareDialogOpen(false);
    },
    onError: () => {
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to share analysis. Please try again.",
      });
    },
  });

  // Handle file upload
  const handleFileUpload = (file: File) => {
    const fileType = file.type.split('/')[0];
    
    if (fileType === 'image' || fileType === 'video') {
      // Check if a specific analysis type is selected
      if (selectedAnalysisType === 'bigfive-image' && fileType === 'image') {
        handleBigFiveImageAnalysis.mutate(file);
      } else if (selectedAnalysisType === 'bigfive-video' && fileType === 'video') {
        handleBigFiveVideoAnalysis.mutate(file);
      } else if (selectedAnalysisType === 'image-mbti' && fileType === 'image') {
        handleImageMBTIAnalysis.mutate(file);
      } else if (selectedAnalysisType === 'video-mbti' && fileType === 'video') {
        handleVideoMBTIAnalysis.mutate(file);
      } else {
        handleUploadMedia.mutate(file);
      }
    } else if (
      file.type === 'application/pdf' || 
      file.type === 'application/msword' || 
      file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' ||
      file.type === 'text/plain'
    ) {
      handleDocumentAnalysis.mutate(file);
    } else {
      toast({
        variant: "destructive",
        title: "Unsupported File Type",
        description: "Please upload an image, video, PDF, DOC, DOCX, or TXT file."
      });
    }
  };

  // Handle text analysis submission
  const handleTextSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (textInput.trim()) {
      handleTextAnalysis.mutate(textInput);
    }
  };
  
  // Handle chat message submission
  const handleChatSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      // Add user message immediately to UI
      setMessages(prev => [...prev, { role: 'user', content: input }]);
      chatMutation.mutate(input);
      setInput("");
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent, submitHandler: (e: React.FormEvent) => void) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitHandler(e as unknown as React.FormEvent);
    }
  };
  
  const onShareSubmit = (data: z.infer<typeof shareSchema>) => {
    shareMutation.mutate(data);
  };
  
  // Generic dropzone for all file types
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      handleFileUpload(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    noClick: true,
    noKeyboard: true,
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024, // 50MB limit
  });

  // Click handlers for different upload types
  const handleImageVideoClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };
  
  const handleDocumentClick = () => {
    if (documentInputRef.current) {
      documentInputRef.current.click();
    }
  };
  
  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>, type: 'media' | 'document') => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (type === 'media') {
        const fileType = file.type.split('/')[0];
        if (fileType === 'image' || fileType === 'video') {
          // Check if a specific analysis type is selected
          if (selectedAnalysisType === 'bigfive-image' && fileType === 'image') {
            handleBigFiveImageAnalysis.mutate(file);
          } else if (selectedAnalysisType === 'bigfive-video' && fileType === 'video') {
            handleBigFiveVideoAnalysis.mutate(file);
          } else if (selectedAnalysisType === 'image-mbti' && fileType === 'image') {
            handleImageMBTIAnalysis.mutate(file);
          } else if (selectedAnalysisType === 'video-mbti' && fileType === 'video') {
            handleVideoMBTIAnalysis.mutate(file);
          } else {
            handleUploadMedia.mutate(file);
          }
        } else {
          toast({
            variant: "destructive",
            title: "Unsupported File Type",
            description: "Please upload an image or video file."
          });
        }
      } else {
        handleDocumentAnalysis.mutate(file);
      }
    }
  };

  return (
    <div className="flex" {...getRootProps()}>
      {/* Left Sidebar - Additional Assessments */}
      <div className="w-56 bg-muted/30 min-h-screen p-4 space-y-2 border-r">
        <h3 className="text-sm font-semibold mb-3 text-muted-foreground">Additional Assessments</h3>
        
        {/* Personality Structure */}
        <Button
          variant={selectedAnalysisType === "personality-structure-text" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-primary to-purple-600 text-white hover:from-primary/90 hover:to-purple-600/90 border-none"
          onClick={async () => {
            setSelectedAnalysisType("personality-structure-text");
            
            if (!textInput.trim()) {
              toast({
                variant: "destructive",
                title: "No Text",
                description: "Please enter text in the Input Preview section below",
              });
              return;
            }
            
            setIsAnalyzing(true);
            setAnalysisProgress(0);
            
            try {
              const data = await analyzePersonalityStructureText(textInput, sessionId, selectedModel);
              
              if (data.messages && data.messages.length > 0) {
                setMessages(prev => [...prev, ...data.messages]);
                setAnalysisId(data.analysisId);
                setAnalysisProgress(100);
                toast({
                  title: "Personality Structure Analysis Complete",
                  description: "Your text has been analyzed across 8 major personality frameworks",
                });
              }
            } catch (error) {
              console.error("Personality Structure text analysis error:", error);
              toast({
                variant: "destructive",
                title: "Analysis Failed",
                description: "Failed to analyze text for consolidated personality structure. Please try again.",
              });
            } finally {
              setIsAnalyzing(false);
            }
          }}
          disabled={isAnalyzing}
          data-testid="button-personality-structure-text"
        >
          🌟 Personality (Text)
        </Button>
        
        <Button
          variant={selectedAnalysisType === "personality-structure-image" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-primary to-purple-600 text-white hover:from-primary/90 hover:to-purple-600/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("personality-structure-image");
            personalityStructureImageInputRef.current?.click();
          }}
          disabled={isAnalyzing}
          data-testid="button-personality-structure-image"
        >
          🌟 Personality (Image)
          <input
            ref={personalityStructureImageInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) {
                handlePersonalityStructureImageAnalysis.mutate(files[0]);
              }
            }}
          />
        </Button>
        
        <Button
          variant={selectedAnalysisType === "personality-structure-video" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-primary to-purple-600 text-white hover:from-primary/90 hover:to-purple-600/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("personality-structure-video");
            personalityStructureVideoInputRef.current?.click();
          }}
          disabled={isAnalyzing}
          data-testid="button-personality-structure-video"
        >
          🌟 Personality (Video)
          <input
            ref={personalityStructureVideoInputRef}
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) {
                handlePersonalityStructureVideoAnalysis.mutate(files[0]);
              }
            }}
          />
        </Button>
        
        <Button
          variant={selectedAnalysisType === "clinical-text" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-red-600 to-orange-600 text-white hover:from-red-600/90 hover:to-orange-600/90 border-none"
          onClick={async () => {
            setSelectedAnalysisType("clinical-text");
            
            if (!textInput.trim()) {
              toast({
                variant: "destructive",
                title: "No Text",
                description: "Please enter text in the Input Preview section below",
              });
              return;
            }
            
            setIsAnalyzing(true);
            setAnalysisProgress(10);
            // Messages now append instead of clear
            
            try {
              const data = await analyzeClinicalText(textInput, sessionId, selectedModel);
              
              if (data.messages && data.messages.length > 0) {
                setMessages(prev => [...prev, ...data.messages]);
                setAnalysisId(data.analysisId);
                setAnalysisProgress(100);
                toast({
                  title: "Clinical Analysis Complete",
                  description: "Your text has been analyzed across 4 major clinical frameworks (MMPI, MCMI, DSM-5 SCID, PID-5)",
                });
              }
            } catch (error) {
              console.error("Clinical text analysis error:", error);
              toast({
                variant: "destructive",
                title: "Analysis Failed",
                description: "Failed to analyze text for clinical/psychopathology assessment. Please try again.",
              });
            } finally {
              setIsAnalyzing(false);
            }
          }}
          disabled={isAnalyzing}
          data-testid="button-clinical-text"
        >
          🏥 Clinical (Text)
        </Button>
        
        <Button
          variant={selectedAnalysisType === "clinical-image" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-red-600 to-orange-600 text-white hover:from-red-600/90 hover:to-orange-600/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("clinical-image");
            clinicalImageInputRef.current?.click();
          }}
          disabled={isAnalyzing}
          data-testid="button-clinical-image"
        >
          🏥 Clinical (Image)
          <input
            ref={clinicalImageInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) {
                handleClinicalImageAnalysis.mutate(files[0]);
              }
            }}
          />
        </Button>
        
        <Button
          variant={selectedAnalysisType === "clinical-video" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-red-600 to-orange-600 text-white hover:from-red-600/90 hover:to-orange-600/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("clinical-video");
            clinicalVideoInputRef.current?.click();
          }}
          disabled={isAnalyzing}
          data-testid="button-clinical-video"
        >
          🏥 Clinical (Video)
          <input
            ref={clinicalVideoInputRef}
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) {
                handleClinicalVideoAnalysis.mutate(files[0]);
              }
            }}
          />
        </Button>
        
        <Button
          variant={selectedAnalysisType === "anxiety-text" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-teal-600 to-cyan-600 text-white hover:from-teal-600/90 hover:to-cyan-600/90 border-none"
          onClick={async () => {
            setSelectedAnalysisType("anxiety-text");
            
            if (!textInput.trim()) {
              toast({
                variant: "destructive",
                title: "No Text",
                description: "Please enter text in the Input Preview section below",
              });
              return;
            }
            
            setIsAnalyzing(true);
            setAnalysisProgress(10);
            
            try {
              const data = await analyzeAnxietyText(textInput, sessionId, selectedModel);
              
              if (data.messages && data.messages.length > 0) {
                setMessages(prev => [...prev, ...data.messages]);
                setAnalysisId(data.analysisId);
                setAnalysisProgress(100);
                toast({
                  title: "Anxiety Analysis Complete",
                  description: "Your text has been analyzed across 5 affective/anxiety scales (BDI, Hamilton Depression, BAI, GAD-7, PHQ-9)",
                });
              }
            } catch (error) {
              console.error("Anxiety text analysis error:", error);
              toast({
                variant: "destructive",
                title: "Analysis Failed",
                description: "Failed to analyze text for anxiety/affective assessment. Please try again.",
              });
            } finally {
              setIsAnalyzing(false);
            }
          }}
          disabled={isAnalyzing}
          data-testid="button-anxiety-text"
        >
          💙 Anxiety (Text)
        </Button>
        
        <Button
          variant={selectedAnalysisType === "anxiety-image" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-teal-600 to-cyan-600 text-white hover:from-teal-600/90 hover:to-cyan-600/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("anxiety-image");
            anxietyImageInputRef.current?.click();
          }}
          disabled={isAnalyzing}
          data-testid="button-anxiety-image"
        >
          💙 Anxiety (Image)
          <input
            ref={anxietyImageInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) {
                handleAnxietyImageAnalysis.mutate(files[0]);
              }
            }}
          />
        </Button>
        
        <Button
          variant={selectedAnalysisType === "anxiety-video" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-teal-600 to-cyan-600 text-white hover:from-teal-600/90 hover:to-cyan-600/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("anxiety-video");
            anxietyVideoInputRef.current?.click();
          }}
          disabled={isAnalyzing}
          data-testid="button-anxiety-video"
        >
          💙 Anxiety (Video)
          <input
            ref={anxietyVideoInputRef}
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) {
                handleAnxietyVideoAnalysis.mutate(files[0]);
              }
            }}
          />
        </Button>
        
        <Button
          variant={selectedAnalysisType === "evo-text" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white hover:from-green-600/90 hover:to-emerald-600/90 border-none"
          onClick={async () => {
            setSelectedAnalysisType("evo-text");
            
            if (!textInput.trim()) {
              toast({
                variant: "destructive",
                title: "No Text",
                description: "Please enter text in the Input Preview section below",
              });
              return;
            }
            
            setIsAnalyzing(true);
            setAnalysisProgress(10);
            
            try {
              const data = await analyzeEvoText(textInput, sessionId, selectedModel, `EVO Psych Text Analysis - ${new Date().toLocaleDateString()}`);
              
              if (data.messages && data.messages.length > 0) {
                setMessages(prev => [...prev, ...data.messages]);
                setAnalysisId(data.analysisId);
                setAnalysisProgress(100);
                toast({
                  title: "EVO Psych Analysis Complete",
                  description: "Your text has been analyzed across 10 evolutionary personality poles",
                });
              }
            } catch (error) {
              console.error("EVO Psych text analysis error:", error);
              toast({
                variant: "destructive",
                title: "Analysis Failed",
                description: "Failed to analyze text for EVO Psych assessment. Please try again.",
              });
            } finally {
              setIsAnalyzing(false);
            }
          }}
          disabled={isAnalyzing}
          data-testid="button-evo-text"
        >
          🧬 EVO Psych (Text)
        </Button>
        
        <Button
          variant={selectedAnalysisType === "evo-image" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white hover:from-green-600/90 hover:to-emerald-600/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("evo-image");
            evoImageInputRef.current?.click();
          }}
          disabled={isAnalyzing}
          data-testid="button-evo-image"
        >
          🧬 EVO Psych (Image)
          <input
            ref={evoImageInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) {
                handleEvoImageAnalysis.mutate(files[0]);
              }
            }}
          />
        </Button>
        
        <Button
          variant={selectedAnalysisType === "evo-video" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-green-600 to-emerald-600 text-white hover:from-green-600/90 hover:to-emerald-600/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("evo-video");
            evoVideoInputRef.current?.click();
          }}
          disabled={isAnalyzing}
          data-testid="button-evo-video"
        >
          🧬 EVO Psych (Video)
          <input
            ref={evoVideoInputRef}
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) {
                handleEvoVideoAnalysis.mutate(files[0]);
              }
            }}
          />
        </Button>
        
        <Button
          variant={selectedAnalysisType === "vh-text" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-indigo-600 to-violet-600 text-white hover:from-indigo-600/90 hover:to-violet-600/90 border-none"
          onClick={async () => {
            setSelectedAnalysisType("vh-text");
            
            if (!textInput.trim()) {
              toast({
                variant: "destructive",
                title: "No Text",
                description: "Please enter text in the Input Preview section below",
              });
              return;
            }
            
            setIsAnalyzing(true);
            setAnalysisProgress(10);
            
            try {
              const data = await analyzeVerticalHorizontalText(textInput, sessionId, selectedModel, `V/H Orientation Analysis - ${new Date().toLocaleDateString()}`);
              
              if (data.messages && data.messages.length > 0) {
                setMessages(prev => [...prev, ...data.messages]);
                setAnalysisId(data.analysisId);
                setAnalysisProgress(100);
                toast({
                  title: "V/H Orientation Analysis Complete",
                  description: "Your text has been analyzed for Vertical vs. Horizontal worldview orientation",
                });
              }
            } catch (error) {
              console.error("V/H Orientation text analysis error:", error);
              toast({
                variant: "destructive",
                title: "Analysis Failed",
                description: "Failed to analyze text for V/H orientation. Please try again.",
              });
            } finally {
              setIsAnalyzing(false);
            }
          }}
          disabled={isAnalyzing}
          data-testid="button-vh-text"
        >
          ⬆️ V/H Orientation (Text)
        </Button>
        
        <Button
          variant={selectedAnalysisType === "vh-image" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-indigo-600 to-violet-600 text-white hover:from-indigo-600/90 hover:to-violet-600/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("vh-image");
            vhImageInputRef.current?.click();
          }}
          disabled={isAnalyzing}
          data-testid="button-vh-image"
        >
          ⬆️ V/H Orientation (Image)
          <input
            ref={vhImageInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) {
                handleVHImageAnalysis.mutate(files[0]);
              }
            }}
          />
        </Button>
        
        <Button
          variant={selectedAnalysisType === "vh-video" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-indigo-600 to-violet-600 text-white hover:from-indigo-600/90 hover:to-violet-600/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("vh-video");
            vhVideoInputRef.current?.click();
          }}
          disabled={isAnalyzing}
          data-testid="button-vh-video"
        >
          ⬆️ V/H Orientation (Video)
          <input
            ref={vhVideoInputRef}
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) {
                handleVHVideoAnalysis.mutate(files[0]);
              }
            }}
          />
        </Button>
        
        <Button
          variant={selectedAnalysisType === "bigfive-text" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-blue-600 to-sky-600 text-white hover:from-blue-600/90 hover:to-sky-600/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("bigfive-text");
            
            if (!textInput.trim()) {
              toast({
                variant: "destructive",
                title: "No Text",
                description: "Please enter text in the Input Preview section below",
              });
              return;
            }
            
            handleBigFiveTextAnalysis.mutate(textInput);
          }}
          disabled={isAnalyzing}
          data-testid="button-bigfive-text"
        >
          Big Five (Text)
        </Button>
        
        <Button
          variant={selectedAnalysisType === "bigfive-image" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-blue-600 to-sky-600 text-white hover:from-blue-600/90 hover:to-sky-600/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("bigfive-image");
            bigFiveImageInputRef.current?.click();
          }}
          disabled={isAnalyzing}
          data-testid="button-bigfive-image"
        >
          Big Five (Image)
          <input
            ref={bigFiveImageInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) {
                handleBigFiveImageAnalysis.mutate(files[0]);
              }
            }}
          />
        </Button>
        
        <Button
          variant={selectedAnalysisType === "bigfive-video" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-blue-600 to-sky-600 text-white hover:from-blue-600/90 hover:to-sky-600/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("bigfive-video");
            bigFiveVideoInputRef.current?.click();
          }}
          disabled={isAnalyzing}
          data-testid="button-bigfive-video"
        >
          Big Five (Video)
          <input
            ref={bigFiveVideoInputRef}
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) {
                handleBigFiveVideoAnalysis.mutate(files[0]);
              }
            }}
          />
        </Button>
        
        <Button
          variant={selectedAnalysisType === "enneagram-text" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-amber-600 to-yellow-500 text-white hover:from-amber-600/90 hover:to-yellow-500/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("enneagram-text");
            
            if (!textInput.trim()) {
              toast({
                variant: "destructive",
                title: "No Text",
                description: "Please enter text in the Input Preview section below",
              });
              return;
            }
            
            handleEnneagramTextAnalysis.mutate(textInput);
          }}
          disabled={isAnalyzing}
          data-testid="button-enneagram-text"
        >
          Enneagram (Text)
        </Button>
        
        <Button
          variant={selectedAnalysisType === "enneagram-image" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-amber-600 to-yellow-500 text-white hover:from-amber-600/90 hover:to-yellow-500/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("enneagram-image");
            enneagramImageInputRef.current?.click();
          }}
          disabled={isAnalyzing}
          data-testid="button-enneagram-image"
        >
          Enneagram (Image)
          <input
            ref={enneagramImageInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) {
                handleEnneagramImageAnalysis.mutate(files[0]);
              }
            }}
          />
        </Button>
        
        <Button
          variant={selectedAnalysisType === "enneagram-video" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-amber-600 to-yellow-500 text-white hover:from-amber-600/90 hover:to-yellow-500/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("enneagram-video");
            enneagramVideoInputRef.current?.click();
          }}
          disabled={isAnalyzing}
          data-testid="button-enneagram-video"
        >
          Enneagram (Video)
          <input
            ref={enneagramVideoInputRef}
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) {
                handleEnneagramVideoAnalysis.mutate(files[0]);
              }
            }}
          />
        </Button>
        
        <Button
          variant={selectedAnalysisType === "darktraits-text" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-slate-700 to-gray-600 text-white hover:from-slate-700/90 hover:to-gray-600/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("darktraits-text");
            
            if (!textInput.trim()) {
              toast({
                variant: "destructive",
                title: "No Text",
                description: "Please enter text in the Input Preview section below",
              });
              return;
            }
            
            handleDarkTraitsTextAnalysis.mutate(textInput);
          }}
          disabled={isAnalyzing}
          data-testid="button-darktraits-text"
        >
          Dark Traits (Text)
        </Button>
        
        <Button
          variant={selectedAnalysisType === "darktraits-image" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-slate-700 to-gray-600 text-white hover:from-slate-700/90 hover:to-gray-600/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("darktraits-image");
            darkTraitsImageInputRef.current?.click();
          }}
          disabled={isAnalyzing}
          data-testid="button-darktraits-image"
        >
          Dark Traits (Image)
          <input
            ref={darkTraitsImageInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) {
                handleDarkTraitsImageAnalysis.mutate(files[0]);
              }
            }}
          />
        </Button>
        
        <Button
          variant={selectedAnalysisType === "darktraits-video" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-slate-700 to-gray-600 text-white hover:from-slate-700/90 hover:to-gray-600/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("darktraits-video");
            darkTraitsVideoInputRef.current?.click();
          }}
          disabled={isAnalyzing}
          data-testid="button-darktraits-video"
        >
          Dark Traits (Video)
          <input
            ref={darkTraitsVideoInputRef}
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) {
                handleDarkTraitsVideoAnalysis.mutate(files[0]);
              }
            }}
          />
        </Button>
        
        <Button
          variant={selectedAnalysisType === "stanford-binet-text" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-rose-600 to-pink-500 text-white hover:from-rose-600/90 hover:to-pink-500/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("stanford-binet-text");
            
            if (!textInput.trim()) {
              toast({
                variant: "destructive",
                title: "No Text",
                description: "Please enter text in the Input Preview section below",
              });
              return;
            }
            
            handleStanfordBinetTextAnalysis.mutate(textInput);
          }}
          disabled={isAnalyzing}
          data-testid="button-stanford-binet-text"
        >
          Stanford-Binet (Text)
        </Button>
        
        <Button
          variant={selectedAnalysisType === "stanford-binet-image" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-rose-600 to-pink-500 text-white hover:from-rose-600/90 hover:to-pink-500/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("stanford-binet-image");
            stanfordBinetImageInputRef.current?.click();
          }}
          disabled={isAnalyzing}
          data-testid="button-stanford-binet-image"
        >
          Stanford-Binet (Image)
          <input
            ref={stanfordBinetImageInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) {
                handleStanfordBinetImageAnalysis.mutate(files[0]);
              }
            }}
          />
        </Button>
        
        <Button
          variant={selectedAnalysisType === "stanford-binet-video" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-rose-600 to-pink-500 text-white hover:from-rose-600/90 hover:to-pink-500/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("stanford-binet-video");
            stanfordBinetVideoInputRef.current?.click();
          }}
          disabled={isAnalyzing}
          data-testid="button-stanford-binet-video"
        >
          Stanford-Binet (Video)
          <input
            ref={stanfordBinetVideoInputRef}
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) {
                handleStanfordBinetVideoAnalysis.mutate(files[0]);
              }
            }}
          />
        </Button>
        
        <Button
          variant={selectedAnalysisType === "vocational-text" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-lime-600 to-green-500 text-white hover:from-lime-600/90 hover:to-green-500/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("vocational-text");
            
            if (!textInput.trim()) {
              toast({
                variant: "destructive",
                title: "No Text",
                description: "Please enter text in the Input Preview section below",
              });
              return;
            }
            
            handleVocationalTextAnalysis.mutate(textInput);
          }}
          disabled={isAnalyzing}
          data-testid="button-vocational-text"
        >
          Motivational
        </Button>
        
        <Button
          variant={selectedAnalysisType === "vocational-image" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-lime-600 to-green-500 text-white hover:from-lime-600/90 hover:to-green-500/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("vocational-image");
            vocationalImageInputRef.current?.click();
          }}
          disabled={isAnalyzing}
          data-testid="button-vocational-image"
        >
          Motivational (Image)
          <input
            ref={vocationalImageInputRef}
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) {
                handleVocationalImageAnalysis.mutate(files[0]);
              }
            }}
          />
        </Button>
        
        <Button
          variant={selectedAnalysisType === "vocational-video" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3 bg-gradient-to-r from-lime-600 to-green-500 text-white hover:from-lime-600/90 hover:to-green-500/90 border-none"
          onClick={() => {
            setSelectedAnalysisType("vocational-video");
            vocationalVideoInputRef.current?.click();
          }}
          disabled={isAnalyzing}
          data-testid="button-vocational-video"
        >
          Motivational (Video)
          <input
            ref={vocationalVideoInputRef}
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              const files = e.target.files;
              if (files && files.length > 0) {
                handleVocationalVideoAnalysis.mutate(files[0]);
              }
            }}
          />
        </Button>
      </div>
      
      {/* Main Content Area */}
      <div className="flex-1">
        <div className="container mx-auto p-4 max-w-6xl">
          <div className="flex justify-between items-center mb-4">
            <a 
              href="mailto:contact@zhisystems.ai" 
              className="text-sm text-primary hover:underline flex items-center gap-1"
              data-testid="link-contact-us"
            >
              Contact Us
            </a>
            <div></div>
          </div>
          <h1 className="text-4xl font-bold text-center mb-8">AI Personality Analysis</h1>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        {/* Left Column - Inputs and Upload */}
        <div className="space-y-6">
          {/* Model Selector */}
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">Step 1: Select AI Model</h2>
            <Select
              value={selectedModel}
              onValueChange={(value) => setSelectedModel(value as ModelType)}
              disabled={isAnalyzing}
            >
              <SelectTrigger>
                <SelectValue placeholder="Select AI Model" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="grok">知 5</SelectItem>
                <SelectItem value="anthropic">知 1</SelectItem>
                <SelectItem value="openai">知 2</SelectItem>
                <SelectItem value="deepseek">知 3</SelectItem>
                <SelectItem value="perplexity">知 4</SelectItem>
              </SelectContent>
            </Select>
          </Card>
          
          {/* Upload Options */}
          <Card className="p-6">
            <h2 className="text-xl font-semibold mb-4">Step 2: Choose Input Type</h2>
            <div className="grid grid-cols-3 gap-3">
              <Button 
                variant={selectedAnalysisType === "image" ? "default" : "outline"}
                className="h-20 flex flex-col items-center justify-center text-xs bg-gradient-to-r from-blue-500 to-cyan-500 text-white hover:from-blue-500/90 hover:to-cyan-500/90 border-none" 
                onClick={() => {
                  setSelectedAnalysisType("image");
                  handleImageVideoClick();
                }}
                disabled={isAnalyzing}
                data-testid="button-image"
              >
                <FileImage className="h-6 w-6 mb-1" />
                <span>Image</span>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*,video/*"
                  style={{ display: 'none' }}
                  onChange={(e) => handleFileInputChange(e, 'media')}
                />
              </Button>
              
              <Button 
                variant={selectedAnalysisType === "document" ? "default" : "outline"}
                className="h-20 flex flex-col items-center justify-center text-xs bg-gradient-to-r from-amber-500 to-orange-500 text-white hover:from-amber-500/90 hover:to-orange-500/90 border-none" 
                onClick={() => {
                  setSelectedAnalysisType("document");
                  handleDocumentClick();
                }}
                disabled={isAnalyzing}
                data-testid="button-document"
              >
                <FileText className="h-6 w-6 mb-1" />
                <span>Document</span>
                <input
                  ref={documentInputRef}
                  type="file"
                  accept=".pdf,.doc,.docx,.txt"
                  style={{ display: 'none' }}
                  onChange={(e) => handleFileInputChange(e, 'document')}
                />
              </Button>
              
              <Button 
                variant={selectedAnalysisType === "video" ? "default" : "outline"}
                className="h-20 flex flex-col items-center justify-center text-xs bg-gradient-to-r from-purple-500 to-pink-500 text-white hover:from-purple-500/90 hover:to-pink-500/90 border-none" 
                onClick={() => {
                  setSelectedAnalysisType("video");
                  handleImageVideoClick();
                }}
                disabled={isAnalyzing}
                data-testid="button-video"
              >
                <Film className="h-6 w-6 mb-1" />
                <span>Video</span>
              </Button>

              <Button 
                variant={selectedAnalysisType === "text-mbti" ? "default" : "outline"}
                className="h-20 flex flex-col items-center justify-center text-xs bg-gradient-to-r from-teal-500 to-emerald-500 text-white hover:from-teal-500/90 hover:to-emerald-500/90 border-none" 
                onClick={async () => {
                  setSelectedAnalysisType("text-mbti");
                  
                  if (!textInput.trim()) {
                    toast({
                      variant: "destructive",
                      title: "No Text",
                      description: "Please enter text in the Input Preview section below",
                    });
                    return;
                  }
                  
                  setIsAnalyzing(true);
                  setAnalysisProgress(0);
                  // Messages now append instead of clear
                  
                  try {
                    const data = await analyzeMBTIText(textInput, sessionId, selectedModel);
                    
                    if (data.messages && data.messages.length > 0) {
                      setMessages(prev => [...prev, ...data.messages]);
                      setAnalysisId(data.analysisId);
                      setAnalysisProgress(100);
                      toast({
                        title: "MBTI Analysis Complete",
                        description: "Your text has been analyzed using the MBTI framework",
                      });
                      setTextInput("");
                    }
                  } catch (error) {
                    console.error("MBTI text analysis error:", error);
                    toast({
                      variant: "destructive",
                      title: "Analysis Failed",
                      description: "Failed to analyze text for MBTI. Please try again.",
                    });
                  } finally {
                    setIsAnalyzing(false);
                  }
                }}
                disabled={isAnalyzing}
                data-testid="button-text-mbti"
              >
                <FileText className="h-6 w-6 mb-1" />
                <span>Text MBTI</span>
              </Button>

              <Button 
                variant={selectedAnalysisType === "image-mbti" ? "default" : "outline"}
                className="h-20 flex flex-col items-center justify-center text-xs bg-gradient-to-r from-teal-500 to-emerald-500 text-white hover:from-teal-500/90 hover:to-emerald-500/90 border-none" 
                onClick={() => {
                  setSelectedAnalysisType("image-mbti");
                  imageMBTIInputRef.current?.click();
                }}
                disabled={isAnalyzing}
                data-testid="button-image-mbti"
              >
                <FileImage className="h-6 w-6 mb-1" />
                <span>Image MBTI</span>
                <input
                  ref={imageMBTIInputRef}
                  type="file"
                  accept="image/*"
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    const files = e.target.files;
                    if (files && files.length > 0) {
                      handleImageMBTIAnalysis.mutate(files[0]);
                    }
                  }}
                />
              </Button>

              <Button 
                variant={selectedAnalysisType === "video-mbti" ? "default" : "outline"}
                className="h-20 flex flex-col items-center justify-center text-xs bg-gradient-to-r from-teal-500 to-emerald-500 text-white hover:from-teal-500/90 hover:to-emerald-500/90 border-none" 
                onClick={() => {
                  setSelectedAnalysisType("video-mbti");
                  videoMBTIInputRef.current?.click();
                }}
                disabled={isAnalyzing}
                data-testid="button-video-mbti"
              >
                <Film className="h-6 w-6 mb-1" />
                <span>Video MBTI</span>
                <input
                  ref={videoMBTIInputRef}
                  type="file"
                  accept="video/*"
                  style={{ display: 'none' }}
                  onChange={(e) => {
                    const files = e.target.files;
                    if (files && files.length > 0) {
                      handleVideoMBTIAnalysis.mutate(files[0]);
                    }
                  }}
                />
              </Button>

              <Button 
                variant={selectedAnalysisType === "image-deepdive" ? "default" : "outline"}
                className="h-20 flex flex-col items-center justify-center text-xs bg-gradient-to-r from-rose-500 to-red-500 text-white hover:from-rose-500/90 hover:to-red-500/90 border-none" 
                onClick={() => {
                  setSelectedAnalysisType("image-deepdive");
                  toast({ title: "Coming Soon", description: "Image Deep Dive functionality will be added soon." });
                }}
                disabled={isAnalyzing}
                data-testid="button-image-deepdive"
              >
                <FileImage className="h-6 w-6 mb-1" />
                <span>Image Deep Dive</span>
              </Button>

              <Button 
                variant={selectedAnalysisType === "text-deepdive" ? "default" : "outline"}
                className="h-20 flex flex-col items-center justify-center text-xs bg-gradient-to-r from-rose-500 to-red-500 text-white hover:from-rose-500/90 hover:to-red-500/90 border-none" 
                onClick={() => {
                  setSelectedAnalysisType("text-deepdive");
                  toast({ title: "Coming Soon", description: "Text Deep Dive functionality will be added soon." });
                }}
                disabled={isAnalyzing}
                data-testid="button-text-deepdive"
              >
                <FileText className="h-6 w-6 mb-1" />
                <span>Text Deep Dive</span>
              </Button>

              <Button 
                variant={selectedAnalysisType === "video-deepdive" ? "default" : "outline"}
                className="h-20 flex flex-col items-center justify-center text-xs bg-gradient-to-r from-rose-500 to-red-500 text-white hover:from-rose-500/90 hover:to-red-500/90 border-none" 
                onClick={() => {
                  setSelectedAnalysisType("video-deepdive");
                  toast({ title: "Coming Soon", description: "Video Deep Dive functionality will be added soon." });
                }}
                disabled={isAnalyzing}
                data-testid="button-video-deepdive"
              >
                <Film className="h-6 w-6 mb-1" />
                <span>Video Deep Dive</span>
              </Button>
            </div>
            
            <p className="text-xs text-amber-600 mt-2">For best results, videos should be under 15 seconds.</p>
            
            {isAnalyzing && (
              <div className="mt-4 space-y-2">
                <div className="flex justify-between">
                  <span>Analyzing...</span>
                  <span>{analysisProgress}%</span>
                </div>
                <Progress value={analysisProgress} className="w-full" />
              </div>
            )}
            
            {/* Drag area info */}
            <div className={`mt-4 p-4 border-2 border-dashed rounded-lg text-center cursor-pointer transition-colors ${isDragActive ? "border-primary bg-primary/5" : "border-muted"}`}>
              <input {...getInputProps()} />
              <p className="text-muted-foreground">
                Drag & drop files here to analyze
              </p>
              <p className="text-xs text-muted-foreground">
                Supports JPG, PNG, MP4, MOV, PDF, DOC, DOCX (max 50MB)
              </p>
            </div>
          </Card>
          
          {/* Input Preview */}
          <Card className="p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">Input Preview</h2>
              {!uploadedMedia && !documentName && (
                <Button 
                  variant="outline" 
                  onClick={() => {
                    // Clear other inputs and focus on text
                    setUploadedMedia(null);
                    setDocumentName("");
                    setTextInput(textInput || "");
                  }}
                  className="flex items-center gap-2"
                  disabled={isAnalyzing}
                >
                  <span>Text Input</span>
                </Button>
              )}
            </div>
            
            {uploadedMedia && mediaType === "image" && (
              <div className="space-y-4">
                <img 
                  src={uploadedMedia} 
                  alt="Uploaded" 
                  className="max-w-full h-auto rounded-lg shadow-md mx-auto"
                />
                <div className="text-center text-sm text-muted-foreground mb-4">
                  Face detection will analyze personality traits and emotions
                </div>
                
                {/* Re-analyze with current model button */}
                <Button 
                  onClick={async () => {
                    if (!mediaData) return;
                    
                    // Clear messages for new analysis
                    // Messages now append instead of clear
                    setIsAnalyzing(true);
                    setAnalysisProgress(0);
                    
                    try {
                      let response;
                      
                      // Check which type of analysis to run based on selectedAnalysisType
                      if (selectedAnalysisType === "clinical-image") {
                        response = await analyzeClinicalImage(mediaData, sessionId, selectedModel);
                      } else if (selectedAnalysisType === "personality-structure-image") {
                        response = await analyzePersonalityStructureImage(mediaData, sessionId, selectedModel);
                      } else {
                        // Default to general image analysis
                        response = await uploadMedia(mediaData, "image", sessionId, { selectedModel, maxPeople: 5 });
                      }
                      
                      setAnalysisProgress(100);
                      
                      if (response && response.analysisId) {
                        setAnalysisId(response.analysisId);
                      }
                      
                      if (response && response.messages && Array.isArray(response.messages)) {
                        setMessages(prev => [...prev, ...response.messages]);
                      }
                      
                      toast({
                        title: "Analysis Complete",
                        description: "Your image has been analyzed with " + 
                          (selectedModel === "grok" ? "知 5" : selectedModel === "openai" ? "知 2" : selectedModel === "anthropic" ? "知 1" : selectedModel === "deepseek" ? "知 3" : "知 4"),
                      });
                    } catch (error) {
                      console.error("Image analysis error:", error);
                      toast({
                        variant: "destructive",
                        title: "Error",
                        description: "Failed to analyze image. Please try again.",
                      });
                    } finally {
                      setIsAnalyzing(false);
                    }
                  }}
                  className="w-full"
                  disabled={isAnalyzing || !mediaData}
                >
                  {messages.length > 0 ? "Re-Analyze" : "Analyze"} with {selectedModel === "grok" ? "知 5" : selectedModel === "openai" ? "知 2" : selectedModel === "anthropic" ? "知 1" : selectedModel === "deepseek" ? "知 3" : "知 4"}
                </Button>
                <div className="pt-4 mt-4 border-t">
                  <p className="text-sm font-semibold mb-2">MBTI Analysis (Image)</p>
                  <Button 
                    onClick={async () => {
                      if (!mediaData) {
                        toast({
                          variant: "destructive",
                          title: "Error",
                          description: "No image data available",
                        });
                        return;
                      }
                      
                      setIsAnalyzing(true);
                      setAnalysisProgress(0);
                      // Messages now append instead of clear
                      
                      try {
                        const data = await analyzeMBTIImage(mediaData, sessionId, selectedModel);
                        
                        if (data.messages && data.messages.length > 0) {
                          setMessages(prev => [...prev, ...data.messages]);
                          setAnalysisId(data.analysisId);
                          setAnalysisProgress(100);
                          toast({
                            title: "MBTI Analysis Complete",
                            description: "Your image has been analyzed using the MBTI framework",
                          });
                        }
                      } catch (error) {
                        console.error("MBTI image analysis error:", error);
                        toast({
                          variant: "destructive",
                          title: "Analysis Failed",
                          description: "Failed to analyze image for MBTI. Please try again.",
                        });
                      } finally {
                        setIsAnalyzing(false);
                      }
                    }}
                    variant="secondary"
                    className="w-full"
                    disabled={isAnalyzing || !mediaData}
                    data-testid="button-mbti-image"
                  >
                    MBTI Image Analysis
                  </Button>
                </div>
              </div>
            )}
            
            {uploadedMedia && mediaType === "video" && (
              <div className="space-y-4">
                <video 
                  ref={videoRef}
                  src={uploadedMedia} 
                  controls
                  className="max-w-full h-auto rounded-lg shadow-md mx-auto"
                />
                <div className="text-center text-sm text-muted-foreground mb-4">
                  Video analysis will extract visual and audio insights
                </div>
                
                {/* Re-analyze with current model button */}
                <Button 
                  onClick={async () => {
                    if (!mediaData) return;
                    
                    // Clear messages for new analysis
                    // Messages now append instead of clear
                    setIsAnalyzing(true);
                    setAnalysisProgress(0);
                    
                    try {
                      let response;
                      
                      // Check which type of analysis to run based on selectedAnalysisType
                      if (selectedAnalysisType === "clinical-video") {
                        response = await analyzeClinicalVideo(mediaData, sessionId, selectedModel);
                      } else if (selectedAnalysisType === "personality-structure-video") {
                        response = await analyzePersonalityStructureVideo(mediaData, sessionId, selectedModel);
                      } else {
                        // Default to general video analysis
                        response = await uploadMedia(mediaData, "video", sessionId, { selectedModel, maxPeople: 5 });
                      }
                      
                      setAnalysisProgress(100);
                      
                      if (response && response.analysisId) {
                        setAnalysisId(response.analysisId);
                      }
                      
                      if (response && response.messages && Array.isArray(response.messages)) {
                        setMessages(prev => [...prev, ...response.messages]);
                      }
                      
                      toast({
                        title: "Analysis Complete",
                        description: "Your video has been analyzed with " + 
                          (selectedModel === "grok" ? "知 5" : selectedModel === "openai" ? "知 2" : selectedModel === "anthropic" ? "知 1" : selectedModel === "deepseek" ? "知 3" : "知 4"),
                      });
                    } catch (error) {
                      console.error("Video analysis error:", error);
                      toast({
                        variant: "destructive",
                        title: "Error",
                        description: "Failed to analyze video. Please try again.",
                      });
                    } finally {
                      setIsAnalyzing(false);
                    }
                  }}
                  className="w-full"
                  disabled={isAnalyzing || !mediaData}
                >
                  {messages.length > 0 ? "Re-Analyze" : "Analyze"} with {selectedModel === "grok" ? "知 5" : selectedModel === "openai" ? "知 2" : selectedModel === "anthropic" ? "知 1" : selectedModel === "deepseek" ? "知 3" : "知 4"}
                </Button>
                <div className="pt-4 mt-4 border-t">
                  <p className="text-sm font-semibold mb-2">MBTI Analysis (Video)</p>
                  <Button 
                    onClick={async () => {
                      if (!mediaData) {
                        toast({
                          variant: "destructive",
                          title: "Error",
                          description: "No video data available",
                        });
                        return;
                      }
                      
                      setIsAnalyzing(true);
                      setAnalysisProgress(0);
                      // Messages now append instead of clear
                      
                      try {
                        const data = await analyzeMBTIVideo(mediaData, sessionId, selectedModel);
                        
                        if (data.messages && data.messages.length > 0) {
                          setMessages(prev => [...prev, ...data.messages]);
                          setAnalysisId(data.analysisId);
                          setAnalysisProgress(100);
                          toast({
                            title: "MBTI Analysis Complete",
                            description: "Your video has been analyzed using the MBTI framework",
                          });
                        }
                      } catch (error) {
                        console.error("MBTI video analysis error:", error);
                        toast({
                          variant: "destructive",
                          title: "Analysis Failed",
                          description: "Failed to analyze video for MBTI. Please try again.",
                        });
                      } finally {
                        setIsAnalyzing(false);
                      }
                    }}
                    variant="secondary"
                    className="w-full"
                    disabled={isAnalyzing || !mediaData}
                    data-testid="button-mbti-video"
                  >
                    MBTI Video Analysis
                  </Button>
                </div>
              </div>
            )}
            
            {documentName && (
              <div className="space-y-4">
                <div className="p-4 bg-muted rounded-lg flex items-center">
                  <FileText className="w-6 h-6 mr-2" />
                  <span>{documentName}</span>
                </div>
                <div className="text-center text-sm text-muted-foreground mb-4">
                  Document content will be analyzed for personality insights
                </div>
                
                {/* Re-analyze with current model button */}
                <Button 
                  onClick={() => {
                    // Here we should trigger re-analysis of the current document
                    // with the currently selected model, but we need actual document data
                    // Since we don't store the file data after upload, we'll need to prompt
                    // user to re-upload the file
                    toast({
                      title: "Re-upload Required",
                      description: "Please re-upload the document to analyze with the new model.",
                    });
                    // Clear document name to allow re-upload
                    setDocumentName("");
                    // Focus on document upload
                    handleDocumentClick();
                  }}
                  className="w-full"
                  disabled={isAnalyzing}
                >
                  Re-Analyze with {selectedModel === "grok" ? "知 5" : selectedModel === "openai" ? "知 2" : selectedModel === "anthropic" ? "知 1" : selectedModel === "deepseek" ? "知 3" : "知 4"}
                </Button>
              </div>
            )}
            
            {!uploadedMedia && !documentName && (
              <form onSubmit={handleTextSubmit} className="space-y-4">
                <div className="flex items-center justify-between mb-2">
                  <label className="text-sm font-medium">Text Input</label>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      const input = document.createElement('input');
                      input.type = 'file';
                      input.accept = '.pdf,.doc,.docx,.txt';
                      input.onchange = async (e) => {
                        const file = (e.target as HTMLInputElement).files?.[0];
                        if (!file) return;
                        
                        const fileExt = file.name.split('.').pop()?.toLowerCase();
                        if (!fileExt || !['pdf', 'doc', 'docx', 'txt'].includes(fileExt)) {
                          toast({
                            variant: "destructive",
                            title: "Unsupported File",
                            description: "Please upload a PDF, DOC, DOCX, or TXT file",
                          });
                          return;
                        }
                        
                        try {
                          toast({
                            title: "Extracting Text",
                            description: `Reading ${file.name}...`,
                          });
                          
                          // Read file as base64
                          const reader = new FileReader();
                          const fileData = await new Promise<string>((resolve) => {
                            reader.onload = (e) => resolve(e.target?.result as string);
                            reader.readAsDataURL(file);
                          });
                          
                          // Call extract-text API
                          const response = await fetch('/api/extract-text', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                              fileData,
                              fileName: file.name,
                              fileType: fileExt
                            })
                          });
                          
                          if (!response.ok) {
                            throw new Error('Failed to extract text');
                          }
                          
                          const data = await response.json();
                          setTextInput(data.text);
                          
                          toast({
                            title: "Text Extracted",
                            description: `Successfully extracted text from ${file.name}`,
                          });
                        } catch (error) {
                          console.error('Document extraction error:', error);
                          toast({
                            variant: "destructive",
                            title: "Extraction Failed",
                            description: "Could not extract text from the document. Please try again.",
                          });
                        }
                      };
                      input.click();
                    }}
                    className="text-xs"
                    disabled={isAnalyzing}
                    data-testid="button-upload-document"
                  >
                    <FileText className="w-4 h-4 mr-1" />
                    Upload Document
                  </Button>
                </div>
                <Textarea
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  onKeyDown={(e) => handleKeyPress(e, handleTextSubmit)}
                  placeholder="Type, paste, or upload a document to analyze..."
                  className="min-h-[250px] resize-y"
                  disabled={isAnalyzing}
                />
                <p className="text-xs text-amber-600 mt-1">For best results, texts should be under 2,000 words.</p>
                <Button 
                  type="submit" 
                  className="w-full" 
                  disabled={!textInput.trim() || isAnalyzing}
                >
                  Analyze Text
                </Button>
                <div className="pt-4 mt-4 border-t">
                  <p className="text-sm font-semibold mb-2 text-primary">🌟 Consolidated Personality Structure (Text)</p>
                  <p className="text-xs text-muted-foreground mb-3">Comprehensive synthesis of 8 frameworks: Big Five, HEXACO, 16PF, MBTI, Keirsey, Socionics, Hogan, DISC</p>
                  <Button 
                    onClick={async () => {
                      if (!textInput.trim()) {
                        toast({
                          variant: "destructive",
                          title: "Error",
                          description: "Please enter some text first",
                        });
                        return;
                      }
                      
                      setIsAnalyzing(true);
                      setAnalysisProgress(0);
                      
                      try {
                        const data = await analyzePersonalityStructureText(textInput, sessionId, selectedModel);
                        
                        if (data.messages && data.messages.length > 0) {
                          setMessages(prev => [...prev, ...data.messages]);
                          setAnalysisId(data.analysisId);
                          setAnalysisProgress(100);
                          toast({
                            title: "Personality Structure Analysis Complete",
                            description: "Your text has been analyzed across 8 major personality frameworks",
                          });
                        }
                      } catch (error) {
                        console.error("Personality Structure text analysis error:", error);
                        toast({
                          variant: "destructive",
                          title: "Analysis Failed",
                          description: "Failed to analyze text for consolidated personality structure. Please try again.",
                        });
                      } finally {
                        setIsAnalyzing(false);
                      }
                    }}
                    variant="default"
                    className="w-full bg-gradient-to-r from-primary to-purple-600 hover:from-primary/90 hover:to-purple-600/90" 
                    disabled={!textInput.trim() || isAnalyzing}
                    data-testid="button-personality-structure-text"
                  >
                    Comprehensive Personality Structure Analysis
                  </Button>
                  
                  <p className="text-sm font-semibold mb-2 mt-6">MBTI Analysis (Text)</p>
                  <Button 
                    onClick={async () => {
                      if (!textInput.trim()) {
                        toast({
                          variant: "destructive",
                          title: "Error",
                          description: "Please enter some text first",
                        });
                        return;
                      }
                      
                      setIsAnalyzing(true);
                      setAnalysisProgress(0);
                      
                      try {
                        const data = await analyzeMBTIText(textInput, sessionId, selectedModel);
                        
                        if (data.messages && data.messages.length > 0) {
                          setMessages(prev => [...prev, ...data.messages]);
                          setAnalysisId(data.analysisId);
                          setAnalysisProgress(100);
                          toast({
                            title: "MBTI Analysis Complete",
                            description: "Your text has been analyzed using the MBTI framework",
                          });
                        }
                      } catch (error) {
                        console.error("MBTI text analysis error:", error);
                        toast({
                          variant: "destructive",
                          title: "Analysis Failed",
                          description: "Failed to analyze text for MBTI. Please try again.",
                        });
                      } finally {
                        setIsAnalyzing(false);
                      }
                    }}
                    variant="secondary"
                    className="w-full" 
                    disabled={!textInput.trim() || isAnalyzing}
                    data-testid="button-mbti-text"
                  >
                    MBTI Text Analysis
                  </Button>
                  
                  <p className="text-sm font-semibold mb-2 mt-4">Big Five Analysis (Text)</p>
                  <Button 
                    onClick={async () => {
                      if (!textInput.trim()) {
                        toast({
                          variant: "destructive",
                          title: "Error",
                          description: "Please enter some text first",
                        });
                        return;
                      }
                      
                      setIsAnalyzing(true);
                      setAnalysisProgress(0);
                      
                      try {
                        const data = await analyzeBigFiveText(textInput, sessionId, selectedModel);
                        
                        if (data.messages && data.messages.length > 0) {
                          setMessages(prev => [...prev, ...data.messages]);
                          setAnalysisId(data.analysisId);
                          setAnalysisProgress(100);
                          toast({
                            title: "Big Five Analysis Complete",
                            description: "Your text has been analyzed using the Five-Factor Model",
                          });
                        }
                      } catch (error) {
                        console.error("Big Five text analysis error:", error);
                        toast({
                          variant: "destructive",
                          title: "Analysis Failed",
                          description: "Failed to analyze text for Big Five. Please try again.",
                        });
                      } finally {
                        setIsAnalyzing(false);
                      }
                    }}
                    variant="secondary"
                    className="w-full" 
                    disabled={!textInput.trim() || isAnalyzing}
                    data-testid="button-bigfive-text-main"
                  >
                    Big Five Text Analysis
                  </Button>
                  
                  <p className="text-sm font-semibold mb-2 mt-4">Enneagram Analysis (Text)</p>
                  <Button 
                    onClick={async () => {
                      if (!textInput.trim()) {
                        toast({
                          variant: "destructive",
                          title: "Error",
                          description: "Please enter some text first",
                        });
                        return;
                      }
                      
                      setIsAnalyzing(true);
                      setAnalysisProgress(0);
                      
                      try {
                        const data = await analyzeEnneagramText(textInput, sessionId, selectedModel);
                        
                        if (data.messages && data.messages.length > 0) {
                          setMessages(prev => [...prev, ...data.messages]);
                          setAnalysisId(data.analysisId);
                          setAnalysisProgress(100);
                          toast({
                            title: "Enneagram Analysis Complete",
                            description: "Your personality type has been identified using the Enneagram framework",
                          });
                        }
                      } catch (error) {
                        console.error("Enneagram text analysis error:", error);
                        toast({
                          variant: "destructive",
                          title: "Analysis Failed",
                          description: "Failed to analyze text for Enneagram. Please try again.",
                        });
                      } finally {
                        setIsAnalyzing(false);
                      }
                    }}
                    variant="secondary"
                    className="w-full" 
                    disabled={!textInput.trim() || isAnalyzing}
                    data-testid="button-enneagram-text-main"
                  >
                    Enneagram Text Analysis
                  </Button>
                </div>
              </form>
            )}
          </Card>
        </div>
        
        {/* Right Column - Results and Chat */}
        <div className="space-y-6">
          {/* ANALYSIS BOX */}
          <Card className="p-6 border-2 border-primary">
            <div className="flex justify-between items-center mb-4">
              <div className="flex items-center gap-2">
                <h2 className="text-xl font-bold">ANALYSIS</h2>
                {/* New Analysis button - always visible */}
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="flex items-center gap-2"
                  onClick={async () => {
                    try {
                      // First clear the session on the server
                      await clearSession(sessionId);
                      
                      // Generate a new session ID to ensure a completely clean state
                      const newSessionId = nanoid();
                      window.location.href = `/?session=${newSessionId}`;
                      
                      // Clear all current state to start a new analysis
                      // Messages now append instead of clear
                      setUploadedMedia(null);
                      setMediaData(null);
                      setDocumentName("");
                      setTextInput("");
                      setAnalysisId(null);
                      setAnalysisProgress(0);
                      toast({
                        title: "New Analysis",
                        description: "Starting a completely new analysis session",
                      });
                    } catch (error) {
                      console.error("Error clearing session:", error);
                      toast({
                        variant: "destructive",
                        title: "Error",
                        description: "Failed to clear previous analysis. Please refresh the page.",
                      });
                    }
                  }}
                  disabled={isAnalyzing}
                >
                  <span>New Analysis</span>
                </Button>
              </div>
              
              {messages.length > 0 && (
                <div className="flex gap-2">
                  {/* Download PDF button */}
                  {analysisId && (
                    <Button 
                      variant="outline" 
                      size="sm" 
                      className="flex items-center gap-2"
                      onClick={() => downloadAnalysis(analysisId, 'pdf')}
                    >
                      <Download className="h-4 w-4" />
                      <span>PDF</span>
                    </Button>
                  )}
                  
                  {/* Download DOCX button */}
                  {analysisId && (
                    <Button 
                      variant="outline" 
                      size="sm" 
                      className="flex items-center gap-2"
                      onClick={() => downloadAnalysis(analysisId, 'docx')}
                    >
                      <File className="h-4 w-4" />
                      <span>DOCX</span>
                    </Button>
                  )}
                  
                  {/* Share button */}
                  {emailServiceAvailable && (
                    <Dialog open={isShareDialogOpen} onOpenChange={setIsShareDialogOpen}>
                      <DialogTrigger asChild>
                        <Button variant="outline" size="sm" className="flex items-center gap-2">
                          <Share2 className="h-4 w-4" />
                          <span>Share</span>
                        </Button>
                      </DialogTrigger>
                      <DialogContent>
                        <DialogHeader>
                          <DialogTitle>Share Analysis</DialogTitle>
                        </DialogHeader>
                        <Form {...shareForm}>
                          <form onSubmit={shareForm.handleSubmit(onShareSubmit)} className="space-y-4">
                            <FormField
                              control={shareForm.control}
                              name="senderEmail"
                              render={({ field }) => (
                                <FormItem>
                                  <FormLabel>Your Email</FormLabel>
                                  <FormControl>
                                    <Input {...field} type="email" placeholder="youremail@example.com" />
                                  </FormControl>
                                  <FormMessage />
                                </FormItem>
                              )}
                            />
                            <FormField
                              control={shareForm.control}
                              name="recipientEmail"
                              render={({ field }) => (
                                <FormItem>
                                  <FormLabel>Recipient's Email</FormLabel>
                                  <FormControl>
                                    <Input {...field} type="email" placeholder="recipient@example.com" />
                                  </FormControl>
                                  <FormMessage />
                                </FormItem>
                              )}
                            />
                            <DialogFooter>
                              <Button 
                                type="submit" 
                                disabled={shareMutation.isPending}
                                className="w-full"
                              >
                                {shareMutation.isPending ? "Sending..." : "Share Analysis"}
                              </Button>
                            </DialogFooter>
                          </form>
                        </Form>
                      </DialogContent>
                    </Dialog>
                  )}
                </div>
              )}
            </div>
            
            <div className="h-[400px] flex flex-col">
              {messages.length === 0 ? (
                <div className="flex flex-col items-center justify-center space-y-4 h-full text-center text-muted-foreground">
                  <AlertCircle className="h-12 w-12" />
                  <div>
                    <p className="text-lg font-medium">No analysis yet</p>
                    <p>Upload media, enter text, or select a document to analyze.</p>
                    <p className="text-xs mt-4">Debug: {JSON.stringify({ messageCount: messages.length, sessionId })}</p>
                  </div>
                </div>
              ) : (
                <>
                  <div className="flex justify-end mb-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        const analysisText = messages
                          .filter(m => m.role === "assistant")
                          .map(m => m.content)
                          .join("\n\n---\n\n");
                        navigator.clipboard.writeText(analysisText);
                        setCopied(true);
                        setTimeout(() => setCopied(false), 2000);
                      }}
                      className="gap-2"
                      data-testid="button-copy-analysis"
                    >
                      {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                      {copied ? "Copied!" : "Copy"}
                    </Button>
                  </div>
                  <ScrollArea className="flex-1 pr-4 mb-4">
                    <div className="space-y-4">
                    {messages.filter(message => message.role === "assistant").map((message, index) => (
                      <div
                        key={index}
                        className="flex flex-col p-4 rounded-lg bg-white border border-gray-200 shadow-sm"
                      >
                        <div 
                          className="whitespace-pre-wrap text-md"
                          dangerouslySetInnerHTML={{ 
                            __html: message.content
                              .replace(/\n/g, '<br/>')
                              .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                              .replace(/^(#+)\s+(.*?)$/gm, (_: string, hashes: string, text: string) => 
                                `<h${hashes.length} class="font-bold text-lg mt-3 mb-1">${text}</h${hashes.length}>`)
                              .replace(/- (.*?)$/gm, '<li class="ml-4">• $1</li>')
                          }} 
                        />
                      </div>
                    ))}
                    <div ref={messagesEndRef} />
                  </div>
                </ScrollArea>
                </>
              )}
            </div>
          </Card>
          
          {/* CHAT BOX */}
          <Card className="p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">Chat</h2>
            </div>
            
            <div className="h-[300px] flex flex-col">
              {messages.length === 0 ? (
                <div className="flex flex-col items-center justify-center space-y-4 h-full text-center text-muted-foreground">
                  <div>
                    <p>No conversation yet. Ask questions about the analysis after it's generated.</p>
                  </div>
                </div>
              ) : (
                <ScrollArea className="flex-1 pr-4 mb-4">
                  <div className="space-y-4">
                    {messages.filter(message => message.role === "user" || (message.role === "assistant" && messages.some(m => m.role === "user"))).map((message, index) => (
                      <div
                        key={index}
                        className={`flex flex-col p-4 rounded-lg ${
                          message.role === "user" ? "bg-primary/10 ml-8" : "bg-primary/5 mr-4"
                        }`}
                      >
                        <span className="font-semibold text-sm mb-2">
                          {message.role === "user" ? "You" : "AI"}
                        </span>
                        <div 
                          className="whitespace-pre-wrap text-md"
                          dangerouslySetInnerHTML={{ 
                            __html: message.content
                              .replace(/\n/g, '<br/>')
                              .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                          }} 
                        />
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              )}
              
              <form onSubmit={handleChatSubmit} className="mt-auto">
                <div className="flex gap-2">
                  <Textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => handleKeyPress(e, handleChatSubmit)}
                    placeholder="Ask a question about the analysis..."
                    className="min-h-[80px] resize-none"
                  />
                  <Button 
                    type="submit" 
                    className="self-end"
                    disabled={!input.trim() || chatMutation.isPending}
                  >
                    <Send className="h-4 w-4" />
                  </Button>
                </div>
              </form>
            </div>
          </Card>
        </div>
      </div>
        </div>
      </div>
    </div>
  );
}