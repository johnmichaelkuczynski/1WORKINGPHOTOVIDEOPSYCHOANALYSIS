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
import { uploadMedia, sendMessage, shareAnalysis, getSharedAnalysis, analyzeText, analyzeDocument, downloadAnalysis, clearSession, analyzeMBTIText, analyzeMBTIImage, analyzeMBTIVideo, analyzeMBTIDocument, analyzeBigFiveText, ModelType, MediaType } from "@/lib/api";
import { Upload, Send, FileImage, Film, Share2, AlertCircle, FileText, File, Download } from "lucide-react";
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
  const [sessionId] = useState(() => nanoid());
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
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedModel, setSelectedModel] = useState<ModelType>("openai");
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
            setMessages(data.messages);
            
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
        setMessages([]);
        
        const response = await analyzeText(text, sessionId, selectedModel);
        
        setAnalysisProgress(80);
        setAnalysisId(response.analysisId);
        
        if (response.messages && response.messages.length > 0) {
          setMessages(response.messages);
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
        setMessages([]);
        
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
          setMessages(response.messages);
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
        setMessages([]);
        
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
          setMessages(response.messages);
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
        setMessages([]);
        
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
          setMessages(response.messages);
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
        setMessages([]);
        
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
          setMessages(response.messages);
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
        setMessages([]);
        
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
          setMessages(response.messages);
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

  // Media upload and analysis
  const handleUploadMedia = useMutation({
    mutationFn: async (file: File) => {
      try {
        setIsAnalyzing(true);
        setAnalysisProgress(0);
        setMessages([]);
        
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
          setMessages(response.messages);
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
      handleUploadMedia.mutate(file);
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
          handleUploadMedia.mutate(file);
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
      <div className="w-48 bg-muted/30 min-h-screen p-4 space-y-2 border-r">
        <h3 className="text-sm font-semibold mb-3 text-muted-foreground">Additional Assessments</h3>
        
        <Button
          variant={selectedAnalysisType === "bigfive-text" ? "default" : "outline"}
          className="w-full justify-start text-xs h-auto py-3"
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
      </div>
      
      {/* Main Content Area */}
      <div className="flex-1">
        <div className="container mx-auto p-4 max-w-6xl">
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
                className="h-20 flex flex-col items-center justify-center text-xs" 
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
                className="h-20 flex flex-col items-center justify-center text-xs" 
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
                className="h-20 flex flex-col items-center justify-center text-xs" 
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
                className="h-20 flex flex-col items-center justify-center text-xs" 
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
                  setMessages([]);
                  
                  try {
                    const data = await analyzeMBTIText(textInput, sessionId, selectedModel);
                    
                    if (data.messages && data.messages.length > 0) {
                      setMessages(data.messages);
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
                className="h-20 flex flex-col items-center justify-center text-xs" 
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
                className="h-20 flex flex-col items-center justify-center text-xs" 
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
                className="h-20 flex flex-col items-center justify-center text-xs" 
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
                className="h-20 flex flex-col items-center justify-center text-xs" 
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
                className="h-20 flex flex-col items-center justify-center text-xs" 
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
                  onClick={() => {
                    if (mediaData) {
                      // Clear messages for new analysis
                      setMessages([]);
                      setIsAnalyzing(true);
                      setAnalysisProgress(0);
                      
                      // Use the stored media data directly
                      uploadMedia(
                        mediaData, 
                        "image", 
                        sessionId, 
                        { 
                          selectedModel, 
                          maxPeople: 5 
                        }
                      ).then(response => {
                        setAnalysisProgress(100);
                        
                        if (response && response.analysisId) {
                          setAnalysisId(response.analysisId);
                        }
                        
                        if (response && response.messages && Array.isArray(response.messages)) {
                          setMessages(response.messages);
                        }
                        
                        toast({
                          title: "Analysis Complete",
                          description: "Your image has been re-analyzed with " + 
                            (selectedModel === "openai" ? "知 2" : selectedModel === "anthropic" ? "知 1" : selectedModel === "deepseek" ? "知 3" : "知 4"),
                        });
                      }).catch(error => {
                        toast({
                          variant: "destructive",
                          title: "Error",
                          description: "Failed to re-analyze image. Please try again.",
                        });
                      }).finally(() => {
                        setIsAnalyzing(false);
                      });
                    }
                  }}
                  className="w-full"
                  disabled={isAnalyzing || !mediaData}
                >
                  Re-Analyze with {selectedModel === "openai" ? "知 2" : selectedModel === "anthropic" ? "知 1" : selectedModel === "deepseek" ? "知 3" : "知 4"}
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
                      setMessages([]);
                      
                      try {
                        const data = await analyzeMBTIImage(mediaData, sessionId, selectedModel);
                        
                        if (data.messages && data.messages.length > 0) {
                          setMessages(data.messages);
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
                  onClick={() => {
                    if (mediaData) {
                      // Clear messages for new analysis
                      setMessages([]);
                      setIsAnalyzing(true);
                      setAnalysisProgress(0);
                      
                      // Use the stored media data directly
                      uploadMedia(
                        mediaData, 
                        "video", 
                        sessionId, 
                        { 
                          selectedModel, 
                          maxPeople: 5 
                        }
                      ).then(response => {
                        setAnalysisProgress(100);
                        
                        if (response && response.analysisId) {
                          setAnalysisId(response.analysisId);
                        }
                        
                        if (response && response.messages && Array.isArray(response.messages)) {
                          setMessages(response.messages);
                        }
                        
                        toast({
                          title: "Analysis Complete",
                          description: "Your video has been re-analyzed with " + 
                            (selectedModel === "openai" ? "知 2" : selectedModel === "anthropic" ? "知 1" : selectedModel === "deepseek" ? "知 3" : "知 4"),
                        });
                      }).catch(error => {
                        toast({
                          variant: "destructive",
                          title: "Error",
                          description: "Failed to re-analyze video. Please try again.",
                        });
                      }).finally(() => {
                        setIsAnalyzing(false);
                      });
                    }
                  }}
                  className="w-full"
                  disabled={isAnalyzing || !mediaData}
                >
                  Re-Analyze with {selectedModel === "openai" ? "知 2" : selectedModel === "anthropic" ? "知 1" : selectedModel === "deepseek" ? "知 3" : "知 4"}
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
                      setMessages([]);
                      
                      try {
                        const data = await analyzeMBTIVideo(mediaData, sessionId, selectedModel);
                        
                        if (data.messages && data.messages.length > 0) {
                          setMessages(data.messages);
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
                  Re-Analyze with {selectedModel === "openai" ? "知 2" : selectedModel === "anthropic" ? "知 1" : selectedModel === "deepseek" ? "知 3" : "知 4"}
                </Button>
              </div>
            )}
            
            {!uploadedMedia && !documentName && (
              <form onSubmit={handleTextSubmit} className="space-y-4">
                <Textarea
                  value={textInput}
                  onChange={(e) => setTextInput(e.target.value)}
                  onKeyDown={(e) => handleKeyPress(e, handleTextSubmit)}
                  placeholder="Type or paste text to analyze..."
                  className="min-h-[250px] resize-y"
                  disabled={isAnalyzing}
                />
                <Button 
                  type="submit" 
                  className="w-full" 
                  disabled={!textInput.trim() || isAnalyzing}
                >
                  Analyze Text
                </Button>
                <div className="pt-4 mt-4 border-t">
                  <p className="text-sm font-semibold mb-2">MBTI Analysis (Text)</p>
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
                          setMessages(data.messages);
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
                      setMessages([]);
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
                <ScrollArea className="flex-1 pr-4 mb-4">
                  <div className="space-y-4">
                    <div className="text-xs text-muted-foreground mb-2">
                      Debug: Found {messages.length} messages, {messages.filter(m => m.role === "assistant").length} are from assistant
                    </div>
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