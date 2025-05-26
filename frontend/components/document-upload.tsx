"use client"

import { useState, useCallback } from "react"
import { motion } from "framer-motion"
import { useDropzone } from "react-dropzone"
import { Upload, FileText, CheckCircle, Loader2 } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { useToast } from "@/hooks/use-toast"

interface DocumentUploadProps {
  onUploadSuccess?: () => void
}

export function DocumentUpload({ onUploadSuccess }: DocumentUploadProps) {
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const { toast } = useToast()

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const file = acceptedFiles[0]
      if (file && file.type === "application/pdf") {
        setUploadedFile(file)
        handleUpload(file)
      } else {
        toast({
          title: "Invalid file type",
          description: "Please upload a PDF file only.",
          variant: "destructive",
        })
      }
    },
    [toast],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
    },
    multiple: false,
  })

  const handleUpload = async (file: File) => {
    setIsUploading(true)
    setUploadProgress(0)

    const formData = new FormData()
    formData.append("file", file)

    try {
      // Simulate progress
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval)
            return prev
          }
          return prev + 10
        })
      }, 200)

      const response = await fetch("http://localhost:5000/process", {
        method: "POST",
        body: formData,
      })

      clearInterval(progressInterval)
      setUploadProgress(100)

      const data = await response.json()

      if (data.success) {
        toast({
          title: "Document processed successfully!",
          description: `${file.name} has been processed and indexed.`,
        })
        onUploadSuccess?.()
      } else {
        throw new Error(data.error || "Upload failed")
      }
    } catch (error) {
      toast({
        title: "Upload failed",
        description: error instanceof Error ? error.message : "An error occurred during upload.",
        variant: "destructive",
      })
    } finally {
      setIsUploading(false)
      setTimeout(() => {
        setUploadProgress(0)
        setUploadedFile(null)
      }, 2000)
    }
  }

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      <Card className="bg-gradient-to-br from-cyan-500/5 to-purple-500/5 border-cyan-500/20">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-cyan-400">
            <Upload className="w-5 h-5" />
            Upload PDF Document
          </CardTitle>
          <CardDescription className="text-gray-400">
            Upload a PDF document to process and index for AI-powered Q&A
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all duration-300 ${
              isDragActive
                ? "border-cyan-400 bg-cyan-400/10"
                : "border-gray-600 hover:border-cyan-500 hover:bg-cyan-500/5"
            }`}
          >
            <input {...getInputProps()} />
            <motion.div animate={isDragActive ? { scale: 1.05 } : { scale: 1 }} className="space-y-4">
              {isUploading ? (
                <Loader2 className="w-12 h-12 text-cyan-400 mx-auto animate-spin" />
              ) : (
                <FileText className="w-12 h-12 text-cyan-400 mx-auto" />
              )}

              <div>
                {isUploading ? (
                  <div className="space-y-2">
                    <p className="text-lg font-medium text-white">Processing {uploadedFile?.name}...</p>
                    <Progress value={uploadProgress} className="w-full max-w-md mx-auto" />
                    <p className="text-sm text-gray-400">
                      {uploadProgress < 90 ? "Uploading..." : "Processing document..."}
                    </p>
                  </div>
                ) : isDragActive ? (
                  <p className="text-lg font-medium text-cyan-400">Drop the PDF file here...</p>
                ) : (
                  <div className="space-y-2">
                    <p className="text-lg font-medium text-white">Drag & drop a PDF file here</p>
                    <p className="text-gray-400">or click to select a file</p>
                  </div>
                )}
              </div>
            </motion.div>
          </div>

          {uploadProgress === 100 && !isUploading && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className="mt-4 p-4 bg-green-500/10 border border-green-500/20 rounded-lg flex items-center gap-2"
            >
              <CheckCircle className="w-5 h-5 text-green-400" />
              <span className="text-green-400 font-medium">Document processed successfully!</span>
            </motion.div>
          )}
        </CardContent>
      </Card>

      <Card className="bg-gradient-to-br from-purple-500/5 to-pink-500/5 border-purple-500/20">
        <CardHeader>
          <CardTitle className="text-purple-400">Processing Pipeline</CardTitle>
          <CardDescription className="text-gray-400">What happens when you upload a document</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {[
              { step: "Text Extraction", desc: "Extract text content from PDF pages" },
              { step: "Topic Analysis", desc: "AI analyzes topics and themes in each page" },
              { step: "Vectorization", desc: "Create embeddings for semantic search" },
              { step: "Indexing", desc: "Store in database for fast retrieval" },
            ].map((item, index) => (
              <motion.div
                key={item.step}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-center gap-3 p-3 bg-white/5 rounded-lg"
              >
                <div className="w-8 h-8 bg-gradient-to-r from-purple-400 to-pink-500 rounded-full flex items-center justify-center text-white font-bold text-sm">
                  {index + 1}
                </div>
                <div>
                  <p className="font-medium text-white">{item.step}</p>
                  <p className="text-sm text-gray-400">{item.desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
