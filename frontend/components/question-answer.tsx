"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { MessageSquare, Send, FileText, Loader2, Bot, User, X } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { useToast } from "@/hooks/use-toast"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { PageScreenshots } from "@/components/page-screenshots"

interface Document {
  filename: string
  total_pages: number
}

interface QAResponse {
  question: string
  answer: string
  context: string
  references: string[]
  topics_used: string[]
  pages_used: number[]
}

interface QuestionAnswerProps {
  onQuestionAnswered?: () => void
}

export function QuestionAnswer({ onQuestionAnswered }: QuestionAnswerProps) {
  const [documents, setDocuments] = useState<Document[]>([])
  const [selectedDocument, setSelectedDocument] = useState<string>("")
  const [question, setQuestion] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [currentQA, setCurrentQA] = useState<QAResponse | null>(null)
  const { toast } = useToast()

  useEffect(() => {
    fetchDocuments()
  }, [])

  const fetchDocuments = async () => {
    try {
      const response = await fetch("http://localhost:5000/documents")
      const data = await response.json()
      if (data.success) {
        setDocuments(data.documents)
        if (data.documents.length > 0) {
          setSelectedDocument(data.documents[0].filename)
        }
      }
    } catch (error) {
      console.error("Error fetching documents:", error)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!question.trim() || !selectedDocument) return

    setIsLoading(true)
    try {
      const response = await fetch("http://localhost:5000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question.trim(),
          filename: selectedDocument,
        }),
      })

      const data = await response.json()
      if (data.success) {
        setCurrentQA(data)
        setQuestion("")
        onQuestionAnswered?.()
        toast({
          title: "Question answered!",
          description: "AI has processed your question successfully.",
        })
      } else {
        throw new Error(data.error || "Failed to get answer")
      }
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to process question",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const clearAnswer = () => {
    setCurrentQA(null)
  }

  const suggestedQuestions = [
    "What are the main topics covered in this document?",
    "Can you provide a summary of the key points?",
    "What are the most important findings or conclusions?",
    "Are there any specific recommendations mentioned?",
  ]

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      {/* Header */}
      <Card className="bg-gradient-to-br from-green-500/5 to-emerald-500/5 border-green-500/20">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-green-400">
            <MessageSquare className="w-5 h-5" />
            Ask Questions
          </CardTitle>
          <CardDescription className="text-gray-400">
            Ask AI-powered questions about your processed documents
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Question Form */}
      <Card className="bg-black/20 border-gray-700">
        <CardContent className="p-6">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="text-sm font-medium text-gray-300 mb-2 block">Select Document</label>
              <Select value={selectedDocument} onValueChange={setSelectedDocument}>
                <SelectTrigger className="bg-white/5 border-gray-600 text-white">
                  <SelectValue placeholder="Choose a document" />
                </SelectTrigger>
                <SelectContent>
                  {documents.map((doc) => (
                    <SelectItem key={doc.filename} value={doc.filename}>
                      <div className="flex items-center gap-2">
                        <FileText className="w-4 h-4" />
                        <span>{doc.filename}</span>
                        <Badge variant="secondary" className="text-xs">
                          {doc.total_pages} pages
                        </Badge>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-sm font-medium text-gray-300 mb-2 block">Your Question</label>
              <div className="flex gap-2">
                <Input
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask anything about the document..."
                  className="bg-white/5 border-gray-600 text-white placeholder:text-gray-400"
                  disabled={isLoading || !selectedDocument}
                />
                <Button
                  type="submit"
                  disabled={isLoading || !question.trim() || !selectedDocument}
                  className="bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600"
                >
                  {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                </Button>
              </div>
            </div>
          </form>

          {/* Suggested Questions */}
          {!currentQA && (
            <div className="mt-6">
              <p className="text-sm font-medium text-gray-300 mb-3">Suggested Questions:</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {suggestedQuestions.map((suggested, index) => (
                  <Button
                    key={index}
                    variant="ghost"
                    size="sm"
                    onClick={() => setQuestion(suggested)}
                    className="text-left justify-start text-gray-400 hover:text-white hover:bg-white/5 h-auto p-3"
                    disabled={!selectedDocument}
                  >
                    {suggested}
                  </Button>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Current Q&A */}
      {currentQA && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}>
          <Card className="bg-black/20 border-gray-700">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <CardTitle className="text-green-400">Current Answer</CardTitle>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={clearAnswer}
                  className="text-gray-400 hover:text-white hover:bg-white/5"
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Question */}
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full flex items-center justify-center">
                  <User className="w-4 h-4 text-white" />
                </div>
                <div className="flex-1">
                  <p className="text-white font-medium">{currentQA.question}</p>
                </div>
              </div>

              {/* Answer */}
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full flex items-center justify-center">
                  <Bot className="w-4 h-4 text-white" />
                </div>
                <div className="flex-1 space-y-4">
                  <div className="prose prose-invert max-w-none">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      className="text-gray-300"
                      components={{
                        h1: ({ children }) => (
                          <h1 className="text-2xl font-bold text-cyan-300 mt-6 mb-4">{children}</h1>
                        ),
                        h2: ({ children }) => (
                          <h2 className="text-xl font-semibold text-cyan-300 mt-5 mb-3">{children}</h2>
                        ),
                        h3: ({ children }) => (
                          <h3 className="text-lg font-medium text-purple-300 mt-4 mb-2">{children}</h3>
                        ),
                        p: ({ children }) => <p className="text-gray-300 mb-3 leading-relaxed">{children}</p>,
                        strong: ({ children }) => <strong className="text-purple-300 font-semibold">{children}</strong>,
                        em: ({ children }) => <em className="text-cyan-300">{children}</em>,
                        code: ({ children }) => (
                          <code className="bg-slate-800 text-cyan-200 px-2 py-1 rounded text-sm font-mono">
                            {children}
                          </code>
                        ),
                        pre: ({ children }) => (
                          <pre className="bg-slate-800 p-4 rounded-lg overflow-x-auto border-l-4 border-cyan-500 my-4">
                            {children}
                          </pre>
                        ),
                        ul: ({ children }) => <ul className="list-disc list-inside space-y-1 mb-3">{children}</ul>,
                        ol: ({ children }) => <ol className="list-decimal list-inside space-y-1 mb-3">{children}</ol>,
                        li: ({ children }) => <li className="text-gray-300">{children}</li>,
                        blockquote: ({ children }) => (
                          <blockquote className="border-l-4 border-purple-500 pl-4 italic text-purple-200 my-4">
                            {children}
                          </blockquote>
                        ),
                      }}
                    >
                      {currentQA.answer}
                    </ReactMarkdown>
                  </div>

                  {/* Page Screenshots */}
                  {currentQA.pages_used && currentQA.pages_used.length > 0 && (
                    <PageScreenshots filename={selectedDocument} pages={currentQA.pages_used} className="mt-4" />
                  )}

                  {/* Metadata */}
                  <div className="flex flex-wrap gap-4 text-sm pt-4 border-t border-gray-700">
                    <div className="flex items-center gap-2">
                      <span className="text-gray-400">Topics:</span>
                      <div className="flex gap-1">
                        {currentQA.topics_used.slice(0, 3).map((topic) => (
                          <Badge key={topic} variant="outline" className="text-xs border-purple-500/30 text-purple-300">
                            {topic}
                          </Badge>
                        ))}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-gray-400">Pages:</span>
                      <div className="flex gap-1">
                        {currentQA.pages_used.slice(0, 5).map((page) => (
                          <Badge key={page} variant="outline" className="text-xs border-green-500/30 text-green-300">
                            {page}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {documents.length === 0 && (
        <Card className="bg-black/20 border-gray-700">
          <CardContent className="p-12 text-center">
            <MessageSquare className="w-16 h-16 text-gray-500 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-white mb-2">No documents available</h3>
            <p className="text-gray-400 mb-4">Upload and process documents first to start asking questions</p>
            <Button className="bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600">
              Upload Document
            </Button>
          </CardContent>
        </Card>
      )}
    </motion.div>
  )
}
