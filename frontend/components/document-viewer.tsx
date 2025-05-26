"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { ArrowLeft, FileText, Hash, Eye, ChevronDown, ChevronRight } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"

interface DocumentViewerProps {
  filename: string
  onBack: () => void
}

interface DocumentTopics {
  filename: string
  table_of_contents: Record<string, number[]>
  total_pages: number
}

interface PageContent {
  page_number: number
  content: string
  topics: string[]
  summary: string
  keywords: string[]
}

export function DocumentViewer({ filename, onBack }: DocumentViewerProps) {
  const [topics, setTopics] = useState<DocumentTopics | null>(null)
  const [selectedPage, setSelectedPage] = useState<number | null>(null)
  const [pageContent, setPageContent] = useState<PageContent | null>(null)
  const [isLoadingTopics, setIsLoadingTopics] = useState(true)
  const [isLoadingPage, setIsLoadingPage] = useState(false)
  const [expandedTopics, setExpandedTopics] = useState<Set<string>>(new Set())

  useEffect(() => {
    fetchTopics()
  }, [filename])

  const fetchTopics = async () => {
    try {
      const response = await fetch(`http://localhost:5000/document/${filename}/topics`)
      const data = await response.json()
      if (data.success) {
        setTopics(data)
      }
    } catch (error) {
      console.error("Error fetching topics:", error)
    } finally {
      setIsLoadingTopics(false)
    }
  }

  const fetchPageContent = async (pageNumber: number) => {
    setIsLoadingPage(true)
    try {
      const response = await fetch(`http://localhost:5000/document/${filename}/page/${pageNumber}`)
      const data = await response.json()
      if (data.success) {
        setPageContent(data)
        setSelectedPage(pageNumber)
      }
    } catch (error) {
      console.error("Error fetching page content:", error)
    } finally {
      setIsLoadingPage(false)
    }
  }

  const toggleTopic = (topic: string) => {
    const newExpanded = new Set(expandedTopics)
    if (newExpanded.has(topic)) {
      newExpanded.delete(topic)
    } else {
      newExpanded.add(topic)
    }
    setExpandedTopics(newExpanded)
  }

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      {/* Header */}
      <Card className="bg-gradient-to-br from-cyan-500/5 to-purple-500/5 border-cyan-500/20">
        <CardHeader>
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={onBack}
              className="text-cyan-400 hover:text-cyan-300 hover:bg-cyan-500/10"
            >
              <ArrowLeft className="w-4 h-4 mr-1" />
              Back
            </Button>
            <div>
              <CardTitle className="flex items-center gap-2 text-cyan-400">
                <FileText className="w-5 h-5" />
                {filename}
              </CardTitle>
              <CardDescription className="text-gray-400">
                {topics && `${topics.total_pages} pages • ${Object.keys(topics.table_of_contents).length} topics`}
              </CardDescription>
            </div>
          </div>
        </CardHeader>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Topics Sidebar */}
        <div className="lg:col-span-1">
          <Card className="bg-black/20 border-gray-700">
            <CardHeader>
              <CardTitle className="text-purple-400">Table of Contents</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              {isLoadingTopics ? (
                <div className="p-4 space-y-2">
                  {[...Array(5)].map((_, i) => (
                    <div key={i} className="h-8 bg-gray-700 rounded animate-pulse" />
                  ))}
                </div>
              ) : topics ? (
                <div className="max-h-96 overflow-y-auto">
                  {Object.entries(topics.table_of_contents).map(([topic, pages]) => (
                    <Collapsible key={topic} open={expandedTopics.has(topic)} onOpenChange={() => toggleTopic(topic)}>
                      <CollapsibleTrigger className="w-full p-3 text-left hover:bg-white/5 transition-colors border-b border-gray-700 last:border-b-0">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-white capitalize">{topic}</span>
                            <Badge variant="secondary" className="text-xs">
                              {pages.length}
                            </Badge>
                          </div>
                          {expandedTopics.has(topic) ? (
                            <ChevronDown className="w-4 h-4 text-gray-400" />
                          ) : (
                            <ChevronRight className="w-4 h-4 text-gray-400" />
                          )}
                        </div>
                      </CollapsibleTrigger>
                      <CollapsibleContent>
                        <div className="px-3 pb-2">
                          <div className="flex flex-wrap gap-1">
                            {pages.map((page) => (
                              <Button
                                key={page}
                                variant="ghost"
                                size="sm"
                                onClick={() => fetchPageContent(page)}
                                className={`text-xs h-6 px-2 ${
                                  selectedPage === page
                                    ? "bg-purple-500/20 text-purple-300"
                                    : "text-gray-400 hover:text-white hover:bg-white/5"
                                }`}
                              >
                                {page}
                              </Button>
                            ))}
                          </div>
                        </div>
                      </CollapsibleContent>
                    </Collapsible>
                  ))}
                </div>
              ) : (
                <div className="p-4 text-center text-gray-400">No topics found</div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Page Content */}
        <div className="lg:col-span-2">
          {selectedPage ? (
            <Card className="bg-black/20 border-gray-700">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-green-400">
                  <Hash className="w-5 h-5" />
                  Page {selectedPage}
                </CardTitle>
                {pageContent && <CardDescription className="text-gray-400">{pageContent.summary}</CardDescription>}
              </CardHeader>
              <CardContent>
                {isLoadingPage ? (
                  <div className="space-y-4">
                    <div className="h-4 bg-gray-700 rounded animate-pulse" />
                    <div className="h-4 bg-gray-700 rounded w-3/4 animate-pulse" />
                    <div className="h-4 bg-gray-700 rounded w-1/2 animate-pulse" />
                  </div>
                ) : pageContent ? (
                  <div className="space-y-6">
                    {/* Keywords */}
                    <div>
                      <h4 className="text-sm font-medium text-cyan-400 mb-2">Keywords</h4>
                      <div className="flex flex-wrap gap-2">
                        {pageContent.keywords.map((keyword) => (
                          <Badge key={keyword} variant="outline" className="text-xs border-cyan-500/30 text-cyan-300">
                            {keyword}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    {/* Topics */}
                    <div>
                      <h4 className="text-sm font-medium text-purple-400 mb-2">Topics</h4>
                      <div className="flex flex-wrap gap-2">
                        {pageContent.topics.map((topic) => (
                          <Badge key={topic} variant="outline" className="text-xs border-purple-500/30 text-purple-300">
                            {topic}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    {/* Content */}
                    <div>
                      <h4 className="text-sm font-medium text-green-400 mb-2">Content</h4>
                      <div className="bg-white/5 rounded-lg p-4 max-h-96 overflow-y-auto">
                        <pre className="text-sm text-gray-300 whitespace-pre-wrap font-mono">{pageContent.content}</pre>
                      </div>
                    </div>
                  </div>
                ) : null}
              </CardContent>
            </Card>
          ) : (
            <Card className="bg-black/20 border-gray-700">
              <CardContent className="p-12 text-center">
                <Eye className="w-16 h-16 text-gray-500 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-white mb-2">Select a page to view</h3>
                <p className="text-gray-400">Click on a page number from the topics to view its content</p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </motion.div>
  )
}
