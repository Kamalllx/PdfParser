"use client"

import { useState, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { ImageIcon, ZoomIn, ChevronLeft, ChevronRight, Loader2 } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"

interface PageScreenshotsProps {
  filename: string
  pages: number[]
  className?: string
}

interface Screenshot {
  filename: string
  page_number: number
  path: string
}

export function PageScreenshots({ filename, pages, className = "" }: PageScreenshotsProps) {
  const [screenshots, setScreenshots] = useState<Screenshot[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedImage, setSelectedImage] = useState<number | null>(null)
  const [currentImageIndex, setCurrentImageIndex] = useState(0)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchScreenshots()
  }, [filename, pages])

  const fetchScreenshots = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await fetch(`http://localhost:5000/document/${filename}/screenshots`)
      const data = await response.json()

      if (data.success) {
        // Filter screenshots to only include the pages we need
        const filteredScreenshots = data.screenshots.filter((screenshot: Screenshot) =>
          pages.includes(screenshot.page_number),
        )
        setScreenshots(filteredScreenshots)
      } else {
        setError(data.error || "Failed to fetch screenshots")
      }
    } catch (error) {
      console.error("Error fetching screenshots:", error)
      setError("Failed to load screenshots")
    } finally {
      setIsLoading(false)
    }
  }

  const openImageModal = (page: number) => {
    const index = screenshots.findIndex((img) => img.page_number === page)
    setCurrentImageIndex(index)
    setSelectedImage(page)
  }

  const navigateImage = (direction: "prev" | "next") => {
    const newIndex =
      direction === "prev"
        ? (currentImageIndex - 1 + screenshots.length) % screenshots.length
        : (currentImageIndex + 1) % screenshots.length

    setCurrentImageIndex(newIndex)
    setSelectedImage(screenshots[newIndex].page_number)
  }

  // Helper function to get the correct image URL
  const getImageUrl = (screenshot: Screenshot) => {
    // Use the direct context path since we added a route to serve files from there
    return `http://localhost:5000/context/${screenshot.filename}`
  }

  if (pages.length === 0) return null

  return (
    <>
      <Card className={`bg-gradient-to-br from-indigo-500/5 to-purple-500/5 border-indigo-500/20 ${className}`}>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-indigo-400 text-sm">
            <ImageIcon className="w-4 h-4" />
            Referenced Pages
            <Badge variant="secondary" className="bg-indigo-500/20 text-indigo-300">
              {pages.length} pages
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-8 h-8 text-indigo-400 animate-spin" />
              <span className="ml-2 text-gray-400">Loading screenshots...</span>
            </div>
          ) : error ? (
            <div className="text-center py-8">
              <ImageIcon className="w-12 h-12 text-gray-500 mx-auto mb-2" />
              <p className="text-red-400 text-sm">{error}</p>
              <Button
                variant="outline"
                size="sm"
                onClick={fetchScreenshots}
                className="mt-2 text-indigo-400 border-indigo-500/30"
              >
                Retry
              </Button>
            </div>
          ) : screenshots.length === 0 ? (
            <div className="text-center py-8">
              <ImageIcon className="w-12 h-12 text-gray-500 mx-auto mb-2" />
              <p className="text-gray-400 text-sm">No screenshots available</p>
              <p className="text-gray-500 text-xs mt-1">Screenshots will appear after asking a question</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
              {screenshots.map((screenshot, index) => (
                <motion.div
                  key={screenshot.page_number}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.1 }}
                  className="relative group cursor-pointer"
                  onClick={() => openImageModal(screenshot.page_number)}
                >
                  <div className="aspect-[3/4] bg-slate-800/50 rounded-lg border border-slate-700 overflow-hidden hover:border-indigo-500/50 transition-all duration-300 group-hover:scale-105">
                    <div className="relative w-full h-full">
                      <img
                        src={getImageUrl(screenshot) || "/placeholder.svg"}
                        alt={`Page ${screenshot.page_number}`}
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          const target = e.target as HTMLImageElement
                          // Try alternative URL format
                          const altUrl = `http://localhost:5000/context/${screenshot.filename}`
                          if (target.src !== altUrl) {
                            target.src = altUrl
                          } else {
                            target.src = `/placeholder.svg?height=400&width=300&text=Page ${screenshot.page_number}`
                          }
                        }}
                        onLoad={() => {
                          console.log(`✅ Screenshot loaded: ${screenshot.filename}`)
                        }}
                      />
                      <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors duration-300 flex items-center justify-center">
                        <ZoomIn className="w-6 h-6 text-white opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                      </div>
                    </div>
                  </div>
                  <div className="absolute -bottom-2 left-1/2 transform -translate-x-1/2">
                    <Badge variant="secondary" className="bg-indigo-500/90 text-white text-xs px-2 py-1 shadow-lg">
                      Page {screenshot.page_number}
                    </Badge>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Image Modal */}
      <Dialog open={selectedImage !== null} onOpenChange={() => setSelectedImage(null)}>
        <DialogContent className="max-w-4xl bg-black/95 border-slate-700">
          <DialogHeader>
            <DialogTitle className="flex items-center justify-between text-indigo-400">
              <span>
                Page {selectedImage} - {filename}
              </span>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => navigateImage("prev")}
                  disabled={screenshots.length <= 1}
                  className="text-indigo-400 hover:text-indigo-300"
                >
                  <ChevronLeft className="w-4 h-4" />
                </Button>
                <span className="text-sm text-gray-400">
                  {currentImageIndex + 1} of {screenshots.length}
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => navigateImage("next")}
                  disabled={screenshots.length <= 1}
                  className="text-indigo-400 hover:text-indigo-300"
                >
                  <ChevronRight className="w-4 h-4" />
                </Button>
              </div>
            </DialogTitle>
          </DialogHeader>
          <div className="relative">
            <AnimatePresence mode="wait">
              {selectedImage && screenshots[currentImageIndex] && (
                <motion.div
                  key={selectedImage}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.9 }}
                  transition={{ duration: 0.3 }}
                  className="flex justify-center"
                >
                  <img
                    src={getImageUrl(screenshots[currentImageIndex]) || "/placeholder.svg"}
                    alt={`Page ${selectedImage}`}
                    className="max-w-full max-h-[70vh] object-contain rounded-lg border border-slate-600"
                    onError={(e) => {
                      const target = e.target as HTMLImageElement
                      const altUrl = `http://localhost:5000/context/${screenshots[currentImageIndex].filename}`
                      if (target.src !== altUrl) {
                        target.src = altUrl
                      } else {
                        target.src = `/placeholder.svg?height=600&width=400&text=Page ${selectedImage}`
                      }
                    }}
                  />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </DialogContent>
      </Dialog>
    </>
  )
}
