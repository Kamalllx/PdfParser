"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { FileText, Upload, MessageSquare, Activity, Zap, ChevronRight } from "lucide-react"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar"
import { Badge } from "@/components/ui/badge"

interface Document {
  filename: string
  upload_date: string
  total_pages: number
}

export function AppSidebar() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    fetchDocuments()
  }, [])

  const fetchDocuments = async () => {
    try {
      const response = await fetch("http://localhost:5000/documents")
      const data = await response.json()
      if (data.success) {
        setDocuments(data.documents)
      }
    } catch (error) {
      console.error("Error fetching documents:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const menuItems = [
    {
      title: "Upload Document",
      icon: Upload,
      href: "upload",
      color: "from-cyan-400 to-blue-500",
    },
    {
      title: "Document Library",
      icon: FileText,
      href: "documents",
      color: "from-purple-400 to-pink-500",
    },
    {
      title: "Ask Questions",
      icon: MessageSquare,
      href: "qa",
      color: "from-green-400 to-emerald-500",
    },
    {
      title: "System Health",
      icon: Activity,
      href: "health",
      color: "from-orange-400 to-red-500",
    },
  ]

  const handleNavigation = (href: string) => {
    // Dispatch custom event to communicate with parent component
    window.dispatchEvent(
      new CustomEvent("sidebar-navigation", {
        detail: { tab: href },
      }),
    )
  }

  return (
    <Sidebar className="border-r border-purple-500/20 bg-black/40 backdrop-blur-sm">
      <SidebarHeader className="border-b border-purple-500/20 p-4">
        <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-r from-cyan-400 to-purple-500 rounded-xl flex items-center justify-center">
            <Zap className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
              RAG System
            </h2>
            <p className="text-xs text-gray-400">Document Intelligence</p>
          </div>
        </motion.div>
      </SidebarHeader>

      <SidebarContent className="p-4">
        <SidebarGroup>
          <SidebarGroupLabel className="text-cyan-400 font-semibold mb-2">Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {menuItems.map((item, index) => (
                <motion.div
                  key={item.title}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      className="group hover:bg-white/5 transition-all duration-200 cursor-pointer"
                      onClick={() => handleNavigation(item.href)}
                    >
                      <div
                        className={`w-8 h-8 bg-gradient-to-r ${item.color} rounded-lg flex items-center justify-center group-hover:scale-110 transition-transform`}
                      >
                        <item.icon className="w-4 h-4 text-white" />
                      </div>
                      <span className="text-gray-300 group-hover:text-white transition-colors">{item.title}</span>
                      <ChevronRight className="w-4 h-4 text-gray-500 group-hover:text-cyan-400 transition-colors ml-auto" />
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                </motion.div>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup className="mt-6">
          <SidebarGroupLabel className="text-purple-400 font-semibold mb-2 flex items-center justify-between">
            Recent Documents
            <Badge variant="secondary" className="bg-purple-500/20 text-purple-300">
              {documents.length}
            </Badge>
          </SidebarGroupLabel>
          <SidebarGroupContent>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {isLoading ? (
                <div className="space-y-2">
                  {[...Array(3)].map((_, i) => (
                    <div key={i} className="h-12 bg-white/5 rounded-lg animate-pulse" />
                  ))}
                </div>
              ) : documents.length > 0 ? (
                documents.slice(0, 5).map((doc, index) => (
                  <motion.div
                    key={doc.filename}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="p-3 bg-white/5 rounded-lg border border-white/10 hover:border-purple-500/30 transition-all duration-200 cursor-pointer group"
                  >
                    <div className="flex items-center gap-2">
                      <FileText className="w-4 h-4 text-purple-400 group-hover:text-purple-300" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm text-white truncate group-hover:text-cyan-300 transition-colors">
                          {doc.filename}
                        </p>
                        <p className="text-xs text-gray-400">{doc.total_pages} pages</p>
                      </div>
                    </div>
                  </motion.div>
                ))
              ) : (
                <div className="text-center py-4">
                  <FileText className="w-8 h-8 text-gray-500 mx-auto mb-2" />
                  <p className="text-sm text-gray-400">No documents yet</p>
                </div>
              )}
            </div>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="border-t border-purple-500/20 p-4">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="flex items-center gap-2 text-xs text-gray-400"
        >
          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
          System Online
        </motion.div>
      </SidebarFooter>
    </Sidebar>
  )
}
