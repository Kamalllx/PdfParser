"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Upload, FileText, MessageSquare, Activity, Zap } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { SidebarTrigger } from "@/components/ui/sidebar"
import { DocumentUpload } from "@/components/document-upload"
import { DocumentList } from "@/components/document-list"
import { QuestionAnswer } from "@/components/question-answer"
import { HealthStatus } from "@/components/health-status"
import { SystemHealthDashboard } from "@/components/system-health-dashboard"

export default function HomePage() {
  const [activeTab, setActiveTab] = useState("upload")
  const [stats, setStats] = useState({
    totalDocuments: 0,
    questionsAnswered: 0,
    systemHealth: "healthy",
  })

  useEffect(() => {
    // Fetch initial stats
    fetchStats()
  }, [])

  useEffect(() => {
    const handleSidebarNavigation = (event: CustomEvent) => {
      setActiveTab(event.detail.tab)
    }

    window.addEventListener("sidebar-navigation", handleSidebarNavigation as EventListener)

    return () => {
      window.removeEventListener("sidebar-navigation", handleSidebarNavigation as EventListener)
    }
  }, [])

  const fetchStats = async () => {
    try {
      const response = await fetch("http://localhost:5000/documents")
      const data = await response.json()
      if (data.success) {
        setStats((prev) => ({
          ...prev,
          totalDocuments: data.total,
        }))
      }
    } catch (error) {
      console.error("Error fetching stats:", error)
    }
  }

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  }

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: "spring",
        stiffness: 100,
      },
    },
  }

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="border-b border-purple-500/20 bg-black/20 backdrop-blur-sm">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center gap-4">
            <SidebarTrigger />
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex items-center gap-2"
            >
              <div className="w-8 h-8 bg-gradient-to-r from-cyan-400 to-purple-500 rounded-lg flex items-center justify-center">
                <Zap className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                  RAG Processor
                </h1>
                <p className="text-xs text-gray-400">AI Document Intelligence</p>
              </div>
            </motion.div>
          </div>
          <HealthStatus />
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 overflow-auto p-6">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="max-w-7xl mx-auto space-y-6"
        >
          {/* Stats Cards */}
          <motion.div variants={itemVariants} className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card className="bg-gradient-to-br from-cyan-500/10 to-blue-500/10 border-cyan-500/20">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-cyan-400">Total Documents</CardTitle>
                <FileText className="h-4 w-4 text-cyan-400" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-white">{stats.totalDocuments}</div>
                <p className="text-xs text-gray-400">Processed and indexed</p>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 border-purple-500/20">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-purple-400">Questions Answered</CardTitle>
                <MessageSquare className="h-4 w-4 text-purple-400" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-white">{stats.questionsAnswered}</div>
                <p className="text-xs text-gray-400">AI responses generated</p>
              </CardContent>
            </Card>

            <Card className="bg-gradient-to-br from-green-500/10 to-emerald-500/10 border-green-500/20">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-green-400">System Status</CardTitle>
                <Activity className="h-4 w-4 text-green-400" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-white capitalize">{stats.systemHealth}</div>
                <p className="text-xs text-gray-400">All systems operational</p>
              </CardContent>
            </Card>
          </motion.div>

          {/* Tab Navigation */}
          <motion.div
            variants={itemVariants}
            className="flex space-x-1 bg-black/20 p-1 rounded-lg border border-purple-500/20"
          >
            {[
              { id: "upload", label: "Upload Document", icon: Upload },
              { id: "documents", label: "Document Library", icon: FileText },
              { id: "qa", label: "Ask Questions", icon: MessageSquare },
              { id: "health", label: "System Health", icon: Activity },
            ].map((tab) => (
              <Button
                key={tab.id}
                variant={activeTab === tab.id ? "default" : "ghost"}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 flex items-center gap-2 transition-all duration-200 ${
                  activeTab === tab.id
                    ? "bg-gradient-to-r from-cyan-500 to-purple-500 text-white"
                    : "text-gray-400 hover:text-white hover:bg-white/5"
                }`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </Button>
            ))}
          </motion.div>

          {/* Tab Content */}
          <motion.div
            variants={itemVariants}
            key={activeTab}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            {activeTab === "upload" && <DocumentUpload onUploadSuccess={fetchStats} />}
            {activeTab === "documents" && <DocumentList />}
            {activeTab === "qa" && (
              <QuestionAnswer
                onQuestionAnswered={() =>
                  setStats((prev) => ({ ...prev, questionsAnswered: prev.questionsAnswered + 1 }))
                }
              />
            )}
            {activeTab === "health" && <SystemHealthDashboard />}
          </motion.div>
        </motion.div>
      </div>
    </div>
  )
}
