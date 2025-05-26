"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Activity, CheckCircle, AlertCircle, XCircle } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"

interface HealthData {
  status: string
  timestamp: string
  version: string
}

export function HealthStatus() {
  const [health, setHealth] = useState<HealthData | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [lastChecked, setLastChecked] = useState<Date | null>(null)

  useEffect(() => {
    checkHealth()
    const interval = setInterval(checkHealth, 30000) // Check every 30 seconds
    return () => clearInterval(interval)
  }, [])

  const checkHealth = async () => {
    try {
      const response = await fetch("http://localhost:5000/health")
      const data = await response.json()
      setHealth(data)
      setLastChecked(new Date())
    } catch (error) {
      console.error("Health check failed:", error)
      setHealth(null)
    } finally {
      setIsLoading(false)
    }
  }

  const getStatusIcon = () => {
    if (isLoading) return <Activity className="w-4 h-4 animate-spin" />
    if (!health) return <XCircle className="w-4 h-4 text-red-400" />
    if (health.status === "healthy") return <CheckCircle className="w-4 h-4 text-green-400" />
    return <AlertCircle className="w-4 h-4 text-yellow-400" />
  }

  const getStatusColor = () => {
    if (isLoading) return "bg-blue-500/20 text-blue-300 border-blue-500/30"
    if (!health) return "bg-red-500/20 text-red-300 border-red-500/30"
    if (health.status === "healthy") return "bg-green-500/20 text-green-300 border-green-500/30"
    return "bg-yellow-500/20 text-yellow-300 border-yellow-500/30"
  }

  const getStatusText = () => {
    if (isLoading) return "Checking..."
    if (!health) return "Offline"
    return health.status === "healthy" ? "Online" : "Warning"
  }

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="sm" className="gap-2">
          <motion.div animate={{ scale: [1, 1.1, 1] }} transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY }}>
            {getStatusIcon()}
          </motion.div>
          <Badge variant="outline" className={getStatusColor()}>
            {getStatusText()}
          </Badge>
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80 bg-black/90 border-gray-700" align="end">
        <Card className="border-0 bg-transparent">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <Activity className="w-4 h-4 text-cyan-400" />
              System Health
            </CardTitle>
            <CardDescription>Backend service status and information</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Status:</span>
              <div className="flex items-center gap-2">
                {getStatusIcon()}
                <span className="text-sm font-medium text-white">{getStatusText()}</span>
              </div>
            </div>

            {health && (
              <>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">Version:</span>
                  <span className="text-sm text-white">{health.version}</span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-400">Server Time:</span>
                  <span className="text-sm text-white">{new Date(health.timestamp).toLocaleTimeString()}</span>
                </div>
              </>
            )}

            {lastChecked && (
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-400">Last Checked:</span>
                <span className="text-sm text-white">{lastChecked.toLocaleTimeString()}</span>
              </div>
            )}

            <div className="pt-2 border-t border-gray-700">
              <Button
                variant="outline"
                size="sm"
                onClick={checkHealth}
                className="w-full text-cyan-400 border-cyan-500/30 hover:bg-cyan-500/10"
              >
                Refresh Status
              </Button>
            </div>
          </CardContent>
        </Card>
      </PopoverContent>
    </Popover>
  )
}
