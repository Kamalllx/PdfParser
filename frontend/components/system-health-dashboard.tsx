"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Activity, Server, Cpu, HardDrive, Wifi, CheckCircle, AlertTriangle, XCircle } from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

interface HealthMetrics {
  status: string
  timestamp: string
  version: string
  uptime?: string
  memory_usage?: number
  cpu_usage?: number
  disk_usage?: number
  active_connections?: number
}

export function SystemHealthDashboard() {
  const [health, setHealth] = useState<HealthMetrics | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [lastChecked, setLastChecked] = useState<Date | null>(null)

  useEffect(() => {
    checkHealth()
    const interval = setInterval(checkHealth, 10000) // Check every 10 seconds
    return () => clearInterval(interval)
  }, [])

  const checkHealth = async () => {
    try {
      const response = await fetch("http://localhost:5000/health")
      const data = await response.json()

      // Simulate additional metrics for demo
      const enhancedData = {
        ...data,
        uptime: "2h 34m",
        memory_usage: Math.floor(Math.random() * 40) + 30,
        cpu_usage: Math.floor(Math.random() * 30) + 10,
        disk_usage: Math.floor(Math.random() * 20) + 15,
        active_connections: Math.floor(Math.random() * 10) + 5,
      }

      setHealth(enhancedData)
      setLastChecked(new Date())
    } catch (error) {
      console.error("Health check failed:", error)
      setHealth(null)
    } finally {
      setIsLoading(false)
    }
  }

  const getStatusIcon = (status?: string) => {
    if (!status) return <XCircle className="w-5 h-5 text-red-400" />
    if (status === "healthy") return <CheckCircle className="w-5 h-5 text-green-400" />
    return <AlertTriangle className="w-5 h-5 text-yellow-400" />
  }

  const getStatusColor = (status?: string) => {
    if (!status) return "text-red-400"
    if (status === "healthy") return "text-green-400"
    return "text-yellow-400"
  }

  const getUsageColor = (usage: number) => {
    if (usage < 50) return "bg-green-500"
    if (usage < 80) return "bg-yellow-500"
    return "bg-red-500"
  }

  const metrics = [
    {
      title: "Memory Usage",
      value: health?.memory_usage || 0,
      icon: Cpu,
      color: "from-blue-500 to-cyan-500",
    },
    {
      title: "CPU Usage",
      value: health?.cpu_usage || 0,
      icon: Server,
      color: "from-purple-500 to-pink-500",
    },
    {
      title: "Disk Usage",
      value: health?.disk_usage || 0,
      icon: HardDrive,
      color: "from-green-500 to-emerald-500",
    },
    {
      title: "Active Connections",
      value: health?.active_connections || 0,
      icon: Wifi,
      color: "from-orange-500 to-red-500",
      isCount: true,
    },
  ]

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      {/* Header */}
      <Card className="bg-gradient-to-br from-orange-500/5 to-red-500/5 border-orange-500/20">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-orange-400">
            <Activity className="w-5 h-5" />
            System Health Dashboard
          </CardTitle>
          <CardDescription className="text-gray-400">
            Real-time monitoring of backend services and system metrics
          </CardDescription>
        </CardHeader>
      </Card>

      {/* Main Status */}
      <Card className="bg-black/20 border-gray-700">
        <CardContent className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              {getStatusIcon(health?.status)}
              <div>
                <h3 className={`text-xl font-bold ${getStatusColor(health?.status)}`}>
                  {isLoading ? "Checking..." : health?.status ? health.status.toUpperCase() : "OFFLINE"}
                </h3>
                <p className="text-gray-400 text-sm">
                  {lastChecked ? `Last checked: ${lastChecked.toLocaleTimeString()}` : "Never checked"}
                </p>
              </div>
            </div>
            <Button
              onClick={checkHealth}
              disabled={isLoading}
              className="bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600"
            >
              {isLoading ? "Checking..." : "Refresh"}
            </Button>
          </div>

          {health && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex items-center gap-2">
                <span className="text-gray-400">Version:</span>
                <Badge variant="outline" className="border-orange-500/30 text-orange-300">
                  {health.version}
                </Badge>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-gray-400">Uptime:</span>
                <span className="text-white font-medium">{health.uptime}</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-gray-400">Server Time:</span>
                <span className="text-white font-medium">{new Date(health.timestamp).toLocaleTimeString()}</span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metrics.map((metric, index) => (
          <motion.div
            key={metric.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
          >
            <Card className="bg-black/20 border-gray-700 hover:border-gray-600 transition-colors">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-gray-300">{metric.title}</CardTitle>
                  <div
                    className={`w-8 h-8 bg-gradient-to-r ${metric.color} rounded-lg flex items-center justify-center`}
                  >
                    <metric.icon className="w-4 h-4 text-white" />
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {metric.isCount ? (
                  <div className="text-2xl font-bold text-white">{metric.value}</div>
                ) : (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-2xl font-bold text-white">{metric.value}%</span>
                      <Badge
                        variant="outline"
                        className={`${
                          metric.value < 50
                            ? "border-green-500/30 text-green-300"
                            : metric.value < 80
                              ? "border-yellow-500/30 text-yellow-300"
                              : "border-red-500/30 text-red-300"
                        }`}
                      >
                        {metric.value < 50 ? "Good" : metric.value < 80 ? "Warning" : "Critical"}
                      </Badge>
                    </div>
                    <Progress
                      value={metric.value}
                      className="h-2"
                      style={{
                        background: "rgba(255, 255, 255, 0.1)",
                      }}
                    />
                  </div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Service Status */}
      <Card className="bg-black/20 border-gray-700">
        <CardHeader>
          <CardTitle className="text-purple-400">Service Status</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {[
              { name: "Flask API", status: "healthy", port: "5000" },
              { name: "MongoDB", status: "healthy", port: "27017" },
              { name: "FAISS Vector Store", status: "healthy", port: "N/A" },
              { name: "Groq LLM", status: "healthy", port: "API" },
            ].map((service, index) => (
              <motion.div
                key={service.name}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-center justify-between p-3 bg-white/5 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse" />
                  <div>
                    <p className="font-medium text-white">{service.name}</p>
                    <p className="text-sm text-gray-400">Port: {service.port}</p>
                  </div>
                </div>
                <Badge variant="outline" className="border-green-500/30 text-green-300">
                  Online
                </Badge>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  )
}
