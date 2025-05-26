"use client"

import { useMemo } from "react"

interface MarkdownRendererProps {
  content: string
  className?: string
}

export function MarkdownRenderer({ content, className = "" }: MarkdownRendererProps) {
  const renderedContent = useMemo(() => {
    // Convert markdown to HTML-like structure
    let html = content

    // Handle headers
    html = html.replace(
      /\*\*(.*?)\*\*/g,
      '<strong class="text-cyan-300 font-semibold text-lg block mt-4 mb-2">$1</strong>',
    )

    // Handle bold text (single asterisks or remaining double)
    html = html.replace(/\*(.*?)\*/g, '<span class="text-purple-300 font-medium">$1</span>')

    // Handle equations in brackets
    html = html.replace(
      /\[([^\]]+)\]/g,
      '<span class="bg-purple-500/20 text-purple-200 px-2 py-1 rounded text-sm font-mono">$1</span>',
    )

    // Handle page references
    html = html.replace(
      /Page (\d+)/g,
      '<span class="bg-green-500/20 text-green-300 px-2 py-1 rounded text-sm font-medium">Page $1</span>',
    )

    // Handle equations with variables
    html = html.replace(
      /([A-Za-z_]+)\s*=\s*([^.\n]+)/g,
      '<div class="bg-slate-800/50 p-3 rounded-lg my-2 font-mono text-cyan-200 border-l-4 border-cyan-500">$1 = $2</div>',
    )

    // Handle bullet points
    html = html.replace(
      /^[\s]*[-*]\s+(.+)$/gm,
      '<div class="flex items-start gap-2 my-1"><span class="text-purple-400 mt-1">•</span><span class="text-gray-300">$1</span></div>',
    )

    // Handle numbered lists
    html = html.replace(
      /^[\s]*(\d+)\.\s+(.+)$/gm,
      '<div class="flex items-start gap-2 my-1"><span class="text-cyan-400 font-medium min-w-[1.5rem]">$1.</span><span class="text-gray-300">$2</span></div>',
    )

    // Handle line breaks
    html = html.replace(/\n\n/g, '<div class="my-3"></div>')
    html = html.replace(/\n/g, "<br />")

    return html
  }, [content])

  return (
    <div
      className={`prose prose-invert max-w-none ${className}`}
      dangerouslySetInnerHTML={{ __html: renderedContent }}
    />
  )
}
