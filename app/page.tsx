"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { useTheme } from "next-themes"
import { Moon, Sun, Github, Copy, Check } from "lucide-react"
import Link from "next/link"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"


type PdfContentResponse = {
  plain: string[]
  markdown: string[]
}

export default function Home() {
  const { setTheme } = useTheme()
  const [pdfUrl, setPdfUrl] = useState<string>("")
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [generatedContent, setGeneratedContent] = useState<PdfContentResponse | null>(null)
  const [copied, setCopied] = useState<"plain" | "markdown" | null>(null)

  const handlePdfUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setPdfUrl(e.target.value)
    if (error) setError(null)
  }

  const handleCancelClick = () => {
    setPdfUrl("")
    setError(null)
    setGeneratedContent(null)
  }

  const handleGenerateClick = async () => {
    if (!pdfUrl) {
      setError("Please enter a PDF URL")
      return
    }

    try {
      setIsLoading(true)
      setError(null)
      setGeneratedContent(null)

      // Call the FastAPI backend
      const response = await fetch("/api/py/pdf_extract", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url: pdfUrl, get_images: false}),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || "Failed to extract text from PDF")
      }

      const data = await response.json()
      setGeneratedContent(data.data.content)
      console.log(data.data.content)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred")
    } finally {
      setIsLoading(false)
    }
  }

  const copyToClipboard = async (type: "plain" | "markdown") => {
    if (!generatedContent) return

    const textToCopy = generatedContent[type].join("\n\n")
    await navigator.clipboard.writeText(textToCopy)

    setCopied(type)
    setTimeout(() => setCopied(null), 2000)
  }

  return (
    <div className="bg-background text-foreground min-h-screen flex flex-col flex-grow container mx-auto px-4 py-8 md:py-12">
      <header className="sticky top-0 z-10 border-b bg-background/80 backdrop-blur-sm flex justify-between py-4">
        <div className="container mx-auto px-4 mt-auto">
          <h1 className="text-xl font-semibold">PDF Text Extractor</h1>
          <p className="text-sm text-muted-foreground hidden sm:block">Extract text and markdown from PDFs</p>
        </div>
        <div className="mx-auto mt-auto">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="icon">
                <Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
                <Moon className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
                <span className="sr-only">Toggle theme</span>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => setTheme("light")}>
                Light
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setTheme("dark")}>
                Dark
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setTheme("system")}>
                System
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </header>

      <main className="flex-grow container mx-auto py-8 md:py-12 px-4">
        <div className="md:flex justify-between gap-12">
          <Card className="w-full mx-auto mb-10">
            <CardHeader>
              <CardTitle>Extract text from PDF</CardTitle>
              <CardDescription>Enter a URL to a PDF file to extract its content</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid w-full items-center gap-4">
                <div className="flex flex-col space-y-1.5">
                  <Label htmlFor="pdf-url">PDF URL</Label>
                  <Input
                    id="pdf-url"
                    type="url"
                    placeholder="https://example.com/document.pdf"
                    value={pdfUrl}
                    onChange={handlePdfUrlChange}
                    disabled={isLoading}
                  />
                </div>
                {error && <p className="text-sm text-red-600 dark:text-red-400">{error}</p>}
              </div>
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline" onClick={handleCancelClick} disabled={isLoading}>
                Clear
              </Button>
              <Button onClick={handleGenerateClick} disabled={isLoading || !pdfUrl}>
                {isLoading ? "Extracting..." : "Extract Text"}
              </Button>
            </CardFooter>
          </Card>

          <div className="w-full">
            {isLoading && (
              <Card className="w-full">
                <CardHeader>
                  <Skeleton className="h-6 bg-muted rounded w-3/4"></Skeleton>
                </CardHeader>
                <CardContent>
                  <Skeleton className="bg-muted rounded-md mb-4 w-full h-60"></Skeleton>
                  <div className="space-y-3 mt-4 p-3">
                    <Skeleton className="h-4 bg-muted rounded w-1/4"></Skeleton>
                    <Skeleton className="h-4 bg-muted rounded w-full"></Skeleton>
                    <Skeleton className="h-4 bg-muted rounded w-5/6"></Skeleton>
                  </div>
                </CardContent>
              </Card>
            )}

            {generatedContent && (
              <Card className="w-full mx-auto mb-10">
                <CardHeader>
                  <CardTitle>Extracted Content</CardTitle>
                </CardHeader>
                <CardContent>
                  <Tabs defaultValue="plain" className="w-full">
                    <TabsList className="grid w-full grid-cols-2">
                      <TabsTrigger value="plain">Plain Text</TabsTrigger>
                      <TabsTrigger value="markdown">Markdown</TabsTrigger>
                    </TabsList>
                    <TabsContent value="plain" className="mt-4">
                      <div className="relative">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="absolute top-2 right-2"
                          onClick={() => copyToClipboard("plain")}
                        >
                          {copied === "plain" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                        </Button>
                        <div className="bg-muted/50 p-4 rounded-md max-h-[400px] overflow-y-auto whitespace-pre-wrap text-sm">
                          {generatedContent.plain.join("\n\n")}
                        </div>
                      </div>
                    </TabsContent>
                    <TabsContent value="markdown" className="mt-4">
                      <div className="relative">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="absolute top-2 right-2"
                          onClick={() => copyToClipboard("markdown")}
                        >
                          {copied === "markdown" ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                        </Button>
                        <div className="bg-muted/50 p-4 rounded-md max-h-[400px] overflow-y-auto whitespace-pre-wrap text-sm">
                          {generatedContent.markdown.join("\n\n")}
                        </div>
                      </div>
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>
            )}

            {!isLoading && !generatedContent && (
              <Card className="w-full border-dashed">
                <CardContent className="flex flex-col items-center justify-center p-10 min-h-[200px]">
                  <p className="text-muted-foreground text-center">Extracted content will appear here</p>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </main>
      <footer className="top-0 z-10 border-t bg-background/80 backdrop-blur-sm flex justify-between py-2">
          <h1 className="text-sm font-semibold my-auto hidden sm:block">Built with the <a href="https://mesolitica.com/" className="underline underline-offset-4">PyMuPDF</a> library</h1>
          <Button variant="ghost" asChild>
            <Link href="https://github.com/JustinTzeJi/pdf-text-extractor">
            <Github/> Repo
            </Link>
          </Button>
      </footer>
    </div>
  );
}