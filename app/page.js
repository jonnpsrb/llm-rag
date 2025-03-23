// app/page.js
"use client"

import { useState, useRef, useEffect } from 'react';
import { Toaster, toast } from 'sonner';

export default function Home() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!input.trim()) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }

      const data = await response.json();
      const assistantMessage = { role: 'assistant', content: data.response };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
      toast.error('Failed to get response from RAG system');
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, I encountered an error while processing your request. Please try again.'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
      <div className="flex flex-col min-h-screen bg-gray-50">
        <Toaster position="top-center" />

        <header className="sticky top-0 z-10 border-b bg-white p-4 shadow-sm">
          <div className="container mx-auto flex items-center justify-between">
            <h1 className="text-xl font-bold text-blue-600">iButler Knowledge Assistant</h1>
            <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded-full">
            Powered by Llama and Groq
          </span>
          </div>
        </header>

        <main className="container mx-auto flex-1 p-4 max-w-4xl">
          <div className="bg-white rounded-lg shadow-sm border p-4 h-[calc(100vh-12rem)] flex flex-col">
            <div className="flex-1 overflow-y-auto mb-4">
              {messages.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-center p-8 text-gray-500">
                    <div className="mb-2">
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-12 h-12 text-blue-200 mx-auto mb-4">
                        <path fillRule="evenodd" d="M10.5 3.75a6.75 6.75 0 100 13.5 6.75 6.75 0 000-13.5zM2.25 10.5a8.25 8.25 0 1114.59 5.28l4.69 4.69a.75.75 0 11-1.06 1.06l-4.69-4.69A8.25 8.25 0 012.25 10.5z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-medium mb-1">Ask me anything about iButler</h3>
                    <p className="text-sm max-w-md">I'll bend space/time to retrieve the relevant results.</p>
                  </div>
              ) : (
                  <div className="space-y-4">
                    {messages.map((message, index) => (
                        <div
                            key={index}
                            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                          <div
                              className={`max-w-[80%] rounded-lg px-4 py-2 ${
                                  message.role === 'user'
                                      ? 'bg-blue-600 text-white rounded-br-none'
                                      : 'bg-gray-100 text-gray-800 rounded-bl-none'
                              }`}
                          >
                            <p className="whitespace-pre-wrap text-sm">{message.content}</p>
                          </div>
                        </div>
                    ))}
                    {isLoading && (
                        <div className="flex justify-start">
                          <div className="bg-gray-100 rounded-lg px-4 py-2 rounded-bl-none">
                            <div className="flex space-x-1 items-center">
                              <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0ms' }}></div>
                              <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '150ms' }}></div>
                              <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '300ms' }}></div>
                            </div>
                          </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                  </div>
              )}
            </div>

            <form onSubmit={handleSubmit} className="flex items-center space-x-2">
              <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask a question..."
                  className="flex-1 rounded-full border border-gray-300 px-4 py-2 outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  disabled={isLoading}
              />
              <button
                  type="submit"
                  disabled={isLoading || !input.trim()}
                  className="bg-blue-600 text-white rounded-full p-2 w-10 h-10 flex items-center justify-center hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                  <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
                </svg>
              </button>
            </form>
          </div>
        </main>
      </div>
  );
}