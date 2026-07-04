"use client";

import { useState } from "react";
import { useStream } from "@langchain/react";
import { Bot, FileText, Search, User, Wrench } from "lucide-react";

import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { getMessageText, toolLabel } from "@/lib/messages";

import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { Message, MessageContent } from "@/components/ai-elements/message";
import { Response } from "@/components/ai-elements/response";
import { Loader } from "@/components/ai-elements/loader";
import { Suggestion, Suggestions } from "@/components/ai-elements/suggestion";
import { Tool, ToolOutput } from "@/components/ai-elements/tool";
import {
  PromptInput,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputToolbar,
} from "@/components/ai-elements/prompt-input";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "/api";

type StreamMessage = ReturnType<typeof useStream>["messages"][number];

const SUGGESTIONS = [
  "How often should I deworm my cat?",
  "What vaccinations do kittens need?",
  "What are signs of feline dehydration?",
];

function toolIcon(name?: string) {
  if (name === "retrieve_information") return <FileText className="size-4" />;
  if (name?.startsWith("tavily")) return <Search className="size-4" />;
  return <Wrench className="size-4" />;
}

export function Chat({ assistantId }: { assistantId: string }) {
  const stream = useStream({ apiUrl: API_URL, assistantId });
  const { messages, isLoading, error } = stream;

  const [input, setInput] = useState("");

  const send = (text: string) => {
    const content = text.trim();
    if (!content || isLoading) return;
    stream.submit({ messages: [{ type: "human", content }] });
    setInput("");
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <Conversation>
        <ConversationContent>
          {messages.length === 0 && (
            <ConversationEmptyState className="mt-10">
              <div className="flex flex-col items-center gap-6">
                <div className="flex size-14 items-center justify-center rounded-full bg-muted">
                  <Bot className="size-7 text-muted-foreground" />
                </div>
                <div className="space-y-1">
                  <h2 className="text-lg font-medium">Ask the cat health agent</h2>
                  <p className="text-sm text-muted-foreground">
                    Streams from your LangGraph deployment via a secure proxy.
                  </p>
                </div>
                <Suggestions>
                  {SUGGESTIONS.map((s) => (
                    <Suggestion key={s} suggestion={s} onClick={send} />
                  ))}
                </Suggestions>
              </div>
            </ConversationEmptyState>
          )}

          {messages.map((message, i) => (
            <MessageRow key={message.id ?? i} message={message} />
          ))}

          {isLoading && <ThinkingRow />}

          {error != null && (
            <Card className="border-destructive/40">
              <CardContent className="text-sm text-destructive">
                {error instanceof Error ? error.message : "Something went wrong."}
              </CardContent>
            </Card>
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      <div className="border-t bg-background">
        <div className="mx-auto w-full max-w-3xl px-4 py-3">
          <PromptInput
            onSubmit={(e) => {
              e.preventDefault();
              send(input);
            }}
          >
            <PromptInputTextarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onSubmit={() => send(input)}
              placeholder="Message the agent..."
              disabled={isLoading}
              autoFocus
            />
            <PromptInputToolbar>
              <span className="text-xs text-muted-foreground">
                Enter to send · Shift+Enter for a new line
              </span>
              <PromptInputSubmit
                status={isLoading ? "streaming" : "ready"}
                disabled={isLoading || input.trim().length === 0}
              />
            </PromptInputToolbar>
          </PromptInput>
        </div>
      </div>
    </div>
  );
}

function MessageRow({ message }: { message: StreamMessage }) {
  const isHuman = message.type === "human";
  const isTool = message.type === "tool";
  const text = getMessageText(message.content);
  const toolCalls =
    message.type === "ai"
      ? (
          message as unknown as {
            tool_calls?: Array<{ name?: string; id?: string }>;
          }
        ).tool_calls ?? []
      : [];

  if (isTool) {
    return (
      <Tool
        icon={toolIcon(message.name)}
        title={toolLabel(message.name)}
        state="completed"
      >
        <ToolOutput>{text}</ToolOutput>
      </Tool>
    );
  }

  return (
    <div className={cn("flex w-full items-start gap-3", isHuman && "flex-row-reverse")}>
      <Avatar>
        <AvatarFallback>
          {isHuman ? <User className="size-4" /> : <Bot className="size-4" />}
        </AvatarFallback>
      </Avatar>

      <Message from={isHuman ? "user" : "assistant"} className="max-w-[80%]">
        {toolCalls.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {toolCalls.map((tc, idx) => (
              <span
                key={tc.id ?? idx}
                className="inline-flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-1 text-xs font-medium text-muted-foreground [&_svg]:size-3"
              >
                {toolIcon(tc.name)}
                {toolLabel(tc.name)}
              </span>
            ))}
          </div>
        )}

        {text && (
          <MessageContent>
            {isHuman ? (
              <span className="whitespace-pre-wrap">{text}</span>
            ) : (
              <Response>{text}</Response>
            )}
          </MessageContent>
        )}
      </Message>
    </div>
  );
}

function ThinkingRow() {
  return (
    <div className="flex w-full items-start gap-3">
      <Avatar>
        <AvatarFallback>
          <Bot className="size-4" />
        </AvatarFallback>
      </Avatar>
      <div className="flex items-center gap-2 rounded-2xl bg-muted px-4 py-3 text-sm text-muted-foreground">
        <Loader size={16} />
        Thinking...
      </div>
    </div>
  );
}
