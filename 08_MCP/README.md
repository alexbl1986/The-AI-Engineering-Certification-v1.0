<p align="center" draggable="false"><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719"
     width="200px"
     height="auto"/>
</p>

<h1 align="center" id="heading">Session 8: Model Context Protocol (MCP)</h1>

### [Quicklinks]()

| Session Sheet | Recording | Slides | Repo | Homework | Feedback |
|:--------------|:----------|:-------|:-----|:---------|:---------|
| [Session 8: MCP](https://github.com/AI-Maker-Space/The-AI-Engineering-Certification-v1.0/tree/main/00_Docs/Modules/08_MCP) |[Recording!](https://us02web.zoom.us/rec/share/rqw5I5hwbOOHy8TrGjnu0IjDJi53ykHb0k897jYfyHqZpgRhUuFP4A18d4NrcEKS.18sNk6Do9XwyaVUy) <br> passcode: `E56&^V+8`| [Session 8 Slides](https://canva.link/k8cixqgkfeghdsn) |You are here! | [Session 8 Assignment](https://forms.gle/TcjjChq38ydMjuqn8) | [Feedback 6/25](https://forms.gle/DvcWDgBXatBWCXqi7) |

## Useful Resources

**MCP (Model Context Protocol)**
- [MCP Official Docs](https://modelcontextprotocol.io/) — Spec, tutorials, and guides
- [MCP-UI](https://mcpui.dev/) — Official standard for interactive UI in MCP
- [MCP Auth Guide (Auth0)](https://auth0.com/blog/mcp-specs-update-all-about-auth/) — Deep dive into MCP auth spec updates

## Main Assignment

In this session, you will build an MCP server with OAuth authentication — a cat
shop application that exposes tools for browsing products, managing a cart, and
checking out.

The main entry point is:

```text
server.py
```

The server implementation lives in:

```text
app/
```

Available MCP tools:

- `list_products`
- `get_product`
- `add_to_cart`
- `view_cart`
- `remove_from_cart`
- `checkout`

## Setup

From this folder:

```bash
uv sync
```

Copy the example env file and fill in your OpenAI API key:

```bash
cp .env.example .env
```

## Running the MCP Server

Run the server locally:

```bash
uv run server.py
```

The server starts on `http://localhost:8000`.

### Expose the server with ngrok

In a separate terminal, start an ngrok tunnel:

```bash
ngrok http 8000
```

Copy the ngrok forwarding URL (e.g. `https://xxxx-xx-xx-xx-xx.ngrok-free.app`) and
restart the server with it:

```bash
ISSUER_URL=https://xxxx-xx-xx-xx-xx.ngrok-free.app uv run server.py
```

> **Note:** The `ISSUER_URL` must match the public URL clients use to reach the
> server, otherwise OAuth authentication will fail.

## Outline

### Breakout Room #1

- Set up the MCP server with OAuth and the product database
- Explore the MCP tools: `list_products`, `get_product`, `add_to_cart`, `view_cart`, `remove_from_cart`, `checkout`

### Breakout Room #2

- Connect an MCP client to the server
- Build an end-to-end interaction flow using the MCP tools

## Ship

The completed MCP server and client integration!

### Deliverables

- A short Loom of either:
  - the MCP server you built and a demo of the client interacting with it; or
  - the notebook you created for the Advanced Build

## Share

Make a social media post about your final application!

### Deliverables

- Make a post on any social media platform about what you built!

Here's a template to get you started:

```
🚀 Exciting News! 🚀

I am thrilled to announce that I have just built and shipped an MCP server with OAuth authentication! 🎉🤖

🔍 Three Key Takeaways:
1️⃣
2️⃣
3️⃣

Let's continue pushing the boundaries of what's possible in the world of AI and tool integration. Here's to many more innovations! 🚀
Shout out to @AIMakerspace !

#MCP #ModelContextProtocol #OAuth #Innovation #AI #TechMilestone

Feel free to reach out if you're curious or would like to collaborate on similar projects! 🤝🔥
```

## Submitting Your Homework 

Follow these steps to prepare and submit your homework assignment:

1. Review the MCP server code in `server.py` and the `app/` directory
2. Run the MCP server locally using `uv run server.py`
3. Connect to the server using an MCP client (e.g., Claude Desktop, or a custom client)
4. Test all available tools: browsing products, adding to cart, viewing cart, removing items, and checkout
5. Record a Loom video reviewing what you have learned from this session

## Questions

### Question #1

Why is OAuth important for MCP servers, and what security considerations should you keep in mind when exposing tools to AI clients?

#### Answer

OAuth is not only important for MCP servers, but became the spec recommended authorization framework when a client connects to an MCP over HTTP. Per the MCP specifications (https://modelcontextprotocol.io/specification/2025-06-18/changelog), MCP servers are classified as "OAuth Resource Servers", meaning they hold the necessary discoverable resource metadata for MCP clients to connect to them securely. 
- OAuth enables an MCP server to register different clients trying to connect to them and authorize them. It can also distinguish between different user credentials for each client, isolating their private information (for our cat shop mcp example, different users can connect to it using chatgpt, but each user will have its own cart status, checkout, etc. and won't have access to other user's carts) and permissions. When a user wants to connect an MCP server to his client, he enters his credentials, and according to those credentials an authorization code is created, which is then exchanged for a token allowing the user access to the MCP tools/resources according to his permissions. 

- When exposing tools to AI clients, the following security considerations should be addressed:
  - A login should have a username and password. Currently anyone can login with anyone else's username.
  - Scope enforcement. In the current implementation, scope permissions are defined but not checked when tools are called. For example, if a user only has read permissions, he should be denied from the tool add_to_cart, but this check doesn't occur.
  - Token storage: In the current implementation, tokens aren't encrypted/hashed, so if the sql database is stolen, the tokens can be used to get access to the mcp server user-associated tools/data.
  - Redirect URI validation: An attacker can issue an authorization and send a login link to a user (with a redirect_uri set to the attacker's address), which will enable him to steal an authorization code associated with the user, when he tries to login. PKCE won't save the user because the authorization request didn't come from him. By validating that the redirect URI is associated with the registered client, this can be prevented when the MCP server receives the /authorize request. 
  - Prompt injection. Because the llm decides which tools to call from the server, a malicious actor can, for our cat shop example, inject a checkout tool call as one of the items in the shop. Which is why before sensitive tool calls like checkout:
    - user-in-the-loop verification should be added.
    - The system prompt of the client should contain a message not to call sensitive write tools if a request is not an explicit user message.
    - Tool results can be audited at the server level to check the data is of the intended structure.
  - Rate limiting: tool usage should be rate limited according to the server's capabilities and reasonable user request rate.
  

### Question #2

What is Streamable HTTP transport in MCP, and why might you expose a server publicly with OAuth instead of using a local stdio connection?

#### Answer

- Streamable HTTP transport in MCP is a way to send messages between an MCP server and clients. A client can send HTTP GET/POST requests and the server can stream SSE (server-sent event) responses and supports stateless operation. This means the server can be running on multiple machines, and serving multiple clients. All machines have a shared database, and when its updated, any client which reconnects later to a different machine (for load management) will get the correct and updated data according to the logged in user.
- I would expose a server publicly with OAuth instead of a local stdio connection (all communication is made locally between the user and server which live on the same machine, so trust intrinsic) when I'd like clients which don't live on my local machine to connect to the server (for example allowing chatGPT to connect to my server), but then as its exposed to the internet, authorization is necessary to control anyone who tries to connect to the MCP server, authenticate them, isolate their data/permissions, and all the other necessary features described in answer to question 1.

## Activity 1: Extend the MCP Server

Add at least one new tool to the cat shop MCP server (e.g., `search_products`, `update_cart_quantity`, or `get_order_history`). Ensure the new tool integrates properly with the existing database and OAuth authentication. Demo the new tool through an MCP client and include it in your Loom video.

## Advanced Activity: Build a Custom MCP Client

Build a custom MCP client that connects to the cat shop server over Streamable HTTP, authenticates via OAuth, and orchestrates a multi-step shopping flow (browse → add to cart → checkout). Compare the developer experience of MCP-based tool integration vs. traditional REST API calls.

Include your findings and a demo in your Loom video.
