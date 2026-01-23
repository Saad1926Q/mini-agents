<h1 align="center">mini-agents</h1>

<p align="center">
Trying to build agents from scratch for fine-grained control
</p>


---

## Motivation

I’ve heard from a lot of people that it’s better to build agents from scratch instead of relying on frameworks like LangGraph and similar tools.  
The main argument is that these frameworks abstract away too many details, which can be limiting when you want fine-grained control over agent behavior.

So, I decided to give it a try myself.

To understand how things work at a lower level, I went through parts of the
[openai-agents-python](https://github.com/openai/openai-agents-python) codebase and used it as a reference for ideas and patterns.

> **Note:** This is not “from-scratch scratch.”  
> I will of course be using the OpenAI API to access an LLM.  
> The focus here is on building the *agent architecture* and control flow myself, rather than relying on higher-level agent frameworks.

## What This Repo Is About

In this repository, I’m trying to implement some of the most fundamental building blocks of agent systems:

- [ ] **Agent Loop**
- [ ] **Tools**
- [ ] **Guardrails**
- [ ] **Handoffs**

## End Goal

By the end, I want to have something that makes it easy to build **multi-agent architectures** without hiding important details.

For example:
- A system that can support **deep research–style workflows**
- Multiple agents with clear responsibilities
- Explicit handoffs 

## Philosophy

I just want to understand the internals well enough that I can confidently design and extend agent systems when needed.

<br><br>
