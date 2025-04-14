# OpenAI [![Go Reference](https://pkg.go.dev/badge/github.com/picatz/openai.svg)](https://pkg.go.dev/github.com/picatz/openai) [![Go Report Card](https://goreportcard.com/badge/github.com/picatz/openai)](https://goreportcard.com/report/github.com/picatz/openai) [![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0) 
 
An unofficial community-maintained CLI application for [OpenAI](https://openai.com/).

## Installation

```console
$ go install github.com/picatz/openai/cmd/openai@latest
```

> [!IMPORTANT] 
> To use the CLI you must have a valid `OPENAI_API_KEY` environment variable set. You can get one [here](https://platform.openai.com/).

> [!TIP]
> You can customize which model is used by setting the `OPENAI_MODEL` environment variable. The default is `gpt-4o` today, but it may change in the future.

### Usage

```console
$ openai --help
OpenAI CLI

Usage:
  openai [flags]
  openai [command]

Available Commands:
  assistant   Start an interactive assistant chat session
  chat        Chat with the OpenAI API
  completion  Generate the autocompletion script for the specified shell
  help        Help about any command
  image       Generate an image with DALL·E
  responses   Manage the OpenAI Responses API

Flags:
  -h, --help   help for openai

Use "openai [command] --help" for more information about a command.
```

```console
$ openai assistant --help
Interact with the OpenAI API using the assistant API.

This can be used to create a temporary assistant, or interact with an existing assistant.

Usage:
  openai assistant [flags]
  openai assistant [command]

Examples:
  $ openai assistant      # create a temporary assistant and start chatting
  $ openai assistant chat # same as above
  $ openai assistant create --name "Example" --model "gpt-4-turbo-preview" --description "..." --instructions "..." --code-interpreter --retrieval
  $ openai assistant list
  $ openai assistant info <assistant-id>
  $ openai assistant chat <assistant-id>
  $ openai assistant delete <assistant-id>

Available Commands:
  chat        Start an interactive assistant chat session
  create      Create an assistant
  delete      Delete an assistant
  file        Manage assistant files
  info        Get information about an assistant
  list        List assistants
  update      Update an assistant

Flags:
  -h, --help   help for assistant

Use "openai assistant [command] --help" for more information about a command.
```

> [!TIP]
>
> If provided no arguments, the CLI will default to the `assistant` command with an ephemeral session,
> meaning messages and files will be deleted after exiting the session.

#### With Ollama

You can use the CLI with [Ollama](https://ollama.com/) to use models that are run locally, such as [IBM Granite](https://ollama.com/library/granite3.1-dense).

```console
$ brew install ollama
...
$ ollama serve &
...
$ ollama run granite3.1-dense:2b
...
$ OPENAI_MODEL="granite3.1-dense:2b" OPENAI_EMBEDDING_MODEL="granite3.1-dense:2b" OPENAI_API_URL="http://localhost:11434/v1/" openai chat
```