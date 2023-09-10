# OpenAI [![Go Reference](https://pkg.go.dev/badge/github.com/picatz/openai.svg)](https://pkg.go.dev/github.com/picatz/openai) [![Go Report Card](https://goreportcard.com/badge/github.com/picatz/openai)](https://goreportcard.com/report/github.com/picatz/openai) [![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0) 
 
An unofficial community-maintained Go client package and CLI for OpenAI.

## Installation

To use this package in your own Go project:

```console
$ go get github.com/picatz/openai@latest
```

To use the `openai` CLI:

```console
$ go install github.com/picatz/openai/cmd/openai@latest
```

> **Note**
> To use this package, you must have a valid OpenAI API key. You can get one [here](https://platform.openai.com/).

## Usage

```go
import "github.com/picatz/openai"

client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
```

```go
resp, _ := client.CreateCompletion(ctx, &openai.CreateComletionRequest{
	Model:     openai.ModelDavinci,
	Prompt:    []string{"Once upon a time"},
	MaxTokens: 5,
})

fmt.Println(resp.Choices[0].Text)
// Once upon a time ...max of 5 tokens...
```

```go
resp, _ := client.CreateEdit(ctx, &openai.CreateEditRequest{
	Model:       openai.ModelTextDavinciEdit001,
	Instruction: "Change the word 'test' to 'example'",
	Input:       "This is a test",
})

fmt.Println(resp.Choices[0].Text)
// This is an example
```

```go
resp, _ := client.CreateImage(ctx, &openai.CreateImageRequest{
	Prompt:         "Golang-style gopher mascot wearing an OpenAI t-shirt",
	N:              1,
	Size:           "256x256",
	ResponseFormat: "url",
})

fmt.Println(*resp.Data[0].URL)
// https://...
```

```go
var history []openai.ChatMessage{
	{
		Role:    openai.ChatRoleSystem,
		Content: "You are a helpful assistant for this example.",
	},
	{
		Role:    openai.ChatRoleUser,
		Content: "Hello!", // Get input from user.
	},
}

resp, _ := client.CreateChat(ctx, &openai.CreateChatRequest{
	Model: openai.ModelGPT35Turbo,
	Messages: history,
})

fmt.Println(resp.Choices[0].Message.Content)
// Hello how may I help you today?

// Update history, summarize, forget, etc. Then repeat.
history = appened(history, resp.Choices[0].Message)
```

### `openai` CLI

Use OpenAI's chat or edit and completion features on the command-line.

```console
$ go install github.com/picatz/openai/cmd/openai@latest
```

#### Modes

* **Chat**: chat mode is enabled using the `chat` command line argument. This is like ChatGPT for your command-line.
* **Edit**: edit mode is enabled when you pipe (`|`) information into the program's STDIN.
* **Complete**: completion mode is enabled when one or more command line arguments are provided.

> **Note**: do not rely on this for reliable output.

```console
$ openai chat

Welcome to the OpenAI API CLI chat mode. Type 'exit' to quit.

> Hello!

Hello there! How may I assist you today?

> How to generate SSH key?

To generate an SSH key, follow these steps:

1. Open the Terminal on your computer.
2. Type in the following command and press 
...
```

```console
$ echo "1,2,3,4,5" | openai convert to JSON                  
[1,2,3,4,5]
```

```console
$ echo "hello world" | openai capitalize        
Hello World
```

```console
$ echo "eye lov golang" | openai spellcheck
I love golang
```

```console
$ openai list of 5 prime numbers
2, 3, 5, 7, 11
```

```console
$ openai name the two founders of apple          
Steve Jobs and Steve Wozniak
```

```console
$ openai what is command to show who is logged in on unix systems
The command to show who is logged in on Unix systems is "who".
```

```console
$ openai simple terraform resource for an AWS S3 bucket    
resource "aws_s3_bucket" "my_s3_bucket" {
  bucket = "my-s3-bucket"
  acl    = "private"
}
```

```console
$ openai generate SSH key                           
1. Open a terminal
2. Run the ssh-keygen command
3. Enter the desired file in which to save the key when prompted
4. Enter a secure passphrase when prompted
5. Your key will now be generated
```

```console
$ openai generate SSH key PEM private key output
-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEAz2HyiiLE...3n3h1fwuI1MGKNw==
-----END RSA PRIVATE KEY-----
```

```console
$ openai markdown table of prices                    
| Item                        | Price  |
| --------------------------- | ------ |
| Apple                       | $1.00  |
| Banana                      | $0.50  |
| Orange                      | $0.75  |
| Pear                        | $1.25  |
| Pineapple                   | $2.00  |
```

```console
$ openai how do I convert from f to c 
To convert from Fahrenheit (F) to Celsius (C), use the following formula:
C = (F - 32) / 1.8

For example, if the temperature is 100°F, then the temperature in Celsius is 37.78°C.
C = (100 - 32) / 1.8
C = 68 / 1.8
C = 37.78°C
```

```console
$ openai ascii art
░░░░░▄▄▄▄▀▀▀▀▀▀▀▀▄▄▄▄▄▄
░░░░░█░░░░▒▒▒▒▒▒▒▒▒▒▒▒░░▀▀▄
░░░░█░░░▒▒▒▒▒▒░░░░░░░░▒▒▒░░█
░░░█░░░░░░▄██▀▄▄░░░░░▄▄▄░░░█
░▄▀▒▄▄▄▒░█▀▀▀▀▄▄█░░░██▄▄█░░░█
█░▒█▒▄░▀▄▄▄▀░░░░░░░░█░░░▒▒▒▒▒█
█░▒█░█▀▄▄░░░░░█▀░░░░▀▄░░▄▀▀▀▄▒█
░█░▀▄░█▄░█▀▄▄░▀░▀▀░▄▄▀░░░░█░░█
░░█░░░▀▄▀█▄▄░█▀▀▀▄▄▄▄▀▀█▀██░█
░░░█░░░░██░░▀█▄▄▄█▄▄█▄████░█
░░░░█░░░░▀▀▄░█░░░█░█▀██████░█
░░░░░▀▄░░░░░▀▀▄▄▄█▄█▄█▄█▄▀░░█
░░░░░░░▀▄▄░▒▒▒▒░░░░░░░░░░▒░░░█
░░░░░░░░░░▀▀▄▄░▒▒▒▒▒▒▒▒▒▒░░░░█
░░░░░░░░░░░░░░▀▄▄▄▄▄░░░░░░░░█
```

```console
$ openai ascii art cat           
  \    /\
  )  ( ')
 (  /  )
  \(__)|
```

```console
$ openai ascii art cat with ball of yarn
  /\_/\
 ( o.o )
 > ^ <
 (  _  )
 /   \/
  \_^_/
   \_/
```

```console
$ openai ascii art picat
 
            _.-^^---....,,--
        _--                  --_
       <                        >)
       |                         |
        \._                   _./
           ```--. . , ; .--'''
               | |   |
            .-=||  | |=-.
            `-=#$%&%$#=-'
               | ;  :|
            _.-' \_/ \`-._
        - -^\$\$\$\$\$\$\$\$\$/^^- -
           _-^ \$$\$/\$$\$\$-_
          <   \$\$ | >-\$-   >
           \$\$\$  |  \$\$\$\$
            `\$\$\$\$--\$\$\$\$'
                |  |
                |  |
                |  |
                |  |
                |  |
        _____/ \__/ \_____
```
