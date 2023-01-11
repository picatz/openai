# OpenAI
 
> **Warning**: this is a work in progress Go client implementation for OpenAI's API.

```console
$ go get -v github.com/picatz/openai
```

```go
import "github.com/picatz/openai"

client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))
```

```go
resp, _ := client.CreateCompletion(ctx, &openai.CreateComletionRequest{
	Model:     openai.ModelDavinci,
	Prompt:    []string{"This is a test"},
	MaxTokens: 5,
})

for _, choice := range resp.Choices {
    fmt.Println(choice.Text)
}
```

```go
resp, _ := client.CreateEdit(ctx, &openai.CreateEditRequest{
	Model:       openai.ModelTextDavinciEdit001,
	Instruction: "Change the word 'test' to 'example'",
	Input:       "This is a test",
})

for _, choice := range resp.Choices {
    fmt.Println(choice.Index, choice.Text)
}
```

```go
resp, _ := client.CreateImage(ctx, &openai.CreateImageRequest{
	Prompt:         "Golang-style gopher mascot wearing an OpenAI t-shirt",
	N:              1,
	Size:           "256x256",
	ResponseFormat: "url",
})

fmt.Println(*resp.Data[0].URL)
```

### `openai` CLI

Use OpenAI's edit and completion features on the command-line.

```console
$ go install github.com/picatz/openai/cmd/openai@latest
```

> **Note**: do not rely on this for reliable output.

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
