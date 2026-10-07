package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"
)

func main() {
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	code := run(ctx, os.Args[1:], os.Stdin, os.Stdout, os.Stderr)
	cancel()
	os.Exit(code)
}

// run keeps process exit and diagnostics testable without a real API or terminal.
func run(ctx context.Context, args []string, in io.Reader, out, errOut io.Writer) int {
	cmd := newRootCommand()
	cmd.SetArgs(args)
	cmd.SetIn(in)
	cmd.SetOut(out)
	cmd.SetErr(errOut)
	if err := cmd.ExecuteContext(ctx); err != nil {
		fmt.Fprintln(errOut, "openai:", err)
		if errors.Is(err, context.Canceled) {
			return 130
		}
		return 1
	}
	return 0
}
