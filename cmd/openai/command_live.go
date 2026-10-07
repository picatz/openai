package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/picatz/openai/internal/liveclient"
	"github.com/picatz/openai/internal/safefile"
	"github.com/picatz/openai/internal/terminal"
	"github.com/spf13/cobra"
)

type syncedAudioFile struct {
	*os.File
	once     sync.Once
	closeErr error
}

func (f *syncedAudioFile) Close() error {
	f.once.Do(func() { f.closeErr = errors.Join(f.File.Sync(), f.File.Close()) })
	return f.closeErr
}

func newLiveCommand(app *application) *cobra.Command {
	var inputPath, outputPath, model, backendModel, voice, instructions, eventLogPath string
	var rate, backendMaxTokens int
	var duration, listenAfter time.Duration
	var controls bool
	cmd := &cobra.Command{Use: "live", Short: "Run a bounded GPT-Live session with explicit PCM files", Args: cobra.NoArgs, RunE: func(cmd *cobra.Command, args []string) error {
		if app.webSearch {
			return fmt.Errorf("web-search tools are not configured in this bounded Live client")
		}
		if len(instructions) > 16<<10 {
			return fmt.Errorf("Live conversation instructions exceed 16 KiB")
		}
		if app.stream {
			return fmt.Errorf("Live already streams over its primary WebSocket; omit --stream")
		}
		if duration <= 0 || duration > 10*time.Minute || listenAfter < 0 || listenAfter > duration {
			return fmt.Errorf("duration must be within (0,10m] and listen-after within [0,duration]")
		}
		if rate != 16000 && rate != 24000 {
			return fmt.Errorf("sample rate must be 16000 or 24000 Hz")
		}
		if strings.TrimSpace(model) == "" || strings.TrimSpace(backendModel) == "" || strings.TrimSpace(voice) == "" {
			return fmt.Errorf("Live model, backend model, and voice are required")
		}
		if backendMaxTokens < 16 || backendMaxTokens > 16384 {
			return fmt.Errorf("backend-max-output-tokens must be between 16 and 16384")
		}
		if inputPath == "" || outputPath == "" || inputPath == "-" || outputPath == "-" {
			return fmt.Errorf("Live requires explicit regular input/output PCM file paths; stdin/stdout are not audio devices")
		}
		if _, err := liveclient.Endpoint(app.baseURL); err != nil {
			return err
		}
		input, err := safefile.OpenRegular(inputPath)
		if err != nil {
			return err
		}
		defer input.Close()
		info, err := input.Stat()
		if err != nil {
			return err
		}
		maxInput := int64(rate) * 2 * int64(duration) / int64(time.Second)
		if info.Size() == 0 || info.Size()%2 != 0 || info.Size() > maxInput {
			return fmt.Errorf("input must contain complete PCM16 samples and fit within the duration limit")
		}
		var header [12]byte
		n, _ := input.ReadAt(header[:], 0)
		if n == 12 && string(header[:4]) == "RIFF" && string(header[8:12]) == "WAVE" {
			return fmt.Errorf("input is WAV; convert to raw mono PCM16 at the selected sample rate first")
		}
		if _, err := os.Lstat(outputPath); err == nil {
			return fmt.Errorf("output file exists; choose a new path")
		} else if !os.IsNotExist(err) {
			return err
		}
		output, err := os.CreateTemp(filepath.Dir(outputPath), ".openai-live-*")
		if err != nil {
			return err
		}
		defer os.Remove(output.Name())
		sink := &syncedAudioFile{File: output}
		defer sink.Close()
		var events chan liveclient.Event
		var eventFile *os.File
		var logDone chan error
		if eventLogPath != "" {
			if eventLogPath == "-" {
				return fmt.Errorf("event-log must be a new regular file path")
			}
			audioAbs, _ := filepath.Abs(outputPath)
			logAbs, _ := filepath.Abs(eventLogPath)
			if audioAbs == logAbs {
				return fmt.Errorf("event-log and output-pcm must use different paths")
			}
			if _, err := os.Lstat(eventLogPath); err == nil {
				return fmt.Errorf("event log already exists")
			} else if !os.IsNotExist(err) {
				return err
			}
			eventFile, err = os.CreateTemp(filepath.Dir(eventLogPath), ".openai-live-events-*")
			if err != nil {
				return err
			}
			defer os.Remove(eventFile.Name())
			events = make(chan liveclient.Event, 16)
			logDone = make(chan error, 1)
			go func() {
				for event := range events {
					if event.Type == "session.output_audio.delta" {
						continue
					}
					if err := json.NewEncoder(eventFile).Encode(event.Raw); err != nil {
						logDone <- err
						return
					}
				}
				logDone <- nil
			}()
		}
		var finishLogOnce sync.Once
		var logErr error
		finishLog := func() error {
			finishLogOnce.Do(func() {
				if events != nil {
					close(events)
					logErr = errors.Join(<-logDone, eventFile.Sync(), eventFile.Close())
				}
			})
			return logErr
		}
		defer finishLog()
		controlCtx, stopControls := context.WithCancel(cmd.Context())
		defer stopControls()
		var commands <-chan liveclient.Command
		var finishControls func()
		if controls {
			commands, finishControls, err = liveControls(controlCtx, cmd.InOrStdin(), cmd.ErrOrStderr())
			if err != nil {
				return err
			}
			defer finishControls()
		}
		socket, err := liveclient.Dial(cmd.Context(), liveclient.ConnectionConfig{BaseURL: app.baseURL, APIKey: os.Getenv("OPENAI_API_KEY"), Organization: os.Getenv("OPENAI_ORG_ID"), Project: os.Getenv("OPENAI_PROJECT_ID")})
		if err != nil {
			return err
		}
		session := liveclient.DefaultSession()
		session.Model = model
		session.Audio.Format.Rate = rate
		session.Audio.Output.Voice = voice
		session.Instructions = instructions
		session.Delegation.Responses.Model = backendModel
		session.Delegation.Responses.MaxOutputTokens = backendMaxTokens
		result, runErr := liveclient.Run(cmd.Context(), socket, liveclient.Options{Session: session, Input: input, Output: sink, Controls: commands, MaxDuration: duration, ListenAfterEOF: listenAfter, Events: events})

		stopControls()
		runErr = errors.Join(runErr, finishLog())
		logSaved := false
		if eventFile != nil && logErr == nil {
			if err := os.Link(eventFile.Name(), eventLogPath); err != nil {
				runErr = errors.Join(runErr, fmt.Errorf("publish event log: %w", err))
			} else {
				logSaved = true
			}
		}
		saved := false
		if runErr == nil && result.Finalized {
			if err := os.Link(output.Name(), outputPath); err != nil {
				runErr = fmt.Errorf("publish Live audio without replacing data: %w", err)
			} else {
				saved = true
			}
		}
		report := struct {
			liveclient.Result
			OutputPath    string `json:"output_path,omitempty"`
			OutputSaved   bool   `json:"output_saved"`
			EventLogSaved bool   `json:"event_log_saved"`
		}{Result: result, OutputPath: outputPath, OutputSaved: saved, EventLogSaved: logSaved}
		if app.output == "json" {
			return errors.Join(runErr, json.NewEncoder(cmd.OutOrStdout()).Encode(report))
		}
		_, printErr := fmt.Fprintf(cmd.OutOrStdout(), "Session: %s\nFinal usage confirmed: %t\nVoice seconds: %.3f\nAudio saved: %t\n", result.SessionID, result.Finalized, result.VoiceSeconds, saved)
		if saved && printErr == nil {
			_, printErr = fmt.Fprintln(cmd.OutOrStdout(), outputPath)
		}
		return errors.Join(runErr, printErr)
	}}
	cmd.Flags().StringVar(&inputPath, "input-pcm", "", "Raw mono PCM16 input file (not WAV)")
	cmd.Flags().StringVar(&outputPath, "output-pcm", "", "New file for returned raw PCM16 audio")
	cmd.Flags().StringVar(&model, "model", "gpt-live-1", "Live voice model")
	cmd.Flags().StringVar(&backendModel, "backend-model", "gpt-6-luna", "Responses delegation model (no tools configured)")
	cmd.Flags().StringVar(&voice, "voice", "marin", "Live voice")
	cmd.Flags().StringVar(&instructions, "instructions", "", "Short conversation instructions")
	cmd.Flags().IntVar(&backendMaxTokens, "backend-max-output-tokens", 1024, "Maximum backend output tokens (16 to 16384)")
	cmd.Flags().IntVar(&rate, "sample-rate", 24000, "Raw PCM sample rate: 16000 or 24000")
	cmd.Flags().DurationVar(&duration, "duration", time.Minute, "Hard session duration limit (maximum 10m)")
	cmd.Flags().DurationVar(&listenAfter, "listen-after", 5*time.Second, "Time to send silence/listen after input EOF before closing")
	cmd.Flags().StringVar(&eventLogPath, "event-log", "", "Optional new JSON-lines file for control/transcript/backend events")
	cmd.Flags().BoolVar(&controls, "controls", false, "Read mute, unmute, and close commands from stdin")
	cmd.MarkFlagRequired("input-pcm")
	cmd.MarkFlagRequired("output-pcm")
	return cmd
}

func liveControls(ctx context.Context, input io.Reader, diagnostics io.Writer) (<-chan liveclient.Command, func(), error) {
	controlCtx, cancel := context.WithCancel(ctx)
	var closeReader io.Closer
	if file, ok := input.(*os.File); ok {
		info, err := file.Stat()
		if err != nil {
			cancel()
			return nil, nil, err
		}
		if !info.Mode().IsRegular() {
			reader, err := terminal.NewInput(controlCtx, file)
			if err != nil {
				cancel()
				return nil, nil, err
			}
			input = reader
			closeReader = reader
		}
	}
	done := make(chan struct{})
	commands := make(chan liveclient.Command, 4)
	go func() {
		defer close(done)
		defer close(commands)
		scanner := bufio.NewScanner(input)
		scanner.Buffer(make([]byte, 256), 4096)
		for scanner.Scan() {
			command := liveclient.Command(strings.ToLower(strings.TrimSpace(scanner.Text())))
			if command == "" {
				continue
			}
			if command != liveclient.Mute && command != liveclient.Unmute && command != liveclient.Close {
				command = "invalid"
			}
			select {
			case commands <- command:
			case <-controlCtx.Done():
				return
			}
		}
	}()
	finish := func() {
		cancel()
		if closeReader != nil {
			<-done
			closeReader.Close()
		}
	}
	return commands, finish, nil
}
