package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"mime"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"github.com/openai/openai-go/v3"
	"github.com/picatz/openai/internal/httpguard"
	"github.com/picatz/openai/internal/safefile"
	"github.com/spf13/cobra"
	"golang.org/x/term"
)

const maxAudioUploadBytes = 25_000_000
const maxSpeechBytes = 64 << 20

func newAudioCommand(app *application) *cobra.Command {
	cmd := &cobra.Command{Use: "audio", Short: "Transcribe audio files or generate speech (no device access)"}
	var transcriptionModel, language, prompt string
	transcribe := &cobra.Command{Use: "transcribe <file>", Short: "Transcribe an audio file up to 25 MB", Args: cobra.ExactArgs(1), RunE: func(cmd *cobra.Command, args []string) error {
		if app.stream {
			return fmt.Errorf("streaming transcription is not supported by this command")
		}
		if strings.TrimSpace(transcriptionModel) == "" {
			return fmt.Errorf("transcription model must not be empty")
		}
		extension := strings.ToLower(filepath.Ext(args[0]))
		allowed := map[string]string{".mp3": "audio/mpeg", ".mp4": "video/mp4", ".mpeg": "audio/mpeg", ".mpga": "audio/mpeg", ".m4a": "audio/mp4", ".wav": "audio/wav", ".webm": "audio/webm", ".flac": "audio/flac", ".ogg": "audio/ogg"}
		contentType, ok := allowed[extension]
		if !ok {
			return fmt.Errorf("unsupported audio extension %q", extension)
		}
		f, err := safefile.OpenRegular(args[0])
		if err != nil {
			return err
		}
		defer f.Close()
		info, err := f.Stat()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() {
			return fmt.Errorf("audio input must be a regular file")
		}
		if info.Size() <= 0 || info.Size() > maxAudioUploadBytes {
			return fmt.Errorf("audio file must be nonempty and no larger than 25 MB")
		}
		data, err := io.ReadAll(io.LimitReader(f, maxAudioUploadBytes+1))
		if err != nil {
			return err
		}
		if len(data) > maxAudioUploadBytes {
			return fmt.Errorf("audio file grew beyond 25 MB")
		}
		if transcriptionModel == "gpt-4o-transcribe-diarize" {
			return fmt.Errorf("diarization requires a separate chunking and speaker-aware workflow; choose another transcription model")
		}
		params := openai.AudioTranscriptionNewParams{Model: transcriptionModel, File: openai.File(bytes.NewReader(data), filepath.Base(args[0]), contentType), ResponseFormat: openai.AudioResponseFormatJSON}
		if language != "" {
			params.Language = openai.String(language)
		}
		if prompt != "" {
			params.Prompt = openai.String(prompt)
		}
		result, err := app.client.Audio.Transcriptions.New(cmd.Context(), params, httpguard.LimitJSONResponse(8<<20, 1<<20))
		if err != nil {
			return fmt.Errorf("transcribe audio: %w", err)
		}
		if result == nil {
			return fmt.Errorf("transcription endpoint returned no result")
		}
		var validated struct {
			Text *string `json:"text"`
		}
		if err := json.Unmarshal([]byte(result.RawJSON()), &validated); err != nil || validated.Text == nil {
			return fmt.Errorf("transcription response must be a JSON object with a text string")
		}
		if app.output == "json" {
			return json.NewEncoder(cmd.OutOrStdout()).Encode(json.RawMessage(result.RawJSON()))
		}
		_, err = fmt.Fprintln(cmd.OutOrStdout(), *validated.Text)
		return err
	}}
	transcribe.Flags().StringVar(&transcriptionModel, "model", "gpt-transcribe", "Transcription model")
	transcribe.Flags().StringVar(&language, "language", "", "Optional ISO-639-1 language code")
	transcribe.Flags().StringVar(&prompt, "prompt", "", "Optional transcription context")
	var speechModel, voice, format, output, instructions string
	var speed float64
	speech := &cobra.Command{Use: "speech [text]", Aliases: []string{"speak"}, Short: "Generate AI speech into a new audio file", RunE: func(cmd *cobra.Command, args []string) error {
		if app.output != "text" || app.stream {
			return fmt.Errorf("speech writes binary audio; use --file (and --format) rather than --output or --stream")
		}
		formats := map[string]bool{"mp3": true, "opus": true, "aac": true, "flac": true, "wav": true, "pcm": true}
		if !formats[format] {
			return fmt.Errorf("unsupported speech format %q", format)
		}
		if speed < 0.25 || speed > 4 || math.IsNaN(speed) || math.IsInf(speed, 0) {
			return fmt.Errorf("speech speed must be between 0.25 and 4")
		}
		if strings.TrimSpace(voice) == "" || strings.TrimSpace(speechModel) == "" {
			return fmt.Errorf("speech model and voice must not be empty")
		}
		if utf8.RuneCountInString(instructions) > 4096 {
			return fmt.Errorf("speech instructions exceed 4096 characters")
		}
		if instructions != "" && (speechModel == "tts-1" || speechModel == "tts-1-hd") {
			return fmt.Errorf("speech instructions require a model that supports them")
		}
		if strings.TrimSpace(output) == "" {
			return fmt.Errorf("--file must specify a new path or -")
		}
		if output == "-" {
			if f, ok := cmd.OutOrStdout().(*os.File); ok && term.IsTerminal(int(f.Fd())) {
				return fmt.Errorf("refusing to write binary audio to a terminal; redirect stdout or use a file")
			}
		} else {
			if _, err := os.Lstat(output); err == nil {
				return fmt.Errorf("output file exists; choose a new path")
			} else if !os.IsNotExist(err) {
				return err
			}
			dir, err := os.Stat(filepath.Dir(output))
			if err != nil {
				return err
			}
			if !dir.IsDir() {
				return fmt.Errorf("output parent must be a directory")
			}
		}
		text, err := readPrompt(cmd.Context(), cmd.InOrStdin(), args)
		if err != nil {
			return err
		}
		if utf8.RuneCountInString(text) > 4096 {
			return fmt.Errorf("speech text exceeds 4096 characters")
		}
		params := openai.AudioSpeechNewParams{Model: speechModel, Input: text, Voice: openai.AudioSpeechNewParamsVoiceUnion{OfString: openai.String(voice)}, ResponseFormat: openai.AudioSpeechNewParamsResponseFormat(format), Speed: openai.Float(speed)}
		if instructions != "" {
			params.Instructions = openai.String(instructions)
		}
		result, err := app.client.Audio.Speech.New(cmd.Context(), params, httpguard.LimitResponse(maxSpeechBytes, 1<<20))
		if err != nil {
			return fmt.Errorf("generate speech: %w", err)
		}
		defer result.Body.Close()
		contentType, _, err := mime.ParseMediaType(result.Header.Get("Content-Type"))
		if err != nil || !(strings.HasPrefix(contentType, "audio/") || contentType == "application/octet-stream") {
			return fmt.Errorf("speech endpoint returned non-audio content type %q", result.Header.Get("Content-Type"))
		}
		if output == "-" {
			n, err := io.Copy(cmd.OutOrStdout(), io.LimitReader(result.Body, maxSpeechBytes+1))
			if err != nil {
				return err
			}
			if n > maxSpeechBytes {
				return fmt.Errorf("speech exceeds 64 MiB")
			}
			if n == 0 {
				return fmt.Errorf("speech endpoint returned no audio")
			}
			return nil
		}
		tmp, err := os.CreateTemp(filepath.Dir(output), ".openai-speech-*")
		if err != nil {
			return err
		}
		defer os.Remove(tmp.Name())
		n, err := io.Copy(tmp, io.LimitReader(result.Body, maxSpeechBytes+1))
		if err != nil {
			tmp.Close()
			return err
		}
		if n > maxSpeechBytes {
			tmp.Close()
			return fmt.Errorf("speech exceeds 64 MiB")
		}
		if n == 0 {
			tmp.Close()
			return fmt.Errorf("speech endpoint returned no audio")
		}
		if err := tmp.Sync(); err != nil {
			tmp.Close()
			return err
		}
		if err := tmp.Close(); err != nil {
			return err
		}
		// Link publishes a complete file without overwriting a destination created
		// concurrently. The temporary file is on the same filesystem.
		if err := os.Link(tmp.Name(), output); err != nil {
			return fmt.Errorf("save speech without replacing existing data: %w", err)
		}
		_, err = fmt.Fprintln(cmd.OutOrStdout(), output)
		return err
	}}
	speech.Flags().StringVar(&speechModel, "model", "gpt-4o-mini-tts", "Speech model")
	speech.Flags().StringVar(&voice, "voice", "marin", "Built-in voice")
	speech.Flags().StringVar(&format, "format", "mp3", "Audio format: mp3, opus, aac, flac, wav, pcm")
	speech.Flags().StringVar(&output, "file", "", "New output path, or - for binary stdout")
	speech.Flags().StringVar(&instructions, "instructions", "", "Optional speaking-style instructions")
	speech.Flags().Float64Var(&speed, "speed", 1, "Speech speed (0.25 to 4)")
	speech.MarkFlagRequired("file")
	cmd.AddCommand(transcribe, speech)
	return cmd
}
