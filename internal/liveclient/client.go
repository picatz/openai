// Package liveclient implements the GPT-Live primary WebSocket lifecycle.
// It does not use the different Realtime API protocol or access audio devices.
// Reference: https://developers.openai.com/api/docs/guides/voice-websockets?api=live
package liveclient

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"time"
)

type Socket interface {
	Read(context.Context) ([]byte, error)
	Write(context.Context, []byte) error
	Close() error
}
type SessionConfig struct {
	Model        string      `json:"model"`
	Instructions string      `json:"instructions,omitempty"`
	Store        bool        `json:"store"`
	Audio        AudioConfig `json:"audio"`
	Delegation   Delegation  `json:"delegation"`
}
type AudioConfig struct {
	Format Format `json:"format"`
	Output Voice  `json:"output"`
}
type Format struct {
	Type string `json:"type"`
	Rate int    `json:"rate"`
}
type Voice struct {
	Voice string `json:"voice"`
}
type Delegation struct {
	Type      string        `json:"type"`
	Responses BackendConfig `json:"responses"`
}
type BackendConfig struct {
	Model           string `json:"model"`
	MaxOutputTokens int    `json:"max_output_tokens"`
}
type Command string

const (
	Mute   Command = "mute"
	Unmute Command = "unmute"
	Close  Command = "close"
)

type Event struct {
	Type          string `json:"type"`
	ClientEventID string `json:"client_event_id,omitempty"`
	Delta         string `json:"delta,omitempty"`
	Reason        string `json:"reason,omitempty"`
	Session       struct {
		ID string `json:"id"`
	} `json:"session,omitempty"`
	Usage *struct {
		Seconds *float64 `json:"seconds"`
	} `json:"usage,omitempty"`
	Error *struct {
		Message       string `json:"message"`
		ClientEventID string `json:"client_event_id"`
	} `json:"error,omitempty"`
	Event json.RawMessage `json:"event,omitempty"`
	Raw   json.RawMessage `json:"-"`
}
type Result struct {
	SessionID    string            `json:"session_id"`
	Finalized    bool              `json:"finalized"`
	VoiceSeconds float64           `json:"voice_seconds"`
	BackendUsage []json.RawMessage `json:"backend_usage,omitempty"`
	Reason       string            `json:"reason,omitempty"`
	InputMuted   bool              `json:"input_muted"`
}
type Options struct {
	Session              SessionConfig
	Input                io.ReadCloser
	Output               io.WriteCloser
	Controls             <-chan Command
	Events               chan<- Event
	EventDeliveryTimeout time.Duration
	MaxDuration          time.Duration
	ListenAfterEOF       time.Duration
	StartupTimeout       time.Duration
	CloseTimeout         time.Duration
	// FrameDuration is injectable for protocol tests; production uses 20 ms.
	FrameDuration time.Duration
}

func DefaultSession() SessionConfig {
	return SessionConfig{Model: "gpt-live-1", Audio: AudioConfig{Format: Format{Type: "audio/pcm", Rate: 24000}, Output: Voice{Voice: "marin"}}, Delegation: Delegation{Type: "responses", Responses: BackendConfig{Model: "gpt-6-luna", MaxOutputTokens: 1024}}}
}
func (o *Options) defaults() error {
	if o.Input == nil || o.Output == nil {
		return fmt.Errorf("Live requires explicit audio input and output streams")
	}
	if o.Session.Model == "" || o.Session.Audio.Output.Voice == "" || o.Session.Delegation.Type != "responses" || o.Session.Delegation.Responses.Model == "" {
		return fmt.Errorf("Live model, voice, and Responses backend model are required")
	}
	if o.Session.Delegation.Responses.MaxOutputTokens < 16 || o.Session.Delegation.Responses.MaxOutputTokens > 16384 {
		return fmt.Errorf("backend output token limit must be between 16 and 16384")
	}
	if len(o.Session.Instructions) > 16<<10 {
		return fmt.Errorf("Live conversation instructions exceed 16 KiB")
	}
	if o.Session.Store {
		return fmt.Errorf("recording storage is not supported by this client")
	}
	if o.Session.Audio.Format.Type != "audio/pcm" || (o.Session.Audio.Format.Rate != 16000 && o.Session.Audio.Format.Rate != 24000) {
		return fmt.Errorf("input must be mono PCM16 at 16000 or 24000 Hz")
	}
	if o.MaxDuration == 0 {
		o.MaxDuration = time.Minute
	}
	if o.MaxDuration < 0 || o.MaxDuration > 10*time.Minute {
		return fmt.Errorf("Live duration must be greater than zero and at most 10 minutes")
	}
	if o.StartupTimeout == 0 {
		o.StartupTimeout = 15 * time.Second
	}
	if o.CloseTimeout == 0 {
		o.CloseTimeout = 15 * time.Second
	}
	if o.FrameDuration == 0 {
		o.FrameDuration = 20 * time.Millisecond
	}
	if o.ListenAfterEOF < 0 || o.ListenAfterEOF > o.MaxDuration {
		return fmt.Errorf("listen-after duration must be nonnegative and no longer than the session limit")
	}
	if o.EventDeliveryTimeout == 0 {
		o.EventDeliveryTimeout = 250 * time.Millisecond
	}
	if o.EventDeliveryTimeout < 0 || o.EventDeliveryTimeout > time.Second {
		return fmt.Errorf("event delivery timeout must be within (0,1s]")
	}
	if o.StartupTimeout < 0 || o.CloseTimeout < 0 || o.FrameDuration < time.Millisecond || o.FrameDuration > time.Second {
		return fmt.Errorf("invalid Live time bounds")
	}
	return nil
}

type received struct {
	data []byte
	err  error
}
type audioFrame struct {
	data []byte
	eof  bool
	err  error
}

func Run(ctx context.Context, socket Socket, opts Options) (result Result, returnErr error) {
	if socket == nil {
		return result, fmt.Errorf("Live socket is required")
	}
	defer socket.Close()
	if opts.Input != nil {
		defer opts.Input.Close()
	}
	if opts.Output != nil {
		defer func() { returnErr = errors.Join(returnErr, opts.Output.Close()) }()
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	if err := opts.defaults(); err != nil {
		return result, err
	}

	// Parent cancellation initiates graceful finalization rather than killing the
	// receiver that must observe session.closed and final usage.
	ioCtx, stopIO := context.WithTimeout(context.WithoutCancel(ctx), opts.StartupTimeout+opts.MaxDuration+opts.CloseTimeout+5*time.Second)
	defer stopIO()
	release := context.AfterFunc(ioCtx, func() { socket.Close(); opts.Input.Close(); opts.Output.Close() })
	defer release()
	stopOutputOnCancel := context.AfterFunc(ctx, func() { opts.Output.Close() })
	defer stopOutputOnCancel()
	receivedEvents := make(chan received, 4)
	go func() {
		for {
			data, err := socket.Read(ioCtx)
			select {
			case receivedEvents <- received{data, err}:
			case <-ioCtx.Done():
				return
			}
			if err != nil {
				return
			}
		}
	}()
	write := func(value any) error {
		data, err := json.Marshal(value)
		if err != nil {
			return err
		}
		writeCtx, cancel := context.WithTimeout(ioCtx, 5*time.Second)
		defer cancel()
		return socket.Write(writeCtx, data)
	}
	if err := write(map[string]any{"type": "session.start", "event_id": "start", "session": opts.Session}); err != nil {
		return result, fmt.Errorf("start Live: %w", err)
	}
	startup := time.NewTimer(opts.StartupTimeout)
	defer startup.Stop()
	var duration, tail, closingTimer *time.Timer
	var durationC, tailC, closingC <-chan time.Time
	defer func() {
		for _, timer := range []*time.Timer{duration, tail, closingTimer} {
			if timer != nil {
				timer.Stop()
			}
		}
	}()
	audioCtx, stopAudio := context.WithCancel(ioCtx)
	defer stopAudio()
	frames := make(chan audioFrame, 1)
	started, closing := false, false
	discardOutput := false
	commands := opts.Controls
	parentDone := ctx.Done()
	var closeCause error
	var pendingID string
	var queued Command
	localMuted, desiredMuted := false, false
	notificationsDisabled := false
	var outputBytes, usageBytes int64
	var pendingMute bool
	var sequence int
	notify := func(event Event) error {
		if opts.Events == nil || notificationsDisabled {
			return nil
		}
		select {
		case opts.Events <- event:
			return nil
		default:
		}
		timer := time.NewTimer(opts.EventDeliveryTimeout)
		defer timer.Stop()
		select {
		case opts.Events <- event:
			return nil
		case <-ctx.Done():
			return ctx.Err()
		case <-ioCtx.Done():
			return ioCtx.Err()
		case <-timer.C:
			return fmt.Errorf("Live event consumer is too slow")
		}
	}

	sendControl := func(command Command) error {
		sequence++
		pendingID = fmt.Sprintf("input_%d", sequence)
		pendingMute = command == Mute
		return write(map[string]any{"type": "session.input_audio." + string(command), "event_id": pendingID})
	}

	beginClose := func(cause error) error {
		if closing {
			closeCause = errors.Join(closeCause, cause)
			return nil
		}
		closing = true
		closeCause = cause
		stopAudio()
		opts.Input.Close()
		parentDone = nil
		durationC = nil
		tailC = nil
		commands = nil
		if err := write(map[string]any{"type": "session.close"}); err != nil {
			return err
		}
		closingTimer = time.NewTimer(opts.CloseTimeout)
		closingC = closingTimer.C
		return nil
	}
	for {
		select {
		case <-ioCtx.Done():
			return result, errors.Join(closeCause, fmt.Errorf("Live exceeded its bounded transport lifetime; final usage is unconfirmed"))
		case <-parentDone:
			if !started {
				return result, ctx.Err()
			}
			if err := beginClose(ctx.Err()); err != nil {
				return result, errors.Join(ctx.Err(), err)
			}
		case <-startup.C:
			if !started {
				return result, fmt.Errorf("Live startup timed out before session.started")
			}
		case <-durationC:
			if err := beginClose(nil); err != nil {
				return result, err
			}
		case <-tailC:
			if err := beginClose(nil); err != nil {
				return result, err
			}
		case <-closingC:
			return result, errors.Join(closeCause, fmt.Errorf("Live finalization timed out before session.closed; final usage is unconfirmed"))
		case command, ok := <-commands:
			if !ok {
				commands = nil
				continue
			}
			if closing {
				continue
			}
			if command != Close && command != Mute && command != Unmute {
				err := fmt.Errorf("unknown Live control; expected mute, unmute, or close")
				if !started {
					return result, err
				}
				if closeErr := beginClose(err); closeErr != nil {
					return result, errors.Join(err, closeErr)
				}
				continue
			}
			if command == Mute {
				desiredMuted = true
				localMuted = true
			}
			if command == Unmute {
				desiredMuted = false
			}
			if !started {
				if command == Close {
					return result, fmt.Errorf("Live was closed during startup; final usage is unconfirmed")
				}
				queued = command
				continue
			}
			if command == Close {
				if err := beginClose(nil); err != nil {
					return result, err
				}
				continue
			}
			if pendingID != "" {
				queued = command
				continue
			}
			if err := sendControl(command); err != nil {
				return result, err
			}

		case frame := <-frames:
			if !started || closing {
				continue
			}
			if frame.err != nil {
				if err := beginClose(frame.err); err != nil {
					return result, errors.Join(frame.err, err)
				}
				continue
			}
			if frame.eof && tail == nil {
				tail = time.NewTimer(opts.ListenAfterEOF)
				tailC = tail.C
			}
			// Once mute is requested, stop locally immediately. Resume only after a
			// matching unmute acknowledgment. Muted playback sends silence, not the file.
			if localMuted {
				clear(frame.data)
			}
			if len(frame.data) > 0 {
				if err := write(map[string]any{"type": "session.input_audio.append", "audio": base64.StdEncoding.EncodeToString(frame.data)}); err != nil {
					return result, err
				}
			}
		case incoming := <-receivedEvents:
			if incoming.err != nil {
				return result, errors.Join(closeCause, fmt.Errorf("Live connection ended before session.closed; final usage is unconfirmed: %w", incoming.err))
			}
			var event Event
			if err := json.Unmarshal(incoming.data, &event); err != nil || event.Type == "" {
				return result, fmt.Errorf("invalid Live event JSON")
			}
			event.Raw = append(json.RawMessage(nil), incoming.data...)
			switch event.Type {
			case "session.started":
				if started || closing || event.Session.ID == "" {
					return result, fmt.Errorf("invalid or duplicate session.started")
				}
				started = true
				startup.Stop()
				result.SessionID = event.Session.ID
				duration = time.NewTimer(opts.MaxDuration)
				durationC = duration.C
				if queued != "" {
					if err := sendControl(queued); err != nil {
						return result, err
					}
					queued = ""
				}
				go streamPCM(audioCtx, opts.Input, opts.Session.Audio.Format.Rate, opts.FrameDuration, frames)
			case "session.input_audio.muted", "session.input_audio.unmuted":
				target := event.Type == "session.input_audio.muted"
				if event.ClientEventID == pendingID && pendingID != "" && target == pendingMute {
					result.InputMuted = target
					if !target && !desiredMuted {
						localMuted = false
					}
					pendingID = ""
					if queued != "" && !closing {
						if err := sendControl(queued); err != nil {
							return result, err
						}
						queued = ""
					}
				}
			case "session.output_audio.delta":
				if discardOutput {
					continue
				}
				if !started {
					return result, fmt.Errorf("received audio before session.started")
				}
				pcm, err := base64.StdEncoding.DecodeString(event.Delta)
				if err != nil || len(pcm)%2 != 0 {
					return result, fmt.Errorf("invalid PCM16 audio delta")
				}
				outputBytes += int64(len(pcm))
				maxOutput := int64(opts.Session.Audio.Format.Rate) * 2 * int64(opts.MaxDuration+opts.CloseTimeout) / int64(time.Second)
				if outputBytes > maxOutput {
					discardOutput = true
					if err := beginClose(fmt.Errorf("Live audio output exceeded the configured duration bound")); err != nil {
						return result, err
					}
					continue
				}
				n, err := opts.Output.Write(pcm)
				if err == nil && n != len(pcm) {
					err = io.ErrShortWrite
				}
				if err != nil {
					discardOutput = true
					cause := fmt.Errorf("write Live audio: %w", err)
					if ctx.Err() != nil {
						cause = ctx.Err()
					}
					if err := beginClose(cause); err != nil {
						return result, errors.Join(cause, err)
					}
				}
			case "session.usage.updated":
				if event.Usage == nil || event.Usage.Seconds == nil || *event.Usage.Seconds < 0 {
					return result, fmt.Errorf("invalid Live usage update")
				}
				result.VoiceSeconds = *event.Usage.Seconds

			case "response.event":
				var nested struct {
					Type     string `json:"type"`
					Response struct {
						Usage json.RawMessage `json:"usage"`
					} `json:"response"`
				}
				if json.Unmarshal(event.Event, &nested) == nil && nested.Type == "response.completed" && len(nested.Response.Usage) > 0 {
					usageBytes += int64(len(nested.Response.Usage))
					if usageBytes > 1<<20 {
						return result, fmt.Errorf("Live backend usage metadata exceeded 1 MiB")
					}
					result.BackendUsage = append(result.BackendUsage, append(json.RawMessage(nil), nested.Response.Usage...))
				}
			case "session.closed":
				if !started || event.Usage == nil || event.Usage.Seconds == nil || *event.Usage.Seconds < 0 {
					return result, fmt.Errorf("session.closed is missing startup or final usage")
				}
				result.Finalized = true
				result.VoiceSeconds = *event.Usage.Seconds
				result.Reason = event.Reason
				notificationErr := notify(event)
				return result, errors.Join(closeCause, ctx.Err(), notificationErr)

			case "error":
				message := "Live rejected an event"
				if event.Error != nil && event.Error.Message != "" {
					message = event.Error.Message
				}
				if !started {
					return result, fmt.Errorf("Live startup error: %s", message)
				}
				if err := beginClose(fmt.Errorf("Live error: %s", message)); err != nil {
					return result, err
				}
			}
			if err := notify(event); err != nil {
				notificationsDisabled = true
				if !started {
					return result, err
				}
				if closeErr := beginClose(err); closeErr != nil {
					return result, errors.Join(err, closeErr)
				}
			}
		}
	}
}
func streamPCM(ctx context.Context, input io.Reader, rate int, frameDuration time.Duration, frames chan<- audioFrame) {
	size := int(int64(rate) * 2 * int64(frameDuration) / int64(time.Second))
	size -= size % 2
	ticker := time.NewTicker(frameDuration)
	defer ticker.Stop()
	eof := false
	for {
		frame := audioFrame{data: make([]byte, size)}
		if !eof {
			n, err := io.ReadFull(input, frame.data)
			if err != nil && err != io.EOF && err != io.ErrUnexpectedEOF {
				frame.err = err
			} else if n%2 != 0 {
				frame.err = fmt.Errorf("PCM16 input ended with an incomplete sample")
			} else if err != nil {
				eof = true
				frame.eof = true
			}
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
		select {
		case <-ctx.Done():
			return
		case frames <- frame:
		}
		if frame.err != nil {
			return
		}
	}
}
