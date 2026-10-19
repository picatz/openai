package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
)

// livePreflightReport describes only checks that can be made without starting a
// session. Raw PCM has no metadata to verify its actual format or sample rate.
type livePreflightReport struct {
	Check                  string   `json:"check"`
	LocalValidation        string   `json:"local_validation"`
	InputPath              string   `json:"input_path"`
	InputBytes             int64    `json:"input_bytes"`
	InputSeconds           float64  `json:"input_seconds_at_selected_rate"`
	OutputPath             string   `json:"output_path"`
	EventLogPath           string   `json:"event_log_path,omitempty"`
	FormatAssumption       string   `json:"format_assumption"`
	SampleRate             int      `json:"sample_rate"`
	Model                  string   `json:"model"`
	BackendModel           string   `json:"backend_model"`
	Voice                  string   `json:"voice"`
	BackendMaxOutputTokens int      `json:"backend_max_output_tokens"`
	Duration               string   `json:"duration"`
	ListenAfter            string   `json:"listen_after"`
	Controls               bool     `json:"controls"`
	Unverified             []string `json:"unverified"`
}

func validateLiveEventLogPaths(outputPath, eventLogPath string) error {
	if eventLogPath == "" {
		return nil
	}
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
	return nil
}

func reportLivePreflight(cmd *cobra.Command, output string, report livePreflightReport) error {
	// Existing-path rejection is shared with the normal run. Here, inspect only
	// parent metadata: creating a probe file would violate the offline contract,
	// and mode bits alone cannot prove that the later write will succeed.
	for _, path := range []string{report.OutputPath, report.EventLogPath} {
		if path == "" {
			continue
		}
		info, err := os.Stat(filepath.Dir(path))
		if err != nil {
			return fmt.Errorf("check destination parent: %w", err)
		}
		if !info.IsDir() {
			return fmt.Errorf("destination parent must be an existing directory")
		}
	}
	report.Check = "offline"
	report.LocalValidation = "passed"
	report.FormatAssumption = "raw mono signed 16-bit little-endian PCM (not resampled or fully decoded)"
	report.InputSeconds = float64(report.InputBytes) / (float64(report.SampleRate) * 2)
	report.Unverified = []string{
		"actual audio format, sample rate, and quality",
		"output/event-log write permissions and later path changes",
		"credentials and service/model/voice availability",
		"microphone and playback hardware (not used by this file client)",
	}
	if output == "json" {
		return json.NewEncoder(cmd.OutOrStdout()).Encode(report)
	}
	_, err := fmt.Fprintf(cmd.OutOrStdout(), "Local file/configuration checks passed (offline).\nInput: %s (%d bytes, %.3fs at %d Hz)\nFormat assumed: %s\nOutput: %s\nLive model: %s; voice: %s; backend: %s (%d max output tokens)\nDuration limit: %s; listen after EOF: %s; controls: %t (not read)\n",
		report.InputPath, report.InputBytes, report.InputSeconds, report.SampleRate,
		report.FormatAssumption, report.OutputPath, report.Model, report.Voice, report.BackendModel,
		report.BackendMaxOutputTokens, report.Duration, report.ListenAfter, report.Controls)
	if err != nil {
		return err
	}
	if report.EventLogPath != "" {
		if _, err := fmt.Fprintln(cmd.OutOrStdout(), "Event log:", report.EventLogPath); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintln(cmd.OutOrStdout(), "No network connection, output creation, or device access. Unverified:"); err != nil {
		return err
	}
	for _, item := range report.Unverified {
		if _, err := fmt.Fprintln(cmd.OutOrStdout(), "-", item); err != nil {
			return err
		}
	}
	return nil
}
