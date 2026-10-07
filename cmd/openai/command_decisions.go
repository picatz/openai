package main

import (
	"encoding/json"
	"fmt"
	"github.com/picatz/openai/internal/decisions"
	"github.com/spf13/cobra"
	"strings"
)

func newDecisionsCommand(app *application) *cobra.Command {
	cmd := &cobra.Command{Use: "decisions", Short: "Evaluate typed predicate, choice, or score questions (beta)"}
	for _, kind := range []string{"predicate", "choice", "score"} {
		var question, name, model string
		var choices, levels []string
		child := &cobra.Command{Use: kind + " [input]", Short: "Evaluate a " + kind + " question", RunE: func(cmd *cobra.Command, args []string) error {
			if app.stream {
				return fmt.Errorf("Decisions streaming is not supported by this command")
			}
			input, err := readPrompt(cmd.Context(), cmd.InOrStdin(), args)
			if err != nil {
				return err
			}
			q := decisions.Question{Type: kind, Name: name, Instructions: question}
			for _, value := range choices {
				label, description, ok := strings.Cut(value, "=")
				if !ok {
					return fmt.Errorf("choice must be value=description")
				}
				q.Choices = append(q.Choices, decisions.Choice{Value: strings.TrimSpace(label), Description: strings.TrimSpace(description)})
			}
			for _, value := range levels {
				label, description, ok := strings.Cut(value, "=")
				if !ok {
					return fmt.Errorf("level must be label=description")
				}
				q.Levels = append(q.Levels, decisions.Level{Label: strings.TrimSpace(label), Description: strings.TrimSpace(description)})
			}
			result, err := decisions.Create(cmd.Context(), &app.client, decisions.Request{Model: model, Input: input, Questions: []decisions.Question{q}})
			if err != nil {
				return err
			}
			if app.output == "json" {
				return json.NewEncoder(cmd.OutOrStdout()).Encode(result.Raw)
			}
			a := result.Answers[0]
			switch a.Type {
			case "refusal":
				_, err = fmt.Fprintf(cmd.OutOrStdout(), "%s\trefusal", a.Name)
			case "predicate":
				_, err = fmt.Fprintf(cmd.OutOrStdout(), "%s\tprobability=%.6f\n", a.Name, *a.Probability)
			case "choice":
				_, err = fmt.Fprintf(cmd.OutOrStdout(), "%s\t%s", a.Name, a.Choice)
			case "score":
				_, err = fmt.Fprintf(cmd.OutOrStdout(), "%s\tscore=%.6f", a.Name, *a.Score)
			}
			if err != nil {
				return err
			}
			if a.Type != "predicate" {
				if a.Confidence != nil {
					_, err = fmt.Fprintf(cmd.OutOrStdout(), "\tconfidence=%.6f", *a.Confidence)
					if err != nil {
						return err
					}
				}
				_, err = fmt.Fprintln(cmd.OutOrStdout())
			}
			return err
		}}
		child.Flags().StringVar(&model, "model", "gpt-6-luna", "Decisions model")
		child.Flags().StringVar(&question, "question", "", "Question instructions")
		child.Flags().StringVar(&name, "name", "result", "Answer name")
		if kind == "choice" {
			child.Flags().StringArrayVar(&choices, "choice", nil, "Choice value=description (repeat in order)")
		}
		if kind == "score" {
			child.Flags().StringArrayVar(&levels, "level", nil, "Score level label=description (repeat from lowest to highest)")
		}
		child.MarkFlagRequired("question")
		cmd.AddCommand(child)
	}
	return cmd
}
