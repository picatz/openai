package main

import (
	"fmt"

	"github.com/charmbracelet/glamour"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/spf13/cobra"
)

func newImageCommand(app *application) *cobra.Command {
	imageCommand := &cobra.Command{
		Use:   "image <prompt>",
		Short: "Generate an image with DALL·E",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			prompt := args[0]

			model := cmd.Flag("model").Value.String()
			quality := cmd.Flag("quality").Value.String()
			style := cmd.Flag("style").Value.String()
			size := cmd.Flag("size").Value.String()
			n, err := cmd.Flags().GetInt64("n")
			if err != nil {
				return err
			}

			resp, err := app.client.Images.Generate(cmd.Context(), openai.ImageGenerateParams{
				Prompt:  prompt,
				Model:   model,
				N:       param.NewOpt(n),
				Quality: openai.ImageGenerateParamsQuality(quality),
				Style:   openai.ImageGenerateParamsStyle(style),
				Size:    openai.ImageGenerateParamsSize(size),
			})
			if err != nil {
				return err
			}

			for _, data := range resp.Data {
				if data.JSON.RevisedPrompt.Raw() != "" {
					rp, err := glamour.Render(data.RevisedPrompt, "dark")
					if err != nil {
						return err
					}
					if _, err := fmt.Fprintln(cmd.OutOrStdout(), rp); err != nil {
						return err
					}
				}

				if _, err := fmt.Fprintln(cmd.OutOrStdout(), data.URL); err != nil {
					return err
				}
			}

			return nil
		},
	}

	imageCommand.Flags().String("quality", "hd", "image quality")
	imageCommand.Flags().String("style", "vivid", "image style")
	imageCommand.Flags().String("model", openai.ImageModelDallE3, "model to use")
	imageCommand.Flags().String("size", "1792x1024", "image size")
	imageCommand.Flags().Int64("n", 1, "number of images to generate")

	return imageCommand
}
