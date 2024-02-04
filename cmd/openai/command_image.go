package main

import (
	"fmt"

	"github.com/charmbracelet/glamour"
	"github.com/picatz/openai"
	"github.com/spf13/cobra"
)

var imageCommand = &cobra.Command{
	Use:   "image <prompt>",
	Short: "Generate an image with DALL·E",
	Args:  cobra.ExactArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		prompt := args[0]

		model := cmd.Flag("model").Value.String()
		quality := cmd.Flag("quality").Value.String()
		style := cmd.Flag("style").Value.String()
		size := cmd.Flag("size").Value.String()
		n, err := cmd.Flags().GetInt("n")
		if err != nil {
			return err
		}

		resp, err := client.CreateImage(cmd.Context(), &openai.CreateImageRequest{
			Prompt:  prompt,
			Model:   model,
			N:       n,
			Quality: quality,
			Size:    size,
			Style:   style,
		})
		if err != nil {
			return err
		}

		for _, data := range resp.Data {
			if data.RevisedPrompt != nil {
				rp, err := glamour.Render(*data.RevisedPrompt, "dark")
				if err != nil {
					return err
				}
				fmt.Println(rp)
			}

			fmt.Println(*data.URL)
		}

		return nil
	},
}

func init() {
	imageCommand.Flags().String("quality", "hd", "image quality")
	imageCommand.Flags().String("style", "vivid", "image style")
	imageCommand.Flags().String("model", openai.ModelDallE3, "model to use")
	imageCommand.Flags().String("size", "1792x1024", "image size")
	imageCommand.Flags().Int("n", 1, "number of images to generate")

	rootCmd.AddCommand(imageCommand)
}
