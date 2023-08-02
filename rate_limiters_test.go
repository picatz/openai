package openai_test

import (
	"testing"
	"time"

	"github.com/picatz/openai"
	"golang.org/x/time/rate"
)

func TestNewChatRequestRateLimiter(t *testing.T) {
	limiter := openai.RateLimits.Chat.Requests

	// Verify that the rate limiter allows up to 3,500 requests per minute.
	if limiter.Limit() != rate.Every(1*time.Minute) {
		t.Fatalf("unexpected rate limit interval: got %v, want %v", limiter.Limit(), rate.Every(1*time.Minute))
	}
	if limiter.Burst() != 3500 {
		t.Fatalf("unexpected burst limit: got %d, want %d", limiter.Burst(), 3500)
	}

	// Now actually test the rate limiter.
	t.Log("making 3,500 requests")
	for i := 0; i < 3500; i++ {
		if !limiter.Allow() {
			t.Fatalf("unexpected rate limit at %d", i)
		}
	}

	// Now verify that the rate limiter blocks after 3,500 requests.
	t.Log("checking for rate limit")
	if limiter.Allow() {
		t.Fatalf("unexpected rate limit at %d", 3500)
	}
	t.Log("hit rate limit (yay!)")

	// Now verify that the rate limiter allows 3,500 requests after 1 minute.
	t.Log("waiting 1 minute")
	time.Sleep(1 * time.Minute)

	t.Log("making 3,500 requests one last time")
	for i := 0; i < 3500; i++ {
		if !limiter.Allow() {
			t.Fatalf("unexpected rate limit at %d", i)
		}
	}
}
