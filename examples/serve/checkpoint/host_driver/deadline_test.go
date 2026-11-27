// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"
)

func TestRequireHostTimeNamespace(t *testing.T) {
	for _, different := range []bool{false, true} {
		checked := false
		err := requireHostNamespaces(func(path string) (string, error) {
			if path == "/host/proc/1/ns/time" {
				checked = true
				if different {
					return "time:[2]", nil
				}
			}
			return "ns:[1]", nil
		})
		if !checked || (err != nil) != different {
			t.Fatalf("time namespace checked=%t mismatch=%t err=%v", checked, different, err)
		}
	}
}

func TestBindDeadlineChargesStartupAndPreflight(t *testing.T) {
	now := time.Unix(1, 0)
	anchor := now
	// The request was originally issued at host time 10 for 60s. Its helper starts at 64.25.
	d, err := bindDeadline(70, 60, func() time.Time { return now }, func() (float64, error) {
		now = now.Add(250 * time.Millisecond)
		return 64.25, nil
	})
	if err != nil || d.end.Sub(anchor) != 5750*time.Millisecond {
		t.Fatalf("invalid bound: %v, %v", d.end, err)
	}
	now = now.Add(2 * time.Second)
	invoked := false
	err = d.start(context.Background(), func() error {
		now = now.Add(750 * time.Millisecond)
		return nil
	}, func(seconds int) error {
		invoked = true
		if seconds != 2 {
			t.Fatalf("executor received %ds instead of 2 whole seconds", seconds)
		}
		return nil
	})
	if err != nil || !invoked {
		t.Fatalf("start: invoked=%t err=%v", invoked, err)
	}
}

func TestBindDeadlineRejectsInvalidOrExpiredInput(t *testing.T) {
	read := func() (float64, error) { return 64.25, nil }
	for _, end := range []float64{0, -1, 64.24, 64.25, math.NaN(), math.Inf(1), 64.25 + 7201} {
		if _, err := bindDeadline(end, 60, time.Now, read); err == nil {
			t.Fatalf("accepted absolute deadline %v", end)
		}
	}
	if _, err := bindDeadline(70, 60, time.Now, func() (float64, error) {
		return 0, errors.New("clock unavailable")
	}); err == nil {
		t.Fatal("accepted an unreadable monotonic clock")
	}
}

func TestDeadlinePreventsLateMutation(t *testing.T) {
	for _, stage := range []string{"before claim", "during claim", "context cancelled during claim", "subsecond budget"} {
		t.Run(stage, func(t *testing.T) {
			now := time.Unix(1, 0)
			d := operationDeadline{now.Add(5 * time.Second), func() time.Time { return now }}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			switch stage {
			case "before claim":
				now = d.end
			case "subsecond budget":
				now = d.end.Add(-time.Nanosecond)
			}
			claims := 0
			err := d.start(ctx, func() error {
				claims++
				if stage == "context cancelled during claim" {
					cancel()
				} else {
					now = d.end
				}
				return nil
			}, func(int) error {
				t.Fatal("executor dispatched after deadline or cancellation")
				return nil
			})
			want, wantClaims := context.DeadlineExceeded, 0
			switch stage {
			case "context cancelled during claim":
				want, wantClaims = context.Canceled, 1
			case "during claim":
				wantClaims = 1
			}
			if !errors.Is(err, want) || claims != wantClaims {
				t.Fatalf("claims=%d err=%v; want %d and %v", claims, err, wantClaims, want)
			}
		})
	}
}
