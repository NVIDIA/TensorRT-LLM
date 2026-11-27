// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"
)

func requireHostNamespaces(readlink func(string) (string, error)) error {
	for _, namespace := range []string{"pid", "cgroup", "time"} {
		self, err := readlink("/proc/self/ns/" + namespace)
		if err != nil {
			return err
		}
		host, err := readlink("/host/proc/1/ns/" + namespace)
		if err != nil {
			return err
		}
		if self != host {
			return fmt.Errorf("host driver must join host %s namespace", namespace)
		}
	}
	return nil
}

type operationDeadline struct {
	end time.Time
	now func() time.Time
}

func validDeadline(seconds float64) bool {
	return seconds > 0 && !math.IsInf(seconds, 0) && !math.IsNaN(seconds)
}

func bindDeadline(absolute float64, timeout int, now func() time.Time, monotonic func() (float64, error)) (operationDeadline, error) {
	if !validDeadline(absolute) || timeout < 1 || timeout > 7200 {
		return operationDeadline{}, errors.New("invalid host operation deadline or timeout")
	}
	// Sample Go's monotonic clock first: a delay between the samples can only shorten authority.
	anchor := now()
	host, err := monotonic()
	if err != nil {
		return operationDeadline{}, err
	}
	left := absolute - host
	if left <= 0 {
		return operationDeadline{}, context.DeadlineExceeded
	}
	if left > 7200 {
		return operationDeadline{}, errors.New("absolute host deadline is outside the supported window")
	}
	end := anchor.Add(time.Duration(min(left, float64(timeout)) * float64(time.Second)))
	return operationDeadline{end: end, now: now}, nil
}

func (d operationDeadline) remaining(ctx context.Context) (time.Duration, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	// A Go timer's cancellation callback need not be scheduled exactly at its deadline.
	left := d.end.Sub(d.now())
	if left <= 0 {
		return 0, context.DeadlineExceeded
	}
	return left, nil
}

func (d operationDeadline) check(ctx context.Context) error {
	_, err := d.remaining(ctx)
	return err
}

func (d operationDeadline) seconds(ctx context.Context) (int, error) {
	left, err := d.remaining(ctx)
	if err != nil {
		return 0, err
	}
	if left < time.Second {
		return 0, context.DeadlineExceeded
	}
	return int(left / time.Second), nil
}

func (d operationDeadline) start(ctx context.Context, claim func() error, invoke func(int) error) error {
	if _, err := d.seconds(ctx); err != nil {
		return err
	}
	if err := claim(); err != nil {
		return err
	}
	seconds, err := d.seconds(ctx)
	if err != nil {
		return err
	}
	return invoke(seconds)
}
