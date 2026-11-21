// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Run on the operator host; only the PageBroker capabilities query is exposed.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/ai-dynamo/snapshot/agent/internal/pagebroker"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "usage: trtllm-broker-health /absolute/control/socket")
		os.Exit(2)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	client := pagebroker.Client{ControlSocketPath: os.Args[1]}
	caps, err := client.Capabilities(ctx)
	if err == nil {
		err = json.NewEncoder(os.Stdout).Encode(map[string]bool{"custom_storage": caps.CustomStorageAvailable})
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
